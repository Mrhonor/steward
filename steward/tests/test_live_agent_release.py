"""Run the shipped Ego Harness against one lost release command in MuJoCo."""

import json
import math
import os
import shutil
import time
from pathlib import Path

import pytest

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.world.chat import ChatConfig, load_env_file
from steward.world.server import StewardHost


STEWARD = Path(__file__).resolve().parents[1]
ROBOT = "w:robot-1"
CUP = "w:cup-1"
TABLE = "w:table-1"
TASK = "请把杯子放到餐桌上。"


class DropFirstCupRelease(RobotLink):
    """Discard one open command for the cup over the table at the simulated actuator link."""

    def __init__(self, url, sim):
        super().__init__(url)
        self.sim = sim
        self.dropped = None

    def command(self, cmd):
        opening = cmd.get("gripper")
        if isinstance(opening, dict):
            opening, = opening.values()
        if opening == "open" and self.dropped is None:
            with self.sim.lock:
                if self.sim.robots[ROBOT].attached == CUP:
                    cup = self.sim._entity_pose(CUP)["p"]
                    table = self.sim.entity_obb(TABLE)
                    if all(abs(cup[i] - table.center[i]) < table.size[i] / 2 for i in (0, 1)):
                        self.dropped = {"stamp": self.sim.time, "cupPosition": cup}
                        cmd = {key: value for key, value in cmd.items() if key != "gripper"}
        super().command(cmd)


def _cup_on_table(world):
    return any(r["type"] == "supported-by" and r["a"]["entity"] == CUP and r["b"]["entity"] == TABLE
               for r in world["relations"])


@pytest.mark.skipif(os.environ.get("STEWARD_LIVE_AGENT") != "1", reason="requires a live DeepSeek run")
def test_deepseek_places_the_cup_after_one_release_command_is_lost(kitchen_root, tmp_path):
    env_file = STEWARD / ".env"
    assert (load_env_file(env_file).get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")), "configure DeepSeek"
    launcher = STEWARD / "dsh/node_modules/.bin" / ("dsh.CMD" if os.name == "nt" else "dsh")
    assert launcher.is_file() and shutil.which("node"), "install the Steward DSH workspace and put Node on PATH"
    source_home = Path(os.environ.get("STEWARD_DSH_VAR", STEWARD / "var/dsh")) / "robot-1"
    source = (source_home / "profiles/ego").resolve()
    assert (source / "node_modules").is_dir(), "set up the Steward ego profile"

    home = tmp_path / "robot"
    profile = home / "profiles/ego"
    shutil.copytree(source, profile, ignore=shutil.ignore_patterns("node_modules"))
    modules = profile / "node_modules"
    if os.name == "nt":
        from _winapi import CreateJunction

        CreateJunction(str(source / "node_modules"), str(modules))
    else:
        modules.symlink_to(source / "node_modules", target_is_directory=True)

    host = runtime = api = None
    try:
        config = ChatConfig(dsh=launcher, cwd=tmp_path, env_file=env_file, robot_home=home, timeout_s=240)
        host = StewardHost(kitchen_root, scenario="phase0-kitchen", speed=math.inf, seed=11,
                           run_dir=tmp_path, chat=config, freeze_idle=True).start()
        assert not _cup_on_table(host.sim.world_state())
        link = DropFirstCupRelease(host.robot_servers[ROBOT].url, host.sim)
        runtime = EgoRuntime(link, steward_bot().to_dict())
        api = ego_api(runtime).start()
        config.ego_api = api.url
        host.attach_ego_api(ROBOT, api.url)
        runtime.start()
        host.chat.send("robot", TASK)
        deadline = time.monotonic() + config.timeout_s + 30
        while time.monotonic() < deadline:
            host.pump(0.05)
            job = runtime.executor.current
            if not host.chat.agents["robot"].busy and (job is None or job.finished is not None):
                break

        chat = host.chat.state()
        with host.sim.lock:
            world = host.sim.world_state()
            actual_held = host.sim.robots[ROBOT].attached
        jobs = [job.to_dict() for job in runtime.executor.jobs.values()]
        result = {"task": TASK, "fault": link.dropped, "cupOnTable": _cup_on_table(world),
                  "actualHeld": actual_held, "egoHeld": runtime.estimator.held_id,
                  "jobs": jobs, "chat": chat}
        evidence = tmp_path / "release-evidence.json"
        evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Evidence: {evidence}")
        assert not host.chat.agents["robot"].busy, f"Harness still running; see {evidence}"
        assert not any(m["role"] == "error" for m in chat["messages"]), f"Harness error; see {evidence}"
        assert any(m["role"] == "tool" and m["status"] == "completed" for m in chat["messages"]), evidence
        assert link.dropped is not None, f"Agent never attempted to release the cup over the table; see {evidence}"
        assert result["cupOnTable"] and actual_held != CUP, f"Cup not placed after a single lost release; see {evidence}"
    finally:
        if host is not None:
            host.chat.stop()
        if api is not None:
            api.stop()
        if runtime is not None:
            runtime.stop()
        if host is not None:
            host.stop()
        if os.name == "nt":
            modules.rmdir()
        else:
            modules.unlink()
