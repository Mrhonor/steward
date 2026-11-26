"""Measure scripted retries through Ego Runtime with deterministic MuJoCo faults."""

import json
import math
import threading

import pytest

from steward.assets.generators import building, dining_table, mug, printed_board, steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.net import HttpError
from steward.world.sim import PerceptionConfig, WorldSim


ROBOT = "w:robot"
TARGET = "w:target"
DECOY = "w:decoy"
TABLE = "w:table"


class FaultLink(RobotLink):
    """Step physics on demand and expose ideal detections without object references."""

    def __init__(self, sim):
        super().__init__("http://unused")
        self.sim = sim
        self.poll = None
        self.next_poll = 0.0
        self.close_count = 0
        self.open_count = 0
        self.missed_closes = 0
        self.swap_at_close = False
        self.block_release = False
        self.stale_holding = False

    def sensors(self):
        frame = self.sim.sensors(ROBOT)
        for observation in frame["detections"]:
            observation.pop("objectRef", None)
        for grip in frame["grippers"].values():
            grip.pop("heldObjectRef", None)
            if self.stale_holding and grip["state"] == "open":
                grip["holding"] = True
        return frame

    def command(self, cmd):
        grip = cmd.get("gripper")
        if isinstance(grip, dict):
            grip, = grip.values()
        if grip == "close":
            self.close_count += 1
            if self.swap_at_close:
                self.swap_at_close = False
                self.sim.edit_move(TARGET, {"on": TABLE, "offset": [0.10, -0.22]})
                self.sim.edit_move(DECOY, {"on": TABLE, "offset": [-0.23, 0.0]})
        elif grip == "open":
            self.open_count += 1
            if self.block_release:
                cmd = {key: value for key, value in cmd.items() if key != "gripper"}
        self.sim.command(ROBOT, cmd)

    def arm(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass

    def wait_until(self, t=None, *args, **kwargs):
        if t is not None:
            self.sim.step_seconds(max(0.0, t - self.sim.time))
        if self.poll is not None and self.sim.time >= self.next_poll:
            self.poll()
            self.next_poll = self.sim.time + 0.1
        return self.sim.time


@pytest.fixture
def workbench(monkeypatch, request):
    monkeypatch.delenv("STEWARD_SEGMENTER_URL", raising=False)
    room, robot, table = (factory().to_dict() for factory in (building, steward_bot, dining_table))
    board = getattr(request, "param", None) == "board"
    target = (printed_board(category="puzzle_piece") if board else mug(color="black")).to_dict()
    decoy = mug(color="red").to_dict()
    entities = (room, robot, table, target, decoy)
    for entity, eid in zip(entities, ("w:room", ROBOT, TABLE, TARGET, DECOY), strict=True):
        entity["id"] = eid
    for entity, position in ((table, [1.13, 0.0, 0.0]), (target, [0.90, 0.0, 0.752]),
                             (decoy, [1.22, 0.22, 0.752])):
        entity["pose"]["value"]["p"] = position
    sim = WorldSim({"mode": "world", "entities": list(entities[:-1] if board else entities)}, seed=3,
                   perception=PerceptionConfig(odom_pos_sigma=0, odom_yaw_sigma=0,
                                               pos_sigma_base=0, pos_sigma_per_m=0, yaw_sigma=0))
    link = FaultLink(sim)
    runtime = EgoRuntime(link, robot)
    finished = threading.Event()
    on_finish = runtime._skill_finished
    attach = sim._try_attach

    def finish(job):
        try:
            on_finish(job)
        finally:
            finished.set()

    def attach_unless_missed(*args):
        if link.missed_closes is None:
            return
        if link.missed_closes:
            link.missed_closes -= 1
        else:
            attach(*args)

    monkeypatch.setattr(runtime.executor, "_on_finish", finish)
    monkeypatch.setattr(sim, "_try_attach", attach_unless_missed)
    try:
        link.poll = runtime._poll_once
        for _ in range(4):
            link.wait_until(sim.time + 0.12)
        yield runtime, link, finished
    finally:
        try:
            if runtime.executor.current is not None and not finished.is_set():
                runtime.executor.cancel()
                assert finished.wait(30), "skill worker did not finish after cancellation"
        finally:
            runtime.stop()
            sim.stop()


def _track(runtime, category, color=None):
    tracks = [t.id for t in runtime.estimator.tracks.values()
              if t.category == category and (color is None or t.color == color)]
    assert len(tracks) == 1, [(t.id, t.category, t.color) for t in runtime.estimator.tracks.values()]
    return tracks[0]


def _attempt(workbench, skill, params):
    runtime, link, finished = workbench
    receipt = runtime.predict(skill, params)
    plan = receipt.get("plan", {})
    row = {"skill": skill, "params": params, "issues": receipt["issues"],
           "plan": {key: plan[key] for key in ("targetP", "dock", "lift", "reach", "graspOffset", "spot")
                    if key in plan}}
    start, closes, opens = link.sim.time, link.close_count, link.open_count
    if receipt["withinRiskLimit"]:
        finished.clear()
        job = runtime.run(receipt["id"])
        job = runtime.wait(job["id"], 30)
        assert job["state"] != "running", (skill, job)
        row.update(job=job["id"], state=job["state"], message=job["message"],
                   postcondition=job.get("postcondition"))
    else:
        row["state"] = "prediction_refused"
    grip = link.sensors()["gripper"]
    row.update(simSeconds=round(link.sim.time - start, 3), closes=link.close_count - closes,
               opens=link.open_count - opens, egoHeld=runtime.estimator.held_id,
               gripperState=grip["state"], sensorHolding=grip["holding"],
               actualHeld=link.sim.robots[ROBOT].attached,
               targetPosition=link.sim._entity_pose(TARGET)["p"],
               egoMugs=[{"id": track.id, "color": track.color, "position": track.pos.tolist()}
                        for track in runtime.estimator.tracks.values() if track.category == "mug"])
    return row


@pytest.mark.parametrize("workbench", ["board"], indirect=True)
def test_pick_reaches_tabletop_board_height_before_first_close(workbench, monkeypatch):
    runtime, link, _ = workbench
    close_heights = []
    command = link.command

    def record_close(cmd):
        grip = cmd.get("gripper")
        if grip == "close" or isinstance(grip, dict) and "close" in grip.values():
            close_heights.append(link.sensors()["joints"]["lift"])
        command(cmd)

    monkeypatch.setattr(link, "command", record_close)
    result = _attempt(workbench, "pick", {"target": _track(runtime, "puzzle_piece")})
    assert close_heights == pytest.approx([result["plan"]["lift"]], abs=runtime.executor.config.joint_tol)
    assert result["state"] == "succeeded", result
    assert result["actualHeld"] == TARGET


@pytest.mark.parametrize("workbench", ["board"], indirect=True)
def test_pick_does_not_close_if_lift_drifts_after_settling(workbench, monkeypatch):
    runtime, _, _ = workbench
    executor = runtime.executor
    reach_over, wait, state = executor._reach_over, executor._wait, executor._state
    extended = False

    def reach(*args):
        nonlocal extended
        reach_over(*args)
        extended = True

    def wait_with_drift(seconds):
        nonlocal extended
        wait(seconds)
        if extended:
            extended = False
            feedback = state()
            feedback["joints"]["lift"] += 2 * executor.config.joint_tol

            def read_drift():
                monkeypatch.setattr(executor, "_state", state)
                return feedback

            monkeypatch.setattr(executor, "_state", read_drift)

    monkeypatch.setattr(executor, "_reach_over", reach)
    monkeypatch.setattr(executor, "_wait", wait_with_drift)
    result = _attempt(workbench, "pick", {"target": _track(runtime, "puzzle_piece")})
    assert result["state"] == "failed", result
    assert "did not settle" in result["message"], result
    assert result["closes"] == 0


def _record(record_property, attempts):
    evidence = json.dumps(attempts, ensure_ascii=False)
    record_property("attempts", evidence)
    print(evidence)


@pytest.mark.parametrize("failure", ["one_close", "one_job", "persistent"])
def test_pick_retry_after_empty_contact(workbench, record_property, failure):
    runtime, link, _ = workbench
    target = _track(runtime, "mug", "black")
    attempts = []
    for attempt in range(3):
        link.missed_closes = {"one_close": 1 if attempt == 0 else 0, "one_job": None if attempt == 0 else 0,
                              "persistent": None}[failure]
        attempts.append(_attempt(workbench, "pick", {"target": target}))
        if attempts[-1]["actualHeld"] == TARGET:
            break
    _record(record_property, attempts)
    assert [row["state"] for row in attempts] == {
        "one_close": ["failed", "succeeded"], "one_job": ["failed", "succeeded"], "persistent": ["failed"] * 3,
    }[failure]
    for row in attempts:
        assert row["closes"] >= 1
        assert row["actualHeld"] == (TARGET if row["state"] == "succeeded" else None)
        assert row["egoHeld"] == (target if row["state"] == "succeeded" else None)
    if failure == "one_close":
        assert [row["closes"] for row in attempts] == [1, 1]
    fault = runtime.fault_status()
    assert [row["job"] for row in fault["actions"]] == [row["job"] for row in attempts]
    assert fault["actions"][-1]["after"]["gripper"]["holding"] == attempts[-1]["sensorHolding"]
    assert fault["actions"][-1]["after"]["source"] == "live"
    if failure == "persistent":
        assert fault["state"] == "DEGRADED"
        assert fault["transition"]["jobs"] == [row["job"] for row in attempts]
        first = fault["actions"][0]
        dock, *probes, backoff_start, backoff_end = first["execution"]["stages"]
        assert "docked" in first["execution"]["log"]
        assert dock["phase"] == "dock_verified" and dock["positionErrorM"] < 0.06
        assert [stage["phase"] for stage in probes] == ["grasp_probe_feedback"] * attempts[0]["closes"]
        assert all(not stage["feedback"]["gripper"]["holding"] for stage in probes)
        assert [backoff_start["phase"], backoff_end["phase"]] == ["backoff_started", "backoff_finished"]
        assert backoff_end["targetXY"] == first["plan"]["predock"]
        odom = backoff_end["feedback"]["odometry"]
        assert math.dist([odom["x"], odom["y"]], first["plan"]["predock"]) < 0.04
        stamps = [stage["feedback"]["stamp"] for stage in first["execution"]["stages"]]
        assert first["started"] <= stamps[0] < stamps[-1] <= first["finished"] and stamps == sorted(stamps)
        close_count = link.close_count
        link.missed_closes = 0
        retry = runtime.predict("pick", {"target": target})
        with pytest.raises(HttpError) as blocked:
            runtime.run(retry["id"])
        assert blocked.value.code == "diagnosis_required"
        assert link.close_count == close_count
        assert runtime.fault_status()["diagnosisRequired"]["job"] == attempts[-1]["job"]
    else:
        assert fault["state"] == "NORMAL"


def test_pick_retry_after_wrong_object(workbench, record_property):
    runtime, link, _ = workbench
    target = _track(runtime, "mug", "black")
    link.swap_at_close = True
    wrong = _attempt(workbench, "pick", {"target": target})
    repeat = _attempt(workbench, "pick", {"target": target})
    table = _track(runtime, "table")
    release = _attempt(workbench, "place", {"on": table, "offset": [-0.18, -0.24]})
    runtime.observe()
    recovered = _attempt(workbench, "pick", {"target": _track(runtime, "mug", "black")})
    _record(record_property, [wrong, repeat, release, recovered])
    assert (wrong["state"], wrong["actualHeld"], wrong["egoHeld"]) == ("succeeded", DECOY, target)
    assert repeat["state"] == "prediction_refused"
    assert release["actualHeld"] is None
    assert recovered["actualHeld"] == TARGET


@pytest.mark.parametrize("workbench", ["board"], indirect=True)
def test_repeated_flip_jobs_degrade_and_block_the_next_flip(workbench):
    runtime, link, _ = workbench
    target = _track(runtime, "puzzle_piece")
    table = _track(runtime, "table")
    flips = []
    for index in range(3):
        pick = _attempt(workbench, "pick", {"target": target})
        assert pick["actualHeld"] == TARGET, pick
        place = _attempt(workbench, "place", {"on": table, "offset": [-0.23 + 0.01 * index, 0.0], "flip": True})
        assert place["state"] == "succeeded" and place["actualHeld"] is None, place
        flips.append(place["job"])
        fault = runtime.fault_status()
        assert fault["state"] == ("NORMAL" if index < 2 else "DEGRADED")
    assert fault["transition"]["code"] == "repeated_flip"
    assert fault["transition"]["jobs"] == flips
    assert fault["actions"][-1]["before"]["gripper"]["holding"]
    assert not fault["actions"][-1]["after"]["gripper"]["holding"]
    assert _attempt(workbench, "pick", {"target": target})["actualHeld"] == TARGET
    open_count = link.open_count
    retry = runtime.predict("place", {"on": table, "offset": [-0.20, 0.0], "flip": True})
    with pytest.raises(HttpError) as blocked:
        runtime.run(retry["id"])
    assert blocked.value.code == "diagnosis_required"
    assert link.open_count == open_count and link.sim.robots[ROBOT].attached == TARGET
    assert runtime.fault_status()["diagnosisRequired"]["job"] == flips[-1]


@pytest.mark.parametrize("failure", ["none", "command_lost", "stale_holding"])
def test_place_retry_after_release(workbench, record_property, failure):
    runtime, link, _ = workbench
    pick = _attempt(workbench, "pick", {"target": _track(runtime, "mug", "black")})
    params = {"on": _track(runtime, "table"), "offset": [-0.18, -0.24]}
    link.block_release = failure == "command_lost"
    link.stale_holding = failure == "stale_holding"
    place = _attempt(workbench, "place", params)
    link.block_release = False
    retry = _attempt(workbench, "place", params)
    _record(record_property, [pick, place, retry])
    assert pick["actualHeld"] == TARGET
    assert place["state"] == "succeeded" and place["opens"] == 1
    assert place["egoHeld"] is None
    assert retry["state"] == "prediction_refused"
    assert retry["opens"] == 0
    if failure == "command_lost":
        assert place["sensorHolding"] and place["gripperState"] == "closed"
        assert place["actualHeld"] == retry["actualHeld"] == TARGET
        assert place["targetPosition"][:2] != pytest.approx(place["plan"]["spot"], abs=0.06)
    else:
        assert place["sensorHolding"] == (failure == "stale_holding")
        assert place["gripperState"] == "open"
        assert place["actualHeld"] is retry["actualHeld"] is None
        assert place["targetPosition"][:2] == pytest.approx(place["plan"]["spot"], abs=0.06)
        assert place["targetPosition"][2] == pytest.approx(0.75, abs=0.03)
