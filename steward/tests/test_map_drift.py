"""Map stability on a physical closed route, with unmodified simulator odometry.

This measures landmark displacement after revisiting a place. It does not certify recovery
from accumulated odometry bias: the simulator has near-ideal odometry and the estimator has
no registration correction. World poses are used only by assertions and diagnostics.
"""

from __future__ import annotations

import json
import math

import numpy as np

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator
from steward.ego.perception import decode_rgbd
from steward.math3d import yaw_from_quat
from steward.scene.validate import validate_ego
from steward.world.sim import PerceptionConfig, WorldSim


ROBOT = "w:robot-1"


def _drive(sim: WorldSim, goal: tuple[float, float, float]) -> None:
    """Move using the base actuator; bound settling by simulation time, not wall time."""
    sim.command(ROBOT, {"base": list(goal)})
    for _ in range(100):
        sim.step_seconds(0.05)
        pose = sim.world_state()["entities"]
        base = next(e["pose"]["value"] for e in pose if e["id"] == ROBOT)
        xy_error = float(np.linalg.norm(np.asarray(base["p"][:2]) - goal[:2]))
        yaw_error = abs((yaw_from_quat(base["q"]) - goal[2] + math.pi) % (2 * math.pi) - math.pi)
        if xy_error < 0.01 and yaw_error < 0.01:
            return
    raise AssertionError(f"base did not reach {goal}: xy={xy_error:.4f} m, yaw={yaw_error:.4f} rad")


def _observe(sim: WorldSim, est: EgoEstimator) -> None:
    for _ in range(3):
        for camera in ("head_camera", "wide_camera"):
            est.update_rgbd(decode_rgbd(sim.rgbd(ROBOT, camera)))
        sim.step_seconds(0.1)


def test_phase1_map_keeps_a_static_landmark_after_closed_loop_motion(
    kitchen_scene, monkeypatch, record_property,
):
    """Two revisits retain furniture and tableware identities and map positions."""
    monkeypatch.delenv("STEWARD_SEGMENTER_URL", raising=False)
    sim = WorldSim(kitchen_scene, seed=3, perception=PerceptionConfig(mode="rgbd", masks=True))
    try:
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        home = (-1.5, 0.0, 1.1)
        route = ((-1.5, 0.8, 0.75), (-0.5, 0.8, math.pi / 2), (-0.5, 0.0, math.pi / 2), home)
        _drive(sim, home)
        _observe(sim, est)
        references = {}
        for category, world_id in (("counter", "w:counter-1"), ("bowl", "w:bowl-1")):
            tracks = [t for t in est.tracks.values() if t.category == category]
            assert len(tracks) == 1, f"initial {category} tracks: {[t.id for t in tracks]}"
            landmark = tracks[0]
            initial = landmark.pos.copy()
            truth = np.asarray(sim.entity_obb(world_id).center).copy()
            initial_error = float(np.linalg.norm(initial - truth))
            assert initial_error < 0.05, f"initial {category} error {initial_error:.4f} m"
            references[category] = (landmark.id, initial, truth, world_id, landmark.observations)
        visits = []

        for lap in range(2):
            for goal in route:
                _drive(sim, goal)
                _observe(sim, est)
            assert validate_ego(est.to_scene().to_dict()) == []
            for category, (identity, initial, truth, world_id, observations) in references.items():
                assert identity in est.tracks, f"{category} identity {identity} lost on lap {lap + 1}"
                landmark = est.tracks[identity]
                assert landmark.category == category
                assert len([t for t in est.tracks.values() if t.category == category]) == 1
                assert landmark.last_seen >= est.stamp - 0.5, f"{category} not observed on revisit"
                assert landmark.observations > observations, f"{category} received no new observations"
                current_truth = np.asarray(sim.entity_obb(world_id).center)
                # Contact settling may move the bowl by micrometres; a moved landmark invalidates
                # this map comparison only once its physical motion exceeds one millimetre.
                np.testing.assert_allclose(current_truth, truth, atol=0.001, rtol=0)
                visits.append({"lap": lap + 1, "category": category, "position": landmark.pos.tolist(),
                               "mapDisplacementM": float(np.linalg.norm(landmark.pos - initial)),
                               "positionErrorM": float(np.linalg.norm(landmark.pos - current_truth))})

        record_property("map_revisits", json.dumps(visits))
        for visit in visits:
            assert visit["mapDisplacementM"] < 0.03, f"map moved after revisit: {visit}"
            assert visit["positionErrorM"] < 0.05, f"map differs from world: {visit}"
    finally:
        sim.stop()
