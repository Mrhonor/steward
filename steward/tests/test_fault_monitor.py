"""Episode diagnostics for repeated actions and Agent reports."""

import io
import json
import math
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import numpy as np

from steward.assets.generators import steward_bot
from steward.ego.fault_monitor import FaultMonitor, FaultMonitorConfig, motion_feedback
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.sensor_analysis import analyze_sensor
from steward.ego.server import _routes, ego_api
from steward.ego.skills import Job
from steward.math3d import quat_rotate
from steward.net import HttpError


def action(number, *, operation="flip", target="e:1", outcome="succeeded", params=None, source="live", holding=False):
    return {"job": f"job-{number}", "skill": "place" if operation == "flip" else operation, "operation": operation,
            "target": target, "outcome": outcome, "started": number - 0.2, "finished": float(number),
            "params": params or {}, "plan": {},
            "after": {"source": source, "gripper": {"holding": holding}}}


def verdict(value, *references):
    return {"verdict": value, "reason": "Review conclusion", "evidence": list(references)}


def test_repeated_flips_of_one_target_count_across_other_actions_and_changed_offsets():
    monitor = FaultMonitor(FaultMonitorConfig())
    for row in (action(1, params={"offset": [0, 0]}), action(2, operation="observe"),
                action(3, target="e:2"), action(4, params={"offset": [0, 0.1]})):
        assert not monitor.record(row)
    assert monitor.status() == {"state": "NORMAL"}
    assert monitor.record(action(5, params={"offset": [0, 0.2]}))
    assert monitor.status() == {"state": "DEGRADED", "transition": {
        "stamp": 5.0, "code": "repeated_flip", "operation": "flip", "target": "e:1",
        "jobs": ["job-1", "job-4", "job-5"]}}
    assert not monitor.record(action(6))
    assert monitor.actions[-1]["job"] == "job-6"


def test_success_and_expired_attempts_break_repeated_failures():
    monitor = FaultMonitor(FaultMonitorConfig(window_s=6))
    for number, outcome in ((1, "failed"), (2, "failed"), (3, "succeeded"),
                            (4, "failed"), (5, "failed"), (12, "failed"), (13, "stopped"), (14, "failed")):
        assert not monitor.record(action(number, operation="pick", outcome=outcome))
    assert monitor.record(action(15, operation="pick", outcome="failed"))
    assert monitor.status()["transition"]["jobs"] == ["job-12", "job-14", "job-15"]


def test_still_holding_after_place_counts_only_with_live_feedback():
    monitor = FaultMonitor(FaultMonitorConfig())
    for number in range(1, 4):
        assert not monitor.record(action(number, operation="place", source="estimator", holding=True))
    for number in (4, 5):
        assert not monitor.record(action(number, operation="place", holding=True))
    assert monitor.record(action(6, operation="place", holding=True))
    assert monitor.status()["transition"]["failure"] == "still_holding_after_place"


def test_agent_degradation_latches_until_a_new_episode():
    runtime = EgoRuntime(RobotLink("http://unused"), steward_bot().to_dict())
    try:
        routes = _routes(runtime)
        report = routes["POST", "/ego/fault/degrade"]
        assert routes["GET", "/ego/fault"]({}, {}) == {"state": "NORMAL", "actions": [], "observations": []}
        with pytest.raises(ValueError, match="only a reason"):
            report({}, {"state": "NORMAL", "reason": "clear"})
        with pytest.raises(ValueError, match="only a reason"):
            report({}, ["reason"])
        with pytest.raises(ValueError, match="needs a reason"):
            report({}, {"reason": " "})
        degraded = report({}, {"reason": "Rechecked the same face without progress"})
        assert degraded["transition"]["code"] == "agent_report"
        assert report({}, {"reason": "another report"}) == degraded
        runtime.acknowledge()
        assert runtime.summary()["fault"] == degraded
        with runtime._lock:
            runtime._restart()
        assert runtime.fault_status() == {"state": "NORMAL", "actions": [], "observations": []}
    finally:
        runtime.stop()


def test_feedback_and_inspection_records_omit_private_references_and_pixel_grids():
    feedback = motion_feedback({"stamp": 1.0, "feedbackSource": "live",
                                "gripper": {"holding": True, "heldObjectRef": "hidden"}})
    assert feedback["gripper"] == {"holding": True}
    monitor = FaultMonitor(FaultMonitorConfig(history_limit=3))
    for stamp in range(5):
        monitor.observe_face({"entity": "e:1", "stamp": stamp, "camera": "head_camera", "coverage": 1,
                              "sourcePx": 30, "plain": True, "interiorColors": 1, "rows": [[None]]})
    assert [x["stamp"] for x in monitor.observations] == [2, 3, 4]
    assert "rows" not in monitor.observations[-1]


def face(monitor, entity, stamp, rgb, coverage=1):
    monitor.observe_face({"entity": entity, "stamp": stamp, "camera": "head_camera", "coverage": coverage,
                          "sourcePx": 40, "plain": True, "interiorColors": 2}, rgb)


def test_diagnosis_includes_triggering_jobs_and_observations_after_other_work():
    monitor = FaultMonitor(FaultMonitorConfig())
    face(monitor, "e:1", 0.5, [161, 137, 99])
    for number in range(1, 4):
        monitor.record(action(number))
    for number in range(4, 24):
        monitor.record(action(number, target="e:2"))
        face(monitor, "e:2", number, [161, 137, 99])
    monitor.begin_recovery("r", 24, None)
    actions, observations = monitor.diagnosis_history("r")
    assert [row["job"] for row in actions[:3]] == ["job-1", "job-2", "job-3"]
    assert actions[-1]["job"] == "job-23"
    assert [observations[0]["entity"], observations[-1]["entity"]] == ["e:1", "e:2"]


def test_diagnosis_keeps_observations_across_the_triggering_attempts():
    monitor = FaultMonitor(FaultMonitorConfig())
    monitor.record(action(1))
    face(monitor, "e:1", 1.1, [161, 137, 99])
    monitor.record(action(2))
    face(monitor, "e:1", 2.1, [35, 92, 147])
    for index in range(20):
        face(monitor, "e:2", 2.2 + index / 100, [140, 150, 160])
    assert monitor.record(action(3))
    for index in range(20):
        face(monitor, "e:2", 4 + index / 100, [140, 150, 160])
    monitor.begin_recovery("r", 6, None)
    _, observations = monitor.diagnosis_history("r")
    assert [row["medianRgb"] for row in observations if row["entity"] == "e:1"] == [
        [161, 137, 99], [35, 92, 147],
    ]


def test_failed_diagnosis_returns_to_degraded_and_expires_the_sensor_packet():
    monitor = FaultMonitor(FaultMonitorConfig())
    monitor.report(0, "unexpected observation")
    monitor.begin_recovery("r", 1, None)
    monitor.finish_diagnosis("r", 2, False)
    assert monitor.status()["state"] == "DEGRADED"
    with pytest.raises(ValueError, match="no longer active"):
        monitor.require_recovery("r", "diagnosing")


def test_degraded_repeat_uses_same_target_and_current_job_history():
    monitor = FaultMonitor(FaultMonitorConfig(window_s=10))
    for number in range(1, 4):
        monitor.record(action(number))
    assert monitor.require_diagnosis_for(4, "flip", "e:2", {}) is None
    assert monitor.require_diagnosis_for(14, "flip", "e:1", {}) is None
    assert monitor.require_diagnosis_for(4, "flip", "e:1", {"offset": [1, 0]}) == {
        "job": "job-3", "operation": "flip", "target": "e:1", "params": {"offset": [1, 0]}}
    assert monitor.status()["transition"]["jobs"] == ["job-1", "job-2", "job-3"]


def test_agent_reported_degradation_blocks_identical_action_without_requiring_a_failed_job():
    monitor = FaultMonitor(FaultMonitorConfig())
    monitor.record(action(1, operation="pick", params={"side": "left"}))
    monitor.report(2, "Repeated handling")
    assert monitor.require_diagnosis_for(2, "pick", "e:1", {"side": "right"}) is None
    assert monitor.require_diagnosis_for(2, "pick", "e:1", {"side": "left"})["job"] == "job-1"
    monitor.begin_recovery("r", 2, None)
    assert monitor.require_recovery("r", "diagnosing")["faultJob"]["job"] == "job-1"


@pytest.mark.parametrize("code", ["import os", "sensor.__class__", "open('scene.json')", "globals()",
                                  "max([sensor, sensor])", "range(1000000000)", "1 / 0"])
def test_sensor_python_cannot_access_host_services_or_unbounded_operations(code):
    with pytest.raises(ValueError):
        analyze_sensor({"body": {"stamp": 0}}, code)


def test_sensor_python_charges_slice_work_against_evaluation_budget():
    with pytest.raises(ValueError, match="evaluation budget"):
        analyze_sensor({"depth": [1] * 4096}, "copies = [sensor['depth'][:] for i in range(12000)]\nlen(copies)")


def test_runtime_passes_detached_rgb_depth_and_proprioception_to_analysis(monkeypatch):
    link = RobotLink("http://unused")
    runtime = EgoRuntime(link, steward_bot().to_dict())
    body = {"stamp": 1, "gripper": {"state": "closed", "holding": True, "heldObjectRef": "ws-ref"},
            "odometry": {"x": 1, "y": 2, "yaw": 0}, "cameras": [{"id": "head_camera"}],
            "detections": [{"objectRef": "ws-ref"}]}
    buf = io.BytesIO()
    np.savez(buf, camera="head_camera", fovy=60, header='{"stamp":1}',
             rgb=np.zeros((2, 3, 3), dtype=np.uint8), depth=np.array([[0, 400, 600], [0, 800, 0]]),
             mask=np.ones((2, 3), dtype=np.uint8))
    monkeypatch.setattr(link, "sensors", lambda: body)
    monkeypatch.setattr(link, "rgbd", lambda camera: buf.getvalue())
    try:
        runtime.report_degraded("inconsistent grip")
        attempt = runtime.begin_recovery(None, "Inspect the camera and gripper")
        sensor = attempt["diagnostic"]["sensor"]
        assert sensor["camera"]["width"] == 3 and "rgb" not in sensor["camera"]
        assert "camera_to_base" in sensor["camera"] and "camera_to_odometry" not in sensor["camera"]
        assert "transformError" in sensor["camera"]
        assert "heldObjectRef" not in sensor["body"]["gripper"]
        assert sensor["body"]["odometry"] == {"x": 1, "y": 2, "yaw": 0}
        assert "detections" not in sensor["body"]
        result = runtime.analyze_recovery(attempt["id"], """
depth = [d for d in sensor['camera']['depth_m'] if d > 0]
{'mean': round(mean(depth), 2), 'holding': sensor['body']['gripper']['holding']}
""")
        assert result == {"value": {"mean": 0.6, "holding": True}}
        with pytest.raises(ValueError):
            runtime.analyze_recovery(attempt["id"], "sensor['camera']['mask']")
        runtime.finish_recovery_diagnosis(attempt["id"], True)
        review = runtime.begin_recovery_review(attempt["id"], [])
        result = runtime.finish_recovery_review(attempt["id"], review["reviewId"],
                                                verdict("resolved", "baseline", "post"))
        assert result["verified"] and result["state"] == "NORMAL"
        with pytest.raises(ValueError, match="no longer active"):
            runtime.analyze_recovery(attempt["id"], "1")
    finally:
        runtime.stop()


@pytest.fixture
def review_robot(monkeypatch):
    link = RobotLink("http://unused")
    runtime = EgoRuntime(link, steward_bot().to_dict())
    sample = {"stamp": 1.0, "depth": 800, "episode": 1}
    runtime._episode = 1

    def sensors():
        return {"stamp": sample["stamp"], "episode": sample["episode"],
                "odometry": {"x": 1.0, "y": 2.0, "yaw": 0.1},
                "gripper": {"state": "open", "holding": False, "heldObjectRef": "private"},
                "cameras": [{"id": "head_camera"}]}

    def rgbd(camera):
        assert camera == "head_camera"
        buf = io.BytesIO()
        np.savez(buf, camera=camera, fovy=60, header=json.dumps(sensors()), mask=np.ones((2, 2), dtype=np.uint8),
                 rgb=np.full((2, 2, 3), [25, 60, 100], dtype=np.uint8),
                 depth=np.full((2, 2), sample["depth"], dtype=np.uint16))
        return buf.getvalue()

    monkeypatch.setattr(link, "sensors", sensors)
    monkeypatch.setattr(link, "rgbd", rgbd)
    yield runtime, sample
    runtime.stop()


def test_recovery_calibration_uses_pixel_stride_and_camera_time_encoders(review_robot, monkeypatch):
    runtime, _ = review_robot
    runtime.config.cameras = ["head_camera"]
    encoders = {"lift": 0.71, "reach": 0.43, "wrist_yaw": 0, "wrist_roll": 0}
    body = {"stamp": 1, "episode": 1, "odometry": {"x": 8, "y": 9, "yaw": 0},
            "joints": {**encoders, "reach": 0.10}, "cameras": [{"id": c} for c in runtime.estimator.cameras]}
    header = {**body, "stamp": 2, "odometry": {"x": 1, "y": 2, "yaw": 0.4}, "joints": encoders,
              "gripper": {"holding": True, "heldObjectRef": "private-id"}, "camera_to_world": "private-pose"}

    def rgbd(camera):
        buf = io.BytesIO()
        np.savez(buf, camera=camera, fovy=100, header=json.dumps(header), rgb=np.zeros((96, 128, 3), dtype=np.uint8),
                 depth=np.full((96, 128), 1000, dtype=np.uint16), mask=np.ones((96, 128), dtype=np.uint8))
        return buf.getvalue()

    monkeypatch.setattr(runtime.link, "sensors", lambda: body)
    monkeypatch.setattr(runtime.link, "rgbd", rgbd)
    monkeypatch.setattr(runtime, "_feedback_header", lambda h: {**h, "odometry": {**h["odometry"], "y": h["odometry"]["y"] + 0.3}})
    runtime.report_degraded("Compare camera and estimated target position")
    recovery_id = runtime.begin_recovery(None, "Locate the object")["id"]
    frame = runtime.capture_recovery_rgbd(recovery_id, None, "wide_camera")
    camera, kinematics = frame["camera"], frame["kinematics"]
    assert camera["name"] == "wide_camera" and camera["stamp"] == kinematics["stamp"] == 2
    assert frame["body"]["stamp"] == 1 and frame["body"]["joints"]["reach"] == 0.10
    assert kinematics["joint_encoders"] == encoders and kinematics["odometry"] == header["odometry"]
    offset = kinematics["armGeometry"]["toolOffsetM"]
    assert np.array(kinematics["gripper_to_base"])[:3, 3] == pytest.approx([offset + encoders["reach"], 0, 0.71])
    assert camera["pixelStride"] == 2 and camera["K"][0][2] == 31.75 and camera["K"][1][2] == 23.75
    assert camera["camera_axes"] == "opencv_x_right_y_down_z_forward"
    odometry_origin = np.array(camera["camera_to_odometry"])[:3, 3]
    ego_origin = np.array(camera["camera_to_ego"])[:3, 3]
    assert ego_origin - odometry_origin == pytest.approx([0, 0.3, 0])
    value = runtime.analyze_recovery(recovery_id, """
c = sensor['camera']
z = c['depth_m'][18*c['width']+12]
k = c['K']
x = (12-k[0][2])*z/k[0][0]
y = (18-k[1][2])*z/k[1][1]
[r[0]*x + r[1]*y + r[2]*z + r[3] for r in c['camera_to_ego'][:3]]
""", ref=frame["ref"])["value"]
    runtime.estimator.joints = encoders
    pose = runtime.estimator.camera_pose({"x": 1, "y": 2.3, "yaw": 0.4}, "wide_camera")
    focal = 48 / math.tan(math.radians(50))
    point = quat_rotate(pose["q"], [1, -(24 - 63.5) / focal, -(36 - 47.5) / focal])
    assert value == pytest.approx([p + t for p, t in zip(point, pose["p"], strict=True)])
    assert "private-" not in json.dumps(frame) and "camera_to_world" not in json.dumps(frame)


def test_sensor_recovery_uses_the_agent_verdict_and_cited_samples(review_robot):
    runtime, sample = review_robot
    sample["depth"] = 0
    runtime.report_degraded("camera depth missing")
    recovery_id = runtime.begin_recovery(None, "Check the depth stream")["id"]
    first = runtime.capture_recovery_rgbd(recovery_id, None, "head_camera")
    assert first["preview"][0][0] == {"rgb": [25, 60, 100], "depthM": 0.0}
    assert runtime.analyze_recovery(recovery_id, "sensor['camera']['depth_m']", ref=first["ref"]) == {"value": [0] * 4}
    assert runtime.analyze_recovery(recovery_id, "sensor.get('camera').get('width')", ref=first["ref"]) == {"value": 2}
    diagnosis = {"cause": "Missing depth", "evidence": "baseline", "recommendation": "Resample",
                 "verification": "Three advancing frames"}
    runtime.finish_recovery_diagnosis(recovery_id, True, diagnosis)
    sample.update(stamp=2.0, depth=800)
    opened = runtime.begin_recovery_review(recovery_id, [])
    review_id = opened["reviewId"]
    assert opened["evidence"]["diagnosis"] == diagnosis
    assert first["ref"].startswith("before-frame-")
    assert runtime.analyze_recovery(recovery_id, "sensor['camera']['depth_m']", review_id, first["ref"]) == {"value": [0] * 4}
    with pytest.raises(ValueError, match="no longer active"):
        runtime.capture_recovery_rgbd(recovery_id, None, "head_camera")
    frame = runtime.capture_recovery_rgbd(recovery_id, review_id, None)
    assert "private" not in json.dumps(frame)
    assert "mask" not in frame["camera"]
    insufficient = runtime.finish_recovery_review(
        recovery_id, review_id, verdict("insufficient_evidence", "baseline", "post", frame["ref"]))
    assert insufficient["state"] == "RECOVERY"
    assert not insufficient["verified"]

    sample["stamp"] = 3.0
    second = runtime.begin_recovery_review(recovery_id, [])["reviewId"]
    assert runtime.fault_status()["state"] == "RECOVERY"
    result = runtime.finish_recovery_review(recovery_id, second, verdict("resolved", "baseline", "post"))
    assert result["verified"] and result["state"] == "NORMAL"
    assert runtime.fault_status()["lastRecovery"]["review"]["evidence"] == ["baseline", "post"]


def test_recovery_receives_skill_log_and_stage_feedback(review_robot):
    runtime, _ = review_robot
    job = Job("job-1", "pick", "rcpt-1", state="failed", started=0, finished=1,
              episode_generation=runtime._generation)
    runtime.receipts[job.receipt] = {"params": {"target": "e:1"}, "plan": {"dock": [0.1, 0], "predock": [-0.25, 0]}}
    job.log.append("docked")
    job.record_stage("dock_verified", {"stamp": 0.5, "odometry": {"x": 0.1, "y": 0, "yaw": 0},
                                       "gripper": {"holding": False, "heldObjectRef": "private-object"}})
    runtime._skill_finished(job)
    runtime.report_degraded("pick missed")
    report = runtime.begin_recovery(job.id, "Diagnose the missed grasp")["diagnostic"]
    action, = report["actions"]
    assert action["plan"]["predock"] == [-0.25, 0]
    assert action["execution"]["log"] == ["docked"]
    stage, = action["execution"]["stages"]
    assert stage["phase"] == "dock_verified" and stage["feedback"]["odometry"]["x"] == 0.1
    assert stage["feedback"]["gripper"] == {"holding": False}
    assert "private-object" not in json.dumps(report)


@pytest.mark.parametrize(("result", "extra_ref", "changes", "approved"), [
    ("resolved", None, None, True),
    ("unresolved", None, None, False),
    ("insufficient_evidence", None, None, False),
    ("resolved", "made-up", None, False),
    ("resolved", None, "job", False),
    ("resolved", None, "episode", False),
    ("resolved", None, "time", False),
    ("resolved", None, "old sample", False),
    ("resolved", None, "job only", False),
    ("resolved", None, "missing baseline", False),
])
def test_review_validates_evidence_without_classifying_the_action(review_robot, result, extra_ref, changes, approved):
    runtime, sample = review_robot
    monitor = runtime.fault_monitor
    goal = {"x": 1, "y": 2, "yaw": 0.1}
    for number in range(1, 4):
        monitor.record(action(number, operation="navigate", target=None, params=goal, outcome="failed"))
    recovery_id = runtime.begin_recovery(None, "Reach the goal")["id"]
    runtime.finish_recovery_diagnosis(recovery_id, True)
    monitor.record(action(5, operation="observe", target="e:2", outcome="failed"))
    sample["stamp"] = 4 if changes == "old sample" else 5
    opened = runtime.begin_recovery_review(recovery_id, [])
    assert opened["evidence"]["faultJob"]["job"] == "job-3"
    changed_action, = opened["evidence"]["actionsSinceDiagnosis"]
    assert (changed_action["job"], changed_action["outcome"]) == ("job-5", "failed")
    review_id = opened["reviewId"]
    if changes == "job":
        monitor.record(action(6, operation="navigate", target=None, params=goal))
        sample["stamp"] = 6
    if changes == "episode":
        sample["episode"] = 2
    if changes == "time":
        sample["stamp"] = 20
    refs = ["baseline", "post", "job-5"] + ([extra_ref] if extra_ref else [])
    if changes == "job only":
        refs.remove("post")
    if changes == "missing baseline":
        refs.remove("baseline")
    finalized = runtime.finish_recovery_review(recovery_id, review_id, verdict(result, *refs))
    assert finalized["verified"] == approved
    assert finalized["state"] == ("NORMAL" if approved else "RECOVERY")
    if approved:
        assert runtime.fault_status()["lastRecovery"]["review"]["evidence"] == refs
    else:
        assert finalized["recovery"]["phase"] == "verifying"
    with pytest.raises(ValueError, match="no longer active"):
        runtime.capture_recovery_rgbd(recovery_id, review_id, None)


def test_live_body_evidence_can_support_recovery_without_an_rgbd_frame(review_robot, monkeypatch):
    runtime, sample = review_robot
    runtime.report_degraded("gripper feedback had been intermittent")
    recovery_id = runtime.begin_recovery(None, "Check gripper feedback")["id"]
    runtime.finish_recovery_diagnosis(recovery_id, True)
    sample["stamp"] = 2

    def unavailable(camera):
        raise TimeoutError("camera unavailable")

    monkeypatch.setattr(runtime.link, "rgbd", unavailable)
    opened = runtime.begin_recovery_review(recovery_id, [])
    assert "camera" not in opened["evidence"]["post"]
    result = runtime.finish_recovery_review(recovery_id, opened["reviewId"], verdict("resolved", "baseline", "post"))
    assert result["verified"] and result["state"] == "NORMAL"


def test_recovery_context_can_pause_diagnosis_and_review_multiple_times(review_robot):
    runtime, sample = review_robot
    runtime.report_degraded("inconsistent grip")
    routes = _routes(runtime)
    start = routes["POST", "/ego/recovery/start"]
    request_context = routes["POST", "/ego/recovery/context/request"]
    answer_context = routes["POST", "/ego/recovery/context/answer"]
    background = "Task constraints. " * 180 + "The success criterion is at the end."
    attempt = start({}, {"context": background})
    assert attempt["diagnostic"]["context"]["operatorContext"] == background
    recovery_id = attempt["id"]
    child_id, parent_id = "session-steward-recovery-existing", "session-eh"
    scope = {"id": recovery_id, "childId": child_id, "parentId": parent_id}
    first = request_context({}, {**scope, "question": "Which target?", "reason": "Choose observations"})
    assert first["needsContext"] and first["recovery"]["phase"] == "diagnosing"
    with pytest.raises(ValueError, match="pending"):
        request_context({}, {**scope, "question": "Which camera?", "reason": "Choose a sample"})
    with pytest.raises(ValueError, match="pending"):
        runtime.finish_recovery_diagnosis(recovery_id, True, child_id=child_id)
    first_id = first["request"]["id"]
    with pytest.raises(ValueError, match="original EH"):
        answer_context({}, {"id": recovery_id, "requestId": first_id, "parentId": "session-other", "answer": "e:1"})
    answer = {"id": recovery_id, "requestId": first_id, "parentId": parent_id, "answer": "Target e:1"}
    resumed = answer_context({}, answer)
    assert resumed["childId"] == child_id and resumed["reviewId"] is None
    assert resumed["evidence"]["answer"] == "Target e:1"
    with pytest.raises(ValueError, match="no longer pending"):
        answer_context({}, answer)

    second = request_context({}, {**scope, "question": "Which camera?", "reason": "Choose a sample"})
    assert second["request"]["id"] != first_id
    face(runtime.fault_monitor, "e:1", 1, [35, 92, 147])
    resumed = answer_context({}, {**answer, "requestId": second["request"]["id"], "answer": "Use head_camera"})
    assert resumed["evidence"]["observations"][-1]["entity"] == "e:1"
    runtime.finish_recovery_diagnosis(recovery_id, True, child_id=child_id)
    sample["stamp"] = 2
    opened = runtime.begin_recovery_review(recovery_id, [])
    review_id = opened["reviewId"]
    assert opened["childId"] == child_id
    third = request_context({}, {**scope, "reviewId": review_id, "question": "What counts as success?", "reason": "Check verdict"})
    assert third["recovery"]["phase"] == "reviewing"
    report = verdict("resolved", "baseline", "post")
    with pytest.raises(ValueError, match="pending"):
        runtime.finish_recovery_review(recovery_id, review_id, report)
    face(runtime.fault_monitor, "e:1", 2, [36, 93, 148])
    resumed = answer_context({}, {**answer, "requestId": third["request"]["id"], "answer": "Stable sensor feedback"})
    assert resumed["childId"] == child_id and resumed["reviewId"] == review_id
    new_inspection = resumed["evidence"]["observations"][-1]["ref"]
    assert runtime.finish_recovery_review(recovery_id, review_id, verdict("resolved", "baseline", new_inspection))["verified"]


@pytest.mark.parametrize(("report_fields", "verified"), [
    ({"verdict": "resolved", "taskCompleted": True}, True),
    ({"verdict": "resolved", "taskCompleted": False}, False),
    ({"verdict": "unresolved", "taskCompleted": True}, False),
    ({"verdict": "resolved"}, False),
])
def test_task_closeout_requires_verified_completion_and_resolved_fault(review_robot, report_fields, verified):
    runtime, sample = review_robot
    monitor = runtime.fault_monitor
    task = "Leave the four pieces face-up on the table"
    claim = "All four are face-up"
    child_id = "session-steward-recovery-closeout"
    assert runtime.begin_task_recovery_review(task, claim, child_id) == {"state": "NORMAL"}
    for number in range(1, 4):
        monitor.record(action(number))
    face(monitor, "e:1", 3, [35, 92, 147])
    sample["stamp"] = 3
    route = _routes(runtime)["POST", "/ego/recovery/task-review"]
    opened = route({}, {"task": task, "claim": claim, "childId": child_id})
    assert opened["newChild"] and opened["childId"] == child_id
    assert opened["evidence"]["taskCompletion"] == {"task": task, "claim": claim}
    assert [row["job"] for row in opened["evidence"]["faultHistory"]] == ["job-1", "job-2", "job-3"]
    assert len(monitor.actions) == 3
    inspection, = opened["evidence"]["inspections"]
    report = {**verdict("resolved", "baseline", inspection["ref"]), **report_fields}
    result = runtime.finish_recovery_review(opened["id"], opened["reviewId"], report)
    assert result["verified"] == verified
    assert result["state"] == ("NORMAL" if verified else "RECOVERY")
    if not verified:
        reopened = runtime.begin_task_recovery_review(task, claim, "session-steward-recovery-other")
        assert not reopened["newChild"] and reopened["childId"] == child_id


def test_task_closeout_context_question_resumes_the_same_review(review_robot):
    runtime, sample = review_robot
    runtime.report_degraded("inconsistent grip")
    task, claim = "Verify the empty gripper", "Gripper is empty"
    child_id, parent_id = "session-steward-recovery-closeout", "session-eh"
    opened = runtime.begin_task_recovery_review(task, claim, child_id)
    pending = runtime.request_recovery_context(opened["id"], opened["reviewId"], child_id, parent_id,
                                               "What was the initial gripper state?", "Check the task")
    assert runtime.begin_task_recovery_review(task, claim, child_id)["request"] == pending["request"]
    resumed = runtime.answer_recovery_context(opened["id"], pending["request"]["id"], parent_id, "The gripper was open")
    assert resumed["completionReview"] and resumed["childId"] == child_id and resumed["reviewId"] == opened["reviewId"]
    report = {**verdict("resolved", "baseline", "post"), "taskCompleted": True}
    assert runtime.finish_recovery_review(opened["id"], opened["reviewId"], report)["verified"]


def test_repeat_blocks_dispatch_until_diagnosis_report_arrives(review_robot, monkeypatch):
    runtime, sample = review_robot
    monitor = runtime.fault_monitor
    sample["stamp"] = 3
    for number in range(1, 4):
        monitor.record(action(number))
    receipt_id = "receipt-repeat"
    receipt = {"id": receipt_id, "skill": "place", "params": {"flip": True}, "plan": {"object": "e:1"},
               "feasible": True, "expiresAt": 100, "risk": 0.1}
    runtime.receipts[receipt_id] = receipt
    runtime._receipt_generation[receipt_id] = runtime._generation
    started = []
    def start(skill, _receipt, **_kwargs):
        started.append(skill)
        return Job("job-4", skill, receipt_id)

    monkeypatch.setattr(runtime.executor, "start", start)
    monkeypatch.setattr(runtime, "apply", lambda _request, handler: handler())
    api = ego_api(runtime).start()
    try:
        req = Request(f"{api.url}/skill/run", data=json.dumps({"receipt": receipt_id}).encode(),
                      headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as pause:
            urlopen(req, timeout=5)
        with pause.value as response:
            assert response.code == 409 and json.load(response)["code"] == "diagnosis_required"
    finally:
        api.stop()
    assert not started and not receipt.get("used")
    assert runtime.fault_state()["diagnosisRequired"]["job"] == "job-3"
    skipped = runtime.begin_task_recovery_review("Flip e:1", "Done", "session-steward-recovery-test")
    assert skipped["state"] == "DEGRADED" and "reviewId" not in skipped
    with pytest.raises(HttpError) as bypass:
        runtime.explore()
    assert bypass.value.code == "diagnosis_required"

    abandoned = runtime.begin_recovery(None, "Check repeated flips")["id"]
    runtime.finish_recovery_diagnosis(abandoned, False)
    with pytest.raises(HttpError):
        runtime.run(receipt_id)

    opened = runtime.begin_recovery(None, "Check repeated flips")
    assert opened["diagnostic"]["fault"]["diagnosisRequired"]["job"] == "job-3"
    child_id, parent_id = "session-steward-recovery-test", "session-eh"
    question = runtime.request_recovery_context(opened["id"], None, child_id, parent_id, "Task?", "Find intended side")
    with pytest.raises(HttpError):
        runtime.run(receipt_id)
    runtime.answer_recovery_context(opened["id"], question["request"]["id"], parent_id, "Printed side up")
    with pytest.raises(ValueError, match="requires a diagnosis report"):
        runtime.finish_recovery_diagnosis(opened["id"], True, child_id=child_id)
    with pytest.raises(HttpError):
        runtime.run(receipt_id)

    report = {"cause": "Ambiguous color", "evidence": "job-3", "recommendation": "Inspect the face",
              "verification": "Compare the backing color"}
    diagnosed = runtime.finish_recovery_diagnosis(opened["id"], True, report, child_id)
    assert diagnosed["state"] == "RECOVERY" and "diagnosisRequired" not in diagnosed
    assert not started and not receipt.get("used")
    assert runtime.run(receipt_id)["id"] == "job-4"
    assert started == ["place"]
