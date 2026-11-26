"""Episode-local diagnostic records for repeated robot operations."""

import copy
import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultMonitorConfig:
    """Size of the diagnostic history and repetition window in simulation seconds."""

    repeat_count: int = 3
    window_s: float = 180.0
    history_limit: int = 64
    recovery_observation_s: float = 10.0

    def __post_init__(self) -> None:
        if not 2 <= self.repeat_count <= self.history_limit:
            raise ValueError("fault repeat_count must be at least two and fit in history_limit")
        if not math.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError("fault window_s must be finite and positive")
        if not math.isfinite(self.recovery_observation_s) or self.recovery_observation_s <= 0:
            raise ValueError("recovery_observation_s must be finite and positive")


def motion_feedback(state: dict) -> dict:
    """Copy controller feedback without exporting simulator object references or ES estimates."""
    def grip(report: dict) -> dict:
        return {key: report[key] for key in ("state", "holding", "holdingHandle") if key in report}

    odom = state.get("odometry", {})
    return {"stamp": state["stamp"], "source": state.get("feedbackSource", "unknown"),
            "odometry": {key: odom[key] for key in ("x", "y", "yaw") if key in odom},
            "joints": dict(state.get("joints", {})), "gripper": grip(state.get("gripper", {})),
            "grippers": {name: grip(report) for name, report in state.get("grippers", {}).items()}}


class FaultMonitor:
    """Record evidence and latch a diagnostic state under the owning Runtime lock."""

    def __init__(self, config: FaultMonitorConfig):
        self.config = config
        self.actions: deque[dict] = deque(maxlen=config.history_limit)
        self.observations: deque[dict] = deque(maxlen=config.history_limit)
        self.transition: dict | None = None
        self.recovery: dict | None = None
        self.required_diagnosis: dict | None = None
        self.last_recovery: dict | None = None
        self._sequence = 0
        self._verified_sequence = 0
        self._degraded_sequence = 0

    def status(self) -> dict:
        """Return the state and the evidence that first caused degradation."""
        if self.transition is None:
            return {"state": "NORMAL"}
        state = {"state": "RECOVERY" if self.recovery is not None else "DEGRADED",
                 "transition": copy.deepcopy(self.transition)}
        if self.recovery is not None:
            state["recovery"] = copy.deepcopy({k: v for k, v in self.recovery.items()
                                               if k not in ("sensor", "captures", "review", "contextOwner")})
        if self.required_diagnosis is not None:
            state["diagnosisRequired"] = copy.deepcopy(self.required_diagnosis)
        return state

    def report(self, stamp: float, reason: str) -> bool:
        """Latch an Agent report and indicate whether the state changed."""
        return self._degrade(stamp, "agent_report", reason=reason)

    def observe_face(self, face: dict, median_rgb: list[float] | None = None) -> None:
        """Keep compact inspection evidence, including coverage for empty samples."""
        self._sequence += 1
        self.observations.append({**{key: face[key] for key in
                                    ("entity", "stamp", "camera", "coverage", "sourcePx", "plain", "interiorColors")},
                                  "medianRgb": median_rgb, "sequence": self._sequence})

    def begin_recovery(self, recovery_id: str, stamp: float, job_id: str | None) -> None:
        """Bind a diagnosis to recorded fault evidence in this episode."""
        if self.transition is None or self.recovery is not None:
            raise ValueError("recovery requires DEGRADED with no active diagnosis")
        ids = self.transition.get("jobs", [])
        mandatory_job = self.required_diagnosis["job"] if self.required_diagnosis is not None else None
        if mandatory_job is not None and job_id not in (None, mandatory_job):
            raise ValueError("diagnose the repeated action before selecting another job")
        if mandatory_job is None and job_id is not None and ids and job_id not in ids:
            raise ValueError("choose a job from the current fault")
        selected = mandatory_job or job_id or (ids[-1] if ids else None)
        failed = next((a for a in self.actions if a["job"] == selected), None)
        if mandatory_job is None and job_id is not None and (failed is None or
                                                              (failed["operation"] != "flip" and self._failure(failed) is None)):
            raise ValueError("recovery job must identify a recorded failure or flip")
        if failed is None and selected:
            raise ValueError("the fault's job evidence has expired")
        self.recovery = {"id": recovery_id, "phase": "diagnosing", "started": stamp, "faultJob": failed}

    def require_recovery(self, recovery_id: str, phase: str) -> dict:
        if self.recovery is None or self.recovery["id"] != recovery_id or self.recovery["phase"] != phase:
            raise ValueError("recovery is no longer active in the requested phase")
        return self.recovery

    def diagnosis_history(self, recovery_id: str, phase: str = "diagnosing") -> tuple[list[dict], list[dict]]:
        """Include the fault interval and recent changes even after the robot has continued working."""
        attempt = self.require_recovery(recovery_id, phase)
        transition = self.transition
        actions = list(self.actions)
        observations = list(self.observations)
        selected = attempt["faultJob"]
        cutoff = selected["finished"] if selected else transition["stamp"]
        fault_jobs = set(transition.get("jobs", []))
        fault_start = min((row["started"] for row in actions if row["job"] in fault_jobs),
                          default=selected["started"] if selected else cutoff)
        interval_actions = [row for row in actions if fault_start <= row["finished"] <= cutoff]
        nearby_actions = [row for row in actions if row["finished"] <= cutoff][-16:]
        jobs = fault_jobs | {row["job"] for row in interval_actions + nearby_actions + actions[-12:]}
        interval_faces = [row for row in observations if fault_start <= row["stamp"] <= cutoff]
        nearby_faces = [row for row in observations if row["stamp"] < fault_start][-8:]
        faces = {row["sequence"] for row in interval_faces + nearby_faces + observations[-16:]}
        return (copy.deepcopy([row for row in actions if row["job"] in jobs]),
                copy.deepcopy([row for row in observations if row["sequence"] in faces]))

    def finish_diagnosis(self, recovery_id: str, stamp: float, completed: bool, report: dict | None = None) -> None:
        if not completed:
            if self.recovery is None or self.recovery["id"] != recovery_id:
                raise ValueError("recovery is no longer active")
            self.recovery = None
            return
        attempt = self.require_recovery(recovery_id, "diagnosing")
        if self.required_diagnosis is not None:
            if report is None:
                raise ValueError("the repeated action requires a diagnosis report")
            self.required_diagnosis = None
        attempt.update(phase="verifying", diagnosisFinished=stamp, diagnosisSequence=self._sequence, diagnosis=report)

    def require_diagnosis_for(self, stamp: float, operation: str, target: str | None, params: dict) -> dict | None:
        """Latch a repeated command before motion and retain its evidence until diagnosis completes."""
        if self.required_diagnosis is not None or self.transition is None or self.recovery is not None:
            return self.required_diagnosis
        for previous in reversed(self.actions):
            if (previous["sequence"] <= self._verified_sequence or previous["outcome"] == "stopped"
                    or not 0 <= stamp - previous["finished"] <= self.config.window_s
                    or previous["operation"] != operation or previous["target"] != target
                    or ((operation != "flip" or target is None) and previous["params"] != params)):
                continue
            if (previous["sequence"] >= self._degraded_sequence
                    or previous["job"] in self.transition.get("jobs", []) or self.transition["code"] == "agent_report"):
                self.required_diagnosis = {"job": previous["job"], "operation": operation,
                                           "target": target, "params": copy.deepcopy(params)}
                return self.required_diagnosis
            break
        return None

    def resolve(self, recovery_id: str, stamp: float, report: dict) -> None:
        """Record the accepted review and clear the fault in the owning Runtime transaction."""
        self.require_recovery(recovery_id, "reviewing")
        self.last_recovery = {"id": recovery_id, "stamp": stamp, "fault": self.transition,
                              "review": copy.deepcopy(report)}
        self.transition = self.recovery = None
        self._verified_sequence = self._sequence

    def record(self, action: dict) -> bool:
        """Record a completed Job and report a newly detected repetition."""
        self._sequence += 1
        action = {**action, "sequence": self._sequence}
        self.actions.append(action)
        if self.transition is not None or action["outcome"] == "stopped":
            return False
        flip = action["operation"] == "flip" and action["target"] is not None
        failure = self._failure(action)
        if not flip and failure is None:
            return False
        matching = []
        for previous in reversed(self.actions):
            if previous["sequence"] <= self._verified_sequence:
                continue
            if not 0 <= action["finished"] - previous["finished"] <= self.config.window_s:
                continue
            if previous["operation"] != action["operation"] or previous["target"] != action["target"]:
                continue
            if previous["outcome"] == "stopped":
                continue
            if not flip:
                if previous["params"] != action["params"]:
                    continue
                if self._failure(previous) != failure:
                    break
            matching.append(previous["job"])
            if len(matching) == self.config.repeat_count:
                return self._degrade(action["finished"], "repeated_flip" if flip else "repeated_failure",
                                     operation=action["operation"], target=action["target"], jobs=matching[::-1],
                                     **({"failure": failure} if not flip else {}))
        return False

    @staticmethod
    def _failure(action: dict) -> str | None:
        if action["outcome"] == "failed":
            return "job_failed"
        after = action["after"] or {}
        if (action["skill"] == "place" and action["outcome"] == "succeeded"
                and after.get("source") == "live" and after["gripper"].get("holding") is True):
            return "still_holding_after_place"
        return None

    def _degrade(self, stamp: float, code: str, **evidence) -> bool:
        if self.transition is not None:
            return False
        self.transition = {"stamp": stamp, "code": code, **evidence}
        self._degraded_sequence = self._sequence + 1
        return True
