"""Immutable task, observation, and execution identities at the controller boundary."""

from dataclasses import dataclass, fields
from hashlib import sha256
import json

import numpy as np

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.state import VoxelState


def snapshot(state: VoxelState) -> VoxelState:
    """Own every array so external observation or plant mutation cannot change a decision."""
    values = {}
    for field in fields(state):
        value = getattr(state, field.name)
        if isinstance(value, np.ndarray):
            value = value.copy()
            value.setflags(write=False)
        elif field.name == "provenance":
            value = dict(value)
        values[field.name] = value
    return VoxelState(**values)


def static_fingerprint(state: VoxelState, frame_id: str, authority_id: str) -> str:
    """Bind task geometry and calibrated action meaning, excluding current occupancy."""
    digest = sha256(json.dumps([frame_id, authority_id, state.spacing_mm,
                               state.plane_z_mm]).encode())
    for name in ("x_axis_mm", "y_axis_mm", "z_axis_mm", "initial_tissue",
                 "target_mask", "constraint_mask", "constraint_surface_z_mm"):
        values = getattr(state, name)
        digest.update(name.encode())
        if values is not None:
            array = np.ascontiguousarray(values)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DesignatedTask:
    state: VoxelState
    frame_id: str
    authority_id: str

    def __post_init__(self):
        if not self.frame_id or not self.authority_id:
            raise ValueError("Task requires frame and action-authority identities")
        object.__setattr__(self, "state", snapshot(self.state))

    @property
    def task_id(self):
        return static_fingerprint(self.state, self.frame_id, self.authority_id)


@dataclass(frozen=True)
class ControllerObservation:
    state: VoxelState
    task_id: str
    scan_id: str
    sequence: int
    timestamp_s: float
    command_id: str | None

    def __post_init__(self):
        if not self.task_id or not self.scan_id or self.sequence < 0:
            raise ValueError("Observation requires task/scan identities and a nonnegative sequence")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("Observation timestamp must be finite")
        object.__setattr__(self, "state", snapshot(self.state))


@dataclass(frozen=True)
class ActionRequest:
    command_id: str
    task_id: str
    trajectory_id: str
    active_prefix: int
    action: PhysicalAction
    issued_at_s: float


@dataclass(frozen=True)
class ExecutionReceipt:
    command_id: str
    requested_action: PhysicalAction
    achieved_action: PhysicalAction | None
    pulse_status: str
    completed_at_s: float
    energy_provenance: str
    pose_verified: bool

    def __post_init__(self):
        if self.pulse_status not in {"completed", "not_executed", "uncertain"}:
            raise ValueError("Unknown pulse execution status")
        if not self.command_id or not self.energy_provenance or not np.isfinite(self.completed_at_s):
            raise ValueError("Receipt requires identity, time, and energy provenance")
        if not np.isfinite(self.requested_action.as_array()).all():
            raise ValueError("Requested action must be finite")
        if self.pulse_status == "completed":
            if self.achieved_action is None or not self.pose_verified:
                raise ValueError("Completed pulse requires a verified achieved action")
            if not np.isfinite(self.achieved_action.as_array()).all():
                raise ValueError("Achieved action must be finite")
