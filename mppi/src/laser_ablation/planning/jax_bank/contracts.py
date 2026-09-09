"""Stable data contracts for the proposal-only JAX trajectory bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from laser_ablation.core.actions import PhysicalAction, PhysicalActionBounds
from laser_ablation.core.sdf import SDFState
from laser_ablation.core.state import VoxelState
from laser_ablation.physics.super_gaussian import PhysicsConfig


ACTION_DIMENSION = 5
CHANNEL_ORDER = ("current_tissue", "target", "safe", "remaining", "removed", "overcut")
ACTIVE_ALIGNMENT_THRESHOLD = 0.995
ANCHOR_SIMILARITY_THRESHOLD = 0.990
# A shard is one bounded state-memory payload persisted as a single artifact file.
STATE_SHARD_BYTES = 256 * 1024 * 1024
BANK_DIVERSITY_BONUS = 0.02


def action_hash(actions: np.ndarray) -> str:
    """Hash an ordered float32 action sequence without its padding."""
    values = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    if values.ndim != 2 or values.shape[1] != ACTION_DIMENSION:
        raise ValueError("actions must have shape (pulse, 5)")
    return sha256(values.tobytes()).hexdigest()


def linearization_path_hash(linearization_path: np.ndarray) -> str:
    """Hash one active per-pulse nominal-library route without padded entries."""
    path = np.ascontiguousarray(np.asarray(linearization_path, dtype=np.int32))
    if path.ndim != 2 or path.shape[1] != 2 or len(path) <= 0:
        raise ValueError("linearization_path must have shape (pulse, 2)")
    if np.any(path < 0):
        raise ValueError("active linearization paths must be nonnegative")
    digest = sha256()
    digest.update(str(path.shape).encode())
    digest.update(path.tobytes())
    return digest.hexdigest()


def trajectory_hash(
    actions: np.ndarray,
    initial_current_sdf: np.ndarray,
    linearization_path: np.ndarray,
    source_id: str,
) -> str:
    """Identify an action rollout by state, nominal route, and source provenance."""
    if not source_id:
        raise ValueError("trajectory source_id must be nonempty")
    path = np.ascontiguousarray(np.asarray(linearization_path, dtype=np.int32))
    if path.shape != (len(actions), 2):
        raise ValueError("linearization_path must align with unpadded actions")
    linearization_path_hash(path)
    digest = sha256()
    digest.update(b"source_id\0")
    digest.update(source_id.encode())
    digest.update(b"\0initial_current_sdf\0")
    digest.update(np.ascontiguousarray(np.asarray(initial_current_sdf, dtype=np.float32)).tobytes())
    digest.update(b"\0actions\0")
    digest.update(np.ascontiguousarray(np.asarray(actions, dtype=np.float32)).tobytes())
    digest.update(b"\0linearization_path\0")
    digest.update(path.tobytes())
    return digest.hexdigest()


def task_fingerprint(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> str:
    """Hash all static semantics needed to compare dynamic planner states."""
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    for name in sorted(metadata):
        digest.update(f"{name}={metadata[name]!r}".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class StaticTaskTensors:
    """Task-invariant fields shared by every stored dynamic prefix state."""

    x_axis_mm: np.ndarray
    y_axis_mm: np.ndarray
    z_axis_mm: np.ndarray
    initial_tissue: np.ndarray
    target_mask: np.ndarray
    constraint_mask: np.ndarray
    initial_current_sdf: np.ndarray
    target_sdf: np.ndarray
    safe_sdf: np.ndarray
    physical_clearance_mm: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    plane_z_mm: float
    spacing_mm: float
    truncation_mm: float
    hard_margin_mm: float
    completion_remaining_fraction: float
    physics: PhysicsConfig
    fingerprint: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.initial_tissue.shape)

    @property
    def initial_target_voxels(self) -> int:
        return int(np.count_nonzero(self.initial_tissue & self.target_mask))

    @property
    def initial_healthy_voxels(self) -> int:
        return int(np.count_nonzero(self.initial_tissue & ~self.target_mask))

    @property
    def initial_target_volume_mm3(self) -> float:
        return self.initial_target_voxels * self.spacing_mm**3

    @classmethod
    def from_state(
        cls,
        voxel_state: VoxelState,
        sdf_state: SDFState,
        bounds: PhysicalActionBounds,
        physics: PhysicsConfig,
        completion_remaining_fraction: float,
    ) -> "StaticTaskTensors":
        """Extract immutable planning semantics from one exact/SDF state pair."""
        observation = sdf_state.observation
        truncation = float(np.max(np.abs(observation.target_sdf_mm)))
        if truncation <= 0.0:
            raise ValueError("SDF truncation must be positive")
        arrays = {
            "x_axis_mm": np.asarray(voxel_state.x_axis_mm, dtype=np.float32),
            "y_axis_mm": np.asarray(voxel_state.y_axis_mm, dtype=np.float32),
            "z_axis_mm": np.asarray(voxel_state.z_axis_mm, dtype=np.float32),
            "initial_tissue": np.asarray(voxel_state.initial_tissue, dtype=np.bool_),
            "target_mask": np.asarray(voxel_state.target_mask, dtype=np.bool_),
            "constraint_mask": np.asarray(voxel_state.constraint_mask, dtype=np.bool_),
            "target_sdf": np.asarray(sdf_state.tensor[SDFState.TARGET], dtype=np.float32),
            "safe_sdf": np.asarray(sdf_state.tensor[SDFState.SAFE], dtype=np.float32),
            "physical_clearance_mm": (
                np.full(voxel_state.grid_shape, np.inf, dtype=np.float32)
                if not np.any(voxel_state.constraint_mask)
                else np.asarray(observation.physical_clearance_mm, dtype=np.float32)
            ),
            "lower_bounds": bounds.lower.astype(np.float32),
            "upper_bounds": bounds.upper.astype(np.float32),
        }
        metadata = {
            "plane_z_mm": float(voxel_state.plane_z_mm),
            "spacing_mm": float(voxel_state.spacing_mm),
            "truncation_mm": truncation,
            "hard_margin_mm": float(observation.hard_margin_mm),
            "completion_remaining_fraction": float(completion_remaining_fraction),
            "physics": asdict(physics),
            "channel_order": CHANNEL_ORDER,
        }
        fingerprint = task_fingerprint(arrays, metadata)
        return cls(
            **arrays,
            initial_current_sdf=np.asarray(
                sdf_state.tensor[SDFState.CURRENT_TISSUE], dtype=np.float32
            ),
            plane_z_mm=metadata["plane_z_mm"],
            spacing_mm=metadata["spacing_mm"],
            truncation_mm=metadata["truncation_mm"],
            hard_margin_mm=metadata["hard_margin_mm"],
            completion_remaining_fraction=metadata["completion_remaining_fraction"],
            physics=physics,
            fingerprint=fingerprint,
        )


@dataclass(frozen=True)
class PaddedActionBatch:
    """Fixed-shape complete-plan actions and inert-padding mask."""

    actions: np.ndarray
    action_mask: np.ndarray
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.actions.ndim != 3 or self.actions.shape[2] != ACTION_DIMENSION:
            raise ValueError("actions must have shape (batch, length, 5)")
        if self.action_mask.shape != self.actions.shape[:2]:
            raise ValueError("action_mask must have shape (batch, length)")
        if len(self.source_ids) != self.actions.shape[0]:
            raise ValueError("source_ids must contain one value per batch row")
        if any(not source_id for source_id in self.source_ids):
            raise ValueError("source_ids must be nonempty")

    @property
    def maximum_pulses(self) -> int:
        return int(self.actions.shape[1])

    @classmethod
    def from_sequences(
        cls, sequences: tuple[np.ndarray, ...], source_ids: tuple[str, ...]
    ) -> "PaddedActionBatch":
        """Pad ordered physical sequences without encoding a sentinel action."""
        if not sequences:
            raise ValueError("at least one action sequence is required")
        lengths = np.array([len(sequence) for sequence in sequences], dtype=int)
        if np.any(lengths <= 0):
            raise ValueError("all action sequences must contain a pulse")
        maximum = int(np.max(lengths))
        values = np.zeros((len(sequences), maximum, ACTION_DIMENSION), dtype=np.float32)
        mask = np.zeros((len(sequences), maximum), dtype=np.bool_)
        for row, sequence in enumerate(sequences):
            array = np.asarray(sequence, dtype=np.float32)
            if array.shape != (lengths[row], ACTION_DIMENSION):
                raise ValueError("each action sequence must have shape (pulse, 5)")
            values[row, : lengths[row]] = array
            mask[row, : lengths[row]] = True
        return cls(values, mask, source_ids)


@dataclass(frozen=True)
class QuickRolloutBatch:
    """Parallel SDF-prefix traces and screening values for one padded batch."""

    current_sdf: np.ndarray
    remaining_voxels: np.ndarray
    overcut_voxels: np.ndarray
    remaining_fraction: np.ndarray
    overcut_fraction: np.ndarray
    healthy_overcut_fraction: np.ndarray
    clearance_mm: np.ndarray
    pulse_count: np.ndarray
    energy_j: np.ndarray
    flags: dict[str, np.ndarray]
    constraint_feasible: np.ndarray
    accepted: np.ndarray


@dataclass(frozen=True)
class BankTrajectory:
    """One fixed source-lineage trajectory with its current hard-feasibility status."""

    trajectory_id: str
    actions: np.ndarray
    linearization_path: np.ndarray
    source_id: str
    active: bool
    current_sdf: np.ndarray
    masks: dict[str, np.ndarray]
    remaining_fraction: np.ndarray
    overcut_fraction: np.ndarray
    clearance_mm: np.ndarray
    pulse_count: np.ndarray
    energy_j: np.ndarray
    flags: dict[str, np.ndarray]
    score: float
    origins: tuple[str, ...]

    def __post_init__(self) -> None:
        path = np.asarray(self.linearization_path, dtype=np.int32)
        if path.shape != (len(self.actions), 2):
            raise ValueError("bank trajectory linearization_path must align with actions")
        if np.any(path < 0):
            raise ValueError("bank trajectory active paths must be nonnegative")
        if not self.source_id:
            raise ValueError("bank trajectory source_id must be nonempty")
        if not isinstance(self.active, (bool, np.bool_)):
            raise TypeError("bank trajectory active status must be boolean")
        path.setflags(write=False)
        object.__setattr__(self, "linearization_path", path)
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "active", bool(self.active))


@dataclass(frozen=True)
class MatchingTail:
    """A non-averaged action suffix beginning at a compatible stored prefix."""

    tail_id: str
    matched_state: np.ndarray
    actions: np.ndarray
    linearization_path: np.ndarray
    source_id: str
    states: np.ndarray
    similarity: float
    trajectory_ids: tuple[str, ...]
    source_records: tuple[dict[str, Any], ...]
    score: float
    remaining_pulses: int
    remaining_energy_j: float

    def __post_init__(self) -> None:
        path = np.asarray(self.linearization_path, dtype=np.int32)
        if path.shape != (len(self.actions), 2):
            raise ValueError("matching-tail linearization_path must align with actions")
        if np.any(path < 0):
            raise ValueError("matching-tail active paths must be nonnegative")
        if not self.source_id:
            raise ValueError("matching-tail source_id must be nonempty")
        path.setflags(write=False)
        object.__setattr__(self, "linearization_path", path)
        object.__setattr__(self, "source_id", str(self.source_id))


@dataclass(frozen=True)
class MatchingTailBank:
    """All similarity-qualified plan tails supplied unchanged to plan repair."""

    task_fingerprint: str
    tails: tuple[MatchingTail, ...]


@dataclass(frozen=True)
class ProvisionalPlan:
    """Surrogate-ranked plan that intentionally lacks exact-verification authority."""

    trajectory_id: str
    actions: tuple[PhysicalAction, ...]
    predicted_states: np.ndarray
    score: float
    task_fingerprint: str
    truncation_mm: float
    parent_trajectory_id: str | None = None

    def __post_init__(self) -> None:
        """Root a global plan at itself and preserve a repaired plan's parent lineage."""
        parent = self.trajectory_id if self.parent_trajectory_id is None else str(
            self.parent_trajectory_id
        )
        if not parent:
            raise ValueError("provisional plan parent trajectory id must be nonempty")
        object.__setattr__(self, "parent_trajectory_id", parent)


@dataclass(frozen=True)
class MPCHandoff:
    """Nominal actions and physical-unit tissue-SDF references for energy MPC."""

    actions: np.ndarray
    predicted_tissue_sdf_mm: np.ndarray
    truncation_mm: float
    task_fingerprint: str


def mpc_handoff(provisional: ProvisionalPlan) -> MPCHandoff:
    """Expose nominal action and SDF references to the still-active MPC route."""
    return MPCHandoff(
        actions=np.asarray([action.as_array() for action in provisional.actions], dtype=np.float32),
        predicted_tissue_sdf_mm=np.asarray(
            provisional.predicted_states[:, SDFState.CURRENT_TISSUE]
            * provisional.truncation_mm,
            dtype=np.float32,
        ),
        truncation_mm=provisional.truncation_mm,
        task_fingerprint=provisional.task_fingerprint,
    )
