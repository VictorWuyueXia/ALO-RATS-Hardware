"""State-aware live plan-repair contracts independent of proposal refinement and MPC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from laser_ablation.planning.jax_bank.contracts import ACTION_DIMENSION, MatchingTailBank, StaticTaskTensors
from laser_ablation.planning.jax_bank.comparison_library import ComparisonROILibrary


class RepairTrigger(str, Enum):
    """External conditions that authorize one live repair attempt."""

    ALIGNMENT_LOSS = "alignment_loss"
    MPC_INFEASIBLE = "mpc_infeasible"
    PERIODIC = "periodic"


class GlobalReplanReason(str, Enum):
    """Exhaustive MPPI outcomes that require a fresh complete-plan search."""

    NO_MATCHING_ANCHOR = "no_matching_anchor"
    ALL_ANCHORS_INFEASIBLE = "all_anchors_infeasible"
    NO_ADMISSIBLE_REPAIR = "no_admissible_repair"


class GlobalReplanRequired(RuntimeError):
    """Signal that repair cannot provide a hard-feasible nominal tail."""

    def __init__(self, reason: GlobalReplanReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True)
class ParallelSegmentBatch:
    """Padded nominal parent-segment rows and their exact nominal start states."""

    actions: np.ndarray
    action_mask: np.ndarray
    nominal_boundary_sdf: np.ndarray
    linearization_path: np.ndarray
    parent_indices: np.ndarray
    segment_indices: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    source_ids: tuple[str, ...]
    parent_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        actions, mask = np.asarray(self.actions, np.float32), np.asarray(self.action_mask, bool)
        boundary = np.asarray(self.nominal_boundary_sdf, np.float32)
        path = np.asarray(self.linearization_path, np.int32)
        parent = np.asarray(self.parent_indices, np.int32)
        segment = np.asarray(self.segment_indices, np.int32)
        starts, stops = np.asarray(self.starts, np.int32), np.asarray(self.stops, np.int32)
        count = len(actions)
        shapes_valid = (
            actions.ndim == 3 and actions.shape[-1] == ACTION_DIMENSION and count > 0
            and mask.shape == actions.shape[:2] and path.shape == actions.shape[:2] + (2,)
            and boundary.ndim == 4 and boundary.shape[0] == count
            and all(values.shape == (count,) for values in (parent, segment, starts, stops))
            and len(self.source_ids) == count and len(self.parent_candidate_ids) == count
        )
        if not shapes_valid:
            raise ValueError("parallel segment arrays and provenance must share a nonempty row axis")
        lengths = np.count_nonzero(mask, axis=1)
        values_valid = (
            not np.any(parent < 0) and not np.any(segment < 0) and not np.any(starts < 0)
            and not np.any(stops <= starts) and not np.any(stops > lengths)
            and not np.any(path[mask] < 0) and not np.any(path[~mask] != -1)
            and not np.any(actions[~mask] != 0.0) and np.all(np.isfinite(actions))
            and np.all(np.isfinite(boundary))
        )
        if not values_valid:
            raise ValueError("parallel segment values violate bounds, routing, padding, or finiteness")
        names = ("actions", "action_mask", "nominal_boundary_sdf", "linearization_path",
                 "parent_indices", "segment_indices", "starts", "stops")
        arrays = (actions, mask, boundary, path, parent, segment, starts, stops)
        for name, values in zip(names, arrays, strict=True):
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def storage_bytes(self) -> int:
        """Count every stored array and UTF-8 provenance byte in this repair-local batch."""
        arrays = (self.actions, self.action_mask, self.nominal_boundary_sdf, self.linearization_path,
                  self.parent_indices, self.segment_indices, self.starts, self.stops)
        return int(sum(values.nbytes for values in arrays) + sum(
            len(value.encode()) for value in self.source_ids + self.parent_candidate_ids
        ))


@dataclass(frozen=True)
class MPPIRepairConfig:
    """Frozen path-integral repair constants and explicit execution batch size."""

    max_anchors: int = 10
    samples_per_anchor: int = 256
    iterations: int = 1
    rollout_batch_size: int = 64
    beta: float = 0.70
    kappa: tuple[float, float, float, float, float] = (0.03, 0.03, 0.025, 0.025, 0.05)
    lambda_remaining: float = 0.60
    lambda_overcut: float = 0.40
    lambda_anchor: float = 0.0
    temperature: float = 0.025
    segment_minimum_pulses: int = 10
    segment_maximum_count: int = 5
    segment_retain_fraction: float = 0.20
    device_memory_cap_gib: float = 8.0
    segment_working_pool_gib: float = 1.0
    roi_match_cosine_min: float = 0.90
    roi_linearization_cosine_min: float = 0.995

    def __post_init__(self) -> None:
        counts = (
            self.max_anchors,
            self.samples_per_anchor,
            self.iterations,
            self.rollout_batch_size,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("MPPI counts and rollout batch size must be positive")
        if self.samples_per_anchor < 2:
            raise ValueError("MPPI requires a control sample and at least one random sample")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("MPPI beta must lie in [0, 1)")
        if len(self.kappa) != 5 or any(not 0.0 < value <= 0.5 for value in self.kappa):
            raise ValueError("MPPI kappa must contain five values in (0, 0.5]")
        if min(self.lambda_remaining, self.lambda_overcut, self.lambda_anchor) < 0.0:
            raise ValueError("MPPI objective weights must be nonnegative")
        if not np.isclose(
            self.lambda_remaining + self.lambda_overcut + self.lambda_anchor, 1.0
        ):
            raise ValueError("MPPI objective weights must sum to one")
        if self.temperature <= 0.0:
            raise ValueError("MPPI temperature must be positive")
        if self.segment_minimum_pulses <= 0 or self.segment_maximum_count <= 0:
            raise ValueError("segment pulse length and count must be positive")
        if not 0.0 < self.segment_retain_fraction <= 1.0:
            raise ValueError("segment retain fraction must lie in (0, 1]")
        if self.device_memory_cap_gib <= 0.0:
            raise ValueError("device memory cap must be positive")
        if not 0.0 < self.segment_working_pool_gib <= self.device_memory_cap_gib:
            raise ValueError("segment working pool must lie in (0, device memory cap]")
        if not -1.0 <= self.roi_match_cosine_min <= 1.0:
            raise ValueError("ROI match cosine threshold must lie in [-1, 1]")
        if not -1.0 <= self.roi_linearization_cosine_min <= 1.0:
            raise ValueError("ROI linearization cosine threshold must lie in [-1, 1]")
        if self.roi_match_cosine_min > self.roi_linearization_cosine_min:
            raise ValueError("ROI match cosine threshold must not exceed linearization threshold")


@dataclass(frozen=True)
class RepairRequest:
    """Observed plan location and explicit cause for one repair invocation."""

    active_trajectory_id: str
    active_prefix: int
    trigger: RepairTrigger
    active_similarity: float | None
    random_seed: int
    artifact_directory: Path | None = None


@dataclass(frozen=True)
class RepairContext:
    """Validated repair request augmented with measured active-plan alignment."""

    request: RepairRequest
    active_similarity: float | None


@dataclass(frozen=True)
class FeasibleRepairTrajectory:
    """One independently rolled-out hard-feasible MPPI sample and its full trace."""

    trajectory_id: str
    action_hash: str
    anchor_tail_id: str
    source_id: str
    iteration: int
    sample: int
    actions: np.ndarray
    linearization_path: np.ndarray
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
    mppi_cost: float
    cost_components: tuple[float, float, float]
    admissible: bool


@dataclass(frozen=True)
class RepairIterationSummary:
    """Per-iteration anchor population and trajectory-level rejection counts."""

    iteration: int
    active_anchors: tuple[str, ...]
    hard_feasible_samples: int
    admissible_samples: int
    rejection_counts: dict[str, int]
    segment_index: int
    segment_bounds: tuple[tuple[str, int, int], ...]
    parent_count: int
    child_count: int
    retained_count: int
    capacity_records: int
    used_record_bytes: int
    compaction_before_count: int
    compaction_after_count: int
    compaction_before_bytes: int
    compaction_after_bytes: int
    support_rejections: int
    contact_guard_rejections: int
    linearization_trust_rejections: int
    sampling_seconds: float
    rollout_seconds: float
    ranking_seconds: float
    segment_rollout_rows: int = 0
    segment_hard_feasible_rows: int = 0
    shortlisted_rows: int = 0
    suffix_rollout_rows: int = 0
    final_trace_rows: int = 0
    segment_rollout_seconds: float = 0.0
    suffix_rollout_seconds: float = 0.0
    weighted_parent_count: int = 0
    weighted_feasible_child_count: int = 0
    weight_ess_min: float = 0.0
    weight_ess_median: float = 0.0
    weight_entropy_median: float = 0.0
    weighted_verification_seconds: float = 0.0
    parallel_segment_count: int = 0
    nominal_scan_seconds: float = 0.0
    boundary_l2_error_median: float = 0.0
    boundary_l2_error_p95: float = 0.0
    boundary_sign_disagreement_mean: float = 0.0
    parallel_batch_bytes: int = 0


@dataclass(frozen=True)
class RepairDiagnostics:
    """Deterministic counts and identities sufficient to audit one repair."""

    matched_tails: int
    selected_anchors: tuple[str, ...]
    dropped_anchors: tuple[str, ...]
    hard_feasible_samples: int
    admissible_samples: int
    selected_trajectory_id: str
    iterations: tuple[RepairIterationSummary, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class RepairResult:
    """Bounded ranked feasible archive with one discrete selected sample."""

    selected: FeasibleRepairTrajectory
    feasible_trajectories: tuple[FeasibleRepairTrajectory, ...]
    diagnostics: RepairDiagnostics


class PlanRepairer(Protocol):
    """Repair several state-matched tails from the currently observed task state."""

    def repair(
        self,
        context: RepairContext,
        task: StaticTaskTensors,
        anchors: MatchingTailBank,
        library: ComparisonROILibrary,
    ) -> RepairResult:
        """Return one sampled admissible trajectory or request a full global replan."""
