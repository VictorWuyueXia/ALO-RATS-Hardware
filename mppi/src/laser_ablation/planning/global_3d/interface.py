"""Typed contracts at the exact-physics planning boundary."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import struct
from typing import Protocol

import numpy as np

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.plan import GlobalPlan, PlanVerification
from laser_ablation.core.sdf import SDFState
from laser_ablation.core.state import VoxelState


class GlobalPlanningFailure(RuntimeError):
    """Typed terminal failure when no exact/surrogate admissible global plan exists."""


def _canonical_action_matrix(
    actions: tuple[PhysicalAction, ...],
) -> np.ndarray:
    if not actions:
        return np.empty((0, 5), dtype=np.dtype("<f8"))
    values = np.asarray([action.as_array() for action in actions], dtype=np.float64)
    if values.shape != (len(actions), 5) or not np.all(np.isfinite(values)):
        raise ValueError("physical action sequences must have finite shape (N, 5)")
    values = values.copy()
    values[values == 0.0] = 0.0
    return np.ascontiguousarray(values.astype(np.dtype("<f8"), copy=False))


def canonical_action_sequence_hash(
    actions: tuple[PhysicalAction, ...],
) -> str:
    """Hash exact physical values as length-tagged little-endian float64."""
    values = _canonical_action_matrix(actions)
    digest = sha256()
    digest.update(b"laser-ablation-physical-actions-v1\0")
    digest.update(struct.pack("<QQ", values.shape[0], values.shape[1]))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def physical_action_sequences_equal(
    left: tuple[PhysicalAction, ...],
    right: tuple[PhysicalAction, ...],
) -> bool:
    """Compare ordered physical values exactly after signed-zero normalization."""
    return bool(np.array_equal(
        _canonical_action_matrix(left),
        _canonical_action_matrix(right),
    ))


@dataclass(frozen=True)
class PlanCandidate:
    """One complete-plan proposal and its independently auditable factors."""

    actions: tuple[PhysicalAction, ...]
    layout_id: str
    tilt_pattern: str
    energy_pattern: str
    repeat_depth_fraction: float
    pulse_fraction: float
    order_pattern: str

    @property
    def pulse_count(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class VerifiedGlobalPlan:
    """Standalone exact-evaluated Global reference and immutable provenance."""

    actions: tuple[PhysicalAction, ...]
    verification: PlanVerification
    exact_terminal_key: tuple[float, ...]
    completion_remaining_pct: float
    hard_margin_mm: float
    planned_action_hash: str
    executed_prefix_hash: str
    state_hash: str
    verification_authority_hash: str
    candidate_id: str
    layout_id: str
    tilt_pattern: str
    energy_pattern: str
    repeat_depth_fraction: float
    pulse_fraction: float
    order_pattern: str
    provenance: tuple[tuple[str, str], ...]
    verification_cache_hit: bool
    cache_miss_reason: str | None
    exact_replay_count: int

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("a verified Global plan must contain a planned action")
        if canonical_action_sequence_hash(self.actions) != self.planned_action_hash:
            raise ValueError("planned_action_hash does not bind the full action sequence")
        executed = self.verification.executed_actions
        if len(executed) > len(self.actions):
            raise ValueError("exact rollout executed beyond the planned action sequence")
        if not physical_action_sequences_equal(
            executed, self.actions[: len(executed)]
        ):
            raise ValueError("exact executed actions do not match the planned prefix")
        if canonical_action_sequence_hash(executed) != self.executed_prefix_hash:
            raise ValueError("executed_prefix_hash does not bind the exact prefix")
        from laser_ablation.planning.global_3d.terminal_verification import (
            exact_terminal_key,
        )
        expected_key = exact_terminal_key(
            self.verification,
            self.completion_remaining_pct,
            self.hard_margin_mm,
        )
        if self.exact_terminal_key != expected_key:
            raise ValueError("exact_terminal_key does not match exact verification")

        for name in (
            "planned_action_hash",
            "executed_prefix_hash",
            "state_hash",
            "verification_authority_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.candidate_id:
            raise ValueError("candidate_id cannot be empty")
        if self.exact_replay_count not in (0, 1):
            raise ValueError("one binding operation can perform at most one exact replay")
        if self.verification_cache_hit != (self.exact_replay_count == 0):
            raise ValueError("cache telemetry and exact replay count disagree")
        if self.verification_cache_hit and self.cache_miss_reason is not None:
            raise ValueError("a cache hit cannot carry a miss reason")
        if not self.verification_cache_hit and not self.cache_miss_reason:
            raise ValueError("a cache miss must record its reason")

    @property
    def executed_actions(self) -> tuple[PhysicalAction, ...]:
        return self.verification.executed_actions

    def as_plan_candidate(self) -> PlanCandidate:
        """Expose the full physical plan without transferring exact authority."""
        return PlanCandidate(
            self.actions,
            self.layout_id,
            self.tilt_pattern,
            self.energy_pattern,
            self.repeat_depth_fraction,
            self.pulse_fraction,
            self.order_pattern,
        )


@dataclass(frozen=True)
class VolumetricCoverageConfig:
    """Finite candidate-bank controls.

    The default profile contains only factors with a demonstrated selected-plan
    benefit in the Phase-4 development ablation. Disabled factors remain explicit
    opt-ins so historical and audit configurations can still be replayed.
    """

    coverage_pitches_mm: tuple[float, ...] = (0.6, 0.9)
    maximum_pulses: int = 160
    maximum_candidates: int = 300
    maximum_repeats_per_anchor: int = 6
    energy_fractions: tuple[float, ...] = (0.82, 0.20, 1.00, 0.60, 0.08, 0.40)
    repeat_depth_fractions: tuple[float, ...] = (0.35, 0.65, 1.0)
    pulse_count_fractions: tuple[float, ...] = (0.8, 1.0)
    tilt_magnitude_rad: float = np.deg2rad(15.0)
    grid_offsets_xy: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.5))
    include_depth_fit_candidates: bool = False
    include_pulse_count_diversity: bool = False
    include_repeat_depth_diversity: bool = False
    include_radial_tilt_candidates: bool = False
    include_overlap_fit_candidates: bool = True


class ExactSimulator(Protocol):
    def step(self, state: VoxelState, action: PhysicalAction) -> VoxelState:
        ...


class PlanVerifier(Protocol):
    def verify(
        self, state: VoxelState, actions: tuple[PhysicalAction, ...]
    ) -> PlanVerification:
        ...


class GlobalPlanProvider(Protocol):
    def propose(self, voxel_state: VoxelState, sdf_state: SDFState) -> GlobalPlan:
        ...
