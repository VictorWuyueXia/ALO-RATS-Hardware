"""Typed boundary between the upstream plan, energy NLP, and exact verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, TypeAlias

import numpy as np

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.metrics import AblationMetrics
from laser_ablation.core.sdf import SDFState
from laser_ablation.core.state import VoxelState


ConstraintValue: TypeAlias = float | tuple[float, ...]


class EnergyMPCFailureReason(str, Enum):
    """Typed failure returned to the caller; this package chooses no fallback."""

    SOLVER_FAILED = "SOLVER_FAILED"
    SURROGATE_INFEASIBLE = "SURROGATE_INFEASIBLE"
    MPC_EXACT_VERIFICATION_FAILED = "MPC_EXACT_VERIFICATION_FAILED"
    # Backward-readable alias used by the exact-acceptance helper.
    EXACT_VERIFICATION_FAILED = "MPC_EXACT_VERIFICATION_FAILED"
    NO_POSITIVE_REMOVAL = "NO_POSITIVE_REMOVAL"
    CONTACT_INVALID = "CONTACT_INVALID"
    SAFETY_INVALID = "SAFETY_INVALID"


@dataclass(frozen=True)
class EnergyMPCProblem:
    """One energy-only receding-horizon problem at an exact observed state."""

    current_exact_voxel_state: VoxelState
    current_sdf_state: SDFState
    nominal_actions: tuple[PhysicalAction, ...]
    nominal_tissue_sdf_mm: tuple[np.ndarray, ...]
    active_plan_index: int
    horizon: int
    physics_fingerprint: str
    safety_fingerprint: str

    def __post_init__(self) -> None:
        if not self.nominal_actions:
            raise ValueError("nominal_actions cannot be empty")
        if isinstance(self.active_plan_index, bool) or not isinstance(
            self.active_plan_index, int
        ):
            raise TypeError("active_plan_index must be an integer")
        if not 0 <= self.active_plan_index < len(self.nominal_actions):
            raise ValueError("active_plan_index lies outside nominal_actions")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("horizon must be an integer")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(self.nominal_tissue_sdf_mm) != len(self.nominal_actions) + 1:
            raise ValueError(
                "nominal_tissue_sdf_mm must contain the initial field and one field "
                "after every nominal action"
            )
        for name, digest in (
            ("physics_fingerprint", self.physics_fingerprint),
            ("safety_fingerprint", self.safety_fingerprint),
        ):
            _validate_sha256(name, digest)
        _validate_action_sequence(self.nominal_actions)
        _validate_current_observation(
            self.current_exact_voxel_state, self.current_sdf_state
        )
        reference_shape = self.current_exact_voxel_state.grid_shape
        if any(np.asarray(field).shape != reference_shape for field in self.nominal_tissue_sdf_mm):
            raise ValueError("all nominal tissue-SDF fields must share the exact lattice")
        if any(not np.all(np.isfinite(field)) for field in self.nominal_tissue_sdf_mm):
            raise ValueError("nominal tissue-SDF fields must be finite")

    @property
    def effective_horizon(self) -> int:
        """Number of real nominal actions remaining, without fake tail padding."""
        return min(self.horizon, len(self.nominal_actions) - self.active_plan_index)

    @property
    def horizon_actions(self) -> tuple[PhysicalAction, ...]:
        stop = self.active_plan_index + self.effective_horizon
        return self.nominal_actions[self.active_plan_index : stop]

    @property
    def horizon_reference_tissue_sdf_mm(self) -> tuple[np.ndarray, ...]:
        """Nominal tissue-SDF fields after each action in the local horizon."""
        start = self.active_plan_index + 1
        return self.nominal_tissue_sdf_mm[start : start + self.effective_horizon]

    @property
    def horizon_pre_action_tissue_sdf_mm(self) -> tuple[np.ndarray, ...]:
        """Nominal pre-action tissue-SDF fields fixing each contact branch."""
        start = self.active_plan_index
        return self.nominal_tissue_sdf_mm[start : start + self.effective_horizon]


@dataclass(frozen=True)
class EnergyMPCSolution:
    """Final constrained energy-MPC result with optional dormant exact verification."""

    success: bool
    optimized_energies: tuple[float, ...] = ()
    first_action: PhysicalAction | None = None
    predicted_states: tuple[np.ndarray, ...] = ()
    objective_terms: Mapping[str, float] = field(default_factory=dict)
    constraint_values: Mapping[str, ConstraintValue] = field(default_factory=dict)
    ipopt_status: str = "NOT_RUN"
    iterations: int = 0
    solve_time_s: float = 0.0
    exact_verified: bool = False
    exact_verification_metrics: AblationMetrics | None = None
    failure_reason: EnergyMPCFailureReason | None = None

    def __post_init__(self) -> None:
        energies = tuple(float(value) for value in self.optimized_energies)
        if not all(math.isfinite(value) for value in energies):
            raise ValueError("optimized_energies must be finite")
        object.__setattr__(self, "optimized_energies", energies)
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError("iterations must be an integer")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")
        if not math.isfinite(float(self.solve_time_s)) or self.solve_time_s < 0.0:
            raise ValueError("solve_time_s must be finite and nonnegative")
        if not isinstance(self.ipopt_status, str) or not self.ipopt_status:
            raise ValueError("ipopt_status must be a nonempty string")
        _validate_numeric_mapping("objective_terms", self.objective_terms)
        _validate_constraint_mapping(self.constraint_values)
        frozen_states: list[np.ndarray] = []
        for state in self.predicted_states:
            array = np.array(state, dtype=float, copy=True)
            if not np.all(np.isfinite(array)):
                raise ValueError("predicted_states must contain only finite values")
            array.setflags(write=False)
            frozen_states.append(array)
        object.__setattr__(self, "predicted_states", tuple(frozen_states))

        if self.success:
            if self.failure_reason is not None:
                raise ValueError("a successful solution cannot carry a failure_reason")
            if self.first_action is None or not energies:
                raise ValueError("a successful solution requires energies and first_action")
        else:
            if self.failure_reason is None:
                raise ValueError("an unsuccessful solution requires a typed failure_reason")
            if self.exact_verified:
                raise ValueError("an unsuccessful solution cannot be marked exact_verified")
        if self.exact_verified and self.exact_verification_metrics is None:
            raise ValueError("exact_verified requires exact_verification_metrics")
        if self.first_action is not None and energies and not math.isclose(
            self.first_action.energy_j, energies[0], rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("first_action energy must equal optimized_energies[0]")

    def validate_against(self, problem: EnergyMPCProblem) -> None:
        """Assert horizon length and preservation of the nominal spatial action."""
        if self.success and len(self.optimized_energies) != problem.effective_horizon:
            raise ValueError("successful optimized energy vector has the wrong horizon")
        if self.predicted_states and len(self.predicted_states) != len(
            self.optimized_energies
        ) + 1:
            raise ValueError("predicted_states must include phi_0 through phi_H")
        if self.first_action is None:
            return
        nominal = problem.horizon_actions[0]
        spatial = (
            (self.first_action.x_mm, nominal.x_mm),
            (self.first_action.y_mm, nominal.y_mm),
            (self.first_action.tilt_x_rad, nominal.tilt_x_rad),
            (self.first_action.tilt_y_rad, nominal.tilt_y_rad),
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in spatial
        ):
            raise ValueError("energy MPC must preserve nominal x/y/tilt exactly")


def _validate_action_sequence(actions: tuple[PhysicalAction, ...]) -> None:
    for action in actions:
        if not isinstance(action, PhysicalAction) or not np.all(np.isfinite(action.as_array())):
            raise ValueError("nominal_actions must contain finite PhysicalAction values")


def _validate_current_observation(voxel: VoxelState, sdf: SDFState) -> None:
    source = sdf.observation.source_voxel_state
    if source.grid_shape != voxel.grid_shape:
        raise ValueError("current SDF and exact voxel state use different lattices")
    for name in ("tissue", "initial_tissue", "target_mask", "constraint_mask"):
        if not np.array_equal(getattr(source, name), getattr(voxel, name)):
            raise ValueError(f"current SDF observation is stale for exact field {name}")


def _validate_sha256(name: str, digest: str) -> None:
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _validate_numeric_mapping(name: str, values: Mapping[str, float]) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key or not math.isfinite(float(value)):
            raise ValueError(f"{name} must map nonempty strings to finite scalars")


def _validate_constraint_mapping(values: Mapping[str, ConstraintValue]) -> None:
    for key, raw in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("constraint_values keys must be nonempty strings")
        sequence = raw if isinstance(raw, tuple) else (raw,)
        if not sequence or not all(math.isfinite(float(value)) for value in sequence):
            raise ValueError("constraint_values must contain finite scalars or tuples")


__all__ = [
    "ConstraintValue",
    "EnergyMPCFailureReason",
    "EnergyMPCProblem",
    "EnergyMPCSolution",
]
