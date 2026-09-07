"""Dormant exact two-level acceptance retained for energy-only MPC candidates.

The active controller currently relies on the constrained MPC solve and does
not invoke this optional prefix/full-tail verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from laser_ablation.control.mpc.contracts import EnergyMPCFailureReason
from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.metrics import AblationMetrics
from laser_ablation.core.plan import PlanVerification
from laser_ablation.core.state import VoxelState
from laser_ablation.metrics import evaluate_ablation
from laser_ablation.physics.exact_voxel import ContactResult, ExactVoxelSimulator
from laser_ablation.planning.global_3d.terminal_verification import ExactPlanVerifier


@dataclass(frozen=True)
class ExactAcceptanceResult:
    """Result of exact prefix screening and independent full-tail replay."""

    accepted: bool
    first_action: PhysicalAction | None
    optimized_prefix: tuple[PhysicalAction, ...]
    unchanged_tail: tuple[PhysicalAction, ...]
    prefix_verifications: tuple[PlanVerification, ...]
    prefix_target_removal_mm3: tuple[float, ...]
    full_tail_verification: PlanVerification | None
    exact_verification_metrics: AblationMetrics
    failure: EnergyMPCFailureReason | None
    failure_reason: str | None
    verification_time_s: float
    prefix_verification_time_s: float = 0.0
    full_tail_verification_time_s: float = 0.0


@dataclass(frozen=True)
class FixedResponseScaleSimulatorView:
    """Bind the canonical simulator response_scale parameter."""

    simulator: ExactVoxelSimulator
    response_scale: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.response_scale) or self.response_scale <= 0.0:
            raise ValueError("response_scale must be finite and positive")

    @property
    def config(self):
        return self.simulator.config

    @property
    def bounds(self):
        return self.simulator.bounds

    def first_contact(self, state: VoxelState, action: PhysicalAction) -> ContactResult:
        return self.simulator.first_contact(state, action)

    def step(self, state: VoxelState, action: PhysicalAction) -> VoxelState:
        return self.simulator.step(state, action, response_scale=self.response_scale)


class ExactEnergyMPCAcceptor:
    """Apply the exact two-level release gate to an energy-MPC proposal."""

    def __init__(self, verifier: ExactPlanVerifier) -> None:
        self.verifier = verifier

    def verify(
        self,
        current_state: VoxelState,
        optimized_prefix: tuple[PhysicalAction, ...],
        unchanged_tail: tuple[PhysicalAction, ...],
    ) -> ExactAcceptanceResult:
        """Return a releasable first action only after both exact checks pass."""
        if not optimized_prefix:
            raise ValueError("optimized_prefix must contain at least one action")

        start_time = perf_counter()
        prefix_state = current_state.copy()
        prefix_verifications: list[PlanVerification] = []
        target_removals: list[float] = []

        # Level A is a local rollout discarded before the independent Level B.
        for pulse, action in enumerate(optimized_prefix):
            before = evaluate_ablation(prefix_state)
            verification = self.verifier.verify_first_action(prefix_state, action)
            prefix_verifications.append(verification)
            if not verification.valid:
                failure = _classify_verifier_rejection(verification)
                return self._failure_result(
                    optimized_prefix,
                    unchanged_tail,
                    tuple(prefix_verifications),
                    tuple(target_removals),
                    verification.terminal_metrics,
                    failure,
                    f"optimized-prefix pulse {pulse}: {verification.rejection_reason}",
                    start_time,
                )

            after = verification.terminal_metrics
            removed_target_mm3 = (
                before.remaining_volume_mm3 - after.remaining_volume_mm3
            )
            target_removals.append(float(removed_target_mm3))
            prefix_state = verification.terminal_state

        # Level B starts independently from current_state and replays the
        # optimized prefix followed by the unchanged nominal tail.
        complete_actions = optimized_prefix + unchanged_tail
        prefix_time = perf_counter() - start_time
        full_tail_start = perf_counter()
        full_verification = self.verifier.verify(current_state, complete_actions)
        full_tail_time = perf_counter() - full_tail_start
        if not full_verification.valid:
            failure = _classify_verifier_rejection(full_verification)
            return ExactAcceptanceResult(
                accepted=False,
                first_action=None,
                optimized_prefix=optimized_prefix,
                unchanged_tail=unchanged_tail,
                prefix_verifications=tuple(prefix_verifications),
                prefix_target_removal_mm3=tuple(target_removals),
                full_tail_verification=full_verification,
                exact_verification_metrics=full_verification.terminal_metrics,
                failure=failure,
                failure_reason=f"complete optimized-prefix + nominal-tail replay: "
                f"{full_verification.rejection_reason}",
                verification_time_s=perf_counter() - start_time,
                prefix_verification_time_s=prefix_time,
                full_tail_verification_time_s=full_tail_time,
            )

        return ExactAcceptanceResult(
            accepted=True,
            first_action=optimized_prefix[0],
            optimized_prefix=optimized_prefix,
            unchanged_tail=unchanged_tail,
            prefix_verifications=tuple(prefix_verifications),
            prefix_target_removal_mm3=tuple(target_removals),
            full_tail_verification=full_verification,
            exact_verification_metrics=full_verification.terminal_metrics,
            failure=None,
            failure_reason=None,
            verification_time_s=perf_counter() - start_time,
            prefix_verification_time_s=prefix_time,
            full_tail_verification_time_s=full_tail_time,
        )

    @staticmethod
    def _failure_result(
        optimized_prefix: tuple[PhysicalAction, ...],
        unchanged_tail: tuple[PhysicalAction, ...],
        prefix_verifications: tuple[PlanVerification, ...],
        target_removals: tuple[float, ...],
        metrics: AblationMetrics,
        failure: EnergyMPCFailureReason,
        reason: str,
        start_time: float,
    ) -> ExactAcceptanceResult:
        return ExactAcceptanceResult(
            accepted=False,
            first_action=None,
            optimized_prefix=optimized_prefix,
            unchanged_tail=unchanged_tail,
            prefix_verifications=prefix_verifications,
            prefix_target_removal_mm3=target_removals,
            full_tail_verification=None,
            exact_verification_metrics=metrics,
            failure=failure,
            failure_reason=reason,
            verification_time_s=perf_counter() - start_time,
            prefix_verification_time_s=perf_counter() - start_time,
            full_tail_verification_time_s=0.0,
        )


def _classify_verifier_rejection(
    verification: PlanVerification,
) -> EnergyMPCFailureReason:
    """Map the existing verifier result to typed MPC failures."""
    reason = (verification.rejection_reason or "").lower()
    if any(not hit for hit in verification.contact_hits) or "misses tissue" in reason:
        return EnergyMPCFailureReason.CONTACT_INVALID
    metrics = verification.terminal_metrics
    if (
        metrics.hard_violations > 0
        or "clearance" in reason
        or "protected tissue" in reason
        or "safety" in reason
    ):
        return EnergyMPCFailureReason.SAFETY_INVALID
    return EnergyMPCFailureReason.EXACT_VERIFICATION_FAILED
