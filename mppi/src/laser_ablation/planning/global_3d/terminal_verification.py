"""Exact complete-plan evaluation and strict treatment-contract checks."""

from __future__ import annotations

from dataclasses import replace

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.plan import PlanVerification
from laser_ablation.core.state import VoxelState
from laser_ablation.metrics import evaluate_ablation
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator


class ExactPlanEvaluator:
    """Replay a whole plan without classifying safe incompletion as unsafe.

    The rollout terminates at exact completion, a contact failure, or a hard
    safety failure. Strict completion acceptance remains the verifier's job.
    """

    def __init__(
        self,
        simulator: ExactVoxelSimulator,
        completion_remaining_pct: float = 10.0,
        hard_margin_mm: float = 0.25,
    ) -> None:
        self.simulator = simulator
        self.completion_remaining_pct = float(completion_remaining_pct)
        self.hard_margin_mm = float(hard_margin_mm)

    def evaluate(
        self, state: VoxelState, actions: tuple[PhysicalAction, ...]
    ) -> PlanVerification:
        current = state.copy()
        prefix_metrics = []
        contact_hits = []
        rejection: str | None = None
        for pulse, action in enumerate(actions):
            contact = self.simulator.first_contact(current, action)
            contact_hits.append(contact.hit)
            if not contact.hit:
                rejection = f"pulse {pulse} misses tissue"
                break
            next_state = self.simulator.step(current, action)
            current = next_state
            metrics = evaluate_ablation(current)
            prefix_metrics.append(metrics)
            if metrics.hard_violations > 0:
                rejection = f"pulse {pulse} removes protected tissue"
                break
            if metrics.minimum_clearance_mm < self.hard_margin_mm:
                rejection = f"pulse {pulse} violates the hard clearance margin"
                break
            if metrics.is_complete(self.completion_remaining_pct):
                break
        terminal = evaluate_ablation(current)
        return PlanVerification(
            rejection is None,
            current,
            terminal,
            tuple(prefix_metrics),
            tuple(contact_hits),
            tuple(actions[: len(contact_hits)]),
            rejection,
        )

    def verify_first_action(
        self, state: VoxelState, action: PhysicalAction
    ) -> PlanVerification:
        """Require contact and exact safety without requiring useful removal."""
        current = state.copy()
        contact = self.simulator.first_contact(current, action)
        if not contact.hit:
            metrics = evaluate_ablation(current)
            return PlanVerification(
                False, current, metrics, (), (False,), (action,), "action misses tissue"
            )
        next_state = self.simulator.step(current, action)
        metrics = evaluate_ablation(next_state)
        valid = (
            metrics.hard_violations == 0
            and metrics.minimum_clearance_mm >= self.hard_margin_mm
        )
        return PlanVerification(
            valid,
            next_state,
            metrics,
            (metrics,),
            (True,),
            (action,),
            None if valid else "action violates exact safety",
        )


class ExactPlanVerifier:
    """Require exact physical validity, hard safety, and terminal completion."""

    def __init__(
        self,
        simulator: ExactVoxelSimulator,
        completion_remaining_pct: float = 10.0,
        hard_margin_mm: float = 0.25,
    ) -> None:
        self.evaluator = ExactPlanEvaluator(
            simulator,
            completion_remaining_pct,
            hard_margin_mm,
        )
        self.simulator = simulator
        self.completion_remaining_pct = float(completion_remaining_pct)
        self.hard_margin_mm = float(hard_margin_mm)

    def require_complete(self, evaluation: PlanVerification) -> PlanVerification:
        """Apply strict completion acceptance without replaying the plan."""
        if (
            evaluation.valid
            and evaluation.terminal_metrics.remaining_pct
            > self.completion_remaining_pct
        ):
            return replace(
                evaluation,
                valid=False,
                rejection_reason="terminal Remaining exceeds the completion threshold",
            )
        return evaluation

    def verify(
        self, state: VoxelState, actions: tuple[PhysicalAction, ...]
    ) -> PlanVerification:
        return self.require_complete(self.evaluator.evaluate(state, actions))

    def verify_first_action(
        self, state: VoxelState, action: PhysicalAction
    ) -> PlanVerification:
        return self.evaluator.verify_first_action(state, action)


def exact_terminal_key(
    evaluation: PlanVerification,
    completion_remaining_pct: float,
    hard_margin_mm: float,
) -> tuple[float, ...]:
    """Return the lexicographic exact whole-treatment comparison key."""
    metrics = evaluation.terminal_metrics
    safe = (
        all(evaluation.contact_hits)
        and metrics.hard_violations == 0
        and metrics.minimum_clearance_mm >= hard_margin_mm
    )
    complete = safe and metrics.remaining_pct <= completion_remaining_pct
    executed = evaluation.executed_actions
    pulse_count = float(len(executed))
    if complete:
        return (
            0.0,
            pulse_count,
            metrics.total_overcut_pct,
            metrics.remaining_pct,
        )
    if safe:
        return (
            1.0,
            metrics.remaining_pct,
            pulse_count,
            metrics.total_overcut_pct,
        )
    return (
        2.0,
        float(metrics.hard_violations),
        -metrics.minimum_clearance_mm,
        metrics.remaining_pct,
    )


__all__ = [
    "ExactPlanEvaluator",
    "ExactPlanVerifier",
    "exact_terminal_key",
]
