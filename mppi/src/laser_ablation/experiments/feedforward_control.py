"""Open-loop global-plan control under authoritative stochastic physics."""

from __future__ import annotations

from dataclasses import dataclass

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.metrics import AblationMetrics
from laser_ablation.core.state import VoxelState
from laser_ablation.metrics import evaluate_ablation
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator


@dataclass(frozen=True)
class FeedforwardResult:
    """Exact terminal outcome and pulse trace without feedback or plan changes."""

    final_state: VoxelState
    final_metrics: AblationMetrics
    step_records: tuple[dict[str, object], ...]
    contact_misses: int

    def __post_init__(self) -> None:
        if self.contact_misses < 0:
            raise ValueError("feedforward contact misses must be nonnegative")
        if self.contact_misses > len(self.step_records):
            raise ValueError("contact misses cannot exceed commanded pulses")


def run_feedforward_control(
    initial_state: VoxelState,
    actions: tuple[PhysicalAction, ...],
    simulator: ExactVoxelSimulator,
) -> FeedforwardResult:
    """Execute every global-plan pulse without observing or changing the plan."""
    if not actions:
        raise ValueError("feedforward control requires at least one action")
    state = initial_state.copy()
    records: list[dict[str, object]] = []
    misses = 0
    for pulse, action in enumerate(actions):
        contact = simulator.first_contact(state, action).hit
        misses += int(not contact)
        state = simulator.step(state, action)
        metrics = evaluate_ablation(state)
        records.append({
            "pulse": pulse,
            "contact_hit": contact,
            "physics_response_scale": simulator.last_response_scale,
            "remaining_pct": metrics.remaining_pct,
            "overcut_pct": metrics.total_overcut_pct,
            "hard_violations": metrics.hard_violations,
        })
    return FeedforwardResult(state, evaluate_ablation(state), tuple(records), misses)


__all__ = ["FeedforwardResult", "run_feedforward_control"]
