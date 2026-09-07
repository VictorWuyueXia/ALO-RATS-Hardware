"""Plan, verification, and controller result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.metrics import AblationMetrics
from laser_ablation.core.state import VoxelState


@dataclass(frozen=True)
class GlobalPlan:
    actions: tuple[PhysicalAction, ...]
    terminal_metrics: AblationMetrics
    exact_verified: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanVerification:
    valid: bool
    terminal_state: VoxelState
    terminal_metrics: AblationMetrics
    prefix_metrics: tuple[AblationMetrics, ...]
    contact_hits: tuple[bool, ...]
    actions: tuple[PhysicalAction, ...]
    rejection_reason: str | None = None

    @property
    def executed_actions(self) -> tuple[PhysicalAction, ...]:
        """Actions actually applied before completion or physical termination."""
        return self.actions[: len(self.prefix_metrics)]


@dataclass(frozen=True)
class ControllerOutcome:
    final_state: VoxelState
    final_metrics: AblationMetrics
    executed_actions: tuple[PhysicalAction, ...]
    replans: int
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedControllerResult:
    """Terminal exact outcome plus identities needed to audit the closed loop."""

    outcome: ControllerOutcome
    step_records: tuple[dict[str, object], ...]
    repair_count: int
    replan_count: int
    selected_trajectory_ids: tuple[str, ...]
    device_identity: tuple[str, ...]
    artifact_paths: tuple[Path, ...]
