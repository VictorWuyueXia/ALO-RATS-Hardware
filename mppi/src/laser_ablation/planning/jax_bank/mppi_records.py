"""Matched-tail arrays and the shared deterministic repair ranking objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_ablation.planning.jax_bank.contracts import (
    MatchingTail,
    MatchingTailBank,
    PaddedActionBatch,
    StaticTaskTensors,
)
from laser_ablation.planning.jax_bank.repair import FeasibleRepairTrajectory


@dataclass(frozen=True)
class AnchorArrays:
    """Fixed-shape matched tails and their immutable nominal-library routes."""

    tails: tuple[MatchingTail, ...]
    origin_actions: np.ndarray
    action_mask: np.ndarray
    linearization_path: np.ndarray
    source_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]


def prepare_anchors(anchors: MatchingTailBank, maximum: int) -> AnchorArrays:
    """Select nonterminal tails and pad actions with their exact library routes."""
    available = tuple(tail for tail in anchors.tails if len(tail.actions) > 0)
    tails = tuple(sorted(
        available, key=lambda tail: (-tail.similarity, tail.score, tail.tail_id)
    )[:maximum])
    if not tails:
        raise ValueError("anchor preparation requires a nonterminal matching tail")
    batch = PaddedActionBatch.from_sequences(
        tuple(tail.actions for tail in tails), tuple(tail.source_id for tail in tails)
    )
    path = np.full(batch.action_mask.shape + (2,), -1, dtype=np.int32)
    for row, tail in enumerate(tails):
        path[row, :len(tail.actions)] = tail.linearization_path
    return AnchorArrays(
        tails, batch.actions.copy(), batch.action_mask.copy(), path,
        tuple(tail.source_id for tail in tails), tuple(tail.tail_id for tail in tails),
    )


def provisional_metrics_score(
    task: StaticTaskTensors,
    remaining_fraction: float,
    overcut_fraction: float,
    clearance_mm: float,
    pulse_count: int,
    maximum_pulses: int,
) -> float:
    """Apply the bank objective to one terminal local-linear rollout state."""
    clearance = float(clearance_mm)
    penalty = 0.0 if clearance == np.inf else 1.0 - np.clip(
        (clearance - task.hard_margin_mm) / (task.truncation_mm - task.hard_margin_mm),
        0.0,
        1.0,
    )
    return float(
        remaining_fraction
        + 0.90 * overcut_fraction
        + 0.05 * penalty
        + 0.02 * pulse_count / maximum_pulses
    )


def provisional_score(
    task: StaticTaskTensors, trajectory: FeasibleRepairTrajectory, maximum_pulses: int
) -> float:
    """Apply the existing bank objective only as a deterministic secondary key."""
    return provisional_metrics_score(
        task,
        float(trajectory.remaining_fraction[-1]),
        float(trajectory.overcut_fraction[-1]),
        float(trajectory.clearance_mm[-1]),
        int(trajectory.pulse_count[-1]),
        maximum_pulses,
    )
