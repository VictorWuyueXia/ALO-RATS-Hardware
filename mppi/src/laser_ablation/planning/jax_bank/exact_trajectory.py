"""Materialize final exact traces for compact segment-beam candidates."""

from __future__ import annotations

import numpy as np

from laser_ablation.planning.jax_bank.contracts import (
    PaddedActionBatch,
    StaticTaskTensors,
    action_hash,
    trajectory_hash,
)
from laser_ablation.planning.jax_bank.exact_segment import ExactSegmentBeam
from laser_ablation.planning.jax_bank.repair import FeasibleRepairTrajectory
from laser_ablation.planning.jax_bank.rollout import rollout_in_batches


def finalize_exact_beam(
    executor: object,
    state: ExactSegmentBeam,
    task: StaticTaskTensors,
) -> tuple[FeasibleRepairTrajectory, ...]:
    """Build complete SDF traces only for bounded final plan-bank candidates."""
    beam = state.beam
    rollout = rollout_in_batches(
        task, PaddedActionBatch(beam.actions, beam.action_mask, beam.source_ids),
        executor.config.rollout_batch_size, executor.devices,
    )
    flags = np.stack(tuple(rollout.flags.values()), axis=-1)
    hard = ~np.any(flags, axis=(1, 2))
    accepted = hard.copy()
    trajectories = []
    for row in np.flatnonzero(hard):
        length = int(np.count_nonzero(beam.action_mask[row]))
        normalized = (beam.actions[row] - beam.origin_actions[row]) / (
            task.upper_bounds - task.lower_bounds
        )
        deviation = np.sum(np.where(beam.action_mask[row, :, None], normalized**2, 0.0)) / (
            5.0 * length
        )
        components = (
            float(rollout.remaining_fraction[row, -1]),
            float(rollout.healthy_overcut_fraction[row, -1]), float(deviation),
        )
        cost = float(np.dot(components, (
            executor.config.lambda_remaining, executor.config.lambda_overcut,
            executor.config.lambda_anchor,
        )))
        trajectories.append(FeasibleRepairTrajectory(
            trajectory_id=trajectory_hash(
                beam.actions[row, :length], task.initial_current_sdf,
                beam.linearization_path[row, :length], beam.source_ids[row],
            ),
            action_hash=action_hash(beam.actions[row, :length]),
            anchor_tail_id=beam.anchor_ids[row], source_id=beam.source_ids[row],
            iteration=state.iteration, sample=int(state.sample[row]),
            actions=beam.actions[row, :length].copy(),
            linearization_path=beam.linearization_path[row, :length].copy(),
            current_sdf=rollout.current_sdf[row, :length + 1].copy(),
            remaining_voxels=rollout.remaining_voxels[row, :length + 1].copy(),
            overcut_voxels=rollout.overcut_voxels[row, :length + 1].copy(),
            remaining_fraction=rollout.remaining_fraction[row, :length + 1].copy(),
            overcut_fraction=rollout.overcut_fraction[row, :length + 1].copy(),
            healthy_overcut_fraction=rollout.healthy_overcut_fraction[row, :length + 1].copy(),
            clearance_mm=rollout.clearance_mm[row, :length + 1].copy(),
            pulse_count=rollout.pulse_count[row, :length + 1].copy(),
            energy_j=rollout.energy_j[row, :length + 1].copy(),
            flags={name: values[row, :length].copy() for name, values in rollout.flags.items()},
            mppi_cost=cost, cost_components=components, admissible=bool(accepted[row]),
        ))
    return tuple(sorted(trajectories, key=lambda item: (item.mppi_cost, item.trajectory_id)))
