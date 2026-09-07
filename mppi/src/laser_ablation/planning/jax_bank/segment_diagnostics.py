"""Operation-2 scalar artifact assembly for one bounded segment-beam repair."""

from __future__ import annotations

import numpy as np

from laser_ablation.planning.jax_bank.contracts import MatchingTailBank, StaticTaskTensors
from laser_ablation.planning.jax_bank.linear_contracts import LinearizationLibrary
from laser_ablation.planning.jax_bank.mppi_records import provisional_score
from laser_ablation.planning.jax_bank.repair import (
    MPPIRepairConfig,
    RepairContext,
    RepairIterationSummary,
    RepairResult,
)
from laser_ablation.planning.jax_bank.repair_artifacts import save_repair_artifacts


def save_segment_repair_artifacts(
    directory: object,
    context: RepairContext,
    task: StaticTaskTensors,
    anchors: MatchingTailBank,
    library: LinearizationLibrary,
    config: MPPIRepairConfig,
    result: RepairResult,
    summaries: tuple[RepairIterationSummary, ...],
    maximum_pulses: int,
    record_bytes: int,
    memory_accounting: dict[str, int],
    microbatch_size: int,
) -> None:
    """Persist retained linear candidates and scalar stage accounting only."""
    lengths = {tail.tail_id: len(tail.actions) for tail in anchors.tails}
    trace = []
    for summary in summaries:
        for anchor_id, start, stop in summary.segment_bounds:
            trace.append({
                "anchor_tail_id": anchor_id, "segment_index": summary.segment_index,
                "segment_start": start, "segment_stop": stop,
                "tail_length": lengths[anchor_id], "parent_count": summary.parent_count,
                "child_count": summary.child_count,
                "hard_feasible_count": summary.hard_feasible_samples,
                "admissible_count": summary.admissible_samples,
                "retained_count": summary.retained_count,
                "retained_bytes": summary.used_record_bytes,
                "beam_capacity_records": summary.capacity_records,
                "beam_capacity_bytes": summary.capacity_records * record_bytes,
                "compacted": summary.compaction_before_count > 0,
                "pre_compaction_count": summary.compaction_before_count,
                "post_compaction_count": summary.compaction_after_count,
                "support_rejections": summary.support_rejections,
                "sampling_seconds": summary.sampling_seconds,
                "rollout_seconds": summary.rollout_seconds,
                "ranking_seconds": summary.ranking_seconds,
                "elapsed_seconds": (
                    summary.sampling_seconds + summary.rollout_seconds + summary.ranking_seconds
                ),
            })
    ranked = sorted(
        result.feasible_trajectories,
        key=lambda item: (
            item.mppi_cost, provisional_score(task, item, maximum_pulses),
            item.trajectory_id,
        ),
    )
    candidates = [
        {
            "trajectory_id": item.trajectory_id, "action_hash": item.action_hash,
            "anchor_tail_id": item.anchor_tail_id, "source_id": item.source_id,
            "library_fingerprint": library.library_fingerprint, "segment_index": item.iteration,
            "mppi_cost": item.mppi_cost,
            "provisional_score": provisional_score(task, item, maximum_pulses),
            "rank": index + 1, "hard_feasible": True, "admissible": item.admissible,
            "remaining_fraction": float(item.remaining_fraction[-1]),
            "overcut_fraction": float(item.overcut_fraction[-1]),
            "clearance_mm": float(item.clearance_mm[-1]),
            "linearization_support": bool(np.any(item.flags["linearization_support"])),
        }
        for index, item in enumerate(ranked)
    ]
    save_repair_artifacts(
        directory, context, task, anchors, config, result,
        segment_trace=trace, segment_candidates=candidates,
        memory_accounting=memory_accounting,
        timing={"linear_microbatch_size": microbatch_size},
        retained_arrays={
            "selected_actions": result.selected.actions,
            "selected_linearization_path": result.selected.linearization_path,
            "selected_current_sdf": result.selected.current_sdf,
        },
        interpretation_summary=(
            "# Operation-2 segment-beam repair\n\n"
            "The selected trajectory was ranked by stored local-linear dynamics from "
            "the observed SDF. No nonlinear fallback or fresh linearization ran.\n"
        ),
    )
