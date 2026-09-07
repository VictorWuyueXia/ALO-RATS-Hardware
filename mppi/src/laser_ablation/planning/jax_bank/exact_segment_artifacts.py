"""Human and machine-readable evidence for one active exact segment repair."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def save_exact_segment_artifacts(
    directory: Path,
    result: object,
    summaries: tuple[object, ...],
    task: object,
    config: object,
    maximum_pulses: int,
    capacity_records: int,
    beam_budget: int,
    final_record_bytes: int,
    working_record_bytes: int,
    parallel_diagnostics: dict[str, object] | None,
) -> Path:
    """Persist exact path-integral fan-out, suffix, timing, and compact-memory evidence."""
    root = Path(directory)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite exact segment artifacts: {root}")
    machine, human = root / "machine_readables", root / "human_readables"
    machine.mkdir(parents=True)
    human.mkdir()
    columns = (
        "segment_index", "refinement_iteration", "parent_count", "child_count",
        "segment_hard_feasible_count",
        "shortlisted_count", "suffix_verified_count", "hard_feasible_count",
        "admissible_count", "retained_count", "capacity_records", "used_record_bytes",
        "sampling_seconds", "segment_rollout_seconds", "suffix_rollout_seconds",
        "weighted_verification_seconds", "ranking_seconds", "weighted_parent_count",
        "weighted_feasible_child_count", "weight_ess_min", "weight_ess_median",
        "weight_entropy_median", "parallel_segment_count", "nominal_scan_seconds",
        "boundary_l2_error_median", "boundary_l2_error_p95",
        "boundary_sign_disagreement_mean", "parallel_batch_bytes",
    )
    rows = []
    for summary in summaries:
        rows.append({
            "segment_index": summary.segment_index,
            "refinement_iteration": summary.iteration,
            "parent_count": summary.parent_count,
            "child_count": summary.child_count,
            "segment_hard_feasible_count": summary.segment_hard_feasible_rows,
            "shortlisted_count": summary.shortlisted_rows,
            "suffix_verified_count": summary.suffix_rollout_rows,
            "hard_feasible_count": summary.hard_feasible_samples,
            "admissible_count": summary.admissible_samples,
            "retained_count": summary.retained_count,
            "capacity_records": summary.capacity_records,
            "used_record_bytes": summary.used_record_bytes,
            "sampling_seconds": summary.sampling_seconds,
            "segment_rollout_seconds": summary.segment_rollout_seconds,
            "suffix_rollout_seconds": summary.suffix_rollout_seconds,
            "weighted_verification_seconds": summary.weighted_verification_seconds,
            "ranking_seconds": summary.ranking_seconds,
            "weighted_parent_count": summary.weighted_parent_count,
            "weighted_feasible_child_count": summary.weighted_feasible_child_count,
            "weight_ess_min": summary.weight_ess_min,
            "weight_ess_median": summary.weight_ess_median,
            "weight_entropy_median": summary.weight_entropy_median,
            "parallel_segment_count": summary.parallel_segment_count,
            "nominal_scan_seconds": summary.nominal_scan_seconds,
            "boundary_l2_error_median": summary.boundary_l2_error_median,
            "boundary_l2_error_p95": summary.boundary_l2_error_p95,
            "boundary_sign_disagreement_mean": summary.boundary_sign_disagreement_mean,
            "parallel_batch_bytes": summary.parallel_batch_bytes,
        })
    with (machine / "segment_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    timing = {
        "repair_elapsed_seconds": float(result.diagnostics.elapsed_seconds),
        "sampling_seconds": float(sum(row["sampling_seconds"] for row in rows)),
        "exact_segment_rollout_seconds": float(sum(row["segment_rollout_seconds"] for row in rows)),
        "exact_suffix_rollout_seconds": float(sum(row["suffix_rollout_seconds"] for row in rows)),
        "weighted_verification_seconds": float(sum(
            row["weighted_verification_seconds"] for row in rows
        )),
        "ranking_seconds": float(sum(row["ranking_seconds"] for row in rows)),
        "nominal_center_scan_seconds": float(sum(row["nominal_scan_seconds"] for row in rows)),
        "linear_workspace_builds": 0, "linear_rollout_rows": 0,
        "final_full_trace_rows": len(result.feasible_trajectories),
    }
    gib = 1024**3
    voxels = int(np.prod(task.shape))
    memory = {
        "device_memory_cap_bytes": int(config.device_memory_cap_gib * gib),
        "active_linear_library_bytes": 0, "active_linear_workspace_bytes": 0,
        "runtime_reserve_bytes": int(1.5 * gib),
        "rollout_temporary_reserve_bytes": int(3.0 * gib),
        "staging_reserve_bytes": int(0.25 * gib), "beam_capacity_bytes": beam_budget,
        "final_full_record_bytes": final_record_bytes, "working_record_bytes": working_record_bytes,
        "capacity_records": capacity_records,
        "maximum_working_beam_bytes": capacity_records * working_record_bytes,
        "full_trace_batch_bytes": config.rollout_batch_size * (maximum_pulses + 1) * voxels * 4,
        "terminal_state_batch_bytes": config.rollout_batch_size * voxels * 4,
        "parallel_segment_batch_bytes": int(sum(row["parallel_batch_bytes"] for row in rows)),
        "planned_total_bytes": int(4.75 * gib) + beam_budget,
    }
    (machine / "timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (machine / "memory_accounting.json").write_text(
        json.dumps(memory, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    np.savez_compressed(
        machine / "selected_and_retained.npz",
        selected_actions=result.selected.actions,
        selected_routes=result.selected.linearization_path,
        selected_states=result.selected.current_sdf,
        retained_ids=np.asarray([row.trajectory_id for row in result.feasible_trajectories]),
        retained_costs=np.asarray([row.mppi_cost for row in result.feasible_trajectories], np.float32),
    )
    if parallel_diagnostics is not None:
        boundary_columns = (
            "parent_index", "segment_index", "start", "stop", "normalized_l2_error",
            "mean_absolute_sdf_error", "sdf_sign_disagreement",
        )
        boundary_rows = parallel_diagnostics["boundary_rows"]
        with (machine / "parallel_boundary_error.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=boundary_columns)
            writer.writeheader()
            writer.writerows(boundary_rows)
        (machine / "parallel_stitch_diagnostics.json").write_text(
            json.dumps({
                "parent_segment_pairs": len(parallel_diagnostics["segment_batch"].actions),
                "sampled_children": int(parallel_diagnostics["child_count"]),
                "segment_hard_feasible": int(parallel_diagnostics["segment_hard_feasible_count"]),
                "suffix_verified": int(parallel_diagnostics["suffix_verified_count"]),
                "weighted_complete_parents": int(parallel_diagnostics["weighted_parent_count"]),
                "accepted_stitched_parents": int(parallel_diagnostics["admissible_count"]),
                "parallel_segment_batch_bytes": int(parallel_diagnostics["segment_batch"].storage_bytes),
                "full_trace_rows_fetched": int(parallel_diagnostics["full_trace_rows_fetched"]),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        np.savez_compressed(
            machine / "parallel_stitched_candidates.npz",
            actions=parallel_diagnostics["stitched_actions"],
            parent_indices=parallel_diagnostics["complete_parent_indices"],
        )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    stages = np.arange(len(rows))
    stage_labels = [
        f"S{row['segment_index'] + 1}.R{row['refinement_iteration'] + 1}" for row in rows
    ]
    segment_time = np.asarray([row["segment_rollout_seconds"] for row in rows], np.float32)
    suffix_time = np.asarray([row["suffix_rollout_seconds"] for row in rows], np.float32)
    axes[0].bar(stages, segment_time, label="all exact segments")
    axes[0].bar(stages, suffix_time, bottom=segment_time, label="all feasible exact suffixes")
    axes[0].set_xticks(stages, stage_labels, rotation=25)
    axes[0].set(xlabel="segment and refinement", ylabel="wall time (s)")
    labels = ("64-row full traces", "64 terminal states", "working beam", "planned total")
    values = np.asarray((
        memory["full_trace_batch_bytes"], memory["terminal_state_batch_bytes"],
        memory["maximum_working_beam_bytes"], memory["planned_total_bytes"],
    ), np.float64) / 1024**3
    axes[1].bar(labels, values)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(ylabel="GiB per device")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    figure.savefig(human / "exact_segment_compute_memory.png", dpi=180)
    plt.close(figure)
    if parallel_diagnostics is not None:
        boundary = list(parallel_diagnostics["boundary_rows"])
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        indices = np.arange(len(boundary))
        axes[0].bar(indices, [row["normalized_l2_error"] for row in boundary])
        axes[0].set(xlabel="parent-segment pair", ylabel="normalized SDF L2 error")
        axes[1].bar(indices, [100.0 * row["sdf_sign_disagreement"] for row in boundary])
        axes[1].set(xlabel="parent-segment pair", ylabel="SDF sign disagreement (%)")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        figure.savefig(human / "parallel_stitch_boundary_error.png", dpi=180)
        plt.close(figure)
        description = (
            "Every segment was independently perturbed from its exact nominal boundary and all "
            "hard-feasible segment-plus-suffix samples formed one path-integral mean. The means "
            "were stitched per parent, then the complete stitched plans were exact-rolled out; "
            "only accepted stitched plans entered the bank. Independent segment states were "
            "used only for the boundary-error diagnostic."
        )
    else:
        description = (
            "Every sampled current segment and hard-feasible nominal suffix used the exact "
            "nonlinear SDF transition. Their full-tail costs formed one sequential path-integral "
            "mean per parent, which was exact-verified before advancing."
        )
    (human / "interpretation_summary.md").write_text(
        "# Exact path-integral repair\n\n" + description + "\n\n"
        f"The repair evaluated {sum(row['child_count'] for row in rows)} segment children, "
        f"exact-verified {sum(row['suffix_verified_count'] for row in rows)} suffixes, retained "
        f"{len(result.feasible_trajectories)} full trajectories, and completed in "
        f"{timing['repair_elapsed_seconds']:.3f} seconds.\n", encoding="utf-8",
    )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest = {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in files
    }
    (root / "manifest.json").write_text(
        json.dumps({
            "schema": (
                "parallel-segment-stitch-repair-v1" if parallel_diagnostics is not None
                else "sequential-path-integral-repair-v1"
            ),
            "files": manifest,
        }, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return root
