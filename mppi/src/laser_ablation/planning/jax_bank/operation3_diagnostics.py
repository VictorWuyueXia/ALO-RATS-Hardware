"""Compact Operation-3 repair diagnostics built from one recentered workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from laser_ablation.planning.jax_bank.operation3_artifacts import save_operation3_artifacts


def _masked_cosine(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    """Report a host-side artifact value without participating in repair decisions."""
    selected_left = left[mask]
    selected_right = right[mask]
    denominator = np.linalg.norm(selected_left) * np.linalg.norm(selected_right)
    if denominator == 0.0:
        return float("nan")
    return float(selected_left @ selected_right / denominator)


def save_operation3_repair_artifacts(
    directory: Path,
    context: object,
    task: object,
    anchors: object,
    library: object,
    workspace: object,
    config: object,
    result: object,
    summaries: tuple[object, ...],
    memory_accounting: dict[str, object],
    kernel_creations: object,
) -> Path:
    """Persist one active repair without changing its linear-only selection semantics."""
    state_rows, roi_rows, refresh_rows, segment_rows = [], [], [], []
    selected_ids = {result.selected.anchor_tail_id}
    for index, tail in enumerate(anchors.tails):
        route = np.asarray(tail.linearization_path[0], dtype=np.int32)
        seed, step = map(int, route)
        valid = int(np.count_nonzero(workspace.base_roi_mask[seed, step]))
        trajectory_id = tail.trajectory_ids[0]
        decision = "selected" if tail.tail_id in selected_ids else "dropped"
        state_rows.append({
            "context": "matching", "trajectory_id": trajectory_id,
            "anchor_tail_id": tail.tail_id, "anchor_slot": index,
            "library_seed": seed, "nominal_step": step, "valid_roi_voxels": valid,
            "roi_cosine": float(tail.similarity),
            "threshold": float(config.roi_match_cosine_min), "decision": decision,
        })
        roi_rows.append({
            "context": "matching", "library_seed": seed, "nominal_step": step,
            "anchor_slot": index, "valid_roi_voxels": valid,
            "roi_cosine": float(tail.similarity), "valid_norm": True,
            "threshold": float(config.roi_match_cosine_min), "decision": decision,
        })
    for slot in range(workspace.anchor_count):
        for local_step in np.flatnonzero(workspace.center_action_mask[slot]):
            seed, step = map(int, workspace.anchor_routes[slot, local_step])
            use_overlay = bool(workspace.refresh_mask[slot, local_step])
            mask = workspace.base_roi_mask[seed, step]
            indices = workspace.base_roi_indices[seed, step, mask]
            cosine = _masked_cosine(
                workspace.center_states[slot, local_step].reshape(-1)[indices],
                workspace.base_nominal_states[seed, step].reshape(-1)[indices],
                np.ones(len(indices), dtype=bool),
            )
            support = (
                np.all(workspace.center_actions[slot, local_step]
                       >= workspace.base_action_support_lower[seed, step])
                and np.all(workspace.center_actions[slot, local_step]
                           <= workspace.base_action_support_upper[seed, step])
            )
            refresh_rows.append({
                "anchor_slot": slot,
                "anchor_tail_id": anchors.tails[slot].tail_id,
                "local_step": int(local_step), "library_seed": seed,
                "nominal_step": step, "state_cosine": cosine,
                "action_in_support": bool(support), "refreshed": use_overlay,
                "valid": bool(np.isfinite(cosine)),
            })
    for summary in summaries:
        for anchor_tail_id, _, _ in summary.segment_bounds:
            segment_rows.append({
                "anchor_tail_id": anchor_tail_id, "segment_index": summary.segment_index,
                "parent_count": summary.parent_count, "child_count": summary.child_count,
                "hard_feasible_count": summary.hard_feasible_samples,
                "admissible_count": summary.admissible_samples,
                "retained_count": summary.retained_count,
                "capacity_records": summary.capacity_records,
                "linearization_trust_rejections": summary.linearization_trust_rejections,
                "contact_guard_rejections": summary.contact_guard_rejections,
                "support_rejections": summary.support_rejections,
                "sampling_seconds": summary.sampling_seconds,
                "rollout_seconds": summary.rollout_seconds,
                "ranking_seconds": summary.ranking_seconds,
                "elapsed_seconds": (
                    summary.sampling_seconds + summary.rollout_seconds + summary.ranking_seconds
                ),
            })
    return save_operation3_artifacts(
        Path(directory),
        tables={
            "state_comparison_trace": tuple(state_rows), "roi_comparison": tuple(roi_rows),
            "refresh_mask": tuple(refresh_rows), "recenter_comparison": (),
            "segment_trace": tuple(segment_rows),
        },
        arrays={
            "selected_actions": result.selected.actions,
            "selected_linearization_path": result.selected.linearization_path,
            "selected_current_sdf": result.selected.current_sdf,
            "selected_anchor_routes": workspace.anchor_routes[:workspace.anchor_count],
            "selected_refresh_mask": workspace.refresh_mask[:workspace.anchor_count],
        },
        timing={
            "phase_seconds": {
                **workspace.build_phase_seconds,
                "segment_beam_repair": float(result.diagnostics.elapsed_seconds),
            },
            "workspace_kernel_creations": kernel_creations,
        },
        memory_accounting=memory_accounting,
        calibration={
            "roi_match_cosine_min": float(config.roi_match_cosine_min),
            "roi_linearization_cosine_min": float(config.roi_linearization_cosine_min),
            "status": "repair-local configuration; checkpoint calibration is separate",
        },
        interpretation_summary=(
            "# Operation-3 recentered repair\n\n"
            "This repair selected only the recentered local-linear segment beam. "
            "The base library and sparse overlays are listed separately, and no nonlinear "
            "rollout was used to select or rescue a candidate."
        ),
    )
