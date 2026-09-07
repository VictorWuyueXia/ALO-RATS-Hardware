"""Versioned, auditable Operation-3 ROI-recentering evidence artifacts."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


STATE_COMPARISON_COLUMNS = (
    "context", "trajectory_id", "anchor_tail_id", "anchor_slot", "library_seed",
    "nominal_step", "valid_roi_voxels", "roi_cosine", "threshold", "decision",
)
ROI_COMPARISON_COLUMNS = (
    "context", "library_seed", "nominal_step", "anchor_slot", "valid_roi_voxels",
    "roi_cosine", "valid_norm", "threshold", "decision",
)
REFRESH_MASK_COLUMNS = (
    "anchor_slot", "anchor_tail_id", "local_step", "library_seed", "nominal_step",
    "state_cosine", "action_in_support", "refreshed", "valid",
)
RECENTER_COMPARISON_COLUMNS = (
    "trajectory_id", "anchor_tail_id", "anchor_slot", "segment_index", "sample",
    "linear_mppi_cost", "nonlinear_mppi_cost", "state_relative_error",
    "sdf_sign_disagreement", "linear_hard_feasible", "nonlinear_hard_feasible",
    "linear_rank", "nonlinear_rank", "retained",
)
SEGMENT_TRACE_COLUMNS = (
    "anchor_tail_id", "segment_index", "parent_count", "child_count",
    "hard_feasible_count", "admissible_count", "retained_count", "capacity_records",
    "linearization_trust_rejections", "contact_guard_rejections", "support_rejections", "sampling_seconds",
    "rollout_seconds", "ranking_seconds", "elapsed_seconds",
)
CONTACT_GUARD_COLUMNS = (
    "checkpoint", "segment_index", "trajectory_id", "anchor_tail_id", "sample",
    "local_step", "roi_cosine", "contact_displacement_mm",
    "contact_displacement_voxels", "has_contact_mismatch", "entry_status_mismatch",
    "crossing_interval_mismatch", "displacement_exceeds_current",
    "rejected_by_current_guard",
)
CONTACT_SWEEP_COLUMNS = (
    "guard_mode", "contact_tolerance_voxels", "contact_tolerance_mm",
    "requires_entry_match", "requires_interval_match", "candidate_rows",
    "contact_rejected_rows", "linear_hard_feasible_rows", "false_feasible",
    "false_infeasible_rate", "nonlinear_best_coverage", "comparable_rows",
    "median_recentered_error", "p95_recentered_error", "score_rank_correlation", "passes",
)

_TABLE_COLUMNS = {
    "state_comparison_trace": STATE_COMPARISON_COLUMNS,
    "roi_comparison": ROI_COMPARISON_COLUMNS,
    "refresh_mask": REFRESH_MASK_COLUMNS,
    "recenter_comparison": RECENTER_COMPARISON_COLUMNS,
    "segment_trace": SEGMENT_TRACE_COLUMNS,
    "contact_guard_trace": CONTACT_GUARD_COLUMNS,
    "contact_guard_sweep": CONTACT_SWEEP_COLUMNS,
}


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, values: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_table(path: Path, rows: Sequence[Mapping[str, object]], columns: tuple[str, ...]) -> None:
    for row in rows:
        if set(row) != set(columns):
            missing = tuple(name for name in columns if name not in row)
            extra = tuple(name for name in row if name not in columns)
            raise ValueError(f"{path.name} schema mismatch missing={missing} extra={extra}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({
            name: (json.dumps(row[name], sort_keys=True, default=_json_default)
                   if isinstance(row[name], (dict, list, tuple)) else row[name])
            for name in columns
        } for row in rows)


def _write_plots(
    directory: Path,
    roi_rows: Sequence[Mapping[str, object]],
    rebase_rows: Sequence[Mapping[str, object]],
    refresh_rows: Sequence[Mapping[str, object]],
    contact_rows: Sequence[Mapping[str, object]],
    sweep_rows: Sequence[Mapping[str, object]],
    timing: Mapping[str, object],
    memory: Mapping[str, object],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    cosine = np.asarray([float(row["roi_cosine"]) for row in roi_rows], dtype=np.float32)
    threshold = np.asarray([float(row["threshold"]) for row in roi_rows], dtype=np.float32)
    if len(cosine):
        axes[0].plot(np.arange(len(cosine)), cosine, "o", label="routed ROI cosine")
        axes[0].plot(np.arange(len(threshold)), threshold, "--", label="threshold")
    refresh = np.asarray([bool(row["refreshed"]) for row in refresh_rows], dtype=bool)
    if len(refresh):
        axes[1].bar(("reused", "refreshed"), (int(np.count_nonzero(~refresh)), int(np.count_nonzero(refresh))))
    axes[0].set(xlabel="comparison", ylabel="cosine", ylim=(-1.02, 1.02))
    axes[1].set(ylabel="anchor-step pairs")
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(loc="best")
    figure.savefig(directory / "roi_cosine_validation.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    errors = np.asarray([float(row["state_relative_error"]) for row in rebase_rows], dtype=np.float32)
    if len(errors):
        axes[0].plot(np.arange(len(errors)), errors, "o-")
    linear = np.asarray([float(row["linear_mppi_cost"]) for row in rebase_rows], dtype=np.float32)
    nonlinear = np.asarray([float(row["nonlinear_mppi_cost"]) for row in rebase_rows], dtype=np.float32)
    if len(linear):
        axes[1].scatter(linear, nonlinear, s=18)
        bounds = np.asarray((linear.min(), linear.max(), nonlinear.min(), nonlinear.max()))
        axes[1].plot((bounds.min(), bounds.max()), (bounds.min(), bounds.max()), "k--")
    axes[0].set(xlabel="diagnostic candidate", ylabel="relative SDF error")
    axes[1].set(xlabel="linear cost", ylabel="nonlinear diagnostic cost")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(directory / "selective_refresh_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    phase_names = tuple(timing.get("phase_seconds", {}))
    phase_values = tuple(float(timing["phase_seconds"][name]) for name in phase_names)
    if phase_names:
        axes[0].bar(np.arange(len(phase_names)), phase_values)
        axes[0].set_xticks(np.arange(len(phase_names)), phase_names, rotation=45, ha="right")
    samples = memory.get("samples", ())
    peaks = [max(sample.get("process_mib", (0,))) for sample in samples]
    if peaks:
        axes[1].plot(np.arange(len(peaks)), peaks, "o-", label="own-process peak")
        axes[1].axhline(8192, color="#B26A00", linestyle="--", label="8 GiB cap")
        axes[1].legend(loc="best")
    axes[0].set(ylabel="seconds")
    axes[1].set(xlabel="memory sample", ylabel="MiB per GPU")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(directory / "device_memory_timing.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    rejected = [row for row in contact_rows if bool(row["rejected_by_current_guard"])]
    cause_names = ("contact existence", "entry status", "crossing interval", "contact displacement")
    cause_values = (
        sum(bool(row["has_contact_mismatch"]) for row in rejected),
        sum(bool(row["entry_status_mismatch"]) for row in rejected),
        sum(bool(row["crossing_interval_mismatch"]) for row in rejected),
        sum(bool(row["displacement_exceeds_current"]) for row in rejected),
    )
    axes[0].bar(np.arange(len(cause_names)), cause_values)
    axes[0].set_xticks(np.arange(len(cause_names)), cause_names, rotation=30, ha="right")
    for mode in sorted({str(row["guard_mode"]) for row in sweep_rows}):
        selected = [row for row in sweep_rows if row["guard_mode"] == mode]
        axes[1].plot(
            [float(row["contact_tolerance_voxels"]) for row in selected],
            [float(row["false_infeasible_rate"]) for row in selected], "o-", label=mode,
        )
    axes[0].set(ylabel="failed active steps", title="Current guard failure causes")
    axes[1].set(
        xlabel="contact tolerance (voxel spacings)", ylabel="false-infeasible rate",
        title="Cached guard alternatives",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    if axes[1].lines:
        axes[1].legend(loc="best", fontsize=8)
    figure.savefig(directory / "contact_guard_diagnostics.png", dpi=180)
    plt.close(figure)


def _write_manifest(directory: Path) -> None:
    manifest = directory / "manifest.json"
    records = {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "audience": "machine" if "machine_readables" in path.parts else "human",
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != manifest
    }
    _write_json(manifest, {"artifact_count": len(records), "artifacts": records})


def validate_operation3_artifact_manifest(directory: Path) -> None:
    """Reject missing, modified, or unaccounted Operation-3 evidence files."""
    location = Path(directory)
    records = json.loads((location / "manifest.json").read_text(encoding="utf-8"))["artifacts"]
    actual = {
        str(path.relative_to(location)) for path in location.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(records):
        raise ValueError("Operation-3 artifact manifest file set mismatch")
    for name, record in records.items():
        path = location / name
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Operation-3 artifact manifest size mismatch for {name}")
        if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Operation-3 artifact manifest hash mismatch for {name}")


def save_operation3_artifacts(
    directory: Path,
    *,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    arrays: Mapping[str, np.ndarray],
    timing: Mapping[str, object],
    memory_accounting: Mapping[str, object],
    calibration: Mapping[str, object],
    interpretation_summary: str,
) -> Path:
    """Save one immutable Operation-3 bundle with strict table schemas and manifest."""
    location = Path(directory)
    if location.exists():
        raise FileExistsError(f"refusing to overwrite Operation-3 artifacts: {location}")
    machine, human = location / "machine_readables", location / "human_readables"
    machine.mkdir(parents=True)
    human.mkdir()
    for name, columns in _TABLE_COLUMNS.items():
        _write_table(machine / f"{name}.csv", tuple(tables.get(name, ())), columns)
    _write_json(machine / "timing.json", dict(timing))
    _write_json(machine / "memory_accounting.json", dict(memory_accounting))
    _write_json(machine / "threshold_calibration.json", dict(calibration))
    np.savez_compressed(machine / "retained_routes_actions.npz", **arrays)
    comparison_rows = tuple(tables.get("roi_comparison", ()))
    rebase_rows = tuple(tables.get("recenter_comparison", ()))
    _write_plots(
        human, comparison_rows, rebase_rows, tuple(tables.get("refresh_mask", ())),
        tuple(tables.get("contact_guard_trace", ())),
        tuple(tables.get("contact_guard_sweep", ())),
        timing, memory_accounting,
    )
    (human / "interpretation_summary.md").write_text(interpretation_summary.rstrip() + "\n", encoding="utf-8")
    _write_manifest(location)
    return location
