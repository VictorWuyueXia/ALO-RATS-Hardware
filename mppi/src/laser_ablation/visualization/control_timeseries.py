"""Render the periodic exact-repair control-session time-series artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


TRACE_COLUMNS = (
    "scenario",
    "pulse",
    "trajectory_id",
    "parent_trajectory_id",
    "remaining_pct",
    "overcut_pct",
    "control_cycle_time_s",
    "gpu_memory_total_mb",
    "roi_substate_similarity",
    "plan_bank_candidate_count",
    "plan_bank_saved_state_count",
    "administered_energy_j",
    "selected_global_plan_energy_j",
)


def write_control_timeseries(
    rows: Sequence[Mapping[str, object]],
    output_directory: Path,
    data_directory: Path | None = None,
) -> None:
    """Write the raw trace and one aligned five-panel view for each scenario."""
    if not rows:
        raise ValueError("control time-series requires at least one row")
    missing = set(TRACE_COLUMNS) - set(rows[0])
    if missing:
        raise ValueError(f"control time-series rows omit {sorted(missing)}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    data_directory = output_directory if data_directory is None else Path(data_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, data_directory / "control_timeseries.csv")
    for scenario in dict.fromkeys(str(row["scenario"]) for row in rows):
        _plot_trace(
            [row for row in rows if row["scenario"] == scenario],
            output_directory / f"control_timeseries_{scenario}",
        )


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Persist every plotted numerical value in a directly reloadable table."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trace(rows: Sequence[Mapping[str, object]], stem: Path) -> None:
    """Draw outcome, step cost, ROI agreement, plan-bank counts, and energy panels."""
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False,
    })
    pulse = _values(rows, "pulse", int)
    remaining = _values(rows, "remaining_pct", float)
    overcut = _values(rows, "overcut_pct", float)
    cycle_ms = 1_000.0 * _values(rows, "control_cycle_time_s", float)
    gpu_memory_mb = _values(rows, "gpu_memory_total_mb", float)
    roi_similarity = _values(rows, "roi_substate_similarity", float)
    plan_count = _values(rows, "plan_bank_candidate_count", float)
    state_count = _values(rows, "plan_bank_saved_state_count", float)
    energy = _values(rows, "administered_energy_j", float)
    global_energy = _values(rows, "selected_global_plan_energy_j", float)
    parent_trajectory_ids = np.asarray([
        str(row["parent_trajectory_id"]) for row in rows
    ])
    switch_indices = np.flatnonzero(
        parent_trajectory_ids[1:] != parent_trajectory_ids[:-1]
    ) + 1
    boundaries = np.concatenate(([0], switch_indices, [len(rows)]))
    segmented_pulse: list[float] = []
    segmented_global_energy: list[float] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start == 1:
            segmented_pulse.extend((pulse[start] - 0.28, pulse[start] + 0.28))
            segmented_global_energy.extend((global_energy[start], global_energy[start]))
        else:
            segmented_pulse.extend(pulse[start:stop])
            segmented_global_energy.extend(global_energy[start:stop])
        segmented_pulse.append(np.nan)
        segmented_global_energy.append(np.nan)
    figure, axes = plt.subplots(5, 1, figsize=(4.2, 6.8), sharex=True, constrained_layout=True)
    axes[0].plot(pulse, remaining, label="Remaining", color="#2A6F97")
    axes[0].plot(pulse, overcut, label="Overcut", color="#C44536")
    axes[0].set_ylabel("Volume (%)")
    axes[0].legend(fontsize=6, ncol=2)
    cycle_line = axes[1].plot(
        pulse, cycle_ms, color="#6A4C93", label="Step time"
    )[0]
    axes[1].set_ylabel("Step time (ms)")
    memory_axis = axes[1].twinx()
    memory_line = memory_axis.plot(
        pulse, gpu_memory_mb, color="#B26A00", linestyle="--", label="Process GPU memory"
    )[0]
    memory_axis.set_ylabel("GPU memory (MB)")
    memory_axis.spines["right"].set_visible(True)
    axes[1].legend(
        (cycle_line, memory_line), ("Step time", "Process GPU memory"), fontsize=5.5,
    )
    axes[2].plot(
        pulse, roi_similarity, color="#246B55",
    )
    finite_roi = roi_similarity[np.isfinite(roi_similarity)]
    if finite_roi.size:
        axes[2].set_ylim(max(-1.0, min(0.95, float(finite_roi.min()) - 0.03)), 1.01)
    axes[2].set_ylabel("Observed vs planned\nROI-state cosine similarity")
    state_line = axes[3].step(
        pulse, state_count, where="post", color="#B26A00",
        label="Plan-bank saved geometry states",
    )[0]
    axes[3].set_ylabel("Saved states (count)")
    plan_axis = axes[3].twinx()
    plan_line = plan_axis.step(
        pulse, plan_count, where="post", color="#6A4C93", linestyle="--",
        label="Plan-bank candidate action plans",
    )[0]
    plan_axis.set_ylabel("Candidate plans (count)")
    plan_axis.spines["right"].set_visible(True)
    axes[3].legend(
        (state_line, plan_line),
        ("Plan-bank saved geometry states", "Plan-bank candidate action plans"),
        fontsize=5.5,
    )
    axes[4].plot(
        pulse, energy, color="#2A6F97", linewidth=1.8,
        label="Administered pulse energy",
    )
    axes[4].plot(
        segmented_pulse, segmented_global_energy, color="#C44536",
        linestyle="--", linewidth=1.4, zorder=3,
        label="Original global-plan pulse energy",
    )
    axes[4].scatter(
        pulse[switch_indices], global_energy[switch_indices], color="#222222",
        marker="*", s=36, zorder=4, label="Parent-plan switch",
    )
    axes[4].set_ylabel("Pulse energy (J)")
    axes[4].legend(fontsize=5.5)
    axes[4].set_xlabel("Successful physical pulse")
    for extension, options in (
        (".png", {"dpi": 300}), (".svg", {}), (".pdf", {}), (".tiff", {"dpi": 600}),
    ):
        figure.savefig(stem.with_suffix(extension), bbox_inches="tight", **options)
    plt.close(figure)


def _values(rows: Sequence[Mapping[str, object]], key: str, cast: type) -> np.ndarray:
    """Convert a required trace column to a plotting vector."""
    return np.asarray([np.nan if row[key] is None else cast(row[key]) for row in rows], dtype=float)
