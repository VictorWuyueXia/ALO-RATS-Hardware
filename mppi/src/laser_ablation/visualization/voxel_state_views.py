"""Self-contained 3-D exact-voxel views for closed-loop workflow artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from laser_ablation.core.state import VoxelState


def write_voxel_state_views(
    final_state: VoxelState,
    first_step_tissue: np.ndarray,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Render exact occupancy after pulse one and at closed-loop termination."""
    first = np.asarray(first_step_tissue)
    if first.shape != final_state.grid_shape or first.dtype != np.bool_:
        raise ValueError("first-step tissue must be a Boolean array on the exact voxel grid")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    spacing = final_state.spacing_mm
    edge_axes = tuple(
        np.concatenate((axis - 0.5 * spacing, (axis[-1] + 0.5 * spacing,)))
        for axis in (final_state.x_axis_mm, final_state.y_axis_mm, final_state.z_axis_mm)
    )
    xx, yy, zz = np.meshgrid(*edge_axes, indexing="ij")
    target_total = np.count_nonzero(final_state.initial_tissue & final_state.target_mask)
    protected_above = np.zeros_like(final_state.constraint_mask)
    protected_above[:, :, :-1] = final_state.constraint_mask[:, :, 1:]
    protected_surface = final_state.constraint_mask & ~protected_above
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42,
    })
    paths: list[Path] = []
    snapshots = (("After physical pulse 1", first, "voxel_state_after_step_001.png"),
                 ("Final physical state", final_state.tissue, "voxel_state_final.png"))
    for title, tissue, filename in snapshots:
        removed = final_state.initial_tissue & ~tissue
        remaining_target = tissue & final_state.target_mask
        removed_target = removed & final_state.target_mask
        healthy_overcut = removed & ~final_state.target_mask
        healthy = tissue & ~final_state.target_mask
        healthy_above = np.zeros_like(healthy)
        healthy_above[:, :, :-1] = tissue[:, :, 1:]
        exposed_healthy = healthy & ~healthy_above
        remaining_pct = 100.0 * np.count_nonzero(remaining_target) / target_total
        overcut_pct = 100.0 * np.count_nonzero(healthy_overcut) / target_total
        figure = plt.figure(figsize=(5.3, 4.4), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        layers = (
            (exposed_healthy, "#B8C4CE", 0.10, "Exposed healthy-tissue surface"),
            (removed_target, "#4EA8DE", 0.22, "Removed target voxels"),
            (remaining_target, "#D62828", 0.72, "Remaining target voxels"),
            (healthy_overcut, "#F4A261", 0.70, "Removed non-target voxels (overcut)"),
            (protected_surface, "#6A4C93", 0.48, "Protected-tissue upper boundary"),
        )
        for mask, color, alpha, _ in layers:
            if np.any(mask):
                axis.voxels(
                    xx, yy, zz, mask, facecolors=color, edgecolors=color,
                    linewidth=0.06, alpha=alpha, shade=True,
                )
        axis.set(
            xlabel="Lateral x (mm)", ylabel="Lateral y (mm)", zlabel="Depth z (mm)",
            title=f"{title}\nRemaining {remaining_pct:.2f}% | Overcut {overcut_pct:.2f}%",
        )
        axis.set_box_aspect(tuple(float(np.ptp(values)) for values in edge_axes))
        axis.view_init(elev=24, azim=-56)
        axis.legend(
            handles=[Patch(facecolor=color, alpha=alpha, label=label)
                     for _, color, alpha, label in layers],
            loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=6.5,
        )
        path = output / filename
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths[0], paths[1]


__all__ = ["write_voxel_state_views"]
