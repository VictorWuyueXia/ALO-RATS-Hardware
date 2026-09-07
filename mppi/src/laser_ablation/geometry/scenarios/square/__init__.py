"""Default square capability scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build the configured square target and flat tissue surface."""
    bounds = tuple(map(float, values["workspace_xy_mm"]))
    xy_voxels = int(values["xy_voxels"])
    spacing = float(values["voxel_spacing_mm"])
    z = np.asarray(values["z_axis_mm"], dtype=float)
    half_width = float(values["target_half_width_mm"])
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError("square workspace bounds must be increasing")
    if xy_voxels < 3 or spacing <= 0.0 or half_width <= 0.0:
        raise ValueError("square scenario dimensions must be positive")
    if z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0.0):
        raise ValueError("square z axis must be strictly increasing")
    axis = np.linspace(*bounds, xy_voxels)
    xx, yy, zz = np.meshgrid(axis, axis, z, indexing="ij")
    initial = zz <= 0.0
    target = initial & (np.abs(xx) <= half_width) & (np.abs(yy) <= half_width)
    constraint = np.zeros_like(initial)
    return VoxelGeometry(
        axis,
        axis.copy(),
        z,
        initial,
        target,
        constraint,
        spacing,
        0.0,
        True,
        np.zeros((axis.size, axis.size)),
        None,
        {
            "geometry": "square",
            "geometry_class": "height_field",
            "target_shape": "axis_aligned_square",
            "target_half_width_mm": half_width,
            "formal_test": False,
        },
    )


__all__ = ["build"]
