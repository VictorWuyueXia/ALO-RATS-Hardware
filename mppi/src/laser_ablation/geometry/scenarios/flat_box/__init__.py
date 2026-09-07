"""Flat rectangular control scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build a constant-depth rectangular target below a flat surface."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    half_width_x, half_width_y = map(float, values["target_half_width_xy_mm"])
    target_depth = float(values["target_depth_mm"])
    constraint_depth = float(values["constraint_depth_mm"])
    if min(half_width_x, half_width_y, target_depth, constraint_depth) <= 0.0:
        raise ValueError("flat-box target and constraint dimensions must be positive")
    if constraint_depth <= target_depth:
        raise ValueError("flat-box constraint must lie below the target")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    initial = zz <= 0.0
    target = (
        initial
        & (np.abs(xx) <= half_width_x)
        & (np.abs(yy) <= half_width_y)
        & (zz >= -target_depth)
    )
    constraint_surface = np.full((x.size, y.size), -constraint_depth)
    constraint = zz <= constraint_surface[:, :, None]
    return VoxelGeometry(
        x,
        y,
        z,
        initial,
        target,
        constraint,
        spacing,
        0.0,
        True,
        np.zeros((x.size, y.size)),
        constraint_surface,
        {
            "geometry": "flat_box",
            "geometry_class": "height_field",
            "target_shape": "axis_aligned_rectangle",
            "target_depth_mm": target_depth,
            "formal_test": False,
        },
    )


__all__ = ["build"]
