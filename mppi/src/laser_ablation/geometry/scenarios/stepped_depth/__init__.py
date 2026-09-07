"""Historical two-level stepped-depth scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build the archived rectangular two-depth target."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    half_x, half_y = map(float, values["target_half_width_xy_mm"])
    shallow, deep = map(float, values["target_depths_mm"])
    constraint_depth = float(values["constraint_depth_mm"])
    if not 0.0 < shallow < deep < constraint_depth:
        raise ValueError("stepped depths and constraint depth must be ordered")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    initial = zz <= 0.0
    support = (np.abs(xx) < half_x) & (np.abs(yy) < half_y)
    depth = np.where(xx < 0.0, shallow, deep)
    target = initial & support & (zz > -depth)
    constraint_surface = np.full((x.size, y.size), -constraint_depth)
    constraint = zz <= constraint_surface[:, :, None]
    return VoxelGeometry(
        x, y, z, initial, target, constraint, spacing, 0.0, False, None,
        constraint_surface,
        {
            "geometry": "stepped_depth",
            "geometry_class": "native_3d_development",
            "target_shape": "axis_aligned_two_depth_step",
            "archived_source": "geometry/development_cases.py",
            "formal_test": False,
        },
    )


__all__ = ["build"]
