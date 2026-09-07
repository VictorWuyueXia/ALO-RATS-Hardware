"""Historical oblique-surface elliptical scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build an elliptical target at constant depth below a tilted surface."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    radius_x, radius_y = map(float, values["target_radii_xy_mm"])
    slope_x, slope_y = map(float, values["surface_slopes_xy"])
    depth = float(values["target_depth_mm"])
    constraint_depth = float(values["constraint_depth_mm"])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    surface = slope_x * xx + slope_y * yy
    initial = zz <= surface
    support = (xx / radius_x) ** 2 + (yy / radius_y) ** 2 < 1.0
    target = support & (zz <= surface) & (zz > surface - depth)
    constraint_surface = surface[:, :, 0] - constraint_depth
    constraint = zz <= constraint_surface[:, :, None]
    if min(radius_x, radius_y, depth, constraint_depth - depth) <= 0.0:
        raise ValueError("oblique-surface dimensions must be positive and safe")
    return VoxelGeometry(
        x, y, z, initial, target, constraint, spacing, 0.0, False, None,
        constraint_surface,
        {
            "geometry": "oblique_surface",
            "geometry_class": "native_3d_development",
            "target_shape": "tilted_elliptical_column",
            "archived_source": "geometry/development_cases.py",
            "formal_test": False,
        },
    )


__all__ = ["build"]
