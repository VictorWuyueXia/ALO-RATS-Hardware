"""Historical concave notched-volume scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build the archived concave target with sloped top and bottom."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    outer_x, outer_y = map(float, values["outer_radii_xy_mm"])
    notch_x, notch_y = map(float, values["notch_radii_xy_mm"])
    notch_center_x = float(values["notch_center_x_mm"])
    constraint_depth = float(values["constraint_depth_mm"])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    initial = zz <= 0.0
    outer = (xx / outer_x) ** 2 + (yy / outer_y) ** 2 < 1.0
    notch = ((xx - notch_center_x) / notch_x) ** 2 + (yy / notch_y) ** 2 < 1.0
    top = -0.28 - 0.18 * yy + 0.08 * xx
    bottom = -2.35 + 0.22 * xx - 0.10 * yy
    target = initial & outer & ~notch & (zz <= top) & (zz > bottom)
    constraint_surface = np.full((x.size, y.size), -constraint_depth)
    constraint = zz <= constraint_surface[:, :, None]
    if min(outer_x, outer_y, notch_x, notch_y, constraint_depth) <= 0.0:
        raise ValueError("concave-volume dimensions must be positive")
    return VoxelGeometry(
        x, y, z, initial, target, constraint, spacing, 0.0, False, None,
        constraint_surface,
        {
            "geometry": "concave_volume",
            "geometry_class": "native_3d_development",
            "target_shape": "notched_concave_volume",
            "archived_source": "geometry/development_cases.py",
            "formal_test": False,
        },
    )


__all__ = ["build"]
