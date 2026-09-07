"""Deep repeated-contact oracle scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build one narrow target deeper than a single pulse can travel."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    radius_x, radius_y = map(float, values["target_radii_xy_mm"])
    depth = float(values["target_depth_mm"])
    constraint_depth = float(values["constraint_depth_mm"])
    if min(radius_x, radius_y, depth, constraint_depth - depth) <= 0.0:
        raise ValueError("repeat-depth dimensions must be positive and safe")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    initial = zz <= 0.0
    support = (xx / radius_x) ** 2 + (yy / radius_y) ** 2 < 1.0
    target = initial & support & (zz > -depth)
    constraint_surface = np.full((x.size, y.size), -constraint_depth)
    constraint = zz <= constraint_surface[:, :, None]
    return VoxelGeometry(
        x, y, z, initial, target, constraint, spacing, 0.0, False, None,
        constraint_surface,
        {
            "geometry": "repeat_depth",
            "geometry_class": "native_3d_oracle_development",
            "target_shape": "deep_elliptical_column",
            "archived_source": "geometry/development_cases.py",
            "formal_test": False,
        },
    )


__all__ = ["build"]
