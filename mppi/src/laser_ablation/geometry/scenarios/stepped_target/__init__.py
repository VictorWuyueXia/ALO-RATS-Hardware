"""Two-depth stepped-target scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build a rectangular target with an oblique two-depth interface."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    half_width_x, half_width_y = map(float, values["target_half_width_xy_mm"])
    shallow_depth, deep_depth = map(float, values["target_depths_mm"])
    interface_y_slope = float(values["interface_y_slope"])
    constraint_depth = float(values["constraint_depth_mm"])
    if not 0.0 < shallow_depth < deep_depth < constraint_depth:
        raise ValueError("stepped target depths and constraint depth must be ordered")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    initial = zz <= 0.0
    bottom = np.where(
        xx + interface_y_slope * yy < 0.0,
        -shallow_depth,
        -deep_depth,
    )
    target = (
        initial
        & (np.abs(xx) <= half_width_x)
        & (np.abs(yy) <= half_width_y)
        & (zz >= bottom)
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
        False,
        None,
        constraint_surface,
        {
            "geometry": "stepped_target",
            "geometry_class": "native_3d",
            "target_shape": "oblique_two_depth_step",
            "target_depths_mm": (shallow_depth, deep_depth),
            "formal_test": False,
        },
    )


__all__ = ["build"]
