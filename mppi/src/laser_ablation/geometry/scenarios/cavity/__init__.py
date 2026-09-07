"""Internal-cavity native 3-D scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build a target shell around an internal cylindrical cavity."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    cavity_radius = float(values["cavity_radius_mm"])
    shell_radius = float(values["target_shell_radius_mm"])
    cavity_depth = float(values["cavity_center_depth_mm"])
    target_half_width_y = float(values["target_half_width_y_mm"])
    constraint_depth = float(values["constraint_depth_mm"])
    if not 0.0 < cavity_radius < shell_radius < constraint_depth:
        raise ValueError("cavity radii and constraint depth must be positive and ordered")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    solid = zz <= 0.0
    radial_squared = xx**2 + (zz + cavity_depth) ** 2
    cavity = (radial_squared <= cavity_radius**2) & (
        np.abs(yy) <= float(values["cavity_half_width_y_mm"])
    )
    initial = solid & ~cavity
    shell = (radial_squared <= shell_radius**2) & (np.abs(yy) <= target_half_width_y)
    target = initial & shell
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
            "geometry": "cavity",
            "geometry_class": "native_3d",
            "target_shape": "cavity_shell",
            "formal_test": False,
        },
    )


__all__ = ["build"]
