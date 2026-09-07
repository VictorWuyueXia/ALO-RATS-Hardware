"""Historical spherical-cavity sidewall scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build the archived cavity shell and angled access tunnel."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    center = np.asarray(values["cavity_center_xyz_mm"], dtype=float)
    cavity_radius = float(values["cavity_radius_mm"])
    shell_radius = float(values["target_shell_radius_mm"])
    tilt = np.deg2rad(float(values["tunnel_tilt_deg"]))
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    initial = zz <= 0.0
    radial = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    tunnel_x = -0.30 + (-zz) * np.tan(tilt)
    tunnel = ((xx - tunnel_x) ** 2 + yy**2 < float(values["tunnel_radius_mm"]) ** 2)
    tunnel &= (zz <= 0.05) & (zz >= -1.55)
    initial &= ~((radial < cavity_radius) | tunnel)
    downstream = np.sin(tilt) * (xx - center[0]) - np.cos(tilt) * (zz - center[2])
    target = initial & (radial >= cavity_radius) & (radial < shell_radius)
    target &= (downstream > 0.30) & (np.abs(yy) < float(values["target_half_width_y_mm"]))
    constraint_depth = float(values["constraint_depth_mm"])
    constraint_surface = np.full((x.size, y.size), -constraint_depth)
    constraint = zz <= constraint_surface[:, :, None]
    if center.shape != (3,) or not 0.0 < cavity_radius < shell_radius:
        raise ValueError("cavity-sidewall center and radii are invalid")
    return VoxelGeometry(
        x, y, z, initial, target, constraint, spacing, 0.0, False, None,
        constraint_surface,
        {
            "geometry": "cavity_sidewall",
            "geometry_class": "native_3d_development",
            "target_shape": "spherical_cavity_sidewall",
            "archived_source": "geometry/development_cases.py",
            "formal_test": False,
        },
    )


__all__ = ["build"]
