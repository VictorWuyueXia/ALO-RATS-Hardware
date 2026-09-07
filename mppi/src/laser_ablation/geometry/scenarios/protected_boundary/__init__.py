"""Protected-sphere boundary oracle scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Build a shallow elliptical target beside a protected sphere."""
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    center_x = float(values["target_center_x_mm"])
    radius_x, radius_y = map(float, values["target_radii_xy_mm"])
    depth = float(values["target_depth_mm"])
    protected_center = np.asarray(values["protected_center_xyz_mm"], dtype=float)
    protected_radius = float(values["protected_radius_mm"])
    if protected_center.shape != (3,) or min(radius_x, radius_y, depth, protected_radius) <= 0.0:
        raise ValueError("protected-boundary dimensions are invalid")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    initial = zz <= 0.0
    support = ((xx - center_x) / radius_x) ** 2 + (yy / radius_y) ** 2 < 1.0
    target = initial & support & (zz > -depth)
    radius = np.sqrt(
        (xx - protected_center[0]) ** 2
        + (yy - protected_center[1]) ** 2
        + (zz - protected_center[2]) ** 2
    )
    constraint = initial & (radius < protected_radius)
    return VoxelGeometry(
        x, y, z, initial, target, constraint, spacing, 0.0, False, None, None,
        {
            "geometry": "protected_boundary",
            "geometry_class": "native_3d_oracle_development",
            "target_shape": "ellipse_beside_protected_sphere",
            "archived_source": "geometry/development_cases.py",
            "formal_test": False,
        },
    )


__all__ = ["build"]
