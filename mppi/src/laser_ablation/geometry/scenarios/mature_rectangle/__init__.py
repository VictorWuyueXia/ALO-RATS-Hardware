"""Seeded mature-global randomized rectangle scenario."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


def build(values: Mapping[str, Any]) -> VoxelGeometry:
    """Reconstruct the archived seeded rectangle without physical rescaling."""
    seed = int(values["geometry_seed"])
    rng = np.random.default_rng(seed)
    center = rng.uniform(-1.2, 1.2, size=2)
    half_width = rng.uniform(1.6, 3.0, size=2)
    depth = float(rng.uniform(1.2, 2.6))
    margin = float(rng.uniform(4.2, 5.2))
    slope = rng.uniform(-0.025, 0.025, size=2)
    spacing = float(values["voxel_spacing_mm"])
    x = centred_axis(tuple(map(float, values["x_bounds_mm"])), spacing)
    y = centred_axis(tuple(map(float, values["y_bounds_mm"])), spacing)
    z = centred_axis(tuple(map(float, values["z_bounds_mm"])), spacing)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    surface = slope[0] * xx + slope[1] * yy
    lateral = (
        (np.abs(xx - center[0]) < half_width[0])
        & (np.abs(yy - center[1]) < half_width[1])
    )
    initial = zz <= surface
    target = initial & lateral & (zz > surface - depth)
    constraint_surface = surface[:, :, 0] - margin
    constraint = zz <= constraint_surface[:, :, None]
    if not np.any(target) or np.any(target & constraint):
        raise RuntimeError("seeded mature rectangle produced an invalid target")
    return VoxelGeometry(
        x, y, z, initial, target, constraint, spacing, 0.0, True,
        surface[:, :, 0], constraint_surface,
        {
            "geometry": "mature_rectangle",
            "geometry_class": "mature_global_randomized_rectangle",
            "target_shape": "open_axis_aligned_rectangle",
            "geometry_seed": seed,
            "center_xy_mm": tuple(map(float, center)),
            "half_width_xy_mm": tuple(map(float, half_width)),
            "target_depth_mm": depth,
            "constraint_margin_mm": margin,
            "plane_tilt_xy": tuple(map(float, slope)),
            "archived_source": "geometry/mature_global.py",
            "formal_test": False,
        },
    )


__all__ = ["build"]
