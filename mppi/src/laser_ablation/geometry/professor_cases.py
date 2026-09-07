"""Registered professor-requested development geometries.

These definitions match the causal-audit geometries exactly. They remain
outside the active scenario registry until an explicit development run.
"""

from __future__ import annotations

import numpy as np

from laser_ablation.geometry.base import VoxelGeometry, centred_axis


PROFESSOR_CASE_IDS = ("rounded_bottom", "v_bottom")


def professor_development_geometry(case_id: str) -> VoxelGeometry:
    """Return one exact 0.1-mm Rounded- or V-bottom development geometry."""
    if case_id not in PROFESSOR_CASE_IDS:
        raise ValueError(f"unknown professor development case: {case_id}")
    spacing = 0.1
    x = centred_axis((-1.8, 1.8), spacing)
    y = centred_axis((-1.4, 1.4), spacing)
    z = centred_axis((-3.6, 0.8), spacing)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    constraint = np.zeros_like(xx, dtype=bool)

    if case_id == "rounded_bottom":
        support = (xx / 1.10) ** 2 + (yy / 0.75) ** 2 < 1.0
        surface = -0.13 * (xx**2 + 0.65 * yy**2)
        definition = "1.55-mm target layer beneath a smooth rounded accessible surface"
    else:
        support = (np.abs(xx) < 1.10) & (np.abs(yy) < 0.72)
        surface = -np.tan(np.deg2rad(15.0)) * np.abs(xx)
        definition = (
            "1.55-mm target layer beneath two 15-degree accessible planes "
            "meeting along x=0"
        )

    initial = zz <= surface
    target = initial & support & (zz > surface - 1.55)
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
        None,
        {
            "geometry": case_id,
            "geometry_class": "professor_angle_value_development",
            "target_definition": definition,
            "formal_test": False,
            "source_contract": "icra27-global-causal-ablation:e1361ac",
        },
    )


__all__ = ["PROFESSOR_CASE_IDS", "professor_development_geometry"]
