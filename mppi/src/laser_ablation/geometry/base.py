"""Geometry-to-voxel-state boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from laser_ablation.core.state import VoxelState


@dataclass(frozen=True)
class VoxelGeometry:
    """Immutable lattice and task masks used to create the initial truth state."""

    x_axis_mm: np.ndarray
    y_axis_mm: np.ndarray
    z_axis_mm: np.ndarray
    initial_tissue: np.ndarray
    target_mask: np.ndarray
    constraint_mask: np.ndarray
    spacing_mm: float
    plane_z_mm: float
    supports_vertical_surface: bool = False
    initial_vertical_surface_z_mm: np.ndarray | None = None
    constraint_surface_z_mm: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def initial_state(self) -> VoxelState:
        return VoxelState(
            self.x_axis_mm,
            self.y_axis_mm,
            self.z_axis_mm,
            self.initial_tissue.copy(),
            self.initial_tissue,
            self.target_mask,
            self.constraint_mask,
            self.spacing_mm,
            self.plane_z_mm,
            self.supports_vertical_surface,
            None
            if self.initial_vertical_surface_z_mm is None
            else self.initial_vertical_surface_z_mm.copy(),
            self.constraint_surface_z_mm,
            dict(self.provenance),
        )


def centred_axis(bounds_mm: tuple[float, float], spacing_mm: float) -> np.ndarray:
    """Return voxel centres on a half-open physical interval."""
    lower, upper = map(float, bounds_mm)
    count = int(round((upper - lower) / spacing_mm))
    if count <= 0 or not np.isclose(lower + count * spacing_mm, upper):
        raise ValueError("bounds must contain an integer number of voxels")
    return np.round(lower + spacing_mm * np.arange(count), 10)
