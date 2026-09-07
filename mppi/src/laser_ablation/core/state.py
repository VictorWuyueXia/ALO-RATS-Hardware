"""Authoritative voxel occupancy state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VoxelState:
    """Physical truth on an ``(x, y, z)`` voxel-centre lattice.

    Immutable task masks may be shared by copies. Only ``tissue`` is mutated by
    exact transitions. A target voxel is tissue that must be removed; a
    constraint voxel belongs to a protected structure.
    """

    x_axis_mm: np.ndarray
    y_axis_mm: np.ndarray
    z_axis_mm: np.ndarray
    tissue: np.ndarray
    initial_tissue: np.ndarray
    target_mask: np.ndarray
    constraint_mask: np.ndarray
    spacing_mm: float
    plane_z_mm: float
    supports_vertical_surface: bool = False
    vertical_surface_z_mm: np.ndarray | None = None
    constraint_surface_z_mm: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (self.x_axis_mm.size, self.y_axis_mm.size, self.z_axis_mm.size)
        for name in ("tissue", "initial_tissue", "target_mask", "constraint_mask"):
            value = np.asarray(getattr(self, name))
            if value.shape != expected or value.dtype != np.bool_:
                raise ValueError(f"{name} must be a Boolean array of shape {expected}")
        if not np.isfinite(self.spacing_mm) or self.spacing_mm <= 0.0:
            raise ValueError("spacing_mm must be finite and positive")
        if self.vertical_surface_z_mm is not None and not self.supports_vertical_surface:
            raise ValueError("non-height-field states cannot expose vertical_surface_z_mm")

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.tissue.shape

    @property
    def voxel_volume_mm3(self) -> float:
        return self.spacing_mm**3

    @property
    def initial_target_volume_mm3(self) -> float:
        return float(np.count_nonzero(self.initial_tissue & self.target_mask) * self.voxel_volume_mm3)

    def copy(self) -> "VoxelState":
        return VoxelState(
            self.x_axis_mm,
            self.y_axis_mm,
            self.z_axis_mm,
            self.tissue.copy(),
            self.initial_tissue,
            self.target_mask,
            self.constraint_mask,
            self.spacing_mm,
            self.plane_z_mm,
            self.supports_vertical_surface,
            None if self.vertical_surface_z_mm is None else self.vertical_surface_z_mm.copy(),
            self.constraint_surface_z_mm,
            dict(self.provenance),
        )
