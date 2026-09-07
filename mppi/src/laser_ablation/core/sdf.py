"""Derived signed-distance observation contracts; negative values are inside."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_ablation.core.state import VoxelState


@dataclass(frozen=True)
class SDFObservation:
    """Continuous geometry observation reconstructed from exact voxel truth."""

    x_axis_mm: np.ndarray
    y_axis_mm: np.ndarray
    z_axis_mm: np.ndarray
    tissue_sdf_mm: np.ndarray
    initial_tissue_sdf_mm: np.ndarray
    target_sdf_mm: np.ndarray
    constraint_sdf_mm: np.ndarray
    safe_sdf_mm: np.ndarray
    target_points_mm: np.ndarray
    target_point_weights_mm3: np.ndarray
    healthy_points_mm: np.ndarray
    healthy_point_weights_mm3: np.ndarray
    physical_clearance_mm: np.ndarray
    hard_margin_mm: float
    source_voxel_state: VoxelState

    def sample(self, field: np.ndarray, points_mm: np.ndarray) -> np.ndarray:
        """Trilinearly sample one aligned field at physical coordinates."""
        values = np.asarray(field)
        points = np.asarray(points_mm, dtype=float)
        expected = (self.x_axis_mm.size, self.y_axis_mm.size, self.z_axis_mm.size)
        if values.shape != expected or points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("sample requires an aligned field and (...,3) points")
        flat = points.reshape(-1, 3)
        axes = (self.x_axis_mm, self.y_axis_mm, self.z_axis_mm)
        lower_bound = np.array([axis[0] for axis in axes])
        upper_bound = np.array([axis[-1] for axis in axes])
        if np.any(flat < lower_bound) or np.any(flat > upper_bound):
            raise ValueError("SDF query lies outside the observation lattice")
        lower = np.column_stack(
            [np.searchsorted(axis, flat[:, i], side="right") - 1 for i, axis in enumerate(axes)]
        )
        lower = np.clip(lower, 0, np.array(expected) - 2)
        upper = lower + 1
        lower_xyz = np.column_stack([axis[lower[:, i]] for i, axis in enumerate(axes)])
        upper_xyz = np.column_stack([axis[upper[:, i]] for i, axis in enumerate(axes)])
        fraction = (flat - lower_xyz) / (upper_xyz - lower_xyz)
        result = np.zeros(flat.shape[0], dtype=float)
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    indices = (
                        lower[:, 0] if ix == 0 else upper[:, 0],
                        lower[:, 1] if iy == 0 else upper[:, 1],
                        lower[:, 2] if iz == 0 else upper[:, 2],
                    )
                    weight = (
                        (1.0 - fraction[:, 0] if ix == 0 else fraction[:, 0])
                        * (1.0 - fraction[:, 1] if iy == 0 else fraction[:, 1])
                        * (1.0 - fraction[:, 2] if iz == 0 else fraction[:, 2])
                    )
                    result += weight * values[indices]
        return result.reshape(points.shape[:-1])


@dataclass(frozen=True)
class SDFState:
    """Planner input containing six aligned fields in ``(channel,x,y,z)`` order."""

    tensor: np.ndarray
    observation: SDFObservation

    CURRENT_TISSUE = 0
    TARGET = 1
    SAFE = 2
    REMAINING_TARGET = 3
    REMOVED = 4
    OVERCUT = 5

    def __post_init__(self) -> None:
        expected = (6,) + self.observation.tissue_sdf_mm.shape
        if self.tensor.shape != expected:
            raise ValueError(f"SDFState tensor must have shape {expected}")
