"""Exact-voxel to negative-inside signed-distance observation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from laser_ablation.core.sdf import SDFObservation, SDFState
from laser_ablation.core.state import VoxelState


def signed_distance(mask: np.ndarray, spacing_mm: float, truncation_mm: float) -> np.ndarray:
    """Return a voxel-face-corrected SDF: negative inside, positive outside."""
    occupancy = np.asarray(mask, dtype=bool)
    if not np.any(occupancy):
        return np.full(occupancy.shape, truncation_mm, dtype=np.float32)
    if np.all(occupancy):
        return np.full(occupancy.shape, -truncation_mm, dtype=np.float32)
    raw = distance_transform_edt(~occupancy, sampling=spacing_mm) - distance_transform_edt(
        occupancy, sampling=spacing_mm
    )
    corrected = np.sign(raw) * np.maximum(np.abs(raw) - 0.5 * spacing_mm, 0.0)
    return np.clip(corrected, -truncation_mm, truncation_mm).astype(np.float32)


class SDFObserver:
    """Rebuild planner observations from authoritative voxel truth only."""

    def __init__(
        self,
        truncation_mm: float = 2.0,
        hard_margin_mm: float = 0.25,
        max_quadrature_points: int = 4096,
        noise_enabled: bool = False,
        noise_std_mm: float = 0.0,
        random_seed: int | None = None,
    ) -> None:
        if truncation_mm <= 0.0 or hard_margin_mm < 0.0 or max_quadrature_points <= 0:
            raise ValueError("invalid SDF observer configuration")
        if not isinstance(noise_enabled, bool):
            raise TypeError("noise_enabled must be Boolean")
        if noise_std_mm < 0.0 or (noise_enabled and noise_std_mm <= 0.0):
            raise ValueError("enabled SDF noise requires a positive standard deviation")
        if noise_enabled and (isinstance(random_seed, bool) or not isinstance(random_seed, int)):
            raise TypeError("enabled SDF noise requires an integer random seed")
        self.truncation_mm = float(truncation_mm)
        self.hard_margin_mm = float(hard_margin_mm)
        self.max_quadrature_points = int(max_quadrature_points)
        self.noise_enabled = noise_enabled
        self.noise_std_mm = float(noise_std_mm)
        self.random_seed = random_seed
        self._noise_rng = None if not noise_enabled else np.random.default_rng(random_seed)

    def observe(self, state: VoxelState) -> SDFState:
        current = signed_distance(state.tissue, state.spacing_mm, self.truncation_mm)
        if self._noise_rng is not None:
            current = np.clip(
                current + self._noise_rng.normal(0.0, self.noise_std_mm, current.shape),
                -self.truncation_mm,
                self.truncation_mm,
            ).astype(np.float32)
        initial = signed_distance(state.initial_tissue, state.spacing_mm, self.truncation_mm)
        target = signed_distance(state.target_mask, state.spacing_mm, self.truncation_mm)
        constraint = signed_distance(state.constraint_mask, state.spacing_mm, self.truncation_mm)
        safe = np.clip(
            constraint - self.hard_margin_mm, -self.truncation_mm, self.truncation_mm
        ).astype(np.float32)
        target_points, target_weights = self._quadrature(state, state.target_mask)
        healthy_points, healthy_weights = self._quadrature(
            state, state.initial_tissue & ~state.target_mask
        )
        snapshot = state.copy()
        snapshot.tissue.setflags(write=False)
        observation = SDFObservation(
            state.x_axis_mm,
            state.y_axis_mm,
            state.z_axis_mm,
            current,
            initial,
            target,
            constraint,
            safe,
            target_points,
            target_weights,
            healthy_points,
            healthy_weights,
            self._physical_clearance(state),
            self.hard_margin_mm,
            snapshot,
        )
        remaining = state.tissue & state.target_mask
        removed = state.initial_tissue & ~state.tissue
        overcut = removed & ~state.target_mask
        tensor = np.stack(
            (
                current / self.truncation_mm,
                target / self.truncation_mm,
                safe / self.truncation_mm,
                remaining,
                removed,
                overcut,
            ),
            axis=0,
        ).astype(np.float32)
        tensor.setflags(write=False)
        return SDFState(tensor, observation)

    def _quadrature(
        self, state: VoxelState, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.argwhere(mask)
        if indices.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        original_count = indices.shape[0]
        if original_count > self.max_quadrature_points:
            selection = np.linspace(
                0, original_count - 1, self.max_quadrature_points, dtype=int
            )
            indices = indices[selection]
        points = np.column_stack(
            (
                state.x_axis_mm[indices[:, 0]],
                state.y_axis_mm[indices[:, 1]],
                state.z_axis_mm[indices[:, 2]],
            )
        ).astype(np.float32)
        weight = original_count * state.voxel_volume_mm3 / points.shape[0]
        return points, np.full(points.shape[0], weight, dtype=np.float32)

    @staticmethod
    def _physical_clearance(state: VoxelState) -> np.ndarray:
        if state.constraint_surface_z_mm is not None:
            return (
                state.z_axis_mm[None, None, :] - state.constraint_surface_z_mm[:, :, None]
            ).astype(np.float32)
        return (
            distance_transform_edt(~state.constraint_mask, sampling=state.spacing_mm)
            - state.spacing_mm
        ).astype(np.float32)
