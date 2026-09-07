"""Authoritative 0.1-mm Boolean voxel transition and ray contact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_ablation.core.actions import PhysicalAction, PhysicalActionBounds
from laser_ablation.core.state import VoxelState
from laser_ablation.physics.super_gaussian import PhysicsConfig, laser_axis, super_gaussian_depth


@dataclass(frozen=True)
class ContactResult:
    hit: bool
    point_mm: np.ndarray | None


class ExactVoxelSimulator:
    """The only authoritative physical transition in the clean framework."""

    def __init__(
        self,
        config: PhysicsConfig,
        bounds: PhysicalActionBounds,
        response_noise_std: float = 0.0,
        response_scale_bounds: tuple[float, float] = (0.3, 1.3),
        random_seed: int | None = None,
    ) -> None:
        canonical_energy = (config.min_energy_j, config.max_energy_j)
        if not np.allclose(bounds.energy_j, canonical_energy, rtol=0.0, atol=1e-12):
            raise ValueError(
                "action energy bounds must derive from the configured minimum and maximum"
            )
        if not np.isfinite(response_noise_std) or response_noise_std < 0.0:
            raise ValueError("response noise standard deviation must be finite and nonnegative")
        if (
            len(response_scale_bounds) != 2
            or not np.all(np.isfinite(response_scale_bounds))
            or not 0.0 < response_scale_bounds[0] < response_scale_bounds[1]
        ):
            raise ValueError("response scale bounds must be finite, positive, and increasing")
        if response_noise_std > 0.0 and (
            isinstance(random_seed, bool) or not isinstance(random_seed, int)
        ):
            raise TypeError("enabled response noise requires an integer random seed")
        self.config = config
        self.bounds = bounds
        self.response_noise_std = float(response_noise_std)
        self.response_scale_bounds = tuple(map(float, response_scale_bounds))
        self.random_seed = random_seed
        self._response_rng = (
            None if response_noise_std == 0.0 else np.random.default_rng(random_seed)
        )
        self.last_response_scale = 1.0
        self.response_scales: list[float] = []

    def first_contact(self, state: VoxelState, action: PhysicalAction) -> ContactResult:
        if not self.bounds.contains(action):
            raise ValueError("physical action lies outside canonical bounds")
        beam = laser_axis(action.tilt_x_rad, action.tilt_y_rad)
        point = self._first_ray_contact(state, action.as_array()[:2], beam)
        return ContactResult(point is not None, point)

    def step(
        self,
        state: VoxelState,
        action: PhysicalAction,
        response_scale: float | None = None,
    ) -> VoxelState:
        """Apply one contact-anchored crater with a realized physical response."""
        scale = response_scale
        if scale is None:
            scale = 1.0 if self._response_rng is None else float(np.clip(
                self._response_rng.normal(1.0, self.response_noise_std),
                *self.response_scale_bounds,
            ))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("response scale must be finite and positive")
        self.last_response_scale = float(scale)
        if self._response_rng is not None and response_scale is None:
            self.response_scales.append(self.last_response_scale)
        contact = self.first_contact(state, action)
        if not contact.hit:
            return state.copy()
        assert contact.point_mm is not None
        beam = laser_axis(action.tilt_x_rad, action.tilt_y_rad)
        occupied = np.argwhere(state.tissue)
        remove = self._support_for_indices(
            state, occupied, contact.point_mm, beam, action.energy_j, self.last_response_scale
        )
        next_state = state.copy()
        selected = occupied[remove]
        next_state.tissue[tuple(selected.T)] = False
        next_state.vertical_surface_z_mm = None
        next_state.supports_vertical_surface = False
        return next_state

    def ablation_support_mask(
        self,
        state: VoxelState,
        action: PhysicalAction,
        contact_point_mm: np.ndarray | None = None,
        response_scale: float = 1.0,
    ) -> np.ndarray:
        """Return the exact crater support on the immutable initial-tissue lattice.

        This read-only diagnostic shares the transition implementation.  Intersecting
        the result with ``state.tissue`` gives the voxels removed by ``step``;
        intersecting with already removed initial tissue measures crater overlap.
        """
        if not self.bounds.contains(action):
            raise ValueError("physical action lies outside canonical bounds")
        point = contact_point_mm
        if point is None:
            contact = self.first_contact(state, action)
            point = contact.point_mm
        support = np.zeros(state.grid_shape, dtype=bool)
        if point is None:
            return support
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("contact point must be a finite three-vector")
        indices = np.argwhere(state.initial_tissue)
        selected = self._support_for_indices(
            state,
            indices,
            point,
            laser_axis(action.tilt_x_rad, action.tilt_y_rad),
            action.energy_j,
            response_scale,
        )
        support[tuple(indices[selected].T)] = True
        return support

    def _support_for_indices(
        self,
        state: VoxelState,
        indices: np.ndarray,
        contact_point_mm: np.ndarray,
        beam: np.ndarray,
        energy_j: float,
        response_scale: float,
    ) -> np.ndarray:
        points = np.column_stack(
            (
                state.x_axis_mm[indices[:, 0]],
                state.y_axis_mm[indices[:, 1]],
                state.z_axis_mm[indices[:, 2]],
            )
        )
        offsets = points - contact_point_mm
        axial = offsets @ beam
        radial_squared = np.sum(offsets * offsets, axis=1) - axial * axial
        depth = super_gaussian_depth(
            np.maximum(radial_squared, 0.0), energy_j, self.config, response_scale
        )
        return (
            (depth > 0.0)
            & (axial >= -0.5 * state.spacing_mm)
            & (axial <= depth)
        )

    @staticmethod
    def _first_ray_contact(
        state: VoxelState, entry_xy_mm: np.ndarray, beam: np.ndarray
    ) -> np.ndarray | None:
        if beam[2] >= -np.finfo(float).eps:
            raise ValueError("the physical beam must propagate into tissue")
        spacing = state.spacing_mm
        axes = (state.x_axis_mm, state.y_axis_mm, state.z_axis_mm)
        lower = np.array([axis[0] for axis in axes]) - 0.5 * spacing
        upper = np.array([axis[-1] for axis in axes]) + 0.5 * spacing
        reference = np.array([entry_xy_mm[0], entry_xy_mm[1], state.plane_z_mm])
        entry = reference - ((upper[2] - state.plane_z_mm) / -beam[2]) * beam
        ray_step = 0.5 * spacing
        distances = np.arange(0.0, np.linalg.norm(upper - lower) + ray_step, ray_step)
        points = entry[None, :] + distances[:, None] * beam[None, :]
        indices = np.floor((points - lower) / spacing).astype(int)
        valid = np.all((indices >= 0) & (indices < np.array(state.grid_shape)), axis=1)
        occupied = np.zeros(distances.size, dtype=bool)
        occupied[valid] = state.tissue[tuple(indices[valid].T)]
        hits = np.flatnonzero(occupied)
        if hits.size == 0:
            return None
        voxel_index = indices[hits[0]]
        centre = np.array([axis[index] for axis, index in zip(axes, voxel_index)])
        voxel_lower = centre - 0.5 * spacing
        voxel_upper = centre + 0.5 * spacing
        moving = np.abs(beam) > np.finfo(float).eps
        near = np.full(3, -np.inf)
        near[moving] = np.minimum(
            (voxel_lower[moving] - entry[moving]) / beam[moving],
            (voxel_upper[moving] - entry[moving]) / beam[moving],
        )
        return entry + np.max(near) * beam
