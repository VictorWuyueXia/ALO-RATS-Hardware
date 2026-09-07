"""Fixed-pose beam geometry for the energy-only CasADi controller."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.sdf import SDFState
from laser_ablation.core.state import VoxelState
from laser_ablation.physics.super_gaussian import PhysicsConfig, laser_axis


@dataclass(frozen=True)
class BeamContactBracket:
    """Two samples bracketing one fixed spatial beam's first tissue contact."""

    entry_point_mm: np.ndarray
    beam: np.ndarray
    outside_point_mm: np.ndarray
    inside_point_mm: np.ndarray
    outside_distance_mm: float
    inside_distance_mm: float


@dataclass(frozen=True)
class BeamMPCROI:
    """Deterministic affected voxels and exact constants outside the beam tubes."""

    indices: np.ndarray
    points_mm: np.ndarray
    flat_indices: np.ndarray
    full_to_roi: np.ndarray
    target_selector: np.ndarray
    bottom_selector: np.ndarray
    normal_selector: np.ndarray
    tracking_weights: np.ndarray
    outside_remaining_volume_mm3: float
    outside_bottom_volume_mm3: float
    outside_normal_volume_mm3: float
    voxel_volume_mm3: float
    initial_target_volume_mm3: float
    hash_sha256: str

    @property
    def size(self) -> int:
        return int(self.indices.shape[0])


def maximum_active_radius_mm(physics: PhysicsConfig) -> float:
    """Finite transverse support radius at the shared maximum energy."""
    logarithm = np.log(physics.max_energy_j / physics.ablation_threshold)
    return float(
        physics.spot_size_mm
        * np.sqrt(2.0)
        * logarithm ** (1.0 / (2.0 * physics.super_gaussian_power))
    )


def beam_contact_bracket(
    tissue_sdf_mm: np.ndarray,
    state: VoxelState,
    action: PhysicalAction,
) -> BeamContactBracket:
    """Find the first outside-to-inside SDF bracket along an arbitrary beam."""
    field = np.asarray(tissue_sdf_mm, dtype=float)
    if field.shape != state.grid_shape or not np.all(np.isfinite(field)):
        raise ValueError("contact SDF must be finite and match the voxel lattice")
    beam = laser_axis(action.tilt_x_rad, action.tilt_y_rad)
    lower = np.array(
        (state.x_axis_mm[0], state.y_axis_mm[0], state.z_axis_mm[0]), dtype=float
    ) - 0.5 * state.spacing_mm
    upper = np.array(
        (state.x_axis_mm[-1], state.y_axis_mm[-1], state.z_axis_mm[-1]), dtype=float
    ) + 0.5 * state.spacing_mm
    reference = np.array((action.x_mm, action.y_mm, state.plane_z_mm), dtype=float)
    entry = reference - ((upper[2] - state.plane_z_mm) / -beam[2]) * beam
    ray_step = 0.5 * state.spacing_mm
    distances = np.arange(0.0, np.linalg.norm(upper - lower) + ray_step, ray_step)
    points = entry[None, :] + distances[:, None] * beam[None, :]
    valid = np.all((points >= lower) & (points <= upper), axis=1)
    coordinates = np.vstack(
        [np.interp(points[:, axis], values, np.arange(values.size)) for axis, values in enumerate(
            (state.x_axis_mm, state.y_axis_mm, state.z_axis_mm)
        )]
    )
    sampled = map_coordinates(field, coordinates, order=1, mode="constant", cval=float(np.max(field)))
    crossing = valid[:-1] & valid[1:] & (sampled[:-1] >= 0.0) & (sampled[1:] <= 0.0)
    hits = np.flatnonzero(crossing)
    if hits.size == 0:
        raise ValueError("nominal beam has no outside-to-inside tissue contact")
    index = int(hits[0])
    return BeamContactBracket(
        entry.astype(float),
        beam.astype(float),
        points[index].astype(float),
        points[index + 1].astype(float),
        float(distances[index]),
        float(distances[index + 1]),
    )


def build_beam_roi(
    voxel_state: VoxelState,
    sdf_state: SDFState,
    nominal_actions: tuple[PhysicalAction, ...],
    physics: PhysicsConfig,
    *,
    halo_mm: float,
    closure_voxels: int,
    tracking_base_weight: float,
    tracking_boundary_band_mm: float,
) -> BeamMPCROI:
    """Build a vectorized union of maximum-energy tubes for fixed beam poses."""
    if not nominal_actions:
        raise ValueError("the MPC ROI requires at least one nominal action")
    source = sdf_state.observation.source_voxel_state
    if not np.array_equal(source.tissue, voxel_state.tissue):
        raise ValueError("the SDF observation must describe the current exact state")
    if halo_mm < 0.0 or closure_voxels < 0:
        raise ValueError("ROI halo and closure must be nonnegative")
    if not 0.0 <= tracking_base_weight <= 1.0 or tracking_boundary_band_mm < 0.0:
        raise ValueError("tracking-weight configuration is invalid")

    grid = np.stack(
        np.meshgrid(
            voxel_state.x_axis_mm,
            voxel_state.y_axis_mm,
            voxel_state.z_axis_mm,
            indexing="ij",
        ),
        axis=-1,
    )
    points = grid.reshape(-1, 3)
    beams = np.stack(
        [laser_axis(action.tilt_x_rad, action.tilt_y_rad) for action in nominal_actions]
    )
    entries = np.stack(
        [
            np.array((action.x_mm, action.y_mm, voxel_state.plane_z_mm))
            - (
                (voxel_state.z_axis_mm[-1] - voxel_state.plane_z_mm) / -beam[2]
            )
            * beam
            for action, beam in zip(nominal_actions, beams, strict=True)
        ]
    )
    offsets = points[:, None, :] - entries[None, :, :]
    axial = np.einsum("pha,ha->ph", offsets, beams)
    radial_squared = np.sum(offsets * offsets, axis=2) - axial * axial
    radius = maximum_active_radius_mm(physics) + float(halo_mm)
    tube = (axial >= -0.5 * voxel_state.spacing_mm) & (
        radial_squared <= radius**2
    )
    roi_mask = np.any(tube, axis=1).reshape(voxel_state.grid_shape)
    if closure_voxels:
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, :] = True
        structure[1, :, 1] = True
        structure[:, 1, 1] = True
        roi_mask = binary_dilation(roi_mask, structure=structure, iterations=closure_voxels)
    indices = np.argwhere(roi_mask)
    if indices.size == 0:
        raise ValueError("beam ROI is empty")
    flat_indices = np.ravel_multi_index(indices.T, voxel_state.grid_shape)
    full_to_roi = np.full(voxel_state.grid_shape, -1, dtype=np.int32)
    full_to_roi[tuple(indices.T)] = np.arange(indices.shape[0], dtype=np.int32)
    target_columns = np.any(voxel_state.target_mask, axis=2)[:, :, None]
    bottom = voxel_state.initial_tissue & ~voxel_state.target_mask & target_columns
    normal = voxel_state.initial_tissue & ~voxel_state.target_mask & ~target_columns
    observation = sdf_state.observation
    boundary = (
        (np.abs(observation.tissue_sdf_mm) <= tracking_boundary_band_mm)
        | (np.abs(observation.target_sdf_mm) <= tracking_boundary_band_mm)
        | (np.abs(observation.safe_sdf_mm) <= tracking_boundary_band_mm)
    )
    tracking = np.full(indices.shape[0], tracking_base_weight, dtype=float)
    tracking[boundary[tuple(indices.T)]] = 1.0
    outside = ~roi_mask
    removed = voxel_state.initial_tissue & ~voxel_state.tissue
    volume = voxel_state.voxel_volume_mm3
    digest = hashlib.sha256()
    digest.update(np.asarray(voxel_state.grid_shape, np.int64).tobytes())
    digest.update(flat_indices.astype(np.int64, copy=False).tobytes())
    digest.update(np.asarray([radius, closure_voxels], np.float64).tobytes())
    return BeamMPCROI(
        indices.astype(np.int32),
        points[flat_indices].astype(np.float64),
        flat_indices.astype(np.int64),
        full_to_roi,
        voxel_state.target_mask[tuple(indices.T)],
        bottom[tuple(indices.T)],
        normal[tuple(indices.T)],
        tracking,
        float(np.count_nonzero(outside & voxel_state.tissue & voxel_state.target_mask) * volume),
        float(np.count_nonzero(outside & removed & bottom) * volume),
        float(np.count_nonzero(outside & removed & normal) * volume),
        volume,
        voxel_state.initial_target_volume_mm3,
        digest.hexdigest(),
    )
