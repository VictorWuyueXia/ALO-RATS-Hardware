"""Reconstruct fixed-grid occupancy from processed surface or volumetric OCT."""

import numpy as np

from laser_ablation.control.interaction import ControllerObservation, DesignatedTask
from laser_ablation.core.state import VoxelState
from surface_scan import VolumetricScan


SPACING_MM = 0.1


def lattice_axes(bounds):
    """Use voxel centers inside the declared volume with a fixed isotropic spacing."""
    axes = []
    for lower, upper in bounds:
        cells = (upper - lower) / SPACING_MM
        if not np.isclose(cells, round(cells), atol=1e-8, rtol=0):
            raise ValueError("Volume bounds must contain an integer number of 0.1 mm cells")
        axes.append(lower + (np.arange(round(cells)) + 0.5) * SPACING_MM)
    return tuple(axes)


def _occupancy(scan, axes, surface_required):
    if isinstance(scan, VolumetricScan):
        tissue = scan.occupancy_on(axes)
        return tissue, scan.surface_on(*axes[:2]) if surface_required else None
    if not np.isclose(scan.lower_boundary_mm, axes[2][0] - SPACING_MM / 2, atol=1e-8, rtol=0):
        raise ValueError("Scan and task disagree on the trusted lower volume boundary")
    surface = scan.surface_on(*axes[:2])
    if np.any(surface > axes[2][-1] + SPACING_MM / 2):
        raise ValueError("Scan surface extends above the modeled volume")
    return axes[2][None, None, :] <= surface[:, :, None] + 1e-8, surface


def designate_task(scan, designation):
    """Convert the approved footprint/depth definition into immutable volumetric masks."""
    if scan.frame_id != designation.frame_id:
        raise ValueError("Scan and designated task use different planning frames")
    axes = lattice_axes(designation.grid_bounds_mm)
    tissue, surface = _occupancy(scan, axes, True)
    x, y = np.meshgrid(*axes[:2], indexing="ij")
    target = np.zeros(tissue.shape, dtype=bool)
    for region in designation.regions:
        inside, depth = region.columns_and_depth(x, y)
        floor = surface - depth
        if not inside.any() or np.any(floor[inside] < scan.lower_boundary_mm):
            raise ValueError("Target is empty or extends below known tissue")
        target |= inside[:, :, None] & (axes[2][None, None, :] >= floor[:, :, None] - 1e-8)
    target &= tissue
    protected = tissue & (axes[2][None, None, :] <= designation.protected_floor_mm)
    if not target.any() or not protected.any() or np.any(target & protected):
        raise ValueError("Target/protection must be nonempty and disjoint")
    state = VoxelState(
        *axes, tissue, tissue.copy(), target, protected, SPACING_MM, designation.plane_z_mm,
        provenance={"source": scan.provenance, "initial_scan_id": scan.scan_id,
                    "observation_model": scan.representation},
    )
    return DesignatedTask(state, designation.frame_id, designation.authority_id)


def observe_task(scan, task, sequence, command_id):
    """Retain all task masks while rebuilding only current tissue from the fresh scan."""
    if scan.frame_id != task.frame_id:
        raise ValueError("New scan is not registered to the treatment frame")
    original = task.state
    tissue, _ = _occupancy(scan, (original.x_axis_mm, original.y_axis_mm, original.z_axis_mm), False)
    if np.any(tissue & ~original.initial_tissue):
        raise ValueError("Reconstructed tissue extends outside initial tissue")
    state = original.copy()
    state.tissue = tissue
    return ControllerObservation(state, task.task_id, scan.scan_id, sequence,
                                 scan.timestamp_s, command_id)


def export_volume(state, scan_id, timestamp_s, frame_id, scan_pose_base_m):
    """Create an ideal registered processed-OCT volume without sharing mutable plant state."""
    from surface_scan import VolumetricScan

    return VolumetricScan(
        state.x_axis_mm, state.y_axis_mm, state.z_axis_mm, state.tissue,
        np.ones(state.grid_shape, dtype=bool), scan_id, timestamp_s, frame_id,
        scan_pose_base_m, "synthetic_registered_volumetric_OCT_from_achieved_beam",
        "mm", "segmented_occupancy_volume",
    )


def export_surface(state):
    """Export visible voxel faces only when every column is a supported solid height field."""
    occupied = np.asarray(state.tissue)
    counts = occupied.sum(axis=2)
    solid_columns = np.arange(occupied.shape[2])[None, None, :] < counts[:, :, None]
    if not np.array_equal(occupied, solid_columns):
        raise ValueError("Residual contains a cavity or overhang; surface-only observation is unsupported")
    surface = state.z_axis_mm[0] - state.spacing_mm / 2 + counts * state.spacing_mm
    x, y = np.meshgrid(state.x_axis_mm, state.y_axis_mm, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), surface.ravel()))


def export_surface_envelope(state):
    """Render the upper occupancy envelope without replacing volumetric controller data."""
    occupied = np.asarray(state.tissue)
    top = np.where(occupied, np.arange(occupied.shape[2]), -1).max(axis=2)
    surface = np.where(top >= 0, state.z_axis_mm[np.maximum(top, 0)] + state.spacing_mm / 2,
                       state.z_axis_mm[0] - state.spacing_mm / 2)
    x, y = np.meshgrid(state.x_axis_mm, state.y_axis_mm, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), surface.ravel()))
