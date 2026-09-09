"""Processed OCT surface and registered volumetric observation interchange."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator


def rigid_transform(value):
    """Accept a finite proper rigid transform, not a scale or reflected coordinate map."""
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Registration must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
        raise ValueError("Registration must be a proper rigid transform")
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class SurfaceScan:
    points_mm: np.ndarray
    valid: np.ndarray
    scan_id: str
    timestamp_s: float
    frame_id: str
    planning_from_scan_mm: np.ndarray
    scan_pose_base_m: np.ndarray
    lower_boundary_mm: float
    provenance: str
    units: str
    representation: str

    def __post_init__(self):
        points = np.array(self.points_mm, dtype=float, copy=True)
        valid = np.array(self.valid, copy=True)
        if self.units != "mm" or self.representation != "complete_height_field":
            raise ValueError("Scan requires explicit mm units and complete_height_field representation")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError("Scan requires finite XYZ surface samples")
        if valid.dtype != bool or valid.shape != (len(points),) or not valid.all():
            raise ValueError("Incomplete scan coverage is not supported")
        if not self.scan_id or not self.frame_id or not self.provenance:
            raise ValueError("Scan identity, frame, and provenance are required")
        if not np.isfinite([self.timestamp_s, self.lower_boundary_mm]).all():
            raise ValueError("Scan timestamp and trusted lower boundary must be finite")
        points.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "points_mm", points)
        object.__setattr__(self, "valid", valid)
        for name in ("planning_from_scan_mm", "scan_pose_base_m"):
            object.__setattr__(self, name, rigid_transform(getattr(self, name)))
        registered = self.registered_points()
        if len(np.unique(np.round(registered[:, :2], 8), axis=0)) != len(points):
            raise ValueError("Surface has duplicate XY columns or multiple depths per column")
        if np.linalg.matrix_rank(registered[:, :2] - registered[:, :2].mean(axis=0)) < 2:
            raise ValueError("Surface samples do not span a two-dimensional scan area")
        if np.any(registered[:, 2] < self.lower_boundary_mm):
            raise ValueError("Observed surface lies below the trusted volume boundary")

    def registered_points(self):
        return self.points_mm @ self.planning_from_scan_mm[:3, :3].T + self.planning_from_scan_mm[:3, 3]

    def surface_on(self, x_axis_mm, y_axis_mm):
        """Interpolate only within measured coverage; unobserved columns are errors."""
        points = self.registered_points()
        x, y = np.meshgrid(x_axis_mm, y_axis_mm, indexing="ij")
        # Canonicalize sub-nanometer rigid-transform roundoff without extrapolating coverage.
        xy = np.round(points[:, :2], 10)
        surface = LinearNDInterpolator(xy, points[:, 2])(np.round(x, 10), np.round(y, 10))
        if not np.isfinite(surface).all():
            raise ValueError("The requested lattice extends beyond scan coverage")
        return np.asarray(surface)


@dataclass(frozen=True)
class VolumetricScan:
    """Complete segmented occupancy on the registered controller voxel lattice."""

    x_axis_mm: np.ndarray
    y_axis_mm: np.ndarray
    z_axis_mm: np.ndarray
    tissue: np.ndarray
    valid: np.ndarray
    scan_id: str
    timestamp_s: float
    frame_id: str
    scan_pose_base_m: np.ndarray
    provenance: str
    units: str
    representation: str

    def __post_init__(self):
        axes = tuple(np.array(getattr(self, name), dtype=float, copy=True)
                     for name in ("x_axis_mm", "y_axis_mm", "z_axis_mm"))
        tissue, valid = np.array(self.tissue, copy=True), np.array(self.valid, copy=True)
        if self.units != "mm" or self.representation != "segmented_occupancy_volume":
            raise ValueError("Volumetric OCT requires mm units and segmented_occupancy_volume representation")
        if any(axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
               or np.any(np.diff(axis) <= 0) for axis in axes):
            raise ValueError("Volumetric OCT axes must be finite, one-dimensional, and strictly increasing")
        shape = tuple(len(axis) for axis in axes)
        if tissue.dtype != bool or tissue.shape != shape or valid.dtype != bool or valid.shape != shape:
            raise ValueError(f"Volumetric OCT tissue and validity must be Boolean arrays of shape {shape}")
        if not valid.all():
            raise ValueError("Incomplete volumetric OCT coverage is unsupported")
        if not self.scan_id or not self.frame_id or not self.provenance or not np.isfinite(self.timestamp_s):
            raise ValueError("Volumetric OCT identity, frame, provenance, and timestamp are required")
        for name, axis in zip(("x_axis_mm", "y_axis_mm", "z_axis_mm"), axes, strict=True):
            axis.setflags(write=False)
            object.__setattr__(self, name, axis)
        tissue.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "tissue", tissue)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "scan_pose_base_m", rigid_transform(self.scan_pose_base_m))

    def occupancy_on(self, axes):
        """Require the registered OCT segmentation to match the controller lattice exactly."""
        supplied = (self.x_axis_mm, self.y_axis_mm, self.z_axis_mm)
        if any(np.shape(actual) != np.shape(expected)
               or not np.allclose(actual, expected, atol=1e-10, rtol=0)
               for actual, expected in zip(supplied, axes, strict=True)):
            raise ValueError("Volumetric OCT axes do not match the registered controller lattice")
        return self.tissue.copy()

    @property
    def lower_boundary_mm(self):
        spacing = np.diff(self.z_axis_mm)
        if not np.allclose(spacing, spacing[0], atol=1e-10, rtol=0):
            raise ValueError("Volumetric OCT Z spacing must be uniform")
        return float(self.z_axis_mm[0] - spacing[0] / 2)

    def surface_on(self, x_axis_mm, y_axis_mm):
        """Use the upper occupied face only for initial depth-based task designation."""
        if (self.x_axis_mm.shape != np.shape(x_axis_mm) or self.y_axis_mm.shape != np.shape(y_axis_mm)
                or not np.allclose(self.x_axis_mm, x_axis_mm, atol=1e-10, rtol=0)
                or not np.allclose(self.y_axis_mm, y_axis_mm, atol=1e-10, rtol=0)):
            raise ValueError("Volumetric OCT XY axes do not match the designation lattice")
        occupied = np.where(self.tissue, np.arange(len(self.z_axis_mm)), -1).max(axis=2)
        if np.any(occupied < 0):
            raise ValueError("Initial volumetric OCT requires tissue in every designation column")
        return self.z_axis_mm[occupied] + (self.z_axis_mm[0] - self.lower_boundary_mm)


def save_scan(scan, path):
    """Store arrays without pickles and require metadata explicitly on reload."""
    metadata = asdict(scan)
    names = (("points_mm", "valid", "planning_from_scan_mm", "scan_pose_base_m")
             if isinstance(scan, SurfaceScan) else
             ("x_axis_mm", "y_axis_mm", "z_axis_mm", "tissue", "valid", "scan_pose_base_m"))
    arrays = {name: metadata.pop(name) for name in names}
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=json.dumps(metadata, sort_keys=True))


def load_scan(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        classes = {"complete_height_field": SurfaceScan,
                   "segmented_occupancy_volume": VolumetricScan}
        if metadata["representation"] not in classes:
            raise ValueError(f"Unsupported processed OCT representation: {metadata['representation']}")
        cls = classes[metadata["representation"]]
        return cls(**metadata, **{name: archive[name] for name in archive.files if name != "metadata"})
