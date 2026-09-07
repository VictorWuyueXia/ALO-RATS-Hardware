"""Strict import boundary for registered, segmented processed-OCT volumes."""

from pathlib import Path

import numpy as np

from .surface_scan import VolumetricScan, load_scan


def load_processed_volume(path):
    """Load one no-pickle OCT interchange file and reject surface-only or raw scanner output."""
    scan = load_scan(Path(path))
    if not isinstance(scan, VolumetricScan):
        raise ValueError("Hardware workflow requires segmented_occupancy_volume processed OCT")
    if scan.units != "mm" or scan.representation != "segmented_occupancy_volume":
        raise ValueError("Processed OCT must be a registered millimetre occupancy volume")
    if not scan.valid.all() or not np.any(scan.tissue):
        raise ValueError("Processed OCT must have complete validity and nonempty segmented tissue")
    return scan


def volume_summary(scan):
    """Expose OCT identity and lattice bounds without reinterpreting scanner data."""
    return {
        "scan_id": scan.scan_id,
        "timestamp_s": float(scan.timestamp_s),
        "frame_id": scan.frame_id,
        "provenance": scan.provenance,
        "shape": list(scan.tissue.shape),
        "bounds_mm": [[float(axis[0]), float(axis[-1])] for axis in
                      (scan.x_axis_mm, scan.y_axis_mm, scan.z_axis_mm)],
        "occupied_voxels": int(np.count_nonzero(scan.tissue)),
        "scan_pose_base_m": scan.scan_pose_base_m.tolist(),
    }
