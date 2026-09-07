"""Create a registered synthetic occupancy volume for UI and installation checks only."""

import argparse
from pathlib import Path

import numpy as np

from alo_rats_hardware.scan_adapter import designate_task, export_volume
from alo_rats_hardware.surface_scan import SurfaceScan, save_scan
from alo_rats_hardware.task_designation import TargetRegion, TaskDesignation


def nominal_scan():
    """Create the documented synthetic surface only for installation and UI checks."""
    x, y = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61), indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    return SurfaceScan(points, np.ones(len(points), dtype=bool), "nominal_initial", 0.0,
                       "nominal_tissue", np.eye(4), np.eye(4), -3.0,
                       "synthetic_surface_not_measured_OCT", "mm", "complete_height_field")


def nominal_designation():
    """Define the nominal occupancy lattice used only to create the synthetic fixture."""
    return TaskDesignation((TargetRegion("rectangle", (0, 0), (1.0, 0.8), 0.8, None),),
                           ((-2.8, 2.8), (-2.8, 2.8), (-3.0, 0.6)), -2.0, 0.5,
                           "nominal_tissue", "nominal_registration_surface_task_only")


def main():
    """Export the same nominal lattice used by the PyBullet simulation as a processed-OCT fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError(f"Use a new fixture path: {path}")
    task = designate_task(nominal_scan(), nominal_designation())
    scan = export_volume(task.state, "nominal_processed_oct_fixture", 0.0, task.frame_id, np.eye(4))
    save_scan(scan, path)


if __name__ == "__main__":
    main()
