"""Declared nominal surface fixtures; no optical scanner or simulation truth shortcut."""

from dataclasses import replace

import numpy as np

from surface_scan import SurfaceScan, rigid_transform
from task_designation import TargetRegion, TaskDesignation


def nominal_scan(*, tilted=False):
    """Sample a flat or tilted surface with deliberately anisotropic scan spacing."""
    x, y = np.meshgrid(np.linspace(-3, 3, 41), np.linspace(-3, 3, 31), indexing="ij")
    z = 0.06 * x - 0.04 * y if tilted else np.zeros_like(x)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return SurfaceScan(points, np.ones(len(points), dtype=bool), "nominal_initial", 0.0,
                       "nominal_tissue", np.eye(4), np.eye(4), -3.0,
                       "synthetic_surface_not_measured_OCT", "mm", "complete_height_field")


def nominal_designation(shape="rectangle"):
    """Provide an editable default task, not a planner-owned synthetic scenario."""
    return TaskDesignation(
        (TargetRegion(shape, (0, 0), (1.0, 0.8), 0.8, None),),
        ((-2.8, 2.8), (-2.8, 2.8), (-3.0, 0.6)), 2.0, 0.5,
        "nominal_tissue", "nominal_registration_surface_task_only",
    )


def reexpress_scan(scan, planning_from_scan_mm, scan_pose_base_m):
    """Change acquisition coordinates while preserving the registered treatment surface."""
    transform = rigid_transform(planning_from_scan_mm)
    registered = scan.registered_points()
    points = (registered - transform[:3, 3]) @ transform[:3, :3]
    return replace(scan, points_mm=points, planning_from_scan_mm=transform,
                   scan_pose_base_m=scan_pose_base_m)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from surface_scan import save_scan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=False)
    save_scan(nominal_scan(), output / "flat.npz")
    save_scan(nominal_scan(tilted=True), output / "tilted.npz")
