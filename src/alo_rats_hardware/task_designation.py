"""Minimal saved target designation in fixed planning-frame millimeters."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TargetRegion:
    shape: str
    center_xy_mm: tuple[float, float]
    half_size_xy_mm: tuple[float, float]
    depth_mm: float
    right_depth_mm: float | None

    def __post_init__(self):
        if self.shape not in {"rectangle", "ellipse"}:
            raise ValueError("Target footprint must be rectangle or ellipse")
        center, size = np.asarray(self.center_xy_mm), np.asarray(self.half_size_xy_mm)
        if center.shape != (2,) or size.shape != (2,) or not np.isfinite([*center, *size, self.depth_mm]).all():
            raise ValueError("Target center, dimensions, and depth must be finite")
        if np.any(size <= 0) or self.depth_mm <= 0:
            raise ValueError("Target dimensions and depth must be positive")
        if self.right_depth_mm is not None and (not np.isfinite(self.right_depth_mm) or self.right_depth_mm <= 0):
            raise ValueError("The second target depth must be finite and positive")
        object.__setattr__(self, "center_xy_mm", tuple(map(float, center)))
        object.__setattr__(self, "half_size_xy_mm", tuple(map(float, size)))

    def columns_and_depth(self, x, y):
        """Depth is measured vertically from the initial surface, not along its normal."""
        dx = (x - self.center_xy_mm[0]) / self.half_size_xy_mm[0]
        dy = (y - self.center_xy_mm[1]) / self.half_size_xy_mm[1]
        inside = (np.abs(dx) <= 1) & (np.abs(dy) <= 1) if self.shape == "rectangle" else dx**2 + dy**2 <= 1
        depth = np.full(x.shape, self.depth_mm)
        if self.right_depth_mm is not None:
            depth[x >= self.center_xy_mm[0]] = self.right_depth_mm
        return inside, depth


@dataclass(frozen=True)
class TaskDesignation:
    regions: tuple[TargetRegion, ...]
    grid_bounds_mm: tuple[tuple[float, float], ...]
    protected_floor_mm: float
    plane_z_mm: float
    frame_id: str
    authority_id: str

    def __post_init__(self):
        bounds = np.asarray(self.grid_bounds_mm, dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Task requires finite increasing XYZ volume bounds")
        if not self.regions or not self.frame_id or not self.authority_id:
            raise ValueError("Task requires designated regions, frame, and action authority")
        if not np.isfinite([self.protected_floor_mm, self.plane_z_mm]).all():
            raise ValueError("Target reference and protected floor must be finite")
        if not bounds[2, 0] < self.protected_floor_mm < bounds[2, 1]:
            raise ValueError("Protected floor must lie inside the modeled volume")
        for region in self.regions:
            center, size = np.array(region.center_xy_mm), np.array(region.half_size_xy_mm)
            if np.any(center - size < bounds[:2, 0]) or np.any(center + size > bounds[:2, 1]):
                raise ValueError("Target footprint extends beyond the designated lattice")
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "grid_bounds_mm", tuple(tuple(row) for row in bounds))


def save_designation(designation, path):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(asdict(designation), stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_designation(path):
    values = json.loads(Path(path).read_text())
    values["regions"] = tuple(TargetRegion(**region) for region in values["regions"])
    return TaskDesignation(**values)
