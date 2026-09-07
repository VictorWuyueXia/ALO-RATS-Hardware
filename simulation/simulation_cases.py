"""Frozen scan-defined treatment cases and their explicit raster seed inputs."""

from dataclasses import asdict, dataclass, replace

from scan_fixtures import nominal_designation, nominal_scan
from scan_adapter import SPACING_MM
from task_designation import TargetRegion, TaskDesignation


RANDOM_SEED = 101
CASE_NAMES = ("centered_rectangle", "offset_round", "stepped_floor",
              "separated_patches", "protected_boundary", "response_disturbance")
LAUNCH_CASE_NAMES = (*CASE_NAMES, "compact_diagnostic")


@dataclass(frozen=True)
class SimulationCase:
    name: str
    designation: TaskDesignation
    raster_settings: dict
    disturbed_pulse: int | None

    def scan(self):
        return replace(nominal_scan(), lower_boundary_mm=self.designation.grid_bounds_mm[2][0])

    def manifest(self):
        return {"case": self.name, "random_seed": RANDOM_SEED,
                "designation": asdict(self.designation), "raster_settings": self.raster_settings,
                "response_scale": {"pulse": self.disturbed_pulse, "factor": 0.8},
                "scan": {"surface": "flat", "sample_spacing_xy_mm": [0.15, 0.20],
                         "lower_boundary_mm": self.scan().lower_boundary_mm},
                "voxel_spacing_mm": SPACING_MM, "completion_remaining_pct": 10.0,
                "maximum_overcut_pct": 10.0, "minimum_clearance_mm": 0.25}


def simulation_case(name):
    """Freeze geometry before planning; never revise a case in response to its result."""
    # This bounded diagnostic exercises real MPPI but never satisfies the frozen case matrix.
    if name == "compact_diagnostic":
        designation = replace(nominal_designation(),
                              grid_bounds_mm=((-1.2, 1.2), (-1.2, 1.2), (-2.0, 0.4)),
                              regions=(TargetRegion("ellipse", (0, 0), (0.6, 0.6), 0.5, None),),
                              protected_floor_mm=-1.5, plane_z_mm=0.3)
        raster = {"candidate_grid_pitches_xy_mm": ((0.3, 0.3), (0.35, 0.35)),
                  "candidate_grid_shape": (1, 1), "candidate_grid_center_xy_mm": (0, 0),
                  "candidate_depth_repetitions": 1, "candidate_energies_j": (3.8, 4.0)}
        return SimulationCase(name, designation, raster, None)
    if name not in CASE_NAMES:
        raise ValueError(f"Unknown simulation case: {name}")
    region = TargetRegion("rectangle", (0, 0), (1.6, 1.2), 1.0, None)
    regions, center, pitches, shape, protected = (region,), (0, 0), ((0.65, 0.5), (0.675, 0.525)), (5, 5), -2.4
    if name == "offset_round":
        center = (0.5, -0.3)
        regions = (TargetRegion("ellipse", center, (1.1, 1.1), 1.0, None),)
        pitches = ((0.45, 0.45), (0.475, 0.475))
    elif name == "stepped_floor":
        regions = (TargetRegion("rectangle", (0, 0), (1.3, 1.0), 0.6, 1.2),)
        pitches = ((0.525, 0.4), (0.55, 0.425))
    elif name == "separated_patches":
        regions = tuple(TargetRegion("rectangle", (x, 0), (0.6, 0.9), 0.9, None)
                        for x in (-1.05, 1.05))
        pitches, shape = ((0.675, 0.6), (0.7, 0.625)), (5, 3)
    elif name == "protected_boundary":
        center, protected = (0.1, 0), -1.5
        regions = (TargetRegion("rectangle", center, (1.2, 1.0), 1.0, None),)
        pitches = ((0.5, 0.4), (0.525, 0.425))
    designation = replace(nominal_designation(), regions=regions, protected_floor_mm=protected)
    raster = {"candidate_grid_pitches_xy_mm": pitches, "candidate_grid_shape": shape,
              "candidate_grid_center_xy_mm": center, "candidate_depth_repetitions": 16,
              "candidate_energies_j": (2.28, 2.30)}
    return SimulationCase(name, designation, raster, 5 if name == "response_disturbance" else None)
