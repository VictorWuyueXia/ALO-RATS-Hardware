"""Configured scan-defined treatment cases and their raster seed inputs."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from simulation.oct.scan_fixtures import nominal_scan
from simulation.oct.scan_adapter import SPACING_MM
from simulation.oct.task_designation import TargetRegion, TaskDesignation


SIMULATION_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/simulation_cases.yaml"
SIMULATION_CONFIG = yaml.safe_load(SIMULATION_CONFIG_PATH.read_text(encoding="utf-8"))
RANDOM_SEED = int(SIMULATION_CONFIG["random_seed"])
CASE_NAMES = tuple(SIMULATION_CONFIG["acceptance_cases"])
LAUNCH_CASE_NAMES = tuple(SIMULATION_CONFIG["launch_cases"])
if (len(CASE_NAMES) != len(set(CASE_NAMES))
        or len(LAUNCH_CASE_NAMES) != len(set(LAUNCH_CASE_NAMES))
        or not set(CASE_NAMES) < set(LAUNCH_CASE_NAMES)
        or set(LAUNCH_CASE_NAMES) != set(SIMULATION_CONFIG["cases"])):
    raise ValueError("simulation case lists must uniquely cover the configured cases")


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
    """Build one case from the sole root configuration authority."""
    cases = SIMULATION_CONFIG["cases"]
    if name not in cases or name not in LAUNCH_CASE_NAMES:
        raise ValueError(f"Unknown simulation case: {name}")
    values = cases[name]
    geometry = values["designation"]
    regions = tuple(
        TargetRegion(
            region["shape"], tuple(region["center_xy_mm"]),
            tuple(region["half_size_xy_mm"]), float(region["depth_mm"]),
            None if region["right_depth_mm"] is None else float(region["right_depth_mm"]),
        )
        for region in geometry["regions"]
    )
    designation = TaskDesignation(
        regions, tuple(tuple(pair) for pair in geometry["grid_bounds_mm"]),
        float(geometry["constraint_depth_mm"]), float(geometry["plane_z_mm"]),
        str(geometry["frame_id"]), str(geometry["authority_id"]),
    )
    raster_values = values["raster_settings"]
    raster = {
        "candidate_grid_pitches_xy_mm": tuple(
            tuple(pair) for pair in raster_values["candidate_grid_pitches_xy_mm"]),
        "candidate_grid_shape": tuple(raster_values["candidate_grid_shape"]),
        "candidate_grid_center_xy_mm": tuple(raster_values["candidate_grid_center_xy_mm"]),
        "candidate_depth_repetitions": int(raster_values["candidate_depth_repetitions"]),
        "candidate_energies_j": tuple(raster_values["candidate_energies_j"]),
    }
    disturbed = values["disturbed_pulse"]
    return SimulationCase(name, designation, raster, None if disturbed is None else int(disturbed))
