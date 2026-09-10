"""No-device tests for the strict processed-OCT and RTDE isolation boundaries."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from alo_rats_hardware.designation import default_designation
from alo_rats_hardware.processed_oct import load_processed_volume
import alo_rats_hardware.robot as robot_module
from alo_rats_hardware.robot import UR5eConnection
from alo_rats_hardware.scan_adapter import designate_task, export_volume
from alo_rats_hardware.site import SiteConfiguration
from alo_rats_hardware.surface_scan import SurfaceScan, save_scan
from alo_rats_hardware.task_designation import TargetRegion, TaskDesignation


def nominal_scan():
    """Create a complete flat scan fixture without relying on the simulation package path."""
    x, y = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61), indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    return SurfaceScan(points, np.ones(len(points), dtype=bool), "nominal_initial", 0.0,
                       "nominal_tissue", np.eye(4), np.eye(4), -3.0,
                       "synthetic_surface_not_measured_OCT", "mm", "complete_height_field")


def nominal_designation():
    """Match the synthetic fixture lattice to the MPPI controller observation contract."""
    return TaskDesignation((TargetRegion("rectangle", (0, 0), (1.0, 0.8), 0.8, None),),
                           ((-2.8, 2.8), (-2.8, 2.8), (-3.0, 0.6)), 2.0, 0.5,
                           "nominal_tissue", "nominal_registration_surface_task_only")


def test_processed_oct_volume_rebuilds_the_same_voxel_task(tmp_path):
    """A registered occupancy interchange file remains exact across the hardware import boundary."""
    reference = designate_task(nominal_scan(), nominal_designation())
    volume = export_volume(reference.state, "initial_oct", 1.0, reference.frame_id, np.eye(4))
    path = tmp_path / "processed_oct.npz"
    save_scan(volume, path)
    scan = load_processed_volume(path)
    assert np.array_equal(designate_task(scan, nominal_designation()).state.tissue, reference.state.tissue)
    assert default_designation(scan).frame_id == scan.frame_id


def test_processed_oct_rejects_surface_only_input(tmp_path):
    """Raw or surface-only OCT cannot silently become an MPPI controller observation."""
    path = tmp_path / "surface.npz"
    save_scan(nominal_scan(), path)
    with pytest.raises(ValueError, match="segmented_occupancy_volume"):
        load_processed_volume(path)


def test_rtde_is_imported_only_when_connecting():
    """Importing the dry-run package never creates a physical connection or laser dependency."""
    tree = ast.parse(inspect.getsource(robot_module))
    connect = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name == "connect")
    imports = [node for node in ast.walk(connect) if isinstance(node, ast.Import)]
    assert {item.name for node in imports for item in node.names} == {"rtde_control", "rtde_receive"}


def test_site_configuration_rejects_unapproved_motion_values():
    """A hardware motion request requires all six explicit approved joint coordinates."""
    values = yaml.safe_load((Path(__file__).parents[1] / "config/site.example.yaml").read_text())
    values["robot"]["safe_joint_pose_rad"] = np.zeros(5)
    with pytest.raises(ValueError, match="six finite"):
        SiteConfiguration(**values)


@pytest.mark.parametrize(("experiment_number", "case_name"), [
    (2, "protected_boundary"), (3, "response_disturbance"),
])
def test_physical_planning_inputs_match_the_validated_case(experiment_number, case_name):
    """Deployment uses the random seed and raster settings exercised by simulation validation."""
    root = Path(__file__).parents[1]
    cases = yaml.safe_load((root / "config/simulation_cases.yaml").read_text())
    experiment = yaml.safe_load((root / f"config/experiment_{experiment_number}.yaml").read_text())
    assert experiment["random_seed"] == cases["random_seed"]
    assert experiment["raster_settings"] == cases["cases"][case_name]["raster_settings"]
