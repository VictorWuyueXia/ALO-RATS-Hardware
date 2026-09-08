"""Causal workflow and negative-path checks; planner doubles are not MPPI acceptance evidence."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pybullet as p
import pytest

from collision_scene import link_pose
from robot_executor import OperatorAbort, RobotExecutor
from robot_scene import ROOT
from run_records import RunRecords, load_voxels
from scan_adapter import designate_task, export_surface, observe_task
from scan_fixtures import nominal_designation, nominal_scan
from simulated_plant import SimulatedPlant
from simulation_cases import simulation_case
from workflow import run_workflow
from workflow_display import WorkflowDisplay
from laser_ablation.control.interaction import DesignatedTask
from laser_ablation.control.method import load_method
from laser_ablation.control.session import ControllerSession
from laser_ablation.core.actions import PhysicalAction
from laser_ablation.geometry.sdf import SDFObserver
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.terminal_verification import ExactPlanVerifier


class RecordingPlanner:
    """Supply controlled actions to test execution and observation ordering only."""

    def __init__(self, action):
        self.action, self.repair_requests = action, []

    def propose(self, state, observed, raster, directory):
        return SimpleNamespace(trajectory_id="controlled_test_plan", parent_trajectory_id=None,
                               actions=(self.action,) * 40)

    def repair(self, state, observed, request):
        self.repair_requests.append((request, state.tissue.copy()))
        return self.propose(state, observed, None, None)


@pytest.fixture
def workflow_case(tmp_path):
    method = load_method(ROOT / "mppi/configs/controller.yaml", tuple(jax.devices()))
    case = simulation_case("centered_rectangle")
    designation = case.designation
    client = p.connect(p.DIRECT)
    try:
        display = WorkflowDisplay(client, tmp_path / "gui_frames")
        verifier = ExactPlanVerifier(ExactVoxelSimulator(method.physics, method.bounds), 10, 0.25)
        robot = RobotExecutor(client, designation.grid_bounds_mm, verifier, display.poll)
        task = designate_task(case.scan(), designation)
        task = DesignatedTask(task.state, task.frame_id, robot.calibration.identity)
        display.bind(robot, task)
        plant = SimulatedPlant(task.state, method.physics, method.bounds, robot.calibration, task.frame_id, None)
        pose = np.linalg.inv(robot.calibration.world_from_base_m) @ link_pose(robot.model, "ee_helper_link")
        initial_scan = plant.observe(pose)
        observed = observe_task(initial_scan, task, 0, None)
        records = RunRecords(tmp_path)
        planner = RecordingPlanner(PhysicalAction(0, 0, 0, 0, 2.0))
        session = ControllerSession(planner, SDFObserver(), object(), 10, 10, 11, tmp_path, 101)
        yield session, task, observed, robot, plant, records, display
    finally:
        p.disconnect(client)


def test_eleven_receipts_scans_and_periodic_order(workflow_case):
    session, task, observed, robot, plant, records, display = workflow_case
    run_workflow(*workflow_case)
    assert session.confirmed_pulses == session.last_observation.sequence == 11
    assert len(plant.executions) == 11 and session.stopped_reason == "maximum_pulses"
    request, state = session.planner.repair_requests[0]
    assert request.active_prefix == 10
    assert np.array_equal(state, load_voxels(records.output / "observed/prefix_010.npz").tissue)
    events = records.events
    assert sum(row["event"] == "receipt" for row in events) == 11
    for i, row in enumerate(events):
        if row["event"] == "observation":
            prior = [event for event in events[:i] if event["event"] == "receipt"][-1]
            assert row["command_id"] == prior["receipt"]["command_id"]
    assert not np.shares_memory(session.state.tissue, plant._state.tissue)


def test_display_uses_darker_larger_voxel_points(workflow_case, monkeypatch):
    display = workflow_case[-1]
    point_calls = Mock(side_effect=(101, 102))
    monkeypatch.setattr(p, "addUserDebugPoints", point_calls)

    display.draw()

    target, protected = point_calls.call_args_list
    assert target.args[1][0] == [0.05, 0.15, 0.55]
    assert protected.args[1][0] == [0.55, 0.02, 0.04]
    assert target.kwargs["pointSize"] == protected.kwargs["pointSize"] == 4


@pytest.mark.parametrize("failure", ["acquisition", "stale", "abort", "blocked"])
def test_failure_retains_pulse_without_reissue(workflow_case, failure, monkeypatch):
    session, task, observed, robot, plant, records, display = workflow_case
    session.planner.action = PhysicalAction(0, 0, 0, 0, 2.3)
    original = plant.observe
    if failure == "acquisition":
        def reject_scan(pose):
            raise RuntimeError("injected scan acquisition failure")
        monkeypatch.setattr(plant, "observe", reject_scan)
    elif failure == "stale":
        monkeypatch.setattr(plant, "observe", lambda pose: replace(original(pose), timestamp_s=0))
    elif failure == "abort":
        def abort_after_pulse():
            if plant.executions:
                raise OperatorAbort("injected operator abort")
        robot.poll = abort_after_pulse
    else:
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=robot.client)
        body = p.createMultiBody(0, shape, basePosition=link_pose(robot.model, "ee2_link")[:3, 3],
                                 physicsClientId=robot.client)
        robot.scene.obstacles.append(body)
    with pytest.raises((ValueError, RuntimeError)):
        run_workflow(*workflow_case)
    expected = 0 if failure == "blocked" else 1
    assert len(plant.executions) == session.confirmed_pulses == expected
    assert session.last_observation.sequence == 0
    assert session.next_action() is None
    assert (records.output / "workflow_status.json").exists()
    assert len(session.executions) == 1


def test_physical_response_disturbance_and_duplicate_pulse(workflow_case):
    _, task, _, robot, plant, _, _ = workflow_case
    changed = SimulatedPlant(task.state, plant._model.config, plant._model.bounds,
                             robot.calibration, task.frame_id, 5)
    for index in range(1, 6):
        action = PhysicalAction(0, 0, 0, 0, 4.0 if index == 5 else 2.0)
        plant.fire(str(index), action)
        changed.fire(str(index), action)
    assert changed.executions["5"]["response_scale"] == 0.8
    assert plant.executions["5"]["achieved_action"].energy_j == changed.executions["5"]["achieved_action"].energy_j
    assert not np.array_equal(plant._state.tissue, changed._state.tissue)
    pose = np.linalg.inv(robot.calibration.world_from_base_m) @ link_pose(robot.model, "ee_helper_link")
    nominal_observed = observe_task(plant.observe(pose), task, 5, "5")
    changed_observed = observe_task(changed.observe(pose), task, 5, "5")
    assert not np.array_equal(nominal_observed.state.tissue, changed_observed.state.tissue)
    before = changed._state.tissue.copy()
    with pytest.raises(RuntimeError, match="already"):
        changed.fire("5", action)
    assert np.array_equal(changed._state.tissue, before)


def test_exact_model_surface_contract_failure_is_exposed(workflow_case):
    _, _, _, robot, plant, _, _ = workflow_case
    nominal_task = designate_task(nominal_scan(), nominal_designation())
    plant = SimulatedPlant(
        nominal_task.state, plant._model.config, plant._model.bounds,
        robot.calibration, nominal_task.frame_id, None)
    for index in range(3):
        plant.fire(str(index), PhysicalAction(-1.3 + 0.65 * index, -1, 0, 0, 2.28))
    with pytest.raises(ValueError, match="cavity or overhang"):
        export_surface(plant._state)
    pose = np.linalg.inv(robot.calibration.world_from_base_m) @ link_pose(robot.model, "ee_helper_link")
    observed = observe_task(plant.observe(pose), nominal_task, 3, "2")
    assert np.array_equal(observed.state.tissue, plant._state.tissue)
