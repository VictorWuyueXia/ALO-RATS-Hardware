"""Signed frame round trips and fail-stop achieved-beam execution on the real URDF."""

from time import time

import numpy as np
import pybullet as p
import pytest
from scipy.spatial.transform import Rotation

from beam_adapter import BeamCalibration, beam_errors
from collision_scene import link_pose
from robot_executor import OperatorAbort, RobotExecutor, require_coverage
from robot_scene import HOME, ROOT
from scan_adapter import designate_task
from simulation_cases import nominal_scan, simulation_case
from laser_ablation.control.interaction import ActionRequest
from laser_ablation.control.method import load_method
from laser_ablation.core.actions import PhysicalAction
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.terminal_verification import ExactPlanVerifier


@pytest.fixture
def apparatus():
    method = load_method(ROOT / "mppi/configs/controller.yaml")
    case = simulation_case("centered_rectangle")
    state = designate_task(nominal_scan(), case.designation).state
    verifier = ExactPlanVerifier(ExactVoxelSimulator(method.physics, method.bounds), 10, 0.25)
    client = p.connect(p.DIRECT)
    try:
        robot = RobotExecutor(client, case.designation.grid_bounds_mm, verifier, lambda: None)
        yield robot, state, verifier
    finally:
        p.disconnect(client)


@pytest.mark.parametrize("tilts", [(0, 0), (0.15, -0.2), (-0.2, 0.15)])
@pytest.mark.parametrize("xy", [(0.6, -0.4), (-0.7, 0.2)])
def test_beam_transform_round_trip(tilts, xy):
    world_base, base_planning = np.eye(4), np.eye(4)
    world_base[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.5]).as_matrix()
    world_base[:3, 3] = [0.5, -0.4, 0.2]
    base_planning[:3, :3] = Rotation.from_euler("xyz", [-0.1, 0.25, -0.4]).as_matrix()
    base_planning[:3, 3] = [0.3, -0.5, 0.4]
    calibration = BeamCalibration(world_base, base_planning)
    action = PhysicalAction(*xy, *tilts, 2.3)
    pose = calibration.requested_pose(action, 0.5)
    recovered = calibration.achieved_action(pose, 0.5, action.energy_j)
    xy_error, axis_error = beam_errors(action, recovered)
    assert xy_error < 1e-6 and axis_error < 1e-8
    shifted = pose.copy()
    shifted[:3, 3] += calibration.world_from_planning_m[:3, 0] * 0.0001
    actual = calibration.achieved_action(shifted, 0.5, 2.3)
    assert actual.x_mm == pytest.approx(action.x_mm + 0.1, abs=1e-6)


@pytest.mark.parametrize("xy,tilts", [((0, 0), (0, 0)), ((0.8, -0.6), (0.05, -0.04)),
                                     ((-0.8, 0.6), (-0.05, 0.04))])
def test_checked_joint_motion_and_return_scan(apparatus, xy, tilts):
    robot, state, _ = apparatus
    action = PhysicalAction(*xy, *tilts, 2.3)
    request = ActionRequest("test:0", "task", "plan", 0, action, time())
    prepared = robot.prepare(request, state)
    assert prepared.intercept_error_mm <= 0.025 and prepared.axis_error_rad <= 0.001
    actual = np.array(robot.last_motion["achieved_joints_rad"])
    assert len(actual) >= 300 and np.max(np.abs(np.diff(actual, axis=0))) <= 0.01
    assert robot.last_motion["preflight_complete"]
    assert prepared.minimum_collision_distance_m > 0
    chain = robot.registration_chain
    assert np.allclose(chain["base_from_tcp_m"] @ chain["tcp_from_oct_m"] @ chain["oct_from_planning_m"],
                       robot.calibration.base_from_planning_m)
    robot.return_to_scan()
    assert np.max(np.abs(robot.joints() - HOME)) < 1e-5


def test_blocked_motion_does_not_execute(apparatus):
    robot, state, _ = apparatus
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=robot.client)
    obstacle = p.createMultiBody(0, shape, basePosition=link_pose(robot.model, "ee2_link")[:3, 3],
                                 physicsClientId=robot.client)
    robot.scene.obstacles.append(obstacle)
    with pytest.raises(ValueError, match="COLLISION"):
        robot.move(HOME + 0.01)
    assert not robot.last_motion["achieved_joints_rad"]
    assert np.allclose(robot.joints(), HOME)


def test_unreachable_pose_limits_and_operator_abort(apparatus):
    robot, state, _ = apparatus
    impossible = np.eye(4)
    impossible[:3, 3] = [10, 10, 10]
    with pytest.raises(ValueError, match="IK endpoint"):
        robot.move(HOME, impossible)
    with pytest.raises(ValueError, match="limits"):
        robot.move(np.ones(6) * 20)
    def abort():
        raise OperatorAbort("OPERATOR_ABORT")
    robot.poll = abort
    with pytest.raises(OperatorAbort):
        robot.move(HOME + 0.01)
    assert not robot.last_motion["achieved_joints_rad"]


def test_reversed_ray_coverage_and_protected_release(apparatus):
    robot, state, verifier = apparatus
    action = PhysicalAction(0, 0, 0, 0, 2.3)
    pose = robot.calibration.requested_pose(action, state.plane_z_mm)
    pose[:3, :2] *= -1
    pose[:3, 1:] *= -1
    with pytest.raises(ValueError, match="reversed"):
        robot.calibration.achieved_action(pose, state.plane_z_mm, 2.3)
    with pytest.raises(ValueError, match="coverage"):
        require_coverage(state, PhysicalAction(2.5, 0, 0, 0, 2.3), verifier.simulator)
    unsafe = state.copy()
    unsafe.constraint_mask = unsafe.tissue.copy()
    request = ActionRequest("unsafe:0", "task", "plan", 0, action, time())
    with pytest.raises(ValueError, match="ACTION_RELEASE_REJECTED"):
        robot.prepare(request, unsafe)
    assert np.array_equal(unsafe.tissue, state.tissue)


def test_achieved_offset_changes_removal(apparatus):
    robot, state, verifier = apparatus
    action = PhysicalAction(0, 0, 0, 0, 2.3)
    pose = robot.calibration.requested_pose(action, state.plane_z_mm)
    pose[:3, 3] += robot.calibration.world_from_planning_m[:3, 0] * 0.0002
    achieved = robot.calibration.achieved_action(pose, state.plane_z_mm, 2.3)
    requested_after = verifier.simulator.step(state, action)
    achieved_after = verifier.simulator.step(state, achieved)
    assert not np.array_equal(requested_after.tissue, achieved_after.tissue)
