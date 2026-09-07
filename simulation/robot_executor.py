"""Checked joint-stepped laser positioning, with no physical device or pulse implementation."""

from dataclasses import dataclass
from time import perf_counter, sleep

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from beam_adapter import AXIS_TOLERANCE_RAD, INTERCEPT_TOLERANCE_MM, STANDOFF_M, BeamCalibration, beam_errors
from collision_scene import CollisionScene, link_pose
from robot_scene import HOME, load_robot
from laser_ablation.physics.super_gaussian import laser_axis, super_gaussian_depth


MAXIMUM_JOINT_INCREMENT_RAD = 0.01


class OperatorAbort(RuntimeError):
    """The simulated operator requested a stop before another pulse could be released."""


@dataclass(frozen=True)
class PreparedBeam:
    requested_pose: np.ndarray
    achieved_pose: np.ndarray
    achieved_action: object
    intercept_error_mm: float
    axis_error_rad: float
    motion_time_s: float
    release_time_s: float
    minimum_collision_distance_m: float


class RobotExecutor:
    """Retain the original six-joint interpolation and laser-link IK, with fail-stop checks."""

    def __init__(self, client, bounds_mm, verifier, poll):
        self.model = load_robot(client)
        self.client, self.body = client, self.model.body
        self.gui = p.getConnectionInfo(client)["connectionMethod"] == p.GUI
        self.verifier, self.poll = verifier, poll
        world_base = link_pose(self.model, "base")
        world_laser = link_pose(self.model, "ee2_link")
        world_planning = world_laser.copy()
        world_planning[:3, 3] -= world_laser[:3, 2] * STANDOFF_M
        self.calibration = BeamCalibration(world_base, np.linalg.inv(world_base) @ world_planning)
        world_tcp, world_oct = link_pose(self.model, "tool0"), link_pose(self.model, "ee_helper_link")
        self.registration_chain = {
            "base_from_tcp_m": np.linalg.inv(world_base) @ world_tcp,
            "tcp_from_oct_m": np.linalg.inv(world_tcp) @ world_oct,
            "oct_from_planning_m": np.linalg.inv(world_oct) @ world_planning,
        }
        self.scene = CollisionScene(self.model, self.calibration, bounds_mm)
        self.last_motion = {}
        self.hold(HOME)
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setTimeStep(1 / 240, physicsClientId=client)
        self.scene.check()

    def joints(self):
        return np.array([row[0] for row in p.getJointStates(
            self.body, self.model.joints, physicsClientId=self.client)])

    def hold(self, joints):
        p.setJointMotorControlArray(self.body, self.model.joints, p.POSITION_CONTROL,
                                    targetPositions=joints, forces=[150, 150, 150, 28, 28, 28],
                                    positionGains=[0.5] * 6, velocityGains=[1] * 6,
                                    physicsClientId=self.client)

    def check_joints(self, joints):
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("IK returned invalid arm joints")
        if np.any(joints < self.model.limits[:, 0]) or np.any(joints > self.model.limits[:, 1]):
            raise ValueError("Arm joints exceed the URDF limits")

    def move(self, end, target_pose=None):
        """Preflight all samples, restore the start, then move only by motor stepping."""
        self.check_joints(end)
        start = self.joints()
        count = max(240, int(np.ceil(np.max(np.abs(end - start)) / 0.005)) + 1)
        path = np.linspace(start, end, count)
        self.last_motion = {"planned_joints_rad": path, "achieved_joints_rad": [],
                            "minimum_distances_m": [], "preflight_complete": False}
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client)
        try:
            for row in path:
                self.poll()
                for joint, value in zip(self.model.joints, row, strict=True):
                    p.resetJointState(self.body, joint, value, physicsClientId=self.client)
                self.scene.check()
            if target_pose is not None:
                error = np.linalg.inv(target_pose) @ link_pose(self.model, "ee2_link")
                if np.linalg.norm(error[:3, 3]) > 1e-5 or Rotation.from_matrix(error[:3, :3]).magnitude() > 1e-4:
                    raise ValueError("IK endpoint does not reach the requested laser pose")
            self.last_motion["preflight_complete"] = True
        finally:
            for joint, value in zip(self.model.joints, start, strict=True):
                p.resetJointState(self.body, joint, value, physicsClientId=self.client)
            self.hold(start)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)
        previous = start
        for row in np.concatenate((path, np.repeat(end[None, :], 60, axis=0))):
            self.poll()
            self.hold(row)
            p.stepSimulation(physicsClientId=self.client)
            achieved = self.joints()
            self.check_joints(achieved)
            if np.max(np.abs(achieved - previous)) > MAXIMUM_JOINT_INCREMENT_RAD:
                raise ValueError("Achieved joint motion exceeds the checked step bound")
            self.last_motion["achieved_joints_rad"].append(achieved)
            self.last_motion["minimum_distances_m"].append(self.scene.check())
            previous = achieved
            if self.gui:
                sleep(1 / 240)
        if np.max(np.abs(previous - end)) > 1e-5:
            raise ValueError("Joint motors did not settle at the checked endpoint")
        return min(self.last_motion["minimum_distances_m"])

    def prepare(self, request, observed_state):
        """Check the achieved ray against observation-derived tissue before any virtual pulse."""
        self.last_motion = {}
        target = self.calibration.requested_pose(request.action, observed_state.plane_z_mm)
        started = perf_counter()
        end = np.asarray(p.calculateInverseKinematics(
            self.body, self.model.links["ee2_link"], target[:3, 3],
            Rotation.from_matrix(target[:3, :3]).as_quat(), maxNumIterations=2000,
            residualThreshold=1e-8, physicsClientId=self.client))
        minimum = self.move(end, target)
        motion_time = perf_counter() - started
        started = perf_counter()
        pose = link_pose(self.model, "ee2_link")
        action = self.calibration.achieved_action(pose, observed_state.plane_z_mm, request.action.energy_j)
        xy_error, axis_error = beam_errors(request.action, action)
        if xy_error > INTERCEPT_TOLERANCE_MM or axis_error > AXIS_TOLERANCE_RAD:
            raise ValueError("Achieved beam exceeds the numerical pose acceptance tolerances")
        require_coverage(observed_state, action, self.verifier.simulator)
        release = self.verifier.verify_first_action(observed_state, action)
        if not release.valid:
            raise ValueError(f"ACTION_RELEASE_REJECTED: {release.rejection_reason}")
        self.poll()
        return PreparedBeam(target, pose, action, xy_error, axis_error, motion_time,
                            perf_counter() - started, minimum)

    def return_to_scan(self):
        """Use the initial OCT pose for repeat observation, as an explicit nominal scan procedure."""
        self.move(HOME)
        return np.linalg.inv(self.calibration.world_from_base_m) @ link_pose(self.model, "ee_helper_link")


def require_coverage(state, action, simulator):
    """Require measured bounds around the full possible crater, including its slanted extent."""
    contact = simulator.first_contact(state, action)
    if not contact.hit:
        raise ValueError("ACTION_RELEASE_REJECTED: beam misses observed tissue")
    config = simulator.config
    radius = np.sqrt(2 * config.spot_size_mm**2
                     * np.log(action.energy_j / config.ablation_threshold)**(1 / config.super_gaussian_power))
    direction = laser_axis(action.tilt_x_rad, action.tilt_y_rad)
    tip = contact.point_mm + direction * super_gaussian_depth(0.0, action.energy_j, config)
    lower = np.array([axis[0] for axis in (state.x_axis_mm, state.y_axis_mm)]) - state.spacing_mm / 2
    upper = np.array([axis[-1] for axis in (state.x_axis_mm, state.y_axis_mm)]) + state.spacing_mm / 2
    if (np.any(np.minimum(contact.point_mm[:2], tip[:2]) - radius < lower)
            or np.any(np.maximum(contact.point_mm[:2], tip[:2]) + radius > upper)):
        raise ValueError("ACTION_RELEASE_REJECTED: surrounding affected region leaves observed coverage")
