"""Connection-free UR5e motion and achieved laser-ray geometry for the preview."""

import numpy as np
import pybullet as p

from robot_scene import HOME, JOINT_NAMES, ROOT, load_robot

STANDOFF_M = 0.020
POSITION_TOLERANCE_M = 0.0001
AXIS_TOLERANCE_RAD = 0.002


class PreviewRobot:
    """Drive only an explicit PyBullet client; no physical backend exists here."""

    def __init__(self, client):
        self.client = client
        self.model = load_robot(client)
        self.body = self.model.body
        self.joints = self.model.joints
        self.laser_link = self.model.links["ee2_link"]
        self.limits = self.model.limits
        self.hold(HOME)
        position, self.orientation = self.pose()
        self.rotation = np.array(p.getMatrixFromQuaternion(self.orientation)).reshape(3, 3)
        # Place a synthetic specimen along the home laser ray, not a lab calibration.
        self.origin = position - self.rotation[:, 2] * 0.040

    def pose(self):
        """Read the laser link frame, not its inertial center or the OCT link."""
        state = p.getLinkState(
            self.body, self.laser_link, computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return np.array(state[4]), np.array(state[5])

    def joint_positions(self):
        return np.array([state[0] for state in p.getJointStates(
            self.body, self.joints, physicsClientId=self.client,
        )])

    def hold(self, joints):
        p.setJointMotorControlArray(
            self.body, self.joints, p.POSITION_CONTROL, targetPositions=joints,
            forces=[150, 150, 150, 28, 28, 28], physicsClientId=self.client,
        )

    def world(self, points_mm):
        return np.asarray(points_mm) @ self.rotation.T / 1000 + self.origin

    def target_pose(self, xy_mm):
        point = self.world([*xy_mm, 0.0]) + self.rotation[:, 2] * STANDOFF_M
        return point, self.orientation

    def path_to(self, xy_mm=None):
        """Solve laser-link IK and form a slow joint path, or return to home."""
        start = self.joint_positions()
        if xy_mm is None:
            end = HOME.copy()
        else:
            position, orientation = self.target_pose(xy_mm)
            end = np.asarray(p.calculateInverseKinematics(
                self.body, self.laser_link, position, orientation,
                maxNumIterations=2000, residualThreshold=1e-8,
                physicsClientId=self.client,
            ))
        if end.shape != (6,) or not np.isfinite(end).all():
            raise ValueError("IK returned invalid joint values")
        if np.any(end < self.limits[:, 0]) or np.any(end > self.limits[:, 1]):
            raise ValueError("IK exceeds URDF joint limits")
        # Validate forward kinematics temporarily; animation uses motor stepping.
        if xy_mm is not None:
            try:
                for joint, value in zip(self.joints, end):
                    p.resetJointState(self.body, joint, value, physicsClientId=self.client)
                self.require_target(xy_mm)
            finally:
                for joint, value in zip(self.joints, start):
                    p.resetJointState(self.body, joint, value, physicsClientId=self.client)
                self.hold(start)
        return np.linspace(start, end, max(240, int(np.max(np.abs(end - start)) / 0.002) + 1))

    def require_target(self, xy_mm):
        """Reject pulse release when achieved position or beam axis is inaccurate."""
        position, orientation = self.pose()
        target, _ = self.target_pose(xy_mm)
        rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        error = np.linalg.norm(position - target)
        angle = np.arccos(np.clip(rotation[:, 2] @ self.rotation[:, 2], -1, 1))
        if error > POSITION_TOLERANCE_M or angle > AXIS_TOLERANCE_RAD:
            raise ValueError(f"Laser pose rejected: {error * 1000:.4f} mm, {angle:.6f} rad")

    def achieved_action(self, energy_j):
        """Intersect the achieved negative-Z laser ray with the tissue reference plane."""
        position, orientation = self.pose()
        rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        point = self.rotation.T @ (position - self.origin) * 1000
        direction = self.rotation.T @ (-rotation[:, 2])
        if direction[2] >= -0.99 or point[2] <= 0:
            raise ValueError("Laser does not point toward the nominal tissue")
        intercept = point - point[2] / direction[2] * direction
        tx = np.arctan2(direction[1], -direction[2])
        ty = np.arcsin(np.clip(-direction[0], -1, 1))
        return np.array([intercept[0], intercept[1], tx, ty, energy_j])
