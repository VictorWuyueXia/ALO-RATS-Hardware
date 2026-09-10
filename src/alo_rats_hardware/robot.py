"""RTDE state capture and checked joint motion for scan and ALO-RATS laser poses."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .beam import BeamCalibration, beam_errors
from .site import SiteConfiguration


@dataclass
class UR5eConnection:
    """Own the RTDE channels and convert each controller action into one checked moveJ."""

    site: SiteConfiguration
    control: object | None = None
    receive: object | None = None
    calibration: BeamCalibration = field(init=False)
    last_motion: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        values = self.site.registration
        self.calibration = BeamCalibration(np.eye(4), values["base_from_planning_m"],
                                           values["laser_standoff_m"])

    def connect(self):
        """Open the RTDE control and receive channels without issuing a motion."""
        import rtde_control
        import rtde_receive

        self.control = rtde_control.RTDEControlInterface(self.site.robot["ip"])
        self.receive = rtde_receive.RTDEReceiveInterface(self.site.robot["ip"])
        if not self.control.isConnected() or not self.receive.isConnected():
            raise RuntimeError("RTDE connection did not become active")

    def snapshot(self):
        """Read measured joints and TCP pose without issuing a robot command."""
        if self.receive is None or not self.receive.isConnected():
            raise RuntimeError("Robot is not connected")
        joints = np.asarray(self.receive.getActualQ(), dtype=float)
        tcp = np.asarray(self.receive.getActualTCPPose(), dtype=float)
        if joints.shape != (6,) or tcp.shape != (6,) or not np.isfinite([*joints, *tcp]).all():
            raise RuntimeError("RTDE returned invalid joint or TCP state")
        return {"actual_joints_rad": joints.tolist(), "actual_tcp_pose_m_rad": tcp.tolist()}

    def move_to_safe_pose(self):
        """Execute the site-approved safe joint target."""
        self._move_approved_joints(self.site.robot["safe_joint_pose_rad"], "safe_pose")

    def return_to_scan_pose(self):
        """Return to the fixed joint pose used for every OCT acquisition."""
        self._move_approved_joints(self.site.robot["scan_joint_pose_rad"], "scan_pose")
        tcp = np.asarray(self.snapshot()["actual_tcp_pose_m_rad"])
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_rotvec(tcp[3:]).as_matrix()
        matrix[:3, 3] = tcp[:3]
        return matrix

    def execute_action(self, request, task_state, require_confirmation=False):
        """Solve, safety-check, execute, and measure one requested laser pose."""
        if self.control is None or not self.control.isConnected():
            raise RuntimeError("Robot is not connected")
        current = np.asarray(self.snapshot()["actual_joints_rad"])
        requested_laser = self.calibration.requested_pose(request.action, task_state.plane_z_mm)
        requested_tcp = requested_laser @ np.linalg.inv(self.site.registration["tcp_from_laser_m"])
        tcp_pose = np.r_[requested_tcp[:3, 3], Rotation.from_matrix(requested_tcp[:3, :3]).as_rotvec()]
        if not self.control.isPoseWithinSafetyLimits(tcp_pose.tolist()):
            raise ValueError("ROBOT_ACTION_REJECTED: requested TCP pose exceeds robot safety limits")
        joints = np.asarray(self.control.getInverseKinematics(tcp_pose.tolist(), current.tolist()), dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("ROBOT_ACTION_REJECTED: inverse kinematics returned invalid joints")
        if np.max(np.abs(joints - current)) > self.site.robot["maximum_joint_delta_rad"]:
            raise ValueError("ROBOT_ACTION_REJECTED: joint change exceeds the site limit")
        if not self.control.isJointsWithinSafetyLimits(joints.tolist()):
            raise ValueError("ROBOT_ACTION_REJECTED: inverse-kinematics joints exceed robot safety limits")
        if require_confirmation:
            print(f"Requested action [x_mm, y_mm, tilt_x_rad, tilt_y_rad, energy_j]: "
                  f"{request.action.as_array().tolist()}", flush=True)
            print(f"Checked target tool0 TCP [m, rad]: {tcp_pose.tolist()}", flush=True)
            print(f"Checked target joints [rad]: {joints.tolist()}", flush=True)
            print(f"Maximum joint change: {np.max(np.abs(joints - current)):.6f} rad", flush=True)
            approval = input(f"Type MOVE {request.command_id} exactly to execute this motion: ")
            if approval != f"MOVE {request.command_id}":
                raise KeyboardInterrupt("Operator declined robot motion")
        started = perf_counter()
        if not self.control.moveJ(joints.tolist(), self.site.robot["move_speed_rad_s"],
                                  self.site.robot["move_acceleration_rad_s2"]):
            raise RuntimeError("UR5e rejected the requested laser-pose joint motion")
        measured = self.snapshot()
        achieved_tcp_vector = np.asarray(measured["actual_tcp_pose_m_rad"])
        achieved_tcp = np.eye(4)
        achieved_tcp[:3, :3] = Rotation.from_rotvec(achieved_tcp_vector[3:]).as_matrix()
        achieved_tcp[:3, 3] = achieved_tcp_vector[:3]
        achieved_laser = achieved_tcp @ self.site.registration["tcp_from_laser_m"]
        achieved = self.calibration.achieved_action(
            achieved_laser, task_state.plane_z_mm, request.action.energy_j)
        intercept_error, axis_error = beam_errors(request.action, achieved)
        if (intercept_error > self.site.registration["maximum_intercept_error_mm"]
                or axis_error > self.site.registration["maximum_axis_error_rad"]):
            raise ValueError("ROBOT_ACTION_REJECTED: measured beam pose exceeds acceptance tolerances")
        self.last_motion = {
            "command_id": request.command_id, "start_joints_rad": current,
            "planned_joints_rad": joints, "achieved_joints_rad": measured["actual_joints_rad"],
            "requested_tcp_pose_m_rad": tcp_pose,
            "achieved_tcp_pose_m_rad": achieved_tcp_vector,
            "requested_laser_pose_base_m": requested_laser,
            "achieved_laser_pose_base_m": achieved_laser,
            "intercept_error_mm": intercept_error, "axis_error_rad": axis_error,
            "motion_time_s": perf_counter() - started,
        }
        return achieved

    def _move_approved_joints(self, joints, name):
        if self.control is None or not self.control.isConnected():
            raise RuntimeError("Robot is not connected")
        current = np.asarray(self.snapshot()["actual_joints_rad"])
        target = np.asarray(joints, dtype=float)
        if (np.max(np.abs(target - current)) > self.site.robot["maximum_joint_delta_rad"]
                or not self.control.isJointsWithinSafetyLimits(target.tolist())):
            raise ValueError(f"ROBOT_ACTION_REJECTED: {name} exceeds the site or robot limits")
        started = perf_counter()
        if not self.control.moveJ(target.tolist(), self.site.robot["move_speed_rad_s"],
                                  self.site.robot["move_acceleration_rad_s2"]):
            raise RuntimeError(f"UR5e rejected the requested {name} motion")
        achieved = np.asarray(self.snapshot()["actual_joints_rad"])
        if np.max(np.abs(achieved - target)) > self.site.robot["maximum_joint_error_rad"]:
            raise ValueError(f"ROBOT_ACTION_REJECTED: measured {name} joint error exceeds the site limit")
        self.last_motion = {"name": name, "start_joints_rad": current,
                            "planned_joints_rad": target,
                            "achieved_joints_rad": achieved,
                            "motion_time_s": perf_counter() - started}

    def close(self):
        """Stop the RTDE script and release both transport channels deterministically."""
        if self.control is not None:
            self.control.stopScript()
            self.control.disconnect()
        if self.receive is not None:
            self.receive.disconnect()
