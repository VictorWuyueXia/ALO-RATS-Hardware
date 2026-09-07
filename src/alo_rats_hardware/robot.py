"""Minimal RTDE access for explicit robot inspection and operator-approved joint motion."""

from dataclasses import dataclass

import numpy as np

from .site import SiteConfiguration


@dataclass
class UR5eConnection:
    """Keep RTDE transport separate from OCT, planning, and all laser control."""

    site: SiteConfiguration
    control: object | None = None
    receive: object | None = None

    def connect(self):
        """Open the same RTDE control and receive channels used by the collaborator workflow."""
        import rtde_control
        import rtde_receive

        self.control = rtde_control.RTDEControlInterface(self.site.robot_ip)
        self.receive = rtde_receive.RTDEReceiveInterface(self.site.robot_ip)
        if not self.control.isConnected() or not self.receive.isConnected():
            raise RuntimeError("RTDE connection did not become active")

    def snapshot(self):
        """Read measured joints and TCP pose without issuing any robot command."""
        if self.receive is None or not self.receive.isConnected():
            raise RuntimeError("Robot is not connected")
        joints = np.asarray(self.receive.getActualQ(), dtype=float)
        tcp = np.asarray(self.receive.getActualTCPPose(), dtype=float)
        if joints.shape != (6,) or tcp.shape != (6,) or not np.isfinite([*joints, *tcp]).all():
            raise RuntimeError("RTDE returned invalid joint or TCP state")
        return {"actual_joints_rad": joints.tolist(), "actual_tcp_pose_m_rad": tcp.tolist()}

    def move_to_safe_pose(self):
        """Execute only the site-approved joint target after an explicit operator request."""
        if self.control is None or not self.control.isConnected():
            raise RuntimeError("Robot is not connected")
        if not self.control.moveJ(self.site.safe_joint_pose_rad.tolist(), self.site.move_speed_rad_s,
                                  self.site.move_acceleration_rad_s2):
            raise RuntimeError("UR5e rejected the requested safe-pose joint motion")

    def close(self):
        """Stop the RTDE script and release both transport channels deterministically."""
        if self.control is not None:
            self.control.stopScript()
            self.control.disconnect()
        if self.receive is not None:
            self.receive.disconnect()
