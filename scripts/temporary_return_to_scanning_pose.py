"""Operator-confirmed slow return from laser alignment to the current-bench scan pose."""

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    """Validate the live UR5e state, require exact confirmation, and execute one slow moveL."""
    import rtde_control
    import rtde_receive

    robot_ip = "192.168.1.103"
    scanning_tcp_pose_m_rad = np.array(
        [0.50599965, -0.53981022, 0.18069501, 0.64042135, 1.48281747, -0.61867306],
        dtype=float)
    linear_speed_m_s = 0.005
    linear_acceleration_m_s2 = 0.05
    maximum_start_translation_m = 0.120
    maximum_start_rotation_rad = 0.35
    maximum_tcp_speed_m_s = 0.0005
    maximum_joint_speed_rad_s = 0.001

    control = rtde_control.RTDEControlInterface(robot_ip)
    receive = rtde_receive.RTDEReceiveInterface(robot_ip)
    current_joints = np.asarray(receive.getActualQ(), dtype=float)
    current_tcp = np.asarray(receive.getActualTCPPose(), dtype=float)
    tcp_offset = np.asarray(control.getTCPOffset(), dtype=float)
    translation = np.linalg.norm(scanning_tcp_pose_m_rad[:3] - current_tcp[:3])
    rotation = (Rotation.from_rotvec(current_tcp[3:]).inv()
                * Rotation.from_rotvec(scanning_tcp_pose_m_rad[3:])).magnitude()
    target_joints = np.asarray(control.getInverseKinematics(
        scanning_tcp_pose_m_rad.tolist(), current_joints.tolist()), dtype=float)

    if receive.getRobotMode() != 7 or receive.getSafetyMode() != 1:
        raise RuntimeError("Robot must be RUNNING with NORMAL safety mode")
    if receive.isProtectiveStopped() or receive.isEmergencyStopped():
        raise RuntimeError("Robot reports a protective or emergency stop")
    if not np.allclose(tcp_offset, np.zeros(6), atol=1e-10, rtol=0):
        raise RuntimeError("Active TCP must be tool0 with zero offset")
    if (np.linalg.norm(receive.getActualTCPSpeed()[:3]) > maximum_tcp_speed_m_s
            or np.max(np.abs(receive.getActualQd())) > maximum_joint_speed_rad_s):
        raise RuntimeError("Robot must be stationary before the scanning-pose return")
    if translation > maximum_start_translation_m or rotation > maximum_start_rotation_rad:
        raise RuntimeError("Scanning pose is farther from the start than the reviewed limits")
    if (target_joints.shape != (6,) or not np.isfinite(target_joints).all()
            or not control.isPoseWithinSafetyLimits(scanning_tcp_pose_m_rad.tolist())
            or not control.isJointsWithinSafetyLimits(target_joints.tolist())):
        raise RuntimeError("UR controller rejected the scanning pose or its inverse kinematics")

    print(f"Current tool0 TCP [m, rad]: {current_tcp.tolist()}")
    print(f"Scanning tool0 TCP [m, rad]: {scanning_tcp_pose_m_rad.tolist()}")
    print(f"Straight-line translation: {translation * 1000:.1f} mm")
    print(f"Orientation change: {rotation:.3f} rad")
    print(f"Speed: {linear_speed_m_s * 1000:.1f} mm/s; "
          f"acceleration: {linear_acceleration_m_s2:.3f} m/s^2")
    confirmation = input("Type RETURN TO SCANNING POSE exactly to start: ")
    if confirmation != "RETURN TO SCANNING POSE":
        control.stopScript()
        control.disconnect()
        receive.disconnect()
        raise SystemExit("Motion cancelled; confirmation did not match")

    if not control.moveL(scanning_tcp_pose_m_rad.tolist(), linear_speed_m_s,
                         linear_acceleration_m_s2):
        raise RuntimeError("UR controller did not complete the scanning-pose return")
    achieved = np.asarray(receive.getActualTCPPose(), dtype=float)
    print(f"Achieved tool0 TCP [m, rad]: {achieved.tolist()}")
    control.stopScript()
    control.disconnect()
    receive.disconnect()


if __name__ == "__main__":
    main()
