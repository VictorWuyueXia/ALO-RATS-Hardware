"""No-device tests for OCT-folder reconstruction, PWM protocol, and checked RTDE motion."""

import json
from pathlib import Path
import socketserver
from threading import Thread
from time import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from laser_ablation.control.interaction import ActionRequest
from laser_ablation.core.actions import PhysicalAction

from alo_rats_hardware.laser import LaserPWMClient
from alo_rats_hardware.oct_folder import OCTFolderAdapter
from alo_rats_hardware.robot import UR5eConnection
from alo_rats_hardware.site import SiteConfiguration


def site_values(shared_root, *, physical=False, laser_port=8000):
    """Return one compact, explicit laboratory mapping for interface tests."""
    identity = np.eye(4).tolist()
    return {
        "physical_execution_enabled": physical,
        "robot": {
            "ip": "192.0.2.1", "safe_joint_pose_rad": [0.0] * 6,
            "scan_joint_pose_rad": [0.0] * 6, "move_speed_rad_s": 0.1,
            "move_acceleration_rad_s2": 0.2, "maximum_joint_delta_rad": 0.5,
            "maximum_joint_error_rad": 1e-6,
        },
        "registration": {
            "calibration_id": "registration-test", "base_from_planning_m": identity,
            "tcp_from_oct_m": identity, "tcp_from_laser_m": identity,
            "laser_standoff_m": 0.04, "maximum_intercept_error_mm": 1e-6,
            "maximum_axis_error_rad": 1e-6,
        },
        "oct": {
            "shared_root": str(shared_root), "file_pattern": "*.tif", "expected_b_scans": 4,
            "image_shape_px": [32, 32],
            "pixel_spacing_lateral_scan_depth_mm": [0.2, 1.0, 0.1],
            "axis_order": [0, 1, 2], "axis_signs": [1, 1, 1],
            "surface_margin_px": 2, "surface_threshold_u8": 100,
            "planning_volume_bounds_mm": [[-1.0, 1.0], [-1.0, 1.0], [-2.0, 0.0]],
            "planning_frame_id": "test-planning", "stable_observations": 2,
            "poll_interval_s": 0.001, "scan_timeout_s": 0.2,
        },
        "laser": {
            "host": "127.0.0.1", "port": laser_port, "timeout_s": 1.0,
            "frequency_hz": 100.0, "pulse_duration_s": 0.001, "startup_delay_s": 0.0,
            "energy_to_duty_cycle": [[2.0, 10.0], [4.0, 30.0]],
            "status_key": "running", "stopped_value": False,
            "watchdog_qualified": physical, "calibration_id": "laser-test",
        },
    }


class FakeRecords:
    """Capture append-only interface events while exposing one scan output directory."""

    def __init__(self, output):
        self.output = Path(output)
        (self.output / "scans").mkdir(parents=True)
        self.events = []

    def event(self, kind, **values):
        self.events.append({"event": kind, **values})


class PWMHandler(socketserver.BaseRequestHandler):
    """Emulate the deployed one-request-per-connection JSON exchange."""

    def handle(self):
        command = json.loads(self.request.recv(1024).decode("utf-8"))
        self.server.commands.append(command)
        response = {"running": False} if command["action"] == "status" else {"ok": True}
        self.request.sendall(json.dumps(response).encode("utf-8"))


class FakeReceive:
    def __init__(self):
        self.connected, self.joints, self.tcp = True, np.zeros(6), np.zeros(6)

    def isConnected(self):
        return self.connected

    def getActualQ(self):
        return self.joints.tolist()

    def getActualTCPPose(self):
        return self.tcp.tolist()

    def disconnect(self):
        self.connected = False


class FakeControl:
    def __init__(self, receive):
        self.receive, self.connected, self.target_tcp = receive, True, None

    def isConnected(self):
        return self.connected

    def isPoseWithinSafetyLimits(self, pose):
        return True

    def getInverseKinematics(self, pose, current):
        self.target_tcp = np.asarray(pose)
        return [0.1] * 6

    def isJointsWithinSafetyLimits(self, joints):
        return True

    def moveJ(self, joints, speed, acceleration):
        self.receive.joints = np.asarray(joints)
        if self.target_tcp is not None:
            self.receive.tcp = self.target_tcp
        return True

    def stopScript(self):
        return None

    def disconnect(self):
        self.connected = False


def test_oct_folder_reconstructs_numeric_b_scan_sequence(tmp_path):
    shared = tmp_path / "share"
    shared.mkdir()
    site = SiteConfiguration(**site_values(shared))
    records = FakeRecords(tmp_path / "machine")
    adapter = OCTFolderAdapter(site, records)
    folder = shared / "scan_001"
    folder.mkdir()
    for index in (1, 2, 3, 4):
        image = np.zeros((32, 32), dtype=np.uint8)
        image[8:13, :] = 255
        assert cv2.imwrite(str(folder / f"bscan_{index}.tif"), image)
    scan = adapter.acquire("command-0", np.eye(4))
    assert scan.scan_id and scan.frame_id == "test-planning"
    assert scan.tissue.shape == (20, 20, 20)
    assert (records.output / "scans/prefix_000_manifest.json").is_file()
    assert records.events[-1]["event"] == "oct_scan"


def test_laser_client_uses_deployed_protocol_and_interpolates_energy(tmp_path):
    with socketserver.TCPServer(("127.0.0.1", 0), PWMHandler) as server:
        server.commands = []
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        site = SiteConfiguration(**site_values(tmp_path, physical=True,
                                               laser_port=server.server_address[1]))
        client = LaserPWMClient(site, FakeRecords(tmp_path / "machine"))
        action = PhysicalAction(1.0, 0.0, 0.0, 0.0, 3.0)
        request = ActionRequest("command-1", "task", "trajectory", 0, action, time() - 1)
        receipt = client.execute(request, action)
        server.shutdown()
        thread.join()
    assert receipt.pulse_status == "completed"
    assert [command["action"] for command in server.commands] == [
        "start", "set_pwm", "stop", "status"]
    assert server.commands[1]["duty_cycle"] == 20.0
    assert client.emission_possible is False


def test_robot_recovers_the_measured_action_from_rtde_pose(tmp_path):
    site = SiteConfiguration(**site_values(tmp_path, physical=True))
    receive = FakeReceive()
    robot = UR5eConnection(site, FakeControl(receive), receive)
    action = PhysicalAction(1.0, 0.0, 0.0, 0.0, 3.0)
    request = ActionRequest("command-2", "task", "trajectory", 0, action, time() - 1)
    achieved = robot.execute_action(request, SimpleNamespace(plane_z_mm=0.0))
    assert np.allclose(achieved.as_array(), action.as_array(), atol=1e-9)
    assert robot.last_motion["intercept_error_mm"] < 1e-9


def test_robot_requires_exact_operator_confirmation_before_motion(tmp_path, monkeypatch):
    site = SiteConfiguration(**site_values(tmp_path, physical=True))
    receive = FakeReceive()
    robot = UR5eConnection(site, FakeControl(receive), receive)
    action = PhysicalAction(1.0, 0.0, 0.0, 0.0, 3.0)
    request = ActionRequest("command-confirm", "task", "trajectory", 0, action, time() - 1)
    monkeypatch.setattr("builtins.input", lambda message: "CANCEL")
    with pytest.raises(KeyboardInterrupt, match="declined robot motion"):
        robot.execute_action(request, SimpleNamespace(plane_z_mm=0.0), require_confirmation=True)
    assert np.array_equal(receive.joints, np.zeros(6))


def test_robot_rejects_scan_pose_when_measured_joints_do_not_arrive(tmp_path):
    values = site_values(tmp_path)
    values["robot"]["scan_joint_pose_rad"] = [0.1] * 6
    receive = FakeReceive()
    control = FakeControl(receive)
    control.moveJ = lambda joints, speed, acceleration: True
    robot = UR5eConnection(SiteConfiguration(**values), control, receive)
    with pytest.raises(ValueError, match="measured scan_pose joint error"):
        robot.return_to_scan_pose()


def test_physical_configuration_requires_watchdog_and_energy_table(tmp_path):
    values = site_values(tmp_path, physical=True)
    values["laser"]["watchdog_qualified"] = False
    with pytest.raises(ValueError, match="watchdog"):
        SiteConfiguration(**values)


def test_physical_configuration_rejects_placeholder_identity(tmp_path):
    values = site_values(tmp_path, physical=True)
    values["registration"]["calibration_id"] = "REPLACE_WITH_REGISTRATION"
    with pytest.raises(ValueError, match="measured identifiers"):
        SiteConfiguration(**values)
