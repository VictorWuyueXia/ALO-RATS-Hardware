"""Hardware-isolated interaction, laser-frame, and virtual-ablation checks."""

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pytest

from simulation.robot.preview_robot import HOME, ROOT
from simulation.visualization.preview_tissue import TARGETS_MM
from simulation.paths.run_robot_preview import PreviewInteraction


@pytest.fixture
def interaction():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.setTimeStep(1 / 240, physicsClientId=client)
    try:
        yield PreviewInteraction(client)
    finally:
        p.disconnect(client)


def finish_motion(interaction):
    """Drive the same GUI state machine without wall-clock animation delays."""
    for _ in range(4000):
        if interaction.tick():
            return
    pytest.fail("Nominal motion did not finish within its fixed step budget")


def test_urdf_and_link_identity(interaction):
    urdf = ROOT / "assets/ur5e/urdf/ur5e_fixed.urdf"
    root = ET.parse(urdf).getroot()
    for mesh in root.findall(".//mesh"):
        assert (urdf.parent / mesh.attrib["filename"]).exists()
    assert len(interaction.robot.joints) == 6
    info = p.getJointInfo(interaction.robot.body, interaction.robot.laser_link,
                          physicsClientId=interaction.robot.client)
    assert info[12] == b"ee2_link"
    assert np.allclose(interaction.robot.joint_positions(), HOME)


@pytest.mark.parametrize("index", range(5))
def test_select_move_confirm_and_reobserve(interaction, index):
    before = interaction.observation.copy()
    interaction.select(index)
    with pytest.raises(ValueError, match="finish a target motion"):
        interaction.fire()
    interaction.move()
    with pytest.raises(ValueError, match="already in progress"):
        interaction.move()
    with pytest.raises(ValueError, match="Wait for motion"):
        interaction.select(0)
    finish_motion(interaction)
    action = interaction.fire()
    assert np.linalg.norm(action[:2] - TARGETS_MM[index]) < 0.025
    assert np.max(np.abs(action[2:4])) < 0.002
    assert interaction.tissue.pulses == 1
    assert interaction.tissue.remaining_pct < 100
    assert not np.array_equal(before, interaction.observation)
    assert not np.shares_memory(interaction.observation, interaction.tissue.surface)
    with pytest.raises(ValueError, match="finish a target motion"):
        interaction.fire()


def test_two_pulse_poc_stop_and_home(interaction):
    for index in (0, 1):
        interaction.select(index)
        interaction.move()
        finish_motion(interaction)
        interaction.fire()
    assert interaction.complete
    assert interaction.tissue.remaining_pct <= 80
    assert interaction.tissue.pulses == 2
    before = interaction.observation.copy()
    with pytest.raises(ValueError, match="stopping criterion"):
        interaction.move()
    interaction.move(home=True)
    finish_motion(interaction)
    assert np.max(np.abs(interaction.robot.joint_positions() - HOME)) < 1e-4
    assert np.array_equal(before, interaction.observation)
    assert not interaction.ready


def test_achieved_ray_controls_removal_and_pose_failure_blocks_pulse(interaction):
    interaction.move()
    finish_motion(interaction)
    baseline = interaction.robot.achieved_action(3.5)
    # A changed actual joint pose changes the ray, independently of the selected target.
    joint = interaction.robot.joints[0]
    position = interaction.robot.joint_positions()[0]
    p.resetJointState(interaction.robot.body, joint, position + 0.01,
                      physicsClientId=interaction.robot.client)
    changed = interaction.robot.achieved_action(3.5)
    assert np.linalg.norm(changed[:2] - baseline[:2]) > 0.1
    with pytest.raises(ValueError, match="Laser pose rejected"):
        interaction.fire()
    assert interaction.tissue.pulses == 0
    assert interaction.tissue.remaining_pct == 100


def test_invalid_target_and_pulse_do_not_mutate_tissue(interaction):
    with pytest.raises(ValueError, match="Choose target"):
        interaction.select(5)
    for action in ([4, 0, 0, 0, 3.5], [0, 0, 0, 0, 100], [np.nan] * 5):
        with pytest.raises(ValueError):
            interaction.tissue.pulse(action)
    assert interaction.tissue.pulses == 0
    assert interaction.tissue.remaining_pct == 100


def test_standalone_run_has_no_hardware_imports_or_network_connections():
    """Prove isolation in a fresh process, including imports and a completed pulse."""
    code = """
import importlib.abc
import sys
class HardwareForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('rtde_', 'UR5Controller', 'ndyag_', 'oct.', 'laser_ablation')):
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, HardwareForbidden())
def audit(event, arguments):
    if event == 'socket.connect':
        raise AssertionError('Simulation attempted a network connection')
sys.addaudithook(audit)
import pybullet as p
from simulation.paths.run_robot_preview import PreviewInteraction
client = p.connect(p.DIRECT)
workflow = PreviewInteraction(client)
workflow.move()
for _ in range(4000):
    if workflow.tick():
        break
assert workflow.ready
workflow.fire()
assert workflow.tissue.pulses == 1
p.disconnect(client)
print('ISOLATION_PASSED')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISOLATION_PASSED" in result.stdout


@pytest.mark.parametrize("return_key", [p.B3G_RETURN, 10, 13])
def test_keyboard_workflow_and_clean_exit(monkeypatch, return_key):
    """Exercise the actual launcher's key routing in an explicitly headless test."""
    import simulation.paths.run_robot_preview as launcher

    connect = p.connect
    instances = []
    frame = iter(range(2000))
    events = {0: return_key, 490: ord('c'), 500: 32, 501: ord('2'), 502: return_key,
              1000: 32, 1001: ord('h'), 1510: ord('v'), 1600: 27}

    def create_interaction(client):
        result = PreviewInteraction(client)
        instances.append(result)
        return result

    def keyboard(**kwargs):
        index = next(frame)
        return {events[index]: p.KEY_WAS_TRIGGERED} if index in events else {}

    def headless_connection(mode):
        assert mode == p.GUI
        return connect(p.DIRECT)

    monkeypatch.setattr(p, 'connect', headless_connection)
    monkeypatch.setattr(p, 'getKeyboardEvents', keyboard)
    monkeypatch.setattr(launcher, 'PreviewInteraction', create_interaction)
    monkeypatch.setattr(launcher.time, 'sleep', lambda seconds: None)
    launcher.main()
    assert instances[0].complete
    assert instances[0].tissue.pulses == 2
    assert not p.isConnected(instances[0].robot.client)
