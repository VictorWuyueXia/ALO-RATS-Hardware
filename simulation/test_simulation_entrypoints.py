"""Fresh-process isolation, output ownership, and failed-suite accounting checks."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import pybullet as p

from laser_ablation.control.method import load_method
from robot_scene import ROOT
from simulation_cases import CASE_NAMES, LAUNCH_CASE_NAMES, SIMULATION_CONFIG_PATH, simulation_case


@pytest.mark.parametrize(("platform", "count", "profile_name", "rollout_batch_size"), [
    ("cpu", 1, "cpu", 4), ("gpu", 1, "1gpu", 4),
    ("cuda", 4, "4gpu", 64), ("gpu", 8, "8gpu", 64),
])
def test_method_matches_the_detected_jax_device_layout(
        platform, count, profile_name, rollout_batch_size):
    devices = tuple(SimpleNamespace(platform=platform) for _ in range(count))
    method = load_method(ROOT / "mppi/configs/controller.yaml", devices)
    assert method.compute_profile == profile_name
    assert method.mppi.rollout_batch_size == rollout_batch_size
    assert method.mppi.max_anchors == 10
    assert method.mppi.samples_per_anchor == 512
    assert method.maximum_pulses == 64
    assert method.mppi.segment_length_pulses == 10
    assert method.mppi.segment_minimum_pulses == 1
    assert method.mppi.segment_count(method.maximum_pulses) == 7


def test_method_rejects_an_unconfigured_jax_device_layout():
    devices = tuple(SimpleNamespace(platform="gpu") for _ in range(2))
    with pytest.raises(ValueError, match="No unique compute profile"):
        load_method(ROOT / "mppi/configs/controller.yaml", devices)


def test_simulation_compute_profile_activates_before_jax_import():
    code = """
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys
from laser_ablation.control import method
assert "jax" not in sys.modules
profile = Mock()
profile.activate.return_value = (SimpleNamespace(platform="gpu"),)
constructor = Mock(return_value=profile)
method.ComputeProfile = constructor
devices = method.activate_compute_profile(Path.cwd().parent / "mppi/configs/controller.yaml")
assert devices[0].platform == "gpu"
assert constructor.call_args.args[0] == "1gpu"
assert constructor.call_args.args[4] is False
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_simulation_compute_override_selects_cpu_profile():
    code = """
from pathlib import Path
from unittest.mock import Mock
from laser_ablation.control import method
profile = Mock()
profile.activate.return_value = ()
method.ComputeProfile = Mock(return_value=profile)
method.activate_compute_profile(Path.cwd().parent / "mppi/configs/controller.yaml", "cpu")
assert method.ComputeProfile.call_args.args[:4] == ("cpu", "cpu", (0,), 4)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_simulation_guard_rejects_physical_imports_and_connections():
    code = """
from simulation_isolation import enforce_simulation_isolation
violations = enforce_simulation_isolation()
import importlib, socket, sys
sys.path.append(str(__import__('pathlib').Path.cwd().parent))
for name in ('rtde_control', 'oct', 'UR5Controller'):
    try:
        importlib.import_module(name)
    except RuntimeError:
        pass
    else:
        raise AssertionError(name)
with socket.socket() as connection:
    try:
        connection.connect(('192.0.2.1', 9))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Socket connection was not blocked')
with socket.socket() as connection:
    try:
        connection.connect(('127.0.0.1', 9))
    except ConnectionRefusedError:
        pass
    else:
        raise AssertionError('Loopback connection unexpectedly succeeded')
assert len(violations) == 4, violations
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_output_is_untouched_by_live_entrypoint(tmp_path):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("Retain this prior run")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_simulation.py")),
                             "--case", "compact_diagnostic", "--output-dir", str(tmp_path)],
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert sentinel.read_text() == "Retain this prior run"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing.txt"]


def test_diagnostic_does_not_replace_any_frozen_positive_case():
    assert len(CASE_NAMES) == 6 and "compact_diagnostic" not in CASE_NAMES
    assert "compact_diagnostic" in LAUNCH_CASE_NAMES
    case = simulation_case("compact_diagnostic")
    assert case.manifest()["random_seed"] == 20260902
    baseline = simulation_case("centered_rectangle")
    assert SIMULATION_CONFIG_PATH == ROOT / "config/simulation_cases.yaml"
    assert baseline.designation.regions[0].half_size_xy_mm == (2.0, 2.0)
    assert baseline.designation.regions[0].depth_mm == 2.0
    assert baseline.designation.constraint_depth_mm == 3.0
    assert baseline.raster_settings["candidate_energies_j"] == (4.0, 8.0)
    assert case.scan().lower_boundary_mm == case.designation.grid_bounds_mm[2][0] == -2.0
    with pytest.raises(ValueError):
        simulation_case("unrecognized_task")


@pytest.mark.parametrize("key", [10, 13, p.B3G_RETURN])
def test_integrated_start_consumes_return_without_terminal_input(key, tmp_path, monkeypatch):
    from workflow_display import WorkflowDisplay
    client = p.connect(p.DIRECT)
    try:
        display = WorkflowDisplay(client, tmp_path / "frames")
        display.gui = True
        events = iter([{}, {key: p.KEY_WAS_TRIGGERED}])
        monkeypatch.setattr(display, "poll", lambda: next(events))
        monkeypatch.setattr(display, "status", lambda message: None)
        display.wait_for_start()
    finally:
        p.disconnect(client)


def test_killed_worker_without_output_fails_and_preserves_not_run_cases(tmp_path, monkeypatch):
    import check_simulation

    returns = iter([0, -9])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=next(returns))
    monkeypatch.setattr(check_simulation.subprocess, "run",
                        run)
    monkeypatch.chdir(tmp_path)
    summary = check_simulation.check_suite(Path("suite"))
    assert Path(calls[1][-1]).is_absolute()
    assert not summary["URDF_MPPI_TASK_SUITE_COMPLETE"]
    first = summary["direct_cases"]["centered_rectangle"]
    assert not first["accepted"] and first["process_returncode"] == -9
    assert all("not_run" in value for name, value in summary["direct_cases"].items()
               if name != "centered_rectangle")
    saved = json.loads((tmp_path / "suite/centered_rectangle/acceptance.json").read_text())
    assert saved == first


def test_cancelled_gui_preserves_failed_suite_summary(tmp_path, monkeypatch):
    """Controlled worker results test aggregation only, not treatment acceptance."""
    import check_simulation

    def validate(path):
        if path.name == "gui":
            raise FileNotFoundError("Operator cancelled before workflow start")
        return {"accepted": True, "truth_metrics": {}, "requested_actions": [],
                "trajectory_ids": [], "active_prefixes": []}
    monkeypatch.setattr(check_simulation, "validate_run", validate)
    monkeypatch.setattr(check_simulation.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    summary = check_simulation.check_suite(tmp_path / "suite")
    assert not summary["gui_completion"] and not summary["URDF_MPPI_TASK_SUITE_COMPLETE"]
    assert "Operator cancelled" in summary["gui_result"]["incomplete_evidence"]
    assert (tmp_path / "suite/suite_result.json").is_file()
