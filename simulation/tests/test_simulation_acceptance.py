"""Falsify archived evidence without treating a controlled planner as real MPPI validation."""

from dataclasses import replace
import json

import jax
import numpy as np
import pytest

from simulation.robot.collision_scene import link_pose
from simulation.robot.robot_scene import ROOT
from alo_rats_hardware.records import write_json
from simulation.simulation.simulation_cases import simulation_case
from simulation.oct.surface_scan import load_scan, save_scan
from simulation.oct.task_designation import save_designation
from simulation.tests.test_simulation_workflow import workflow_case
from simulation.simulation.validate_simulation import validate_run
from simulation.simulation.workflow import run_workflow
from laser_ablation.control.method import load_method


@pytest.fixture
def recorded_run(workflow_case):
    """Archive one non-completing controlled pulse with genuine robot motion and scan reconstruction."""
    session, task, observed, robot, plant, records, _ = workflow_case
    output = records.output
    session.maximum_pulses = 1
    case = simulation_case("centered_rectangle")
    method = load_method(ROOT / "mppi/configs/controller.yaml", tuple(jax.devices()))
    write_json(output / "case.json", case.manifest())
    write_json(output / "method.json", {"inputs": method.source_values})
    write_json(output / "registration.json", {"calibration": robot.calibration,
                                             "joint_limits_rad": robot.model.limits})
    write_json(output / "isolation.json", {"violations": []})
    save_scan(case.scan(), output / "input_scan.npz")
    save_designation(case.designation, output / "task.json")
    pose = np.linalg.inv(robot.calibration.world_from_base_m) @ link_pose(robot.model, "ee_helper_link")
    scan = replace(plant.observe(pose), timestamp_s=observed.timestamp_s)
    save_scan(scan, output / "scans/prefix_000.npz")
    run_workflow(*workflow_case)
    return output


def test_valid_evidence_is_not_automatically_task_completion(recorded_run):
    result = validate_run(recorded_run)
    assert not result["accepted"] and not result["gates"]["completion"]
    assert all(value for name, value in result["gates"].items() if name != "completion")


@pytest.mark.parametrize("damage,gate", [
    ("pulse_status", "robot_execution"), ("duplicate", None),
    ("joint_jump", "robot_execution"), ("joint_limit", "robot_execution"),
    ("collision", "robot_execution"), ("missing_return", None),
    ("truth", "achieved_action_dynamics"), ("final_observation", "observation_fidelity"),
    ("stale_scan", "feedback"), ("wrong_schedule", "feedback"),
    ("isolation", "hardware_isolation"), ("false_completion", "completion"),
])
def test_corrupted_evidence_cannot_pass(recorded_run, damage, gate):
    """Each mutation targets a distinct accounting, motion, observation, or isolation boundary."""
    output = recorded_run
    events = [json.loads(row) for row in (output / "events.jsonl").read_text().splitlines()]
    receipt = next(row for row in events if row["event"] == "receipt")
    if damage == "pulse_status":
        receipt["receipt"]["pulse_status"] = "uncertain"
    elif damage == "duplicate":
        events.append(receipt)
    elif damage == "wrong_schedule":
        next(row for row in events if row["event"] == "request")["cycle"]["periodic_repair_due"] = True
    elif damage in {"joint_jump", "joint_limit", "collision"}:
        path = output / "motions/pulse_001_laser.npz"
        with np.load(path) as archive:
            arrays = dict(archive)
        if damage == "joint_jump":
            arrays["achieved_joints_rad"][1, 0] += 0.02
        elif damage == "joint_limit":
            arrays["achieved_joints_rad"][:, 0] = 9
        else:
            arrays["minimum_distances_m"][:] = -0.001
        np.savez_compressed(path, **arrays)
    elif damage == "missing_return":
        (output / "motions/pulse_001_scan.npz").unlink()
    elif damage in {"truth", "final_observation"}:
        path = output / ("truth/prefix_001.npz" if damage == "truth" else "observed/final.npz")
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays["tissue"][0, 0, 0] = False
        np.savez_compressed(path, **arrays)
    elif damage == "stale_scan":
        path = output / "scans/prefix_001.npz"
        scan = replace(load_scan(path), timestamp_s=0)
        path.unlink()
        save_scan(scan, path)
        next(row for row in events if row["event"] == "observation")["timestamp_s"] = 0
    else:
        path = output / ("isolation.json" if damage == "isolation" else "workflow_status.json")
        data = json.loads(path.read_text())
        if damage == "isolation":
            data["violations"] = [{"kind": "socket_connect"}]
        else:
            data["terminal_reason"] = "completion_gate"
        path.write_text(json.dumps(data))
    (output / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    if gate is None:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            validate_run(output)
    else:
        result = validate_run(output)
        assert not result["accepted"] and not result["gates"][gate]
