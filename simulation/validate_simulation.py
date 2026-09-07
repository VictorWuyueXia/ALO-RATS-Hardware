"""Derive acceptance from archived actions, robot motion, observations, and exact tissue states."""

import json
from pathlib import Path

import numpy as np

from beam_adapter import AXIS_TOLERANCE_RAD, INTERCEPT_TOLERANCE_MM, BeamCalibration, beam_errors
from laser_ablation.config import bounds_from_mapping, physics_from_mapping
from laser_ablation.core.actions import PhysicalAction
from laser_ablation.metrics import evaluate_ablation
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from run_records import json_value, load_voxels
from scan_adapter import designate_task, observe_task
from simulation_cases import CASE_NAMES, simulation_case
from surface_scan import load_scan
from task_designation import load_designation


def checked_motion(path, limits):
    """Check both laser approach and scan return against the frozen joint execution contract."""
    with np.load(path, allow_pickle=False) as motion:
        actual, planned = motion["achieved_joints_rad"], motion["planned_joints_rad"]
        distances = motion["minimum_distances_m"]
        if (actual.ndim != 2 or planned.ndim != 2 or actual.shape[1] != 6 or planned.shape[1] != 6
                or len(planned) < 240 or len(actual) != len(planned) + 60):
            return False
        return bool(motion["preflight_complete"] and np.isfinite(actual).all()
                    and np.isfinite(planned).all() and np.isfinite(distances).all()
                    and distances.shape == (len(actual),) and np.min(distances) > 0
                    and np.max(np.abs(np.diff(actual, axis=0))) <= 0.01
                    and np.max(np.abs(np.diff(planned, axis=0))) <= 0.01
                    and np.all(actual >= limits[:, 0]) and np.all(actual <= limits[:, 1])
                    and np.all(planned >= limits[:, 0]) and np.all(planned <= limits[:, 1])
                    and np.max(np.abs(actual[-1] - planned[-1])) < 1e-5)


def validate_run(output):
    """A missing, stopped, stale, or tissue-only run cannot satisfy the treatment gates."""
    output = Path(output)
    status = json.loads((output / "workflow_status.json").read_text())
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    case = json.loads((output / "case.json").read_text())
    method = json.loads((output / "method.json").read_text())
    registration = json.loads((output / "registration.json").read_text())
    isolation = json.loads((output / "isolation.json").read_text())
    calibration = BeamCalibration(**registration["calibration"])
    limits = np.array(registration["joint_limits_rad"])
    task = designate_task(load_scan(output / "input_scan.npz"), load_designation(output / "task.json"))
    final_truth, final_observed = load_voxels(output / "truth/final.npz"), load_voxels(output / "observed/final.npz")
    truth_metrics, observed_metrics = evaluate_ablation(final_truth), evaluate_ablation(final_observed)
    receipts = [event for event in events if event["event"] == "receipt"]
    observations = [event for event in events if event["event"] == "observation"]
    requests = [event for event in events if event["event"] == "request"]
    prepared = [event for event in events if event["event"] == "prepared"]
    request_map = {event["request"]["command_id"]: event for event in requests}
    prepared_map = {event["command_id"]: event["prepared"] for event in prepared}
    completed_ids = [event["receipt"]["command_id"] for event in receipts]
    model = ExactVoxelSimulator(physics_from_mapping(method["inputs"]["physics"]),
                                bounds_from_mapping(method["inputs"]["physics"]))
    dynamics_valid = robot_valid = observation_valid = prefix_safe = causal = True
    previous = load_voxels(output / "truth/prefix_000.npz")
    initial = previous
    previous_scan = load_scan(output / "scans/prefix_000.npz")
    reconstructed = observe_task(previous_scan, task, 0, None)
    observation_valid &= np.array_equal(reconstructed.state.tissue, initial.tissue)
    errors = []
    for pulse, event in enumerate(receipts, 1):
        command = event["receipt"]["command_id"]
        achieved = PhysicalAction(**event["receipt"]["achieved_action"])
        requested = PhysicalAction(**request_map[command]["request"]["action"])
        receipt, request = event["receipt"], request_map[command]["request"]
        beam = prepared_map[command]
        recovered = calibration.achieved_action(beam["achieved_pose"], previous.plane_z_mm, achieved.energy_j)
        offset, angle = beam_errors(requested, recovered)
        robot_valid &= (offset <= INTERCEPT_TOLERANCE_MM and angle <= AXIS_TOLERANCE_RAD
                        and np.allclose(recovered.as_array(), achieved.as_array(), atol=1e-8, rtol=0)
                        and receipt["pulse_status"] == "completed" and receipt["pose_verified"]
                        and receipt["energy_provenance"] == "commanded_simulation_energy_j"
                        and achieved.energy_j == requested.energy_j
                        and receipt["requested_action"] == request["action"])
        scale = 0.8 if case["response_scale"]["pulse"] == pulse else 1.0
        truth = load_voxels(output / "truth" / f"prefix_{pulse:03d}.npz")
        prediction = model.step(previous, achieved, response_scale=scale)
        dynamics_valid &= np.array_equal(prediction.tissue, truth.tissue) and event["response_scale"] == scale
        metrics = evaluate_ablation(truth)
        prefix_safe &= metrics.hard_violations == 0 and metrics.minimum_clearance_mm >= 0.25
        for operation in ("laser", "scan"):
            robot_valid &= checked_motion(output / "motions" / f"pulse_{pulse:03d}_{operation}.npz", limits)
        observed_path = output / "observed" / f"prefix_{pulse:03d}.npz"
        if observed_path.exists():
            observed = load_voxels(observed_path)
            scan = load_scan(output / "scans" / f"prefix_{pulse:03d}.npz")
            reconstructed = observe_task(scan, task, pulse, command)
            observation_valid &= (np.array_equal(observed.tissue, truth.tissue)
                                  and np.array_equal(reconstructed.state.tissue, observed.tissue))
            matching = [row for row in observations if row["command_id"] == command and row["sequence"] == pulse]
            causal &= (len(matching) == 1 and matching[0]["timestamp_s"] == scan.timestamp_s
                       and matching[0]["scan_id"] == scan.scan_id and scan.scan_id != previous_scan.scan_id
                       and previous_scan.timestamp_s <= request["issued_at_s"] <= receipt["completed_at_s"] <= scan.timestamp_s
                       and previous_scan.timestamp_s < scan.timestamp_s
                       and request_map[command]["observation_scan_id"] == previous_scan.scan_id
                       and events.index(request_map[command]) < events.index(event) < events.index(matching[0]))
            previous_scan = scan
        else:
            observation_valid = False
            errors.append(f"Pulse {pulse} lacks a reconstructed observation")
        previous = truth
    observation_valid &= np.array_equal(reconstructed.state.tissue, final_observed.tissue)
    frozen_geometry = False
    if case["case"] in CASE_NAMES:
        nominal = simulation_case(case["case"])
        frozen = designate_task(nominal.scan(), nominal.designation).state
        frozen_geometry = all(np.array_equal(getattr(frozen, key), getattr(initial, key)) for key in
                              ("x_axis_mm", "y_axis_mm", "z_axis_mm", "initial_tissue", "target_mask", "constraint_mask"))
    else:
        errors.append("Diagnostic task is not a member of the frozen acceptance matrix")
    # Scheduling checks use the observation prefix recorded before each actual decision.
    schedule_valid = causal and all(event["observation_sequence"] == index for index, event in enumerate(requests))
    schedule_valid &= all(event["cycle"]["periodic_repair_due"] == (index > 0 and index % 10 == 0)
                          for index, event in enumerate(requests))
    if case["case"] == "response_disturbance":
        schedule_valid &= len(receipts) >= 11 and any(event["status"] == "repaired" for event in status["repairs"])
    gates = {
        "frozen_geometry": frozen_geometry,
        "completion": status["terminal_reason"] == "completion_gate" and status["failure"] is None
                      and truth_metrics.remaining_pct <= 10 and observed_metrics.remaining_pct <= 10
                      and 0 < len(receipts) <= 400,
        "protection": bool(initial.constraint_mask.any()) and prefix_safe,
        "overcut": truth_metrics.total_overcut_pct <= 10,
        "robot_execution": robot_valid and len(receipts) == len(prepared)
                           and len(completed_ids) == len(set(completed_ids)) == status["confirmed_pulses"]
                           and len(requests) == len(request_map) == len(receipts),
        "achieved_action_dynamics": dynamics_valid and np.array_equal(previous.tissue, final_truth.tissue),
        "feedback": schedule_valid and len(receipts) == len(observations) == status["observed_pulses"],
        "observation_fidelity": observation_valid and abs(truth_metrics.remaining_pct - observed_metrics.remaining_pct) <= 2,
        "hardware_isolation": not isolation["violations"],
    }
    return json_value({"accepted": all(gates.values()), "gates": gates, "case": case["case"],
                       "truth_metrics": truth_metrics, "observed_metrics": observed_metrics,
                       "confirmed_pulses": len(receipts), "terminal_reason": status["terminal_reason"],
                       "failure": status["failure"], "errors": errors,
                       "requested_actions": [event["request"]["action"] for event in requests],
                       "trajectory_ids": [event["request"]["trajectory_id"] for event in requests],
                       "active_prefixes": [event["request"]["active_prefix"] for event in requests]})
