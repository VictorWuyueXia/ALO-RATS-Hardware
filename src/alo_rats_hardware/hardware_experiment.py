"""Operator-approved UR5e, Raspberry Pi PWM, and feedback-OCT execution for experiments 2 and 3."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter, time

import numpy as np
import yaml

from laser_ablation.control.interaction import DesignatedTask, ExecutionReceipt
from laser_ablation.metrics import evaluate_ablation

from .designation import approve_volume_task
from .laser import LaserPWMClient
from .oct_folder import OCTFolderAdapter
from .records import RunRecords, save_voxels, write_json
from .robot import UR5eConnection
from .scan_adapter import observe_task
from .site import load_site


ROOT = Path(__file__).resolve().parents[2]
MPPI_CONTROLLER = ROOT / "mppi/configs/controller.yaml"


def run_experiment(site_path, experiment_number, metadata_path, output):
    """Run the strict request, motion, pulse, return, scan, and observation sequence."""
    site = load_site(site_path)
    if experiment_number not in (2, 3):
        raise ValueError("Physical entry point accepts experiment 2 or 3")
    if not site.physical_execution_enabled:
        raise RuntimeError("Set physical_execution_enabled only after completing every qualification stage")
    metadata = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8"))
    metadata_keys = {
        "experiment_number", "experiment_run_id", "phantom_id", "phantom_formula",
        "phantom_batch_id", "phantom_preparation_time", "operator_ids",
        "laser_safety_approval_id", "registration_calibration_id", "laser_calibration_id",
        "oct_repeatability_xor_mm3", "random_seed", "raster_settings",
    }
    if not isinstance(metadata, dict) or set(metadata) != metadata_keys:
        raise ValueError(f"Experiment metadata keys must be exactly {sorted(metadata_keys)}")
    raster_keys = {"candidate_grid_pitches_xy_mm", "candidate_grid_shape",
                   "candidate_grid_center_xy_mm", "candidate_depth_repetitions",
                   "candidate_energies_j"}
    raster = metadata["raster_settings"]
    identity_values = (
        metadata["experiment_run_id"], metadata["phantom_id"], metadata["phantom_formula"],
        metadata["phantom_batch_id"], metadata["phantom_preparation_time"],
        metadata["laser_safety_approval_id"], metadata["registration_calibration_id"],
        metadata["laser_calibration_id"],
    )
    if (metadata["experiment_number"] != experiment_number or not metadata["experiment_run_id"]
            or not metadata["phantom_id"] or not metadata["phantom_formula"]
            or not metadata["phantom_batch_id"] or not metadata["phantom_preparation_time"]
            or not isinstance(metadata["operator_ids"], list)
            or not metadata["operator_ids"] or not all(metadata["operator_ids"])
            or not metadata["laser_safety_approval_id"]
            or any("REPLACE" in str(value) for value in (*identity_values, *metadata["operator_ids"]))
            or metadata["registration_calibration_id"] != site.registration["calibration_id"]
            or metadata["laser_calibration_id"] != site.laser["calibration_id"]
            or not np.isfinite(metadata["oct_repeatability_xor_mm3"])
            or metadata["oct_repeatability_xor_mm3"] <= 0
            or not isinstance(metadata["random_seed"], int)
            or not isinstance(raster, dict) or set(raster) != raster_keys):
        raise ValueError("Experiment identity, calibration, repeatability, seed, or raster metadata is invalid")
    raster_settings = {
        "candidate_grid_pitches_xy_mm": tuple(tuple(map(float, pair))
                                                for pair in raster["candidate_grid_pitches_xy_mm"]),
        "candidate_grid_shape": tuple(map(int, raster["candidate_grid_shape"])),
        "candidate_grid_center_xy_mm": tuple(map(float, raster["candidate_grid_center_xy_mm"])),
        "candidate_depth_repetitions": int(raster["candidate_depth_repetitions"]),
        "candidate_energies_j": tuple(map(float, raster["candidate_energies_j"])),
    }

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    human = output / "human_readables"
    machine = output / "machine_readables"
    human.mkdir()
    machine.mkdir()
    records = RunRecords(machine, ("motions", "scans", "observed", "receipts"))
    write_json(machine / "run_metadata.json", metadata)
    write_json(machine / "site.json", {"site_id": site.identity,
                                        "source": Path(site_path).resolve()})

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    from laser_ablation.control.method import load_method
    from laser_ablation.physics.exact_voxel import ExactVoxelSimulator

    devices = tuple(jax.devices())
    method = load_method(MPPI_CONTROLLER, devices)
    components = method.components(f"physical_experiment_{experiment_number}", raster_settings,
                                   devices, metadata["random_seed"])
    predictor = ExactVoxelSimulator(method.physics, method.bounds)
    write_json(machine / "method.json", {
        "inputs": method.source_values, "source_sha256": method.source_hashes,
        "compute_profile": method.compute_profile, "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
    })

    robot = UR5eConnection(site)
    laser = LaserPWMClient(site, records)
    adapter = session = task = request = receipt = None
    phase, failure, stop_failure = "connect", None, None
    mismatch_volumes = []
    started = perf_counter()
    try:
        adapter = OCTFolderAdapter(site, records)
        robot.connect()
        records.event("robot_connected", state=robot.snapshot())
        laser.stop()
        phase = "initial_scan_pose"
        scan_pose = robot.return_to_scan_pose()
        records.motion("prefix_000_scan", robot)
        input("Create the initial volume in the Windows Lumedica application, then press Enter. ")
        phase = "initial_oct"
        scan = adapter.acquire(None, scan_pose)
        phase = "designation"
        task = approve_volume_task(scan, machine / "approved_task")
        (machine / "approved_task/designation.png").replace(human / "designation.png")
        authority = sha256((components.authority_id + robot.calibration.identity + site.identity).encode()).hexdigest()
        task = DesignatedTask(task.state, task.frame_id, authority)
        observation = observe_task(scan, task, 0, None)
        save_voxels(machine / "observed/prefix_000.npz", observation.state)
        write_json(machine / "registration.json", {
            "site_id": site.identity, "task_id": task.task_id,
            "method_authority": components.authority_id,
            "beam_calibration_id": robot.calibration.identity,
            "registration_calibration_id": site.registration["calibration_id"],
        })
        session = components.session(method.periodic_repair_pulses, output, metadata["random_seed"])
        phase = "initial_planning"
        session.start(observation, task)
        records.event("initial_plan", task_id=task.task_id, scan_id=scan.scan_id)

        while True:
            phase = "planning"
            planning_started = perf_counter()
            request = session.next_action()
            planning_time = perf_counter() - planning_started
            if request is None:
                break
            records.event("request", request=request, cycle=session.cycle_record,
                          observation_scan_id=session.last_observation.scan_id,
                          planning_time_s=planning_time)
            approval = input(f"Type MOVE {request.command_id} to approve the checked robot motion: ")
            if approval != f"MOVE {request.command_id}":
                raise KeyboardInterrupt("Operator declined robot motion")
            before_state = session.state
            phase = "robot_motion"
            achieved = robot.execute_action(request, before_state)
            records.motion(f"pulse_{session.confirmed_pulses + 1:03d}_laser", robot)
            predicted = predictor.step(before_state, achieved)
            predicted_metrics = evaluate_ablation(predicted)
            if (np.array_equal(predicted.tissue, before_state.tissue)
                    or predicted_metrics.hard_violations > 0):
                raise ValueError("ACTION_RELEASE_REJECTED: measured beam removes no tissue or protected tissue")
            approval = input(f"Type PULSE {request.command_id} to approve one laser pulse: ")
            if approval != f"PULSE {request.command_id}":
                raise KeyboardInterrupt("Operator declined laser pulse")
            phase = "laser_pulse"
            receipt = laser.execute(request, achieved)
            session.record_execution(receipt)
            write_json(machine / "receipts" / f"prefix_{session.confirmed_pulses:03d}.json", receipt)
            phase = "return_to_scan"
            scan_pose = robot.return_to_scan_pose()
            records.motion(f"prefix_{session.confirmed_pulses:03d}_scan", robot)
            input("Create the feedback volume in the Windows Lumedica application, then press Enter. ")
            phase = "feedback_oct"
            scan = adapter.acquire(request.command_id, scan_pose)
            observation = observe_task(scan, task, session.confirmed_pulses, request.command_id)
            mismatch = np.count_nonzero(predicted.tissue ^ observation.state.tissue) * before_state.voxel_volume_mm3
            mismatch_volumes.append(float(mismatch))
            phase = "controller_update"
            session.update(observation)
            save_voxels(machine / "observed" / f"prefix_{session.confirmed_pulses:03d}.npz",
                        observation.state)
            records.event(
                "observation", sequence=session.confirmed_pulses, scan_id=scan.scan_id,
                command_id=request.command_id, mismatch_volume_mm3=mismatch,
                predicted_removed_volume_mm3=np.count_nonzero(
                    before_state.tissue & ~predicted.tissue) * before_state.voxel_volume_mm3,
                observed_removed_volume_mm3=np.count_nonzero(
                    before_state.tissue & ~observation.state.tissue) * before_state.voxel_volume_mm3,
                metrics=asdict(evaluate_ablation(session.state)),
            )
            receipt = None
    except (Exception, KeyboardInterrupt) as error:
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)}
        records.event("failure", **failure)
        if session is not None and session.started and session.pending is not None and session.receipt is None:
            status = "uncertain" if laser.emission_possible or receipt is not None else "not_executed"
            session.record_execution(ExecutionReceipt(
                session.pending.command_id, session.pending.action, None, status, time(),
                "unknown_outcome" if status == "uncertain" else "no_energy_delivered", False,
            ))
        if session is not None and session.started and session.stopped_reason is None:
            session.stop("OPERATOR_ABORT" if isinstance(error, KeyboardInterrupt)
                         else phase.upper() + "_FAILED")
    finally:
        try:
            laser.stop()
        except Exception as error:
            stop_failure = {"type": type(error).__name__, "message": str(error)}
            records.event("laser_stop_failure", **stop_failure)
        robot.close()
        final_metrics = asdict(evaluate_ablation(session.state)) if session is not None and session.started else None
        accepted = bool(
            failure is None and stop_failure is None and session is not None
            and session.stopped_reason == "completion_gate" and final_metrics["hard_violations"] == 0
            and session.last_observation.sequence == session.confirmed_pulses
            and (experiment_number == 2 or (
                any(value > metadata["oct_repeatability_xor_mm3"] for value in mismatch_volumes)
                and any(event["successful_pulses"] == 10 for event in session.repair_events)))
        )
        flag = f"EXPERIMENT_{experiment_number}_COMPLETE" if accepted else None
        write_json(machine / "workflow_status.json", {
            "completion_flag": flag, "accepted": accepted,
            "terminal_reason": None if session is None else session.stopped_reason,
            "confirmed_pulses": 0 if session is None else session.confirmed_pulses,
            "observed_pulses": 0 if session is None or not session.started
            else session.last_observation.sequence,
            "repairs": [] if session is None else session.repair_events,
            "mismatch_volumes_mm3": mismatch_volumes, "metrics": final_metrics,
            "failure": failure, "laser_stop_failure": stop_failure,
            "elapsed_s": perf_counter() - started,
        })
    if failure is not None:
        raise RuntimeError(f"Physical experiment stopped during {failure['phase']}: {failure['message']}")
    if stop_failure is not None:
        raise RuntimeError(f"Laser stopped status was not verified: {stop_failure['message']}")
    if not accepted:
        raise RuntimeError("Experiment ended without satisfying its acceptance conditions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=int, choices=(2, 3))
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_experiment(args.site, args.experiment, args.metadata, args.output_dir)


if __name__ == "__main__":
    main()
