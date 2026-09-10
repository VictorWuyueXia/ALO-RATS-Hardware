"""Operator-approved UR5e, Raspberry Pi PWM, and feedback-OCT execution for experiments 2 and 3."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter, time

import numpy as np
import yaml

from laser_ablation.control.interaction import ControllerObservation, DesignatedTask, ExecutionReceipt
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


def run_experiment(site_path, experiment_number, metadata_path, output, inert_one_loop=False):
    """Run the strict request, motion, pulse, return, scan, and observation sequence."""
    site = load_site(site_path)
    if experiment_number not in (2, 3):
        raise ValueError("Physical entry point accepts experiment 2 or 3")
    if not inert_one_loop and not site.physical_execution_enabled:
        raise RuntimeError("Set physical_execution_enabled only after completing every qualification stage")
    metadata = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8"))
    metadata_keys = {
        "experiment_number", "experiment_run_id", "laser_safety_approval_id",
        "laser_calibration_id", "random_seed", "raster_settings",
    }
    if not isinstance(metadata, dict) or set(metadata) != metadata_keys:
        raise ValueError(f"Experiment metadata keys must be exactly {sorted(metadata_keys)}")
    raster_keys = {"candidate_grid_pitches_xy_mm", "candidate_grid_shape",
                   "candidate_grid_center_xy_mm", "candidate_depth_repetitions",
                   "candidate_energies_j"}
    raster = metadata["raster_settings"]
    validation_case_name = {2: "protected_boundary", 3: "response_disturbance"}[experiment_number]
    validation_cases = yaml.safe_load((ROOT / "config/simulation_cases.yaml").read_text(encoding="utf-8"))
    validation_case = validation_cases["cases"][validation_case_name]
    identity_values = (metadata["experiment_run_id"], metadata["laser_safety_approval_id"],
                       metadata["laser_calibration_id"])
    if (metadata["experiment_number"] != experiment_number
            or not isinstance(metadata["random_seed"], int)
            or not isinstance(raster, dict) or set(raster) != raster_keys
            or metadata["random_seed"] != validation_cases["random_seed"]
            or raster != validation_case["raster_settings"]
            or (not inert_one_loop and (
                not metadata["experiment_run_id"] or not metadata["laser_safety_approval_id"]
                or any("REPLACE" in str(value) for value in identity_values)
                or metadata["laser_calibration_id"] != site.laser["calibration_id"]))):
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
    write_json(machine / "execution_mode.json", {
        "inert_one_loop": inert_one_loop, "laser_control_present": not inert_one_loop,
        "maximum_feedback_loops": 1 if inert_one_loop else None,
    })

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    from laser_ablation.control.method import load_method
    from laser_ablation.physics.exact_voxel import ExactVoxelSimulator

    devices = tuple(jax.devices())
    method = load_method(MPPI_CONTROLLER, devices)
    components = method.components(validation_case_name, raster_settings,
                                   devices, metadata["random_seed"])
    predictor = ExactVoxelSimulator(method.physics, method.bounds)
    write_json(machine / "method.json", {
        "inputs": method.source_values, "source_sha256": method.source_hashes,
        "compute_profile": method.compute_profile, "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
    })

    robot = UR5eConnection(site)
    laser = None if inert_one_loop else LaserPWMClient(site, records)
    adapter = session = task = request = receipt = None
    phase, failure, stop_failure = "connect", None, None
    mismatch_volumes = []
    started = perf_counter()
    try:
        adapter = OCTFolderAdapter(site, records)
        robot.connect()
        records.event("robot_connected", state=robot.snapshot())
        if laser is not None:
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
        planning_started = perf_counter()
        print("INITIAL_PLANNING_STARTED using the tracked deployment MPPI configuration", flush=True)
        session.start(observation, task)
        planning_elapsed = perf_counter() - planning_started
        memory_stats = {str(device): device.memory_stats() for device in devices}
        write_json(machine / "jax_memory.json", {
            "after_initial_planning": memory_stats,
            "units": "bytes",
        })
        print(
            f"INITIAL_PLANNING_COMPLETE elapsed_s={planning_elapsed:.3f} "
            f"peak_gpu_memory_mib={max(stats['peak_bytes_in_use'] for stats in memory_stats.values()) / 2**20:.1f}",
            flush=True,
        )
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
            before_state = session.state
            phase = "robot_motion"
            achieved = robot.execute_action(request, before_state, require_confirmation=True)
            records.motion(f"pulse_{session.confirmed_pulses + 1:03d}_laser", robot)
            predicted = predictor.step(before_state, achieved)
            predicted_metrics = evaluate_ablation(predicted)
            if (np.array_equal(predicted.tissue, before_state.tissue)
                    or predicted_metrics.hard_violations > 0):
                raise ValueError("ACTION_RELEASE_REJECTED: measured beam removes no tissue or protected tissue")
            if inert_one_loop:
                phase = "inert_zero_energy"
                print("INERT_EXECUTION_RECORDED energy_delivered_j=0; no laser connection was created",
                      flush=True)
                receipt = ExecutionReceipt(
                    request.command_id, request.action, achieved, "completed", time(),
                    "INERT_REHEARSAL_ZERO_ENERGY", True,
                )
                records.event("inert_execution", receipt=receipt,
                              laser_control_present=False, energy_delivered_j=0.0)
            else:
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
            if inert_one_loop:
                raw_tissue = scan.occupancy_on((before_state.x_axis_mm, before_state.y_axis_mm,
                                                before_state.z_axis_mm))
                raw_added = int(np.count_nonzero(raw_tissue & ~before_state.tissue))
                raw_removed = int(np.count_nonzero(before_state.tissue & ~raw_tissue))
                observation = ControllerObservation(
                    before_state, task.task_id, scan.scan_id, session.confirmed_pulses,
                    scan.timestamp_s, request.command_id,
                )
                records.event(
                    "inert_feedback_reconciliation", scan_id=scan.scan_id,
                    raw_added_voxels=raw_added, raw_removed_voxels=raw_removed,
                    state_source="pre_motion_zero_energy_state",
                )
                mismatch = np.count_nonzero(raw_tissue ^ before_state.tissue) * before_state.voxel_volume_mm3
            else:
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
            if inert_one_loop:
                phase = "post_feedback_planning"
                print("POST_FEEDBACK_PLANNING_STARTED", flush=True)
                next_request = session.next_action()
                if next_request is None:
                    raise RuntimeError("Controller produced no request after the feedback update")
                records.event("post_feedback_request", request=next_request,
                              cycle=session.cycle_record,
                              observation_scan_id=session.last_observation.scan_id)
                session.stop("INERT_ONE_FEEDBACK_LOOP_COMPLETE")
                print(f"INERT_ONE_FEEDBACK_LOOP_PASS next_command_id={next_request.command_id}",
                      flush=True)
                break
    except (Exception, KeyboardInterrupt) as error:
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)}
        records.event("failure", **failure)
        if session is not None and session.started and session.pending is not None and session.receipt is None:
            status = "uncertain" if ((laser is not None and laser.emission_possible)
                                     or receipt is not None) else "not_executed"
            session.record_execution(ExecutionReceipt(
                session.pending.command_id, session.pending.action, None, status, time(),
                "unknown_outcome" if status == "uncertain" else "no_energy_delivered", False,
            ))
        if session is not None and session.started and session.stopped_reason is None:
            session.stop("OPERATOR_ABORT" if isinstance(error, KeyboardInterrupt)
                         else phase.upper() + "_FAILED")
    finally:
        if laser is not None:
            try:
                laser.stop()
            except Exception as error:
                stop_failure = {"type": type(error).__name__, "message": str(error)}
                records.event("laser_stop_failure", **stop_failure)
        robot.close()
        final_metrics = asdict(evaluate_ablation(session.state)) if session is not None and session.started else None
        accepted = bool(failure is None and stop_failure is None and session is not None and (
            (inert_one_loop and session.stopped_reason == "INERT_ONE_FEEDBACK_LOOP_COMPLETE"
             and session.confirmed_pulses == 1 and session.last_observation.sequence == 1)
            or (not inert_one_loop and session.stopped_reason == "completion_gate"
                and final_metrics["hard_violations"] == 0
                and session.last_observation.sequence == session.confirmed_pulses
                and (experiment_number == 2 or any(
                    event["successful_pulses"] == 10 for event in session.repair_events)))))
        flag = ("INERT_ONE_FEEDBACK_LOOP_PASS" if inert_one_loop and accepted else
                f"EXPERIMENT_{experiment_number}_COMPLETE" if accepted else None)
        write_json(machine / "workflow_status.json", {
            "completion_flag": flag, "accepted": accepted,
            "inert_one_loop": inert_one_loop, "laser_control_present": laser is not None,
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
        reason = "UNKNOWN" if session is None else session.stopped_reason
        raise RuntimeError(f"Experiment ended without satisfying its acceptance conditions: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=int, choices=(2, 3))
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inert-one-loop", action="store_true")
    args = parser.parse_args()
    run_experiment(args.site, args.experiment, args.metadata, args.output_dir,
                   args.inert_one_loop)


if __name__ == "__main__":
    main()
