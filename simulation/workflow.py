"""One-process observation, MPPI decision, checked motion, pulse, and repeat-scan workflow."""

from dataclasses import asdict
from time import perf_counter, time

from laser_ablation.control.interaction import ExecutionReceipt
from laser_ablation.metrics import evaluate_ablation
from robot_executor import OperatorAbort
from run_records import save_voxels, write_json
from scan_adapter import observe_task
from surface_scan import save_scan


def run_workflow(session, task, initial_observation, robot, plant, records, display):
    """Keep the execution ledger authoritative when any later motion or acquisition fails."""
    output = records.output
    plant.save_evaluation_state(output / "truth/prefix_000.npz")
    save_voxels(output / "observed/prefix_000.npz", initial_observation.state)
    phase, request, failure = "initial_planning", None, None
    started = perf_counter()
    try:
        display.status("Global planning; Esc aborts before the next released action")
        session.start(initial_observation, task)
        records.event("initial_plan", elapsed_s=perf_counter() - started,
                      task_id=task.task_id, scan_id=initial_observation.scan_id)
        while True:
            robot.poll()
            phase = "planning"
            plan_started = perf_counter()
            display.status(f"Planning from scan {session.last_observation.scan_id}; pulses {session.confirmed_pulses}")
            request = session.next_action()
            planning_time = perf_counter() - plan_started
            if request is None:
                break
            robot.poll()
            records.event("request", request=request, observation_scan_id=session.last_observation.scan_id,
                          observation_sequence=session.last_observation.sequence,
                          planning_time_s=planning_time, cycle=session.cycle_record)
            pulse = session.confirmed_pulses + 1
            phase = "motion_and_release"
            display.status(f"Moving to MPPI action {pulse}; energy {request.action.energy_j:.4f} J")
            prepared = robot.prepare(request, session.state)
            records.motion(f"pulse_{pulse:03d}_laser", robot)
            records.event("prepared", command_id=request.command_id, prepared=prepared)
            display.beam(request.action, prepared.achieved_action, False)
            phase = "pulse"
            robot.poll()
            pulse_started = perf_counter()
            completion = plant.fire(request.command_id, prepared.achieved_action)
            receipt = ExecutionReceipt(request.command_id, request.action, prepared.achieved_action,
                                       "completed", completion["completed_at_s"], "commanded_simulation_energy_j", True)
            session.record_execution(receipt)
            records.event("receipt", receipt=receipt, pulse=pulse,
                          response_scale=completion["response_scale"], pulse_time_s=perf_counter() - pulse_started,
                          truth_metrics=plant.evaluation_metrics())
            plant.save_evaluation_state(output / "truth" / f"prefix_{pulse:03d}.npz")
            display.beam(request.action, prepared.achieved_action, True)
            phase = "return_to_scan"
            scan_motion_started = perf_counter()
            scan_pose = robot.return_to_scan()
            records.motion(f"pulse_{pulse:03d}_scan", robot)
            scan_motion_time = perf_counter() - scan_motion_started
            phase = "acquisition"
            acquire_started = perf_counter()
            scan = plant.observe(scan_pose)
            acquire_time = perf_counter() - acquire_started
            save_scan(scan, output / "scans" / f"prefix_{pulse:03d}.npz")
            phase = "reconstruction"
            reconstruct_started = perf_counter()
            observation = observe_task(scan, task, pulse, request.command_id)
            reconstruct_time = perf_counter() - reconstruct_started
            sdf_started = perf_counter()
            session.update(observation)
            sdf_time = perf_counter() - sdf_started
            save_voxels(output / "observed" / f"prefix_{pulse:03d}.npz", observation.state)
            records.event("observation", scan_id=scan.scan_id, command_id=request.command_id, sequence=pulse,
                          timestamp_s=scan.timestamp_s, scan_motion_time_s=scan_motion_time,
                          acquisition_time_s=acquire_time, reconstruction_time_s=reconstruct_time,
                          sdf_observation_time_s=sdf_time, metrics=asdict(evaluate_ablation(session.state)))
            display.observation(observation, session)
        display.status(f"Stopped: {session.stopped_reason}; confirmed pulses {session.confirmed_pulses}")
    except (Exception, KeyboardInterrupt) as error:
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)}
        if phase in {"motion_and_release", "return_to_scan"} and robot.last_motion:
            records.motion(f"rejected_{phase}_{session.confirmed_pulses:03d}", robot)
        # A rejected preparation acknowledges no pulse; a failed scan retains the completed receipt.
        if session.started and session.pending is not None and session.receipt is None:
            status = "uncertain" if phase == "pulse" else "not_executed"
            session.record_execution(ExecutionReceipt(
                session.pending.command_id, session.pending.action, None, status, time(),
                "no_energy_delivered" if status == "not_executed" else "unknown_outcome", False,
            ))
            records.event("rejected_receipt", receipt=session.receipt)
        if session.started and session.stopped_reason is None:
            session.stop("OPERATOR_ABORT" if isinstance(error, (OperatorAbort, KeyboardInterrupt)) else phase.upper() + "_FAILED")
        records.event("failure", **failure)
        raise
    finally:
        plant.save_evaluation_state(output / "truth/final.npz")
        save_voxels(output / "observed/final.npz", session.state if session.started else initial_observation.state)
        write_json(output / "workflow_status.json", {
            "terminal_reason": session.stopped_reason, "confirmed_pulses": session.confirmed_pulses,
            "observed_pulses": session.last_observation.sequence if session.started else 0,
            "repairs": session.repair_events, "replans": session.replans, "failure": failure,
            "elapsed_s": perf_counter() - started,
        })
