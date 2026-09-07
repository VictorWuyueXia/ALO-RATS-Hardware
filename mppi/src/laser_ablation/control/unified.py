"""Exact-voxel experiment adapter for the externally driven controller session."""

from pathlib import Path
from time import perf_counter, time

from laser_ablation.control.interaction import ControllerObservation, DesignatedTask, ExecutionReceipt
from laser_ablation.control.session import ControllerSession
from laser_ablation.control.session_planning import PERIODIC_REPAIR_EVENT_COLUMNS
from laser_ablation.core.plan import ControllerOutcome, UnifiedControllerResult
from laser_ablation.metrics import evaluate_ablation


class UnifiedAblationController:
    """Drive the shared session with the experiment's exact plant and observer."""

    def __init__(self, planner, simulator, observer, completion_remaining_pct,
                 periodic_repair_pulses, maximum_pulses, device_identity):
        if completion_remaining_pct < 0:
            raise ValueError("completion remaining percentage must be nonnegative")
        if periodic_repair_pulses <= 0 or maximum_pulses <= 0:
            raise ValueError("periodic repair interval and maximum pulses must be positive")
        self.planner, self.simulator, self.observer = planner, simulator, observer
        self.completion_remaining_pct = float(completion_remaining_pct)
        self.periodic_repair_pulses = int(periodic_repair_pulses)
        self.maximum_pulses = int(maximum_pulses)
        self.device_identity = device_identity

    def run(self, initial_state, raster_generator, frozen_generator,
            output_directory: Path, random_seed: int) -> UnifiedControllerResult:
        """Keep execution, measurement, and experiment timing outside the decision kernel."""
        session = ControllerSession(
            self.planner, self.observer, raster_generator, frozen_generator,
            self.completion_remaining_pct, self.periodic_repair_pulses, self.maximum_pulses,
            output_directory, random_seed,
        )
        task = DesignatedTask(initial_state, "experiment", "caller-resolved-experiment-method")
        session.start(ControllerObservation(
            initial_state, task.task_id, "scan_000", 0, time(), None,
        ), task)
        records, cumulative_energy_j = [], 0.0
        while True:
            started = perf_counter()
            request = session.next_action()
            if request is None:
                break
            action = request.action
            exact_started = perf_counter()
            next_state = self.simulator.step(session.state, action)
            execution_time = perf_counter() - exact_started
            session.record_execution(ExecutionReceipt(
                request.command_id, action, action, "completed", time(),
                "commanded_simulation_energy_j", True,
            ))
            observation_started = perf_counter()
            session.update(ControllerObservation(
                next_state, task.task_id, f"scan_{session.confirmed_pulses:03d}",
                session.confirmed_pulses, time(), request.command_id,
            ))
            observation_time = perf_counter() - observation_started
            after = evaluate_ablation(session.state)
            cumulative_energy_j += action.energy_j
            bank = self.planner.bank
            if bank is None:
                raise RuntimeError("active plan execution requires a populated plan bank")
            records.append({
                "pulse": session.confirmed_pulses - 1,
                "trajectory_id": request.trajectory_id,
                "parent_trajectory_id": session.plan.parent_trajectory_id,
                "active_prefix": request.active_prefix,
                **session.cycle_record,
                "repair_count": session.repairs, "replan_count": session.replans,
                "control_cycle_time_s": perf_counter() - started,
                "exact_execution_time_s": execution_time,
                "sdf_observation_time_s": observation_time,
                "gpu_memory_total_mb": self.planner.total_device_memory_mb(),
                "trajectory_bank_count": len(bank.trajectories),
                "bank_serialized_dynamic_state_bytes": bank.serialized_dynamic_state_bytes,
                "planned_energy_j": action.energy_j, "administered_energy_j": action.energy_j,
                "cumulative_administered_energy_j": cumulative_energy_j,
                "physics_response_scale": self.simulator.last_response_scale,
                "remaining_pct": after.remaining_pct, "overcut_pct": after.total_overcut_pct,
                "minimum_clearance_mm": after.minimum_clearance_mm,
                "hard_violations": after.hard_violations,
            })
            if session.confirmed_pulses == 1 or session.confirmed_pulses % 10 == 0:
                print(f"controller pulse={session.confirmed_pulses}/{self.maximum_pulses} "
                      f"remaining={after.remaining_pct:.3f}% repairs={session.repairs} "
                      f"replans={session.replans}", flush=True)
        metadata = {"repair_count": session.repairs,
                    "selected_trajectory_ids": tuple(session.selected_ids)}
        if session.selected_ids:
            metadata["periodic_repair_pulses"] = self.periodic_repair_pulses
        outcome = ControllerOutcome(
            session.state, evaluate_ablation(session.state), session.executed_actions,
            session.replans, session.stopped_reason, metadata,
        )
        return UnifiedControllerResult(
            outcome, tuple(records), session.repairs, session.replans,
            tuple(session.selected_ids), self.device_identity,
            (session.machine, session.repairs_directory, session.banks_directory,
             session.observations_directory),
        )
