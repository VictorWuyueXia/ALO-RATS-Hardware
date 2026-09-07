"""One-action controller session with separate execution and observation ownership."""

import csv
from pathlib import Path
from time import time
from uuid import uuid4

import numpy as np

from laser_ablation.control.interaction import (
    ActionRequest, ControllerObservation, DesignatedTask, ExecutionReceipt, static_fingerprint,
)
from laser_ablation.control.session_planning import PERIODIC_REPAIR_EVENT_COLUMNS, propose, select_action
from laser_ablation.metrics import evaluate_ablation


class ControllerSession:
    """Retain plan state while an external workflow owns the plant and scan source."""

    def __init__(self, planner, observer, raster_generator, frozen_generator,
                 completion_remaining_pct, periodic_repair_pulses, maximum_pulses,
                 output_directory, random_seed):
        if completion_remaining_pct < 0 or min(periodic_repair_pulses, maximum_pulses) <= 0:
            raise ValueError("Invalid controller completion or pulse limits")
        self.planner, self.observer = planner, observer
        self.raster_generator, self.frozen_generator = raster_generator, frozen_generator
        self.completion_remaining_pct = float(completion_remaining_pct)
        self.periodic_repair_pulses = int(periodic_repair_pulses)
        self.maximum_pulses = int(maximum_pulses)
        self.machine = Path(output_directory) / "machine_readables"
        self.repairs_directory = self.machine / "repairs"
        self.banks_directory = self.machine / "plan_banks"
        self.observations_directory = self.machine / "observations"
        self.event_path = self.machine / "periodic_repair_events.csv"
        self.session_id = uuid4().hex
        self.repair_seeds = iter(np.random.SeedSequence(random_seed).generate_state(
            maximum_pulses // periodic_repair_pulses + 1, dtype=np.uint32,
        ))
        self.started = False
        self.stopped_reason = None
        self.pending = self.receipt = None
        self.plan = None
        self.prefix = self.confirmed_pulses = self.repairs = self.replans = 0
        self.executions, self.selected_ids, self.repair_events = [], [], []
        self.seen_scans = set()
        self.cycle_record = {}

    def start(self, observation: ControllerObservation, task: DesignatedTask):
        """Bind immutable treatment geometry and build the initial plan exactly once."""
        if self.started:
            raise RuntimeError("Session is already started")
        self.task, self.task_id = task, task.task_id
        self._validate_observation(observation)
        if observation.sequence != 0 or observation.command_id is not None:
            raise ValueError("Initial observation must have sequence zero and no execution")
        for directory in (self.repairs_directory, self.banks_directory, self.observations_directory):
            directory.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=PERIODIC_REPAIR_EVENT_COLUMNS).writeheader()
        self._incorporate(observation)
        self.started = True
        if propose(self, 0):
            self.selected_ids.append(self.plan.trajectory_id)

    def next_action(self):
        """Return one proposal only when no execution or fresh observation is pending."""
        if not self.started:
            raise RuntimeError("Start the session before requesting an action")
        if self.stopped_reason is not None:
            return None
        if self.pending is not None:
            raise RuntimeError("An execution or its fresh observation is still pending")
        # Preserve the legacy loop's pulse-limit test before its completion test.
        if self.confirmed_pulses >= self.maximum_pulses:
            self.stopped_reason = "maximum_pulses"
            return None
        if evaluate_ablation(self.state).is_complete(self.completion_remaining_pct):
            self.stopped_reason = "completion_gate"
            return None
        action = select_action(self)
        if action is None:
            return None
        self.pending = ActionRequest(
            f"{self.session_id}:{self.confirmed_pulses:04d}", self.task_id,
            self.plan.trajectory_id, self.prefix, action, time(),
        )
        return self.pending

    def record_execution(self, receipt: ExecutionReceipt):
        """Record completion once; never advance from an attempt or infer the next scan."""
        if self.stopped_reason is not None or self.pending is None or self.receipt is not None:
            raise RuntimeError("No unacknowledged action is available")
        if receipt.command_id != self.pending.command_id or receipt.requested_action != self.pending.action:
            raise ValueError("Execution does not match the outstanding command")
        if receipt.completed_at_s < self.pending.issued_at_s:
            raise ValueError("Execution predates its command")
        self.executions.append(receipt)
        self.receipt = receipt
        if receipt.pulse_status == "completed":
            self.confirmed_pulses += 1
        else:
            self.stopped_reason = "EXECUTION_FAILED" if receipt.pulse_status == "not_executed" else "EXECUTION_UNCERTAIN"

    def update(self, observation: ControllerObservation):
        """Incorporate a registered post-pulse scan before advancing the active prefix."""
        if self.stopped_reason is not None or self.receipt is None:
            raise RuntimeError("A completed pulse is required before a new observation")
        self._validate_observation(observation)
        if observation.command_id != self.receipt.command_id or observation.sequence != self.confirmed_pulses:
            raise ValueError("Observation does not identify the confirmed pulse prefix")
        if observation.timestamp_s < self.receipt.completed_at_s or observation.timestamp_s <= self.last_observation.timestamp_s:
            raise ValueError("Observation is stale or predates pulse completion")
        self._incorporate(observation)
        self.prefix += 1
        self.pending = self.receipt = None

    def stop(self, reason):
        """Latch an explicit workflow failure without forgetting an already confirmed pulse."""
        if not self.started or not reason or self.stopped_reason is not None:
            raise RuntimeError("Stop requires an active session and an explicit reason")
        self.stopped_reason = str(reason)

    def _validate_observation(self, observation):
        if observation.task_id != self.task_id:
            raise ValueError("Observation belongs to another task")
        fingerprint = static_fingerprint(observation.state, self.task.frame_id, self.task.authority_id)
        if fingerprint != self.task_id:
            raise ValueError("Observation changed immutable task geometry or action meaning")
        if observation.scan_id in self.seen_scans:
            raise ValueError("Observation scan identity was already consumed")

    def _incorporate(self, observation):
        self.state = observation.state
        self.observed = self.observer.observe(self.state)
        self.last_observation = observation
        self.seen_scans.add(observation.scan_id)
        np.savez_compressed(
            self.observations_directory / f"prefix_{observation.sequence:03d}.npz",
            tensor=self.observed.tensor, tissue_sdf_mm=self.observed.observation.tissue_sdf_mm,
            exact_tissue=self.state.tissue,
        )

    @property
    def executed_actions(self):
        return tuple(receipt.achieved_action for receipt in self.executions
                     if receipt.pulse_status == "completed")
