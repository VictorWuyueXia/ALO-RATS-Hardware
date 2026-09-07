"""In-memory store for fixed global-source plan lineages."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from laser_ablation.planning.jax_bank.bank_storage import PlanBankStorage
from laser_ablation.planning.global_3d.interface import GlobalPlanningFailure
from laser_ablation.planning.jax_bank.contracts import (
    BANK_DIVERSITY_BONUS, STATE_SHARD_BYTES, BankTrajectory, PaddedActionBatch,
    QuickRolloutBatch, StaticTaskTensors, action_hash, linearization_path_hash,
    trajectory_hash,
)
from laser_ablation.planning.jax_bank.comparison_library import ComparisonROILibrary
from laser_ablation.planning.jax_bank.linear_contracts import LinearizationLibrary
from laser_ablation.planning.jax_bank.repair import RepairResult


def _dynamic_masks(task: StaticTaskTensors, current_sdf: np.ndarray) -> dict[str, np.ndarray]:
    """Derive exact-mask-relative dynamic channels for every saved prefix."""
    tissue = current_sdf <= 0.0
    removed = task.initial_tissue[None, ...] & ~tissue
    return {
        "remaining": tissue & task.target_mask[None, ...],
        "removed": removed,
        "overcut": removed & ~task.target_mask[None, ...],
    }


def trajectory_score(
    task: StaticTaskTensors, trajectory: BankTrajectory, maximum_pulses: int
) -> float:
    """Apply the fixed dominance-preserving provisional ranking objective."""
    remaining = float(trajectory.remaining_fraction[-1])
    overcut = float(trajectory.overcut_fraction[-1])
    clearance = float(trajectory.clearance_mm[-1])
    clearance_penalty = 0.0 if clearance == np.inf else 1.0 - np.clip(
        (clearance - task.hard_margin_mm) / (task.truncation_mm - task.hard_margin_mm),
        0.0, 1.0,
    )
    normalized_pulses = float(trajectory.pulse_count[-1]) / maximum_pulses
    return float(
        remaining + 0.90 * overcut + 0.05 * clearance_penalty
        + 0.02 * normalized_pulses
    )


class PlanBank(PlanBankStorage):
    """Keep one mutable trajectory slot per immutable global-source lineage."""

    def __init__(self, task: StaticTaskTensors, maximum_pulses: int) -> None:
        if maximum_pulses <= 0:
            raise ValueError("maximum_pulses must be positive")
        self.task = task
        self.maximum_pulses = int(maximum_pulses)
        self._records: dict[str, BankTrajectory] = {}
        self.rejection_ledger: list[dict[str, Any]] = []
        self.comparison_library: ComparisonROILibrary | None = None
        # Dormant Operation-1 research contract; the active planner never assigns it.
        self.linearization_library: LinearizationLibrary | None = None

    @property
    def trajectories(self) -> tuple[BankTrajectory, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def serialized_dynamic_state_bytes(self) -> int:
        """Return the uncompressed dynamic-state payload under the shard encoding."""
        voxel_count = int(np.prod(self.task.shape))
        bytes_per_prefix = np.dtype(np.float16).itemsize * voxel_count + 3 * (
            (voxel_count + 7) // 8
        )
        return sum(len(record.current_sdf) * bytes_per_prefix for record in self.trajectories)

    def add_rollout(
        self,
        batch: PaddedActionBatch,
        rollout: QuickRolloutBatch,
        origins: tuple[str, ...],
        linearization_path: np.ndarray,
    ) -> None:
        """Install every source row and mark hard-infeasible lineages inactive."""
        if len(origins) != batch.actions.shape[0]:
            raise ValueError("origins must align with the padded action batch")
        paths = np.asarray(linearization_path, dtype=np.int32)
        if paths.shape != batch.actions.shape[:2] + (2,):
            raise ValueError("linearization_path must have shape (batch, pulse, 2)")
        if np.any(paths[batch.action_mask] < 0):
            raise ValueError("active linearization paths must be nonnegative")
        if np.any(paths[~batch.action_mask] != -1):
            raise ValueError("masked linearization paths must equal [-1, -1]")
        for row, origin in enumerate(origins):
            length = int(np.count_nonzero(batch.action_mask[row]))
            actions = batch.actions[row, :length]
            path = paths[row, :length]
            source_id = batch.source_ids[row]
            identifier = trajectory_hash(
                actions, rollout.current_sdf[row, 0], path, source_id
            )
            route_hash = linearization_path_hash(path)
            if not rollout.constraint_feasible[row]:
                self.rejection_ledger.append({
                    "origin": origin, "source_id": source_id,
                    "action_hash": action_hash(actions), "trajectory_id": identifier,
                    "linearization_path_hash": route_hash,
                    "remaining_fraction": float(rollout.remaining_fraction[row, -1]),
                    "overcut_fraction": float(rollout.overcut_fraction[row, -1]),
                    "flags": {
                        name: bool(np.any(values[row])) for name, values in rollout.flags.items()
                    },
                })
            if identifier in self._records:
                record = self._records[identifier]
                self._records[identifier] = replace(
                    record,
                    active=bool(rollout.constraint_feasible[row]),
                    origins=tuple(sorted(set(record.origins + (origin,)))),
                )
                continue
            for previous_id, previous in tuple(self._records.items()):
                if previous.source_id == source_id:
                    del self._records[previous_id]
            states = rollout.current_sdf[row, : length + 1]
            record = BankTrajectory(
                trajectory_id=identifier, actions=actions.copy(),
                linearization_path=path.copy(), source_id=source_id,
                active=bool(rollout.constraint_feasible[row]),
                current_sdf=states.copy(),
                masks=_dynamic_masks(self.task, states),
                remaining_fraction=rollout.remaining_fraction[row, : length + 1].copy(),
                overcut_fraction=rollout.overcut_fraction[row, : length + 1].copy(),
                clearance_mm=rollout.clearance_mm[row, : length + 1].copy(),
                pulse_count=rollout.pulse_count[row, : length + 1].copy(),
                energy_j=rollout.energy_j[row, : length + 1].copy(),
                flags={name: values[row, :length] for name, values in rollout.flags.items()},
                score=float("nan"), origins=(origin,),
            )
            self._records[identifier] = replace(
                record, score=trajectory_score(self.task, record, self.maximum_pulses)
            )

    def active(self) -> BankTrajectory:
        active = tuple(record for record in self.trajectories if record.active)
        if not active:
            raise GlobalPlanningFailure("the JAX plan bank has no active hard-feasible trajectory")
        return min(active, key=lambda record: (record.score, record.trajectory_id))

    def retain_top(
        self,
        capacity_bytes: int = STATE_SHARD_BYTES,
        diversity_bonus: float = BANK_DIVERSITY_BONUS,
        required_trajectory_id: str | None = None,
    ) -> None:
        """Validate that the complete fixed-lineage bank fits the persistence budget."""
        voxel_count = int(np.prod(self.task.shape))
        bytes_per_prefix = 2 * voxel_count + 3 * ((voxel_count + 7) // 8)
        if diversity_bonus < 0.0:
            raise ValueError("bank diversity bonus must be nonnegative")
        if required_trajectory_id is not None:
            if required_trajectory_id not in self._records:
                raise KeyError(f"unknown required plan-bank trajectory: {required_trajectory_id}")
            if not self._records[required_trajectory_id].active:
                raise ValueError("required plan-bank trajectory must be active")
        source_ids = [record.source_id for record in self.trajectories]
        if len(source_ids) != len(set(source_ids)):
            raise RuntimeError("plan bank contains multiple trajectories for one source lineage")
        used_bytes = sum(len(record.current_sdf) * bytes_per_prefix for record in self.trajectories)
        if used_bytes > capacity_bytes:
            raise ValueError("the complete fixed-lineage plan bank exceeds its state budget")

    def trajectory(self, trajectory_id: str) -> BankTrajectory:
        if trajectory_id not in self._records:
            raise KeyError(f"unknown plan-bank trajectory: {trajectory_id}")
        return self._records[trajectory_id]

    def add_repair_result(self, result: RepairResult) -> BankTrajectory:
        """Replace repaired source slots; a feasible child can reactivate its lineage."""
        for item in result.feasible_trajectories:
            origin = f"mppi:{item.anchor_tail_id}:{item.iteration}:{item.sample}"
            expected_id = trajectory_hash(
                item.actions, item.current_sdf[0], item.linearization_path, item.source_id
            )
            if item.trajectory_id != expected_id:
                raise ValueError("repair trajectory_id does not cover its source route")
            if not item.admissible:
                self.rejection_ledger.append({
                    "origin": origin, "source_id": item.source_id,
                    "action_hash": item.action_hash, "trajectory_id": item.trajectory_id,
                    "linearization_path_hash": linearization_path_hash(item.linearization_path),
                    "remaining_fraction": float(item.remaining_fraction[-1]),
                    "overcut_fraction": float(item.overcut_fraction[-1]),
                    "flags": {name: bool(np.any(values)) for name, values in item.flags.items()},
                })
                continue
            if item.trajectory_id in self._records:
                record = self._records[item.trajectory_id]
                self._records[item.trajectory_id] = replace(
                    record, active=True,
                    origins=tuple(sorted(set(record.origins + (origin,))))
                )
                continue
            for previous_id, previous in tuple(self._records.items()):
                if previous.source_id == item.source_id:
                    del self._records[previous_id]
            record = BankTrajectory(
                trajectory_id=item.trajectory_id, actions=item.actions.copy(),
                linearization_path=item.linearization_path.copy(), source_id=item.source_id,
                active=True,
                current_sdf=item.current_sdf.copy(),
                masks=_dynamic_masks(self.task, item.current_sdf),
                remaining_fraction=item.remaining_fraction.copy(),
                overcut_fraction=item.overcut_fraction.copy(),
                clearance_mm=item.clearance_mm.copy(), pulse_count=item.pulse_count.copy(),
                energy_j=item.energy_j.copy(),
                flags={name: values.copy() for name, values in item.flags.items()},
                score=float("nan"), origins=(origin,),
            )
            self._records[item.trajectory_id] = replace(
                record, score=trajectory_score(self.task, record, self.maximum_pulses)
            )
        return self.trajectory(result.selected.trajectory_id)

    def state_tensor(self, trajectory: BankTrajectory, prefix: int) -> np.ndarray:
        if prefix < 0 or prefix >= trajectory.current_sdf.shape[0]:
            raise IndexError("prefix lies outside the stored trajectory")
        return np.stack((
            trajectory.current_sdf[prefix], self.task.target_sdf, self.task.safe_sdf,
            trajectory.masks["remaining"][prefix].astype(np.float32),
            trajectory.masks["removed"][prefix].astype(np.float32),
            trajectory.masks["overcut"][prefix].astype(np.float32),
        ), axis=0).astype(np.float32)

    def state_tensors(self, trajectory: BankTrajectory, start: int = 0) -> np.ndarray:
        return np.stack([
            self.state_tensor(trajectory, prefix)
            for prefix in range(start, len(trajectory.current_sdf))
        ])


__all__ = ["PlanBank", "trajectory_score"]
