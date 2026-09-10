"""Orchestrator for JAX proposal screening, bank persistence, and state-tail matching."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import subprocess

import numpy as np

from laser_ablation.core.actions import PhysicalAction, PhysicalActionBounds
from laser_ablation.core.sdf import SDFState
from laser_ablation.core.state import VoxelState
from laser_ablation.physics.super_gaussian import PhysicsConfig
from laser_ablation.planning.jax_bank.bank import PlanBank
from laser_ablation.planning.jax_bank.comparison_library import (
    build_comparison_roi_library,
)
from laser_ablation.planning.jax_bank.contracts import (
    MatchingTailBank,
    PaddedActionBatch,
    ProvisionalPlan,
    QuickRolloutBatch,
    StaticTaskTensors,
)
from laser_ablation.planning.global_3d.interface import GlobalPlanningFailure
from laser_ablation.planning.jax_bank.global_seeds import RasterGlobalSeeds
from laser_ablation.planning.jax_bank.repair import (
    GlobalReplanReason,
    GlobalReplanRequired,
    RepairContext,
    RepairRequest,
)
from laser_ablation.planning.jax_bank.mppi_repairer import MPPIPlanRepairer
from laser_ablation.planning.jax_bank.rollout import rollout_in_batches
from laser_ablation.planning.jax_bank.similarity import (
    _routed_roi_values,
    matching_tail_bank,
)


class JaxPlanBankPlanner:
    """Proposal-only plan-bank route that never returns an exact controller plan."""

    def __init__(
        self,
        maximum_pulses: int,
        physics: PhysicsConfig,
        bounds: PhysicalActionBounds,
        completion_remaining_fraction: float,
        repairer: MPPIPlanRepairer,
        devices: tuple[object, ...],
        rollout_batch_size: int,
        geometry_generator: object | None,
    ) -> None:
        if maximum_pulses <= 0:
            raise ValueError("maximum_pulses must be positive")
        self.maximum_pulses = maximum_pulses
        self.physics = physics
        self.bounds = bounds
        self.completion_remaining_fraction = completion_remaining_fraction
        self.repairer = repairer
        if not devices:
            raise ValueError("JAX plan-bank planner requires explicit devices")
        self.devices = devices
        if rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        self.rollout_batch_size = rollout_batch_size
        self.geometry_generator = geometry_generator
        self.global_planning_calls = 0
        self.bank: PlanBank | None = None
        self._parent_trajectory_ids: dict[str, str] = {}

    def propose(
        self,
        voxel_state: VoxelState,
        sdf_state: SDFState,
        raster_generator: object,
        bank_directory: Path | None = None,
    ) -> ProvisionalPlan:
        """Screen four rasters and add geometry-aware parents after initial planning."""
        task = StaticTaskTensors.from_state(
            voxel_state, sdf_state, self.bounds, self.physics,
            self.completion_remaining_fraction,
        )
        bank = PlanBank(task, self.maximum_pulses)
        include_geometry = self.global_planning_calls > 0
        self.global_planning_calls += 1
        seeds = RasterGlobalSeeds.build(
            voxel_state, raster_generator,
            self.geometry_generator if include_geometry else None,
        )
        generated_actions = tuple(actions[:self.maximum_pulses] for actions in seeds.actions)
        generated_origins = seeds.source_ids
        generated_batch = PaddedActionBatch.from_sequences(generated_actions, generated_origins)
        expanded_actions, expanded_origins = self.repairer.expand_initial(
            task, generated_batch, generated_origins
        )
        if not expanded_actions:
            raise GlobalPlanningFailure("initial MPPI produced zero hard-feasible weighted children")
        if len(expanded_actions) > len(generated_actions):
            raise RuntimeError("global MPPI retained more lineages than its parent population")
        expanded_batch = PaddedActionBatch.from_sequences(expanded_actions, expanded_origins)
        routed_paths = np.full(expanded_batch.action_mask.shape + (2,), -1, dtype=np.int32)
        for row, actions in enumerate(expanded_actions):
            routed_paths[row, :len(actions), 0] = row
            routed_paths[row, :len(actions), 1] = np.arange(len(actions), dtype=np.int32)
        print(
            f"global path-integral verification started lineages={expanded_batch.actions.shape[0]} "
            f"padded_pulses={expanded_batch.actions.shape[1]}",
            flush=True,
        )
        expanded_rollout = self._rollout(task, expanded_batch)
        print("global path-integral verification completed", flush=True)
        comparison_library = build_comparison_roi_library(
            expanded_rollout.current_sdf, expanded_batch.action_mask, expanded_batch.actions[..., 4],
            expanded_origins, task.fingerprint,
        )
        for index, actions in enumerate(expanded_actions):
            print(
                f"path-integral lineage={index} origin={expanded_origins[index]} "
                f"pulses={len(actions)} "
                f"remaining={100.0 * expanded_rollout.remaining_fraction[index, -1]:.3f}% "
                f"overcut={100.0 * expanded_rollout.overcut_fraction[index, -1]:.3f}% "
                f"active={bool(expanded_rollout.constraint_feasible[index])}",
                flush=True,
            )
        bank.add_rollout(
            expanded_batch, expanded_rollout, expanded_origins, routed_paths,
        )
        bank.comparison_library = comparison_library
        bank.retain_top()
        print(
            f"global plan bank retained={len(bank.trajectories)} "
            f"active={sum(record.active for record in bank.trajectories)}",
            flush=True,
        )

        self.bank = bank
        self._parent_trajectory_ids = {
            record.trajectory_id: record.trajectory_id for record in bank.trajectories
        }
        active = bank.active()
        if bank_directory is not None:
            bank.save(
                bank_directory,
                {
                    "planner": type(self).__name__,
                    "generated_candidates": len(generated_actions),
                    "selected_lineages": len(bank.trajectories),
                    "active_lineages": sum(record.active for record in bank.trajectories),
                    "generated_candidate_lengths": [
                        len(actions) for actions in generated_actions
                    ],
                    "repairer": type(self.repairer).__name__,
                    "selected_trajectory": active.trajectory_id,
                    "global_seeds": seeds.provenance,
                    "comparison_roi_library": {
                        "fingerprint": comparison_library.library_fingerprint,
                        "storage_bytes": comparison_library.storage_bytes,
                        "maximum_roi_voxels": int(
                            comparison_library.roi_indices.shape[-1]
                        ),
                    },
                },
            )
        return ProvisionalPlan(
            trajectory_id=active.trajectory_id,
            actions=tuple(PhysicalAction.from_array(row) for row in active.actions),
            predicted_states=bank.state_tensors(active),
            score=active.score,
            task_fingerprint=task.fingerprint,
            truncation_mm=task.truncation_mm,
            parent_trajectory_id=self._parent_trajectory_ids[active.trajectory_id],
        )

    def repair(
        self,
        voxel_state: VoxelState,
        sdf_state: SDFState,
        request: RepairRequest,
    ) -> ProvisionalPlan:
        """Validate one repair trigger and delegate matched tails to the MPPI repairer."""
        if self.bank is None:
            raise RuntimeError("a plan bank must be built before plan repair")
        task = replace(
            self.bank.task,
            initial_current_sdf=np.asarray(
                sdf_state.tensor[SDFState.CURRENT_TISSUE], dtype=np.float32
            ),
        )
        if self.bank.comparison_library is None:
            raise RuntimeError("plan repair requires the persisted ROI comparison library")
        active = self.bank.trajectory(request.active_trajectory_id)
        if not 0 <= request.active_prefix < len(active.actions):
            raise ValueError("repair prefix must select a remaining active action")
        matching = matching_tail_bank(
            self.bank, task, sdf_state, self.bank.comparison_library,
            self.repairer.config.roi_match_cosine_min,
            request.active_trajectory_id, request.active_prefix,
        )
        if not matching.tails:
            raise GlobalReplanRequired(GlobalReplanReason.NO_MATCHING_ANCHOR)
        result = self.repairer.repair(
            RepairContext(request, request.active_similarity), task, matching,
            self.bank.comparison_library,
        )
        anchor_parents = {
            tail.tail_id: tail.trajectory_ids[0] for tail in matching.tails
        }
        direct_parent = anchor_parents[result.selected.anchor_tail_id]
        parent_trajectory_id = self._parent_trajectory_ids.get(
            direct_parent, direct_parent
        )
        selected = self.bank.add_repair_result(result)
        self._parent_trajectory_ids[selected.trajectory_id] = parent_trajectory_id
        self.bank.retain_top(required_trajectory_id=selected.trajectory_id)
        if request.artifact_directory is not None:
            updated_directory = (
                request.artifact_directory.parent
                / f"plan_bank_after_{selected.trajectory_id[:12]}"
            )
            self.bank.save(
                updated_directory,
                {
                    "planner": type(self).__name__,
                    "repairer": type(self.repairer).__name__,
                    "repair_artifacts": str(request.artifact_directory),
                    "selected_trajectory": selected.trajectory_id,
                },
            )
        return ProvisionalPlan(
            trajectory_id=selected.trajectory_id,
            actions=tuple(PhysicalAction.from_array(row) for row in selected.actions),
            predicted_states=self.bank.state_tensors(selected),
            score=selected.score,
            task_fingerprint=task.fingerprint,
            truncation_mm=task.truncation_mm,
            parent_trajectory_id=parent_trajectory_id,
        )

    def match_state(
        self, voxel_state: VoxelState, sdf_state: SDFState
    ) -> MatchingTailBank:
        """Build the complete fingerprint-compatible tail set for live repair."""
        if self.bank is None:
            raise RuntimeError("a plan bank must be built before state-tail matching")
        task = StaticTaskTensors.from_state(
            voxel_state, sdf_state, self.bounds, self.physics,
            self.completion_remaining_fraction,
        )
        if self.bank.comparison_library is None:
            raise RuntimeError("state matching requires the persisted ROI comparison library")
        return matching_tail_bank(
            self.bank, task, sdf_state, self.bank.comparison_library,
            self.repairer.config.roi_match_cosine_min,
        )

    def active_roi_similarity(
        self,
        observed: SDFState,
        predicted_current_sdf: np.ndarray,
        trajectory_id: str,
        active_prefix: int,
    ) -> float:
        """Compare live and predicted geometry in the first remaining route-local ROI."""
        if self.bank is None or self.bank.comparison_library is None:
            raise RuntimeError("active ROI comparison requires a persisted ROI library")
        trajectory = self.bank.trajectory(trajectory_id)
        if not 0 <= active_prefix < len(trajectory.actions):
            raise ValueError("active prefix must select a remaining trajectory action")
        observed_current = np.asarray(
            observed.tensor[SDFState.CURRENT_TISSUE], dtype=np.float32
        )
        predicted = np.asarray(predicted_current_sdf, dtype=np.float32)
        values, valid_norm = _routed_roi_values(
            observed_current, predicted[None],
            trajectory.linearization_path[active_prefix][None],
            self.bank.comparison_library,
        )
        if not bool(valid_norm[0]):
            raise ValueError("active ROI comparison requires finite nonzero selected geometry norms")
        return float(values[0])

    def predict_next(
        self, voxel_state: VoxelState, sdf_state: SDFState, action: PhysicalAction
    ) -> np.ndarray:
        """Rebase one JAX transition on the freshly observed current-tissue SDF."""
        task = StaticTaskTensors.from_state(
            voxel_state, sdf_state, self.bounds, self.physics,
            self.completion_remaining_fraction,
        )
        batch = PaddedActionBatch.from_sequences(
            (action.as_array()[None].astype(np.float32),), ("one_step_rebase",)
        )
        return self._rollout(task, batch).current_sdf[0, 1]

    def total_device_memory_mb(self) -> float:
        """Report this controller process's aggregate NVIDIA memory footprint."""
        query = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        )
        process_id = os.getpid()
        return float(sum(
            float(memory) for pid, memory in (
                line.split(",") for line in query.stdout.splitlines()
            ) if int(pid) == process_id
        ))

    def _rollout(
        self, task: StaticTaskTensors, batch: PaddedActionBatch
    ) -> QuickRolloutBatch:
        """Execute full-state rollouts in explicitly bounded batches."""
        return rollout_in_batches(
            task, batch, self.rollout_batch_size, self.devices
        )
