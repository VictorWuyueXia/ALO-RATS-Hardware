"""Initial nonlinear MPPI population and active sequential path-integral repair."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np

from laser_ablation.planning.jax_bank.contracts import (
    MatchingTailBank,
    PaddedActionBatch,
    StaticTaskTensors,
    action_hash,
)
from laser_ablation.planning.jax_bank.comparison_library import ComparisonROILibrary
from laser_ablation.planning.jax_bank.exact_segment import ExactSegmentBeam
from laser_ablation.planning.jax_bank.exact_segment_artifacts import save_exact_segment_artifacts
from laser_ablation.planning.jax_bank.exact_trajectory import finalize_exact_beam
from laser_ablation.planning.jax_bank.mppi_execution import MPPIExecutor
from laser_ablation.planning.jax_bank.mppi_records import prepare_anchors, provisional_score
from laser_ablation.planning.jax_bank.path_integral_segment import stream_path_integral_segment
from laser_ablation.planning.jax_bank.repair import (
    GlobalReplanReason,
    GlobalReplanRequired,
    FeasibleRepairTrajectory,
    MPPIRepairConfig,
    RepairContext,
    RepairDiagnostics,
    RepairIterationSummary,
    RepairResult,
)
from laser_ablation.planning.jax_bank.segment_beam import SegmentBeam


class MPPIPlanRepairer:
    """Populate globally and causally refine one exact weighted child per parent."""

    def __init__(
        self,
        config: MPPIRepairConfig,
        devices: tuple[object, ...],
        initial_seed: int,
        maximum_pulses: int,
    ) -> None:
        if maximum_pulses <= 0:
            raise ValueError("maximum_pulses must be positive")
        self.config = config
        self.executor = MPPIExecutor(config, devices)
        self.initial_seed = initial_seed
        self.maximum_pulses = int(maximum_pulses)

    def expand_initial(
        self,
        task: StaticTaskTensors,
        batch: PaddedActionBatch,
        origins: tuple[str, ...],
    ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
        """Return only hard-feasible weighted children from the raw global parents."""
        tail_ids = tuple(
            sha256((
                action_hash(batch.actions[row, :int(np.count_nonzero(batch.action_mask[row]))])
                + ":" + origins[row]
            ).encode()).hexdigest()
            for row in range(batch.actions.shape[0])
        )
        source_rows = np.arange(len(tail_ids), dtype=np.int32)
        nominal = batch.actions.copy()
        origin = batch.actions.copy()
        mask = batch.action_mask.copy()
        for iteration in range(self.config.iterations):
            samples, perturbations = self.executor.sample(
                self.initial_seed, iteration, tail_ids, source_rows, nominal, mask, mask, task
            )
            rollout = self.executor.screen(samples, mask, tail_ids, source_rows, task, iteration)
            _, _, weights = self.executor.cost_and_weights(
                samples, origin, mask, rollout, task
            )
            squared_weights = np.sum(weights * weights, axis=1)
            effective_samples = 1.0 / squared_weights[squared_weights > 0.0]
            nominal = self.executor.update(nominal, perturbations, weights, mask, task)
            print(
                f"initial MPPI iteration={iteration + 1}/{self.config.iterations} "
                f"hard_feasible={int(np.count_nonzero(rollout.constraint_feasible))} "
                f"admissible={int(np.count_nonzero(rollout.accepted))} "
                f"median_ess={float(np.median(effective_samples)) if len(effective_samples) else 0.0:.2f} "
                f"flags={{{', '.join(f'{name}:{int(np.count_nonzero(values))}' for name, values in rollout.flags.items())}}}",
                flush=True,
            )
        children = self.executor.screen(
            nominal[:, None], mask, tail_ids, source_rows, task, self.config.iterations,
        )
        selected = np.flatnonzero(children.constraint_feasible)
        print(
            f"initial MPPI weighted-children hard_feasible={len(selected)}/{len(nominal)}",
            flush=True,
        )
        lengths = np.count_nonzero(mask, axis=1)
        retained_lengths = np.minimum(lengths[selected], children.pulse_count[selected])
        return (
            tuple(nominal[row, :length].copy() for row, length in zip(selected, retained_lengths, strict=True)),
            tuple(origins[row] for row in selected),
        )

    def repair(
        self,
        context: RepairContext,
        task: StaticTaskTensors,
        anchors: MatchingTailBank,
        library: ComparisonROILibrary,
    ) -> RepairResult:
        """Refine each segment repeatedly, then advance from its exact weighted boundary."""
        self._validate(task, anchors, library)
        prepared = prepare_anchors(anchors, self.config.max_anchors)
        anchors_count = len(prepared.tails)
        beam = SegmentBeam(
            prepared.origin_actions.copy(), prepared.origin_actions.copy(),
            prepared.action_mask.copy(), prepared.linearization_path.copy(),
            prepared.source_ids, tuple(tail.tail_id for tail in prepared.tails),
            prepared.candidate_ids,
            np.arange(anchors_count, dtype=np.int32),
        )
        initial_scores = np.asarray([tail.score for tail in prepared.tails], np.float32)
        state = ExactSegmentBeam(
            beam, np.broadcast_to(task.initial_current_sdf, (anchors_count,) + task.shape).copy(),
            np.zeros(anchors_count, np.int32), initial_scores.copy(), initial_scores.copy(),
            np.zeros(anchors_count, np.int32), np.zeros(anchors_count, bool), -1,
        )
        gib = 1024**3
        runtime_reserve = int(1.5 * gib)
        rollout_reserve = int(3.0 * gib)
        staging_reserve = int(0.25 * gib)
        cap_bytes = int(self.config.device_memory_cap_gib * gib)
        requested_pool = int(self.config.segment_working_pool_gib * gib)
        remaining_bytes = cap_bytes - runtime_reserve - rollout_reserve - staging_reserve
        beam_budget = min(requested_pool, remaining_bytes)
        record_bytes = beam.record_bytes(task.shape)
        capacity_records = beam_budget // record_bytes
        if capacity_records < anchors_count:
            raise MemoryError("the exact repair working pool cannot retain one final full trajectory")
        planned_total = runtime_reserve + rollout_reserve + staging_reserve + beam_budget
        if planned_total > cap_bytes:
            raise MemoryError("the exact segment repair exceeds its configured device-memory cap")
        microbatch_size = self.config.rollout_batch_size
        started = perf_counter()
        summaries: list[RepairIterationSummary] = []
        segment_count = self.config.segment_count(beam.maximum_pulses)
        for stage in range(segment_count):
            beam = state.beam
            starts, stops, unfinished = beam.segment_bounds(
                stage, self.config.segment_length_pulses, self.config.segment_minimum_pulses,
            )
            if not np.any(unfinished):
                break
            active_rows, complete_rows = np.flatnonzero(unfinished), np.flatnonzero(~unfinished)
            active_start = state.take(active_rows)
            iteration_state = active_start
            iteration_starts, iteration_stops = starts[active_rows], stops[active_rows]
            next_active = None
            for refinement in range(self.config.iterations):
                before = iteration_state
                noise_iteration = stage * self.config.iterations + refinement
                next_active, stats = stream_path_integral_segment(
                    self.executor, context.request.random_seed, noise_iteration, before,
                    iteration_starts, iteration_stops, task, microbatch_size,
                )
                retained_count = 0 if next_active is None else next_active.beam.count
                summaries.append(RepairIterationSummary(
                    iteration=refinement, active_anchors=before.beam.anchor_ids,
                    hard_feasible_samples=int(stats["hard_feasible_count"]),
                    admissible_samples=int(stats["admissible_count"]),
                    rejection_counts=dict(stats["rejection_counts"]), segment_index=stage,
                    segment_bounds=tuple(
                        (before.beam.anchor_ids[row], int(iteration_starts[row]), int(iteration_stops[row]))
                        for row in range(before.beam.count)
                    ),
                    parent_count=before.beam.count, child_count=int(stats["child_count"]),
                    retained_count=retained_count, capacity_records=capacity_records,
                    used_record_bytes=retained_count * record_bytes,
                    compaction_before_count=0, compaction_after_count=0,
                    compaction_before_bytes=0, compaction_after_bytes=0,
                    segment_rollout_rows=int(stats["child_count"]),
                    segment_hard_feasible_rows=int(stats["segment_hard_feasible_count"]),
                    shortlisted_rows=int(stats["shortlisted_count"]),
                    suffix_rollout_rows=int(stats["suffix_verified_count"]),
                    support_rejections=int(stats["support_rejections"]),
                    contact_guard_rejections=int(stats["contact_guard_rejections"]),
                    linearization_trust_rejections=int(stats["linearization_trust_rejections"]),
                    sampling_seconds=float(stats["sampling_seconds"]),
                    rollout_seconds=float(stats["rollout_seconds"]),
                    ranking_seconds=float(stats["ranking_seconds"]),
                    segment_rollout_seconds=float(stats["segment_rollout_seconds"]),
                    suffix_rollout_seconds=float(stats["suffix_rollout_seconds"]),
                    weighted_parent_count=int(stats["weighted_parent_count"]),
                    weighted_feasible_child_count=int(stats["weighted_feasible_child_count"]),
                    weight_ess_min=float(stats["weight_ess_min"]),
                    weight_ess_median=float(stats["weight_ess_median"]),
                    weight_entropy_median=float(stats["weight_entropy_median"]),
                    weighted_verification_seconds=float(stats["weighted_verification_seconds"]),
                ))
                print(
                    f"path-integral segment={stage + 1}/{segment_count} "
                    f"refinement={refinement + 1}/{self.config.iterations} "
                    f"children={stats['child_count']} weighted_parents={stats['weighted_parent_count']} "
                    f"retained={retained_count}/{capacity_records}", flush=True,
                )
                if next_active is None:
                    break
                if refinement + 1 < self.config.iterations:
                    retained = np.asarray(stats["retained_parent_indices"], np.int32)
                    start_rows = before.take(retained)
                    iteration_starts, iteration_stops = (
                        iteration_starts[retained], iteration_stops[retained]
                    )
                    iteration_state = ExactSegmentBeam(
                        next_active.beam, start_rows.boundary_sdf, start_rows.boundary_step,
                        next_active.mppi_cost, next_active.provisional_score,
                        next_active.sample, next_active.admissible, noise_iteration,
                    )
            carried = state.take(complete_rows) if len(complete_rows) else None
            if next_active is None and carried is None:
                raise GlobalReplanRequired(GlobalReplanReason.ALL_ANCHORS_INFEASIBLE)
            if next_active is None:
                state = carried
            elif carried is None:
                state = next_active
            else:
                combined = SegmentBeam(
                    np.concatenate((next_active.beam.actions, carried.beam.actions)),
                    np.concatenate((next_active.beam.origin_actions, carried.beam.origin_actions)),
                    np.concatenate((next_active.beam.action_mask, carried.beam.action_mask)),
                    np.concatenate((next_active.beam.linearization_path, carried.beam.linearization_path)),
                    next_active.beam.source_ids + carried.beam.source_ids,
                    next_active.beam.anchor_ids + carried.beam.anchor_ids,
                    next_active.beam.candidate_ids + carried.beam.candidate_ids,
                    np.concatenate((next_active.beam.anchor_slots, carried.beam.anchor_slots)),
                )
                state = ExactSegmentBeam(
                    combined, np.concatenate((next_active.boundary_sdf, carried.boundary_sdf)),
                    np.concatenate((next_active.boundary_step, carried.boundary_step)),
                    np.concatenate((next_active.mppi_cost, carried.mppi_cost)),
                    np.concatenate((next_active.provisional_score, carried.provisional_score)),
                    np.concatenate((next_active.sample, carried.sample)),
                    np.concatenate((next_active.admissible, carried.admissible)), noise_iteration,
                )
        beam = state.beam
        final_rows = finalize_exact_beam(self.executor, state, task)
        admissible = tuple(item for item in final_rows if item.admissible)
        if not admissible:
            raise GlobalReplanRequired(GlobalReplanReason.NO_ADMISSIBLE_REPAIR)
        selected = min(
            admissible,
            key=lambda item: (
                item.mppi_cost, provisional_score(task, item, beam.maximum_pulses),
                item.trajectory_id,
            ),
        )
        elapsed = perf_counter() - started
        diagnostics = RepairDiagnostics(
            matched_tails=len(anchors.tails),
            selected_anchors=tuple(tail.tail_id for tail in prepared.tails),
            dropped_anchors=tuple(sorted(
                set(tail.tail_id for tail in prepared.tails) - set(beam.anchor_ids)
            )),
            hard_feasible_samples=sum(item.hard_feasible_samples for item in summaries),
            admissible_samples=sum(item.admissible_samples for item in summaries),
            selected_trajectory_id=selected.trajectory_id,
            iterations=tuple(summaries), elapsed_seconds=elapsed,
        )
        result = RepairResult(selected, final_rows, diagnostics)
        if context.request.artifact_directory is not None:
            save_exact_segment_artifacts(
                context.request.artifact_directory, result, tuple(summaries), task, self.config,
                self.maximum_pulses, capacity_records, beam_budget, record_bytes,
                state.working_record_bytes(), parallel_diagnostics=None,
            )
        return result

    @staticmethod
    def _validate(
        task: StaticTaskTensors,
        anchors: MatchingTailBank,
        library: ComparisonROILibrary,
    ) -> None:
        """Reject incompatible stored geometry or undefined repair objectives."""
        if anchors.task_fingerprint != task.fingerprint:
            raise ValueError("MPPI anchors and current task fingerprints disagree")
        if library.task_fingerprint != task.fingerprint:
            raise ValueError("MPPI comparison library and current task fingerprints disagree")
        if task.initial_healthy_voxels <= 0:
            raise ValueError("MPPI healthy-overcut cost requires initial healthy tissue")
