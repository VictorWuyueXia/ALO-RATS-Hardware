"""Exact segment shooting with compact boundaries and exact suffix verification."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np

from laser_ablation.planning.jax_bank.contracts import (
    ACTION_DIMENSION,
    StaticTaskTensors,
)
from laser_ablation.planning.jax_bank.segment_beam import SegmentBeam
from laser_ablation.planning.jax_bank.terminal_rollout import rollout_terminal_in_batches


@dataclass(frozen=True)
class ExactSegmentBeam:
    """Action parents with exact SDFs at their next unsearched segment boundary."""

    beam: SegmentBeam
    boundary_sdf: np.ndarray
    boundary_step: np.ndarray
    mppi_cost: np.ndarray
    provisional_score: np.ndarray
    sample: np.ndarray
    admissible: np.ndarray
    iteration: int

    def __post_init__(self) -> None:
        count = self.beam.count
        sdf = np.asarray(self.boundary_sdf, np.float32)
        steps = np.asarray(self.boundary_step, np.int32)
        costs = np.asarray(self.mppi_cost, np.float32)
        scores = np.asarray(self.provisional_score, np.float32)
        samples = np.asarray(self.sample, np.int32)
        admissible = np.asarray(self.admissible, bool)
        if sdf.ndim != 4 or sdf.shape[0] != count:
            raise ValueError("exact segment boundaries must have shape (row, x, y, z)")
        if any(values.shape != (count,) for values in (steps, costs, scores, samples, admissible)):
            raise ValueError("exact segment scalar arrays must contain one value per parent")
        if np.any(steps < 0) or np.any(steps > self.beam.maximum_pulses):
            raise ValueError("exact segment boundary steps are outside the padded action horizon")
        if not np.all(np.isfinite(sdf)) or not np.all(np.isfinite(costs)) or not np.all(np.isfinite(scores)):
            raise ValueError("exact segment boundaries and ranking values must be finite")
        for values in (sdf, steps, costs, scores, samples, admissible):
            values.setflags(write=False)
        object.__setattr__(self, "boundary_sdf", sdf)
        object.__setattr__(self, "boundary_step", steps)
        object.__setattr__(self, "mppi_cost", costs)
        object.__setattr__(self, "provisional_score", scores)
        object.__setattr__(self, "sample", samples)
        object.__setattr__(self, "admissible", admissible)

    def take(self, rows: np.ndarray) -> "ExactSegmentBeam":
        """Retain a deterministic aligned subset without reconstructing any prefix trace."""
        indices = np.asarray(rows, np.int32)
        return ExactSegmentBeam(
            self.beam.take(indices), self.boundary_sdf[indices].copy(),
            self.boundary_step[indices].copy(), self.mppi_cost[indices].copy(),
            self.provisional_score[indices].copy(), self.sample[indices].copy(),
            self.admissible[indices].copy(), self.iteration,
        )

    def working_record_bytes(self) -> int:
        """Count action provenance, one exact boundary SDF, and compact scalar ranking state."""
        action_bytes = self.beam.record_bytes((1, 1, 1)) - (self.beam.maximum_pulses + 1) * 4
        return int(action_bytes + np.prod(self.boundary_sdf.shape[1:]) * 4 + 21)


def stream_exact_segment(
    executor: object,
    seed: int,
    stage: int,
    state: ExactSegmentBeam,
    starts: np.ndarray,
    stops: np.ndarray,
    task: StaticTaskTensors,
    capacity_records: int,
    microbatch_size: int,
) -> tuple[ExactSegmentBeam | None, dict[str, object]]:
    """Shoot every new segment exactly, then exact-verify only each parent's top fifth."""
    beam = state.beam
    starts, stops = np.asarray(starts, np.int32), np.asarray(stops, np.int32)
    lengths = np.count_nonzero(beam.action_mask, axis=1).astype(np.int32)
    if capacity_records <= 0 or microbatch_size <= 0:
        raise ValueError("exact segment capacity and microbatch size must be positive")
    if starts.shape != (beam.count,) or stops.shape != (beam.count,):
        raise ValueError("exact segment bounds must contain one interval per parent")
    if np.any(starts != state.boundary_step) or np.any(stops <= starts) or np.any(stops > lengths):
        raise ValueError("exact segment parents must start at their cached boundary")

    samples_per_parent = executor.config.samples_per_anchor
    segment_width = max(
        executor.config.segment_minimum_pulses,
        (beam.maximum_pulses + executor.config.segment_maximum_count - 1)
        // executor.config.segment_maximum_count,
    )
    retained: list[tuple[tuple[float, float, str], tuple[object, ...]]] = []
    rejection_counts: dict[str, int] = {}
    sampling_seconds = segment_seconds = suffix_seconds = ranking_seconds = 0.0
    child_count = segment_feasible_count = shortlisted_count = hard_count = admissible_count = 0

    for parent in range(beam.count):
        pulse = np.arange(beam.maximum_pulses, dtype=np.int32)[None]
        perturb = (pulse >= starts[parent]) & (pulse < stops[parent]) & beam.action_mask[parent]
        sampled_at = perf_counter()
        samples, _ = executor.sample(
            seed, stage, (beam.candidate_ids[parent],), np.asarray((0,), np.int32),
            beam.actions[parent:parent + 1], beam.action_mask[parent:parent + 1],
            perturb, task,
        )
        sampling_seconds += perf_counter() - sampled_at
        actions = samples[0]
        sample_count = len(actions)
        width = int(stops[parent] - starts[parent])
        segment_actions = np.zeros((sample_count, segment_width, ACTION_DIMENSION), np.float32)
        segment_mask = np.zeros((sample_count, segment_width), bool)
        segment_actions[:, :width] = actions[:, starts[parent]:stops[parent]]
        segment_mask[:, :width] = True
        segment_at = perf_counter()
        segment = rollout_terminal_in_batches(
            task,
            np.broadcast_to(state.boundary_sdf[parent], (sample_count,) + task.shape).copy(),
            segment_actions, segment_mask, microbatch_size, executor.devices,
        )
        segment_seconds += perf_counter() - segment_at
        child_count += sample_count
        segment_feasible_count += int(np.count_nonzero(segment.constraint_feasible))
        for name, values in segment.flags.items():
            rejection_counts[f"segment:{name}"] = rejection_counts.get(f"segment:{name}", 0) + int(
                np.count_nonzero(values)
            )

        normalized = (actions - beam.origin_actions[parent]) / (task.upper_bounds - task.lower_bounds)
        deviation = np.sum(
            np.where(beam.action_mask[parent, :, None], normalized**2, 0.0), axis=(1, 2),
        ) / (5.0 * lengths[parent])
        local_cost = (
            executor.config.lambda_remaining * segment.remaining_fraction
            + executor.config.lambda_overcut * segment.healthy_overcut_fraction
            + executor.config.lambda_anchor * deviation
        )
        identifiers = []
        for sample in range(sample_count):
            digest = sha256()
            digest.update(beam.candidate_ids[parent].encode())
            digest.update(beam.source_ids[parent].encode())
            digest.update(np.asarray((stage, sample), np.int32).tobytes())
            digest.update(np.ascontiguousarray(actions[sample]).tobytes())
            digest.update(np.ascontiguousarray(beam.linearization_path[parent]).tobytes())
            identifiers.append(digest.hexdigest())
        feasible = np.flatnonzero(segment.constraint_feasible)
        shortlist_limit = max(1, int(np.ceil(executor.config.segment_retain_fraction * sample_count)))
        shortlist = sorted(
            feasible, key=lambda row: (float(local_cost[row]), identifiers[row]),
        )[:shortlist_limit]
        if not shortlist:
            continue
        shortlist = np.asarray(shortlist, np.int32)
        shortlisted_count += len(shortlist)
        suffix_actions = np.zeros((len(shortlist), beam.maximum_pulses, ACTION_DIMENSION), np.float32)
        suffix_mask = np.zeros((len(shortlist), beam.maximum_pulses), bool)
        suffix_length = int(lengths[parent] - stops[parent])
        if suffix_length:
            suffix_actions[:, :suffix_length] = actions[shortlist, stops[parent]:lengths[parent]]
            suffix_mask[:, :suffix_length] = True
        suffix_at = perf_counter()
        suffix = rollout_terminal_in_batches(
            task, segment.current_sdf[shortlist], suffix_actions, suffix_mask,
            microbatch_size, executor.devices,
        )
        suffix_seconds += perf_counter() - suffix_at
        for name, values in suffix.flags.items():
            rejection_counts[f"suffix:{name}"] = rejection_counts.get(f"suffix:{name}", 0) + int(
                np.count_nonzero(values)
            )
        combined_hard = suffix.constraint_feasible
        hard_count += int(np.count_nonzero(combined_hard))
        final_accepted = combined_hard.copy()
        admissible_count += int(np.count_nonzero(final_accepted))
        final_cost = (
            executor.config.lambda_remaining * suffix.remaining_fraction
            + executor.config.lambda_overcut * suffix.healthy_overcut_fraction
            + executor.config.lambda_anchor * deviation[shortlist]
        )
        ranked_at = perf_counter()
        for short_row in np.flatnonzero(combined_hard):
            sample = int(shortlist[short_row])
            clearance = min(float(segment.clearance_mm[sample]), float(suffix.clearance_mm[short_row]))
            penalty = 0.0 if clearance == np.inf else 1.0 - np.clip(
                (clearance - task.hard_margin_mm) / (task.truncation_mm - task.hard_margin_mm), 0.0, 1.0,
            )
            score = float(
                suffix.remaining_fraction[short_row] + 0.90 * suffix.overcut_fraction[short_row]
                + 0.05 * penalty + 0.02 * lengths[parent] / beam.maximum_pulses
            )
            identifier = identifiers[sample]
            values = (
                actions[sample].copy(), beam.origin_actions[parent].copy(),
                beam.action_mask[parent].copy(), beam.linearization_path[parent].copy(),
                beam.source_ids[parent], beam.anchor_ids[parent], identifier,
                np.int32(beam.anchor_slots[parent]), segment.current_sdf[sample].copy(),
                np.int32(stops[parent]), np.int32(sample), bool(final_accepted[short_row]),
                float(final_cost[short_row]), score,
            )
            retained.append(((float(final_cost[short_row]), score, identifier), values))
        retained = sorted(retained, key=lambda item: item[0])[:capacity_records]
        ranking_seconds += perf_counter() - ranked_at

    stats: dict[str, object] = {
        "parent_count": beam.count, "child_count": child_count,
        "segment_hard_feasible_count": segment_feasible_count,
        "shortlisted_count": shortlisted_count, "suffix_verified_count": shortlisted_count,
        "hard_feasible_count": hard_count, "admissible_count": admissible_count,
        "retained_count": len(retained), "capacity_records": capacity_records,
        "rejection_counts": rejection_counts, "support_rejections": 0,
        "contact_guard_rejections": 0, "linearization_trust_rejections": 0,
        "sampling_seconds": sampling_seconds,
        "segment_rollout_seconds": segment_seconds, "suffix_rollout_seconds": suffix_seconds,
        "rollout_seconds": segment_seconds + suffix_seconds, "ranking_seconds": ranking_seconds,
        "full_trace_rows_fetched": 0,
    }
    if not retained:
        return None, stats
    values = tuple(item[1] for item in retained)
    next_beam = SegmentBeam(
        np.stack([item[0] for item in values]), np.stack([item[1] for item in values]),
        np.stack([item[2] for item in values]), np.stack([item[3] for item in values]),
        tuple(item[4] for item in values), tuple(item[5] for item in values),
        tuple(item[6] for item in values), np.asarray([item[7] for item in values], np.int32),
    )
    return ExactSegmentBeam(
        next_beam, np.stack([item[8] for item in values]), np.asarray([item[9] for item in values]),
        np.asarray([item[12] for item in values]), np.asarray([item[13] for item in values]),
        np.asarray([item[10] for item in values]), np.asarray([item[11] for item in values]), stage,
    ), stats
