"""Sequential exact segment MPPI with one weighted child per parent."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np

from laser_ablation.planning.jax_bank.contracts import ACTION_DIMENSION, StaticTaskTensors
from laser_ablation.planning.jax_bank.exact_segment import ExactSegmentBeam
from laser_ablation.planning.jax_bank.segment_beam import SegmentBeam
from laser_ablation.planning.jax_bank.terminal_rollout import rollout_terminal_in_batches


def stream_path_integral_segment(
    executor: object,
    seed: int,
    stage: int,
    state: ExactSegmentBeam,
    starts: np.ndarray,
    stops: np.ndarray,
    task: StaticTaskTensors,
    microbatch_size: int,
) -> tuple[ExactSegmentBeam | None, dict[str, object]]:
    """Exact-score every feasible segment suffix and verify its weighted control mean."""
    beam = state.beam
    starts, stops = np.asarray(starts, np.int32), np.asarray(stops, np.int32)
    lengths = np.count_nonzero(beam.action_mask, axis=1).astype(np.int32)
    if microbatch_size <= 0:
        raise ValueError("path-integral segment microbatch size must be positive")
    if starts.shape != (beam.count,) or stops.shape != (beam.count,):
        raise ValueError("path-integral segment bounds must contain one interval per parent")
    if np.any(starts != state.boundary_step) or np.any(stops <= starts) or np.any(stops > lengths):
        raise ValueError("path-integral segments must begin at each cached boundary")

    sample_count = executor.config.samples_per_anchor
    width = max(
        executor.config.segment_minimum_pulses,
        (beam.maximum_pulses + executor.config.segment_maximum_count - 1)
        // executor.config.segment_maximum_count,
    )
    pulse = np.arange(beam.maximum_pulses, dtype=np.int32)[None]
    perturb_mask = (pulse >= starts[:, None]) & (pulse < stops[:, None]) & beam.action_mask
    sampled_at = perf_counter()
    samples, perturbations = executor.sample(
        seed, stage, beam.candidate_ids, np.arange(beam.count, dtype=np.int32),
        beam.actions, beam.action_mask, perturb_mask, task,
    )
    sampling_seconds = perf_counter() - sampled_at
    flat_samples = np.asarray(samples, np.float32).reshape(-1, beam.maximum_pulses, ACTION_DIMENSION)
    parent_rows = np.repeat(np.arange(beam.count, dtype=np.int32), sample_count)
    segment_actions = np.zeros((len(flat_samples), width, ACTION_DIMENSION), np.float32)
    segment_mask = np.zeros((len(flat_samples), width), bool)
    for parent in range(beam.count):
        rows = slice(parent * sample_count, (parent + 1) * sample_count)
        segment_width = int(stops[parent] - starts[parent])
        segment_actions[rows, :segment_width] = flat_samples[rows, starts[parent]:stops[parent]]
        segment_mask[rows, :segment_width] = True
    segment_at = perf_counter()
    segment = rollout_terminal_in_batches(
        task, state.boundary_sdf, segment_actions, segment_mask, microbatch_size,
        executor.devices, initial_rows=parent_rows,
    )
    segment_seconds = perf_counter() - segment_at
    segment_feasible = np.flatnonzero(segment.constraint_feasible)
    rejection_counts: dict[str, int] = {
        f"segment:{name}": int(np.count_nonzero(values)) for name, values in segment.flags.items()
    }
    if not len(segment_feasible):
        return None, _stats(
            beam.count, len(flat_samples), 0, 0, 0, 0, rejection_counts,
            sampling_seconds, segment_seconds, 0.0, 0.0, 0.0, (), (),
        )

    suffix_actions = np.zeros((len(segment_feasible), beam.maximum_pulses, ACTION_DIMENSION), np.float32)
    suffix_mask = np.zeros((len(segment_feasible), beam.maximum_pulses), bool)
    for row, child in enumerate(segment_feasible):
        parent = int(parent_rows[child])
        suffix_length = int(lengths[parent] - stops[parent])
        if suffix_length:
            suffix_actions[row, :suffix_length] = flat_samples[child, stops[parent]:lengths[parent]]
            suffix_mask[row, :suffix_length] = True
    suffix_at = perf_counter()
    suffix = rollout_terminal_in_batches(
        task, segment.current_sdf[segment_feasible], suffix_actions, suffix_mask,
        microbatch_size, executor.devices,
    )
    suffix_seconds = perf_counter() - suffix_at
    for name, values in suffix.flags.items():
        rejection_counts[f"suffix:{name}"] = int(np.count_nonzero(values))
    hard_children = segment_feasible[suffix.constraint_feasible]
    feasible = np.zeros((beam.count, sample_count), bool)
    remaining = np.zeros((beam.count, sample_count), np.float32)
    healthy = np.zeros((beam.count, sample_count), np.float32)
    feasible.reshape(-1)[hard_children] = True
    remaining.reshape(-1)[segment_feasible] = suffix.remaining_fraction
    healthy.reshape(-1)[segment_feasible] = suffix.healthy_overcut_fraction
    ranked_at = perf_counter()
    costs, _ = executor.cost_compiled(
        samples, beam.origin_actions, beam.action_mask, remaining, healthy,
        task.lower_bounds, task.upper_bounds,
        (executor.config.lambda_remaining, executor.config.lambda_overcut, executor.config.lambda_anchor),
    )
    weights = np.asarray(executor.weight_compiled(
        costs, feasible, temperature=executor.config.temperature,
    ), np.float32)
    updated = np.asarray(executor.update(
        beam.actions, perturbations, weights, beam.action_mask, task,
    ), np.float32)
    surviving = np.flatnonzero(np.sum(weights, axis=1) > 0.0)
    squared_weights = np.sum(weights * weights, axis=1)
    parent_ess = np.zeros_like(squared_weights)
    np.divide(1.0, squared_weights, out=parent_ess, where=squared_weights > 0.0)
    parent_entropy = -np.sum(
        weights * np.log(np.maximum(weights, np.finfo(np.float32).tiny)), axis=1,
    )
    ranking_seconds = perf_counter() - ranked_at
    if not len(surviving):
        return None, _stats(
            beam.count, len(flat_samples), len(segment_feasible), len(segment_feasible), 0, 0,
            rejection_counts, sampling_seconds, segment_seconds, suffix_seconds, 0.0, ranking_seconds,
            
            parent_ess, parent_entropy,
        )

    weighted_segment_actions = np.zeros((len(surviving), width, ACTION_DIMENSION), np.float32)
    weighted_segment_mask = np.zeros((len(surviving), width), bool)
    for row, parent in enumerate(surviving):
        segment_width = int(stops[parent] - starts[parent])
        weighted_segment_actions[row, :segment_width] = updated[parent, starts[parent]:stops[parent]]
        weighted_segment_mask[row, :segment_width] = True
    verification_at = perf_counter()
    weighted_segment = rollout_terminal_in_batches(
        task, state.boundary_sdf[surviving], weighted_segment_actions,
        weighted_segment_mask, microbatch_size, executor.devices,
    )
    weighted_suffix_actions = np.zeros((len(surviving), beam.maximum_pulses, ACTION_DIMENSION), np.float32)
    weighted_suffix_mask = np.zeros((len(surviving), beam.maximum_pulses), bool)
    for row, parent in enumerate(surviving):
        suffix_length = int(lengths[parent] - stops[parent])
        if suffix_length:
            weighted_suffix_actions[row, :suffix_length] = updated[parent, stops[parent]:lengths[parent]]
            weighted_suffix_mask[row, :suffix_length] = True
    weighted_suffix = rollout_terminal_in_batches(
        task, weighted_segment.current_sdf, weighted_suffix_actions,
        weighted_suffix_mask, microbatch_size, executor.devices,
    )
    verification_seconds = perf_counter() - verification_at
    for name, values in weighted_segment.flags.items():
        rejection_counts[f"weighted_segment:{name}"] = int(np.count_nonzero(values))
    for name, values in weighted_suffix.flags.items():
        rejection_counts[f"weighted_suffix:{name}"] = int(np.count_nonzero(values))
    weighted_hard = weighted_segment.constraint_feasible & weighted_suffix.constraint_feasible
    retained = surviving[weighted_hard]
    accepted = weighted_hard & weighted_suffix.accepted
    if not len(retained):
        return None, _stats(
            beam.count, len(flat_samples), len(segment_feasible), len(segment_feasible), 0,
            int(np.count_nonzero(accepted)), rejection_counts, sampling_seconds, segment_seconds,
            suffix_seconds, verification_seconds, ranking_seconds, parent_ess, parent_entropy,
        )
    final_remaining = np.zeros((beam.count, 1), np.float32)
    final_healthy = np.zeros((beam.count, 1), np.float32)
    final_remaining[surviving, 0] = weighted_suffix.remaining_fraction
    final_healthy[surviving, 0] = weighted_suffix.healthy_overcut_fraction
    weighted_cost, _ = executor.cost_compiled(
        updated[:, None], beam.origin_actions, beam.action_mask,
        final_remaining, final_healthy, task.lower_bounds, task.upper_bounds,
        (executor.config.lambda_remaining, executor.config.lambda_overcut, executor.config.lambda_anchor),
    )
    weighted_cost = np.asarray(weighted_cost[:, 0], np.float32)
    scores = np.zeros(len(retained), np.float32)
    identifiers: list[str] = []
    for row, parent in enumerate(retained):
        survivor_row = int(np.flatnonzero(surviving == parent)[0])
        clearance = min(
            float(weighted_segment.clearance_mm[survivor_row]),
            float(weighted_suffix.clearance_mm[survivor_row]),
        )
        penalty = 0.0 if clearance == np.inf else 1.0 - np.clip(
            (clearance - task.hard_margin_mm) / (task.truncation_mm - task.hard_margin_mm), 0.0, 1.0,
        )
        scores[row] = (
            weighted_suffix.remaining_fraction[survivor_row]
            + 0.90 * weighted_suffix.overcut_fraction[survivor_row]
            + 0.05 * penalty + 0.02 * lengths[parent] / beam.maximum_pulses
        )
        digest = sha256()
        digest.update(beam.candidate_ids[parent].encode())
        digest.update(beam.source_ids[parent].encode())
        digest.update(np.asarray((stage,), np.int32).tobytes())
        digest.update(np.ascontiguousarray(updated[parent]).tobytes())
        digest.update(np.ascontiguousarray(beam.linearization_path[parent]).tobytes())
        identifiers.append(digest.hexdigest())
    next_beam = SegmentBeam(
        updated[retained], beam.origin_actions[retained], beam.action_mask[retained],
        beam.linearization_path[retained], tuple(beam.source_ids[row] for row in retained),
        tuple(beam.anchor_ids[row] for row in retained), tuple(identifiers), beam.anchor_slots[retained],
    )
    retained_positions = np.asarray([np.flatnonzero(surviving == parent)[0] for parent in retained], np.int32)
    next_state = ExactSegmentBeam(
        next_beam, weighted_segment.current_sdf[retained_positions], stops[retained],
        weighted_cost[retained], scores, np.full(len(retained), -1, np.int32),
        accepted[retained_positions], stage,
    )
    stats = _stats(
        beam.count, len(flat_samples), len(segment_feasible), len(segment_feasible), len(retained),
        int(np.count_nonzero(accepted)), rejection_counts, sampling_seconds, segment_seconds,
        suffix_seconds, verification_seconds, ranking_seconds, parent_ess, parent_entropy,
    )
    stats["retained_parent_indices"] = retained.copy()
    return next_state, stats


def _stats(
    parents: int, children: int, segment_feasible: int, suffix_verified: int, retained: int,
    accepted: int, rejection_counts: dict[str, int], sampling: float, segment: float,
    suffix: float, verification: float, ranking: float,
    ess: np.ndarray | tuple[()], entropy: np.ndarray | tuple[()],
) -> dict[str, object]:
    """Return the single compact evidence record shared by repair and artifacts."""
    positive_ess = np.asarray(ess, np.float32)[np.asarray(ess) > 0.0]
    positive_entropy = np.asarray(entropy, np.float32)[np.asarray(ess) > 0.0]
    return {
        "parent_count": parents, "child_count": children,
        "segment_hard_feasible_count": segment_feasible,
        "shortlisted_count": suffix_verified, "suffix_verified_count": suffix_verified,
        "hard_feasible_count": retained, "admissible_count": accepted,
        "retained_count": retained, "capacity_records": parents,
        "rejection_counts": rejection_counts, "support_rejections": 0,
        "contact_guard_rejections": 0, "linearization_trust_rejections": 0,
        "sampling_seconds": sampling, "segment_rollout_seconds": segment,
        "suffix_rollout_seconds": suffix, "weighted_verification_seconds": verification,
        "rollout_seconds": segment + suffix, "ranking_seconds": ranking,
        "full_trace_rows_fetched": 0, "weighted_parent_count": int(len(positive_ess)),
        "weighted_feasible_child_count": suffix_verified,
        "weight_ess_min": float(np.min(positive_ess)) if len(positive_ess) else 0.0,
        "weight_ess_median": float(np.median(positive_ess)) if len(positive_ess) else 0.0,
        "weight_entropy_median": float(np.median(positive_entropy)) if len(positive_entropy) else 0.0,
    }
