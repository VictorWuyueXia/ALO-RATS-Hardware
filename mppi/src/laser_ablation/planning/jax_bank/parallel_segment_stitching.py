"""Parallel independent segment MPPI followed by exact stitched-plan verification."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np

from laser_ablation.planning.jax_bank.contracts import ACTION_DIMENSION, PaddedActionBatch, StaticTaskTensors
from laser_ablation.planning.jax_bank.exact_segment import ExactSegmentBeam
from laser_ablation.planning.jax_bank.mppi_records import provisional_metrics_score
from laser_ablation.planning.jax_bank.repair import ParallelSegmentBatch
from laser_ablation.planning.jax_bank.rollout import rollout_in_batches
from laser_ablation.planning.jax_bank.segment_beam import SegmentBeam
from laser_ablation.planning.jax_bank.terminal_rollout import TerminalRolloutBatch, rollout_terminal_in_batches


def propose_parallel_stitched_segments(
    executor: object, seed: int, beam: SegmentBeam, task: StaticTaskTensors,
    microbatch_size: int,
) -> tuple[ExactSegmentBeam | None, dict[str, object]]:
    """Optimize all nominally centered segments together, stitch, and exact-verify."""
    if microbatch_size <= 0:
        raise ValueError("parallel segment microbatch size must be positive")
    lengths = np.count_nonzero(beam.action_mask, axis=1).astype(np.int32)
    row_data: list[tuple[int, int, int, int]] = []
    for stage in range(executor.config.segment_maximum_count):
        starts, stops, active = beam.segment_bounds(
            stage, executor.config.segment_minimum_pulses,
            executor.config.segment_maximum_count,
        )
        for parent in np.flatnonzero(active):
            row_data.append((int(parent), stage, int(starts[parent]), int(stops[parent])))
    parents, segments, starts, stops = np.asarray(row_data, np.int32).T
    nominal_seconds = 0.0
    if np.any(starts > 0):
        nominal_at = perf_counter()
        nominal = rollout_in_batches(
            task, PaddedActionBatch(beam.actions, beam.action_mask, beam.source_ids),
            microbatch_size, executor.devices,
        )
        boundaries = np.stack([
            nominal.current_sdf[parent, start]
            for parent, start in zip(parents, starts, strict=True)
        ])
        nominal_seconds = perf_counter() - nominal_at
    else:
        boundaries = np.broadcast_to(
            task.initial_current_sdf, (len(parents),) + task.shape,
        ).copy()
    segment_batch = ParallelSegmentBatch(
        beam.actions[parents], beam.action_mask[parents], boundaries,
        beam.linearization_path[parents], parents, segments, starts, stops,
        tuple(beam.source_ids[parent] for parent in parents),
        tuple(beam.candidate_ids[parent] for parent in parents),
    )
    stats: dict[str, object] = {
        "segment_batch": segment_batch, "child_count": 0,
        "segment_hard_feasible_count": 0, "suffix_verified_count": 0,
        "hard_feasible_count": 0, "admissible_count": 0, "retained_count": 0,
        "weighted_parent_count": 0, "weighted_feasible_child_count": 0,
        "weight_ess_min": 0.0, "weight_ess_median": 0.0,
        "weight_entropy_median": 0.0, "rejection_counts": {},
        "nominal_scan_seconds": nominal_seconds, "sampling_seconds": 0.0,
        "segment_rollout_seconds": 0.0, "suffix_rollout_seconds": 0.0,
        "weighted_verification_seconds": 0.0, "ranking_seconds": 0.0,
        "boundary_rows": (), "full_trace_rows_fetched": beam.count if np.any(starts > 0) else 0,
    }

    import jax
    import jax.numpy as jnp

    keys = np.stack([
        np.asarray(jax.random.fold_in(
            jax.random.fold_in(jax.random.PRNGKey(seed), int(stage)),
            int(beam.candidate_ids[parent][:8], 16) & 0x7FFFFFFF,
        ))
        for parent, stage in zip(parents, segments, strict=True)
    ])
    pulse = np.arange(beam.maximum_pulses, dtype=np.int32)[None]
    perturb_mask = (
        (pulse >= starts[:, None]) & (pulse < stops[:, None])
        & segment_batch.action_mask
    )
    sampled_at = perf_counter()
    samples, perturbations = executor.sample_compiled(
        jnp.asarray(keys), jnp.asarray(segment_batch.actions),
        jnp.asarray(segment_batch.action_mask), jnp.asarray(perturb_mask),
        jnp.asarray(task.lower_bounds), jnp.asarray(task.upper_bounds),
        samples_per_anchor=executor.config.samples_per_anchor,
        beta=executor.config.beta, kappa=jnp.asarray(executor.config.kappa),
    )
    samples, perturbations = np.asarray(samples), np.asarray(perturbations)
    sampling_seconds = perf_counter() - sampled_at
    sample_count = executor.config.samples_per_anchor
    pair_count = len(segment_batch.actions)
    child_count = pair_count * sample_count
    stats.update(child_count=child_count, sampling_seconds=sampling_seconds)
    flat_samples = samples.reshape(child_count, beam.maximum_pulses, ACTION_DIMENSION)
    child_pairs = np.repeat(np.arange(pair_count, dtype=np.int32), sample_count)
    segment_width = int(np.max(stops - starts))
    offsets = np.arange(segment_width, dtype=np.int32)[None]
    source_steps = starts[child_pairs, None] + offsets
    segment_mask = offsets < (stops - starts)[child_pairs, None]
    segment_actions = flat_samples[
        np.arange(child_count, dtype=np.int32)[:, None],
        np.minimum(source_steps, beam.maximum_pulses - 1),
    ]
    segment_actions = np.where(segment_mask[..., None], segment_actions, 0.0)
    segment_at = perf_counter()
    segment_rollout = rollout_terminal_in_batches(
        task, np.repeat(segment_batch.nominal_boundary_sdf, sample_count, axis=0),
        segment_actions, segment_mask, microbatch_size, executor.devices,
    )
    segment_seconds = perf_counter() - segment_at
    segment_feasible = np.flatnonzero(segment_rollout.constraint_feasible)
    rejection_counts = {
        f"segment:{name}": int(np.count_nonzero(values))
        for name, values in segment_rollout.flags.items()
    }
    stats.update(
        segment_hard_feasible_count=len(segment_feasible),
        rejection_counts=rejection_counts, segment_rollout_seconds=segment_seconds,
    )
    if not len(segment_feasible):
        return None, stats

    feasible_pairs = child_pairs[segment_feasible]
    suffix_width = int(np.max(lengths[parents] - stops))
    if suffix_width:
        suffix_offsets = np.arange(suffix_width, dtype=np.int32)[None]
        suffix_steps = stops[feasible_pairs, None] + suffix_offsets
        suffix_mask = suffix_offsets < (lengths[parents] - stops)[feasible_pairs, None]
        suffix_actions = flat_samples[
            segment_feasible[:, None], np.minimum(suffix_steps, beam.maximum_pulses - 1),
        ]
        suffix_actions = np.where(suffix_mask[..., None], suffix_actions, 0.0)
        suffix_at = perf_counter()
        suffix_rollout = rollout_terminal_in_batches(
            task, segment_rollout.current_sdf[segment_feasible], suffix_actions,
            suffix_mask, microbatch_size, executor.devices,
        )
        suffix_seconds = perf_counter() - suffix_at
    else:
        suffix_seconds = 0.0
        suffix_rollout = TerminalRolloutBatch(
            segment_rollout.current_sdf[segment_feasible],
            segment_rollout.remaining_fraction[segment_feasible],
            segment_rollout.overcut_fraction[segment_feasible],
            segment_rollout.healthy_overcut_fraction[segment_feasible],
            segment_rollout.clearance_mm[segment_feasible],
            segment_rollout.pulse_count[segment_feasible], segment_rollout.energy_j[segment_feasible],
            {name: values[segment_feasible] for name, values in segment_rollout.flags.items()},
            np.ones(len(segment_feasible), bool), segment_rollout.accepted[segment_feasible],
        )
    for name, values in suffix_rollout.flags.items():
        rejection_counts[f"suffix:{name}"] = int(np.count_nonzero(values))
    hard_children = segment_feasible[suffix_rollout.constraint_feasible]
    feasible = np.zeros((pair_count, sample_count), bool)
    remaining = np.zeros((pair_count, sample_count), np.float32)
    healthy = np.zeros((pair_count, sample_count), np.float32)
    feasible.reshape(-1)[hard_children] = True
    remaining.reshape(-1)[segment_feasible] = suffix_rollout.remaining_fraction
    healthy.reshape(-1)[segment_feasible] = suffix_rollout.healthy_overcut_fraction
    ranked_at = perf_counter()
    costs, _ = executor.cost_compiled(
        samples, beam.origin_actions[parents], segment_batch.action_mask,
        remaining, healthy, task.lower_bounds, task.upper_bounds,
        (executor.config.lambda_remaining, executor.config.lambda_overcut, executor.config.lambda_anchor),
    )
    weights = np.asarray(executor.weight_compiled(
        costs, feasible, temperature=executor.config.temperature,
    ), np.float32)
    updated = np.asarray(executor.update(
        segment_batch.actions, perturbations, weights,
        segment_batch.action_mask, task,
    ), np.float32)
    weighted_pairs = np.flatnonzero(np.sum(weights, axis=1) > 0.0)
    squared_weights = np.sum(weights * weights, axis=1)
    ess = np.where(squared_weights > 0.0, 1.0 / squared_weights, 0.0)
    entropy = -np.sum(
        weights * np.log(np.maximum(weights, np.finfo(np.float32).tiny)), axis=1,
    )
    ranking_seconds = perf_counter() - ranked_at
    stats.update(
        suffix_verified_count=len(segment_feasible),
        weighted_feasible_child_count=len(hard_children),
        suffix_rollout_seconds=suffix_seconds, ranking_seconds=ranking_seconds,
        weight_ess_min=float(np.min(ess[weighted_pairs])) if len(weighted_pairs) else 0.0,
        weight_ess_median=float(np.median(ess[weighted_pairs])) if len(weighted_pairs) else 0.0,
        weight_entropy_median=float(np.median(entropy[weighted_pairs])) if len(weighted_pairs) else 0.0,
    )
    if not len(weighted_pairs):
        return None, stats
    weighted_width = int(np.max(stops[weighted_pairs] - starts[weighted_pairs]))
    weighted_offsets = np.arange(weighted_width, dtype=np.int32)[None]
    weighted_steps = starts[weighted_pairs, None] + weighted_offsets
    weighted_mask = weighted_offsets < (stops - starts)[weighted_pairs, None]
    weighted_actions = updated[
        weighted_pairs[:, None], np.minimum(weighted_steps, beam.maximum_pulses - 1),
    ]
    weighted_actions = np.where(weighted_mask[..., None], weighted_actions, 0.0)
    verified_at = perf_counter()
    weighted_segment = rollout_terminal_in_batches(
        task, segment_batch.nominal_boundary_sdf[weighted_pairs], weighted_actions,
        weighted_mask, microbatch_size, executor.devices,
    )
    complete_parents = np.asarray([
        parent for parent in range(beam.count)
        if np.all(np.isin(np.flatnonzero(parents == parent), weighted_pairs))
    ], np.int32)
    stats["weighted_parent_count"] = len(complete_parents)
    if not len(complete_parents):
        stats["weighted_verification_seconds"] = perf_counter() - verified_at
        return None, stats
    stitched = beam.actions[complete_parents].copy()
    for pair in weighted_pairs:
        parent_positions = np.flatnonzero(complete_parents == parents[pair])
        if len(parent_positions):
            stitched[parent_positions[0], starts[pair]:stops[pair]] = updated[pair, starts[pair]:stops[pair]]
    stitched_rollout = rollout_in_batches(
        task,
        PaddedActionBatch(
            stitched, beam.action_mask[complete_parents],
            tuple(beam.source_ids[parent] for parent in complete_parents),
        ),
        microbatch_size, executor.devices,
    )
    verification_seconds = perf_counter() - verified_at
    for name, values in stitched_rollout.flags.items():
        rejection_counts[f"stitched:{name}"] = int(np.count_nonzero(values))
    accepted_positions = np.flatnonzero(stitched_rollout.accepted)
    retained_parents = complete_parents[accepted_positions]
    boundary_rows: list[dict[str, object]] = []
    weighted_lookup = {int(pair): row for row, pair in enumerate(weighted_pairs)}
    stitched_lookup = {int(parent): row for row, parent in enumerate(complete_parents)}
    for pair in weighted_pairs:
        parent = int(parents[pair])
        if parent not in stitched_lookup:
            continue
        independent = weighted_segment.current_sdf[weighted_lookup[int(pair)]]
        coupled = stitched_rollout.current_sdf[stitched_lookup[parent], stops[pair]]
        delta = coupled - independent
        boundary_rows.append({
            "parent_index": parent, "segment_index": int(segments[pair]),
            "start": int(starts[pair]), "stop": int(stops[pair]),
            "normalized_l2_error": float(
                np.linalg.norm(delta.ravel()) / max(np.linalg.norm(coupled.ravel()), 1.0e-12)
            ),
            "mean_absolute_sdf_error": float(np.mean(np.abs(delta))),
            "sdf_sign_disagreement": float(np.mean((coupled <= 0.0) != (independent <= 0.0))),
        })
    stats.update(
        hard_feasible_count=int(np.count_nonzero(stitched_rollout.constraint_feasible)),
        admissible_count=len(accepted_positions), retained_count=len(retained_parents),
        rejection_counts=rejection_counts, weighted_verification_seconds=verification_seconds,
        boundary_rows=tuple(boundary_rows), stitched_actions=stitched.copy(),
        complete_parent_indices=complete_parents.copy(),
        full_trace_rows_fetched=(beam.count if np.any(starts > 0) else 0) + len(complete_parents),
    )
    if not len(retained_parents):
        return None, stats
    final_remaining = stitched_rollout.remaining_fraction[accepted_positions, -1, None]
    final_healthy = stitched_rollout.healthy_overcut_fraction[accepted_positions, -1, None]
    final_cost, _ = executor.cost_compiled(
        stitched[accepted_positions, None], beam.origin_actions[retained_parents],
        beam.action_mask[retained_parents], final_remaining, final_healthy,
        task.lower_bounds, task.upper_bounds,
        (executor.config.lambda_remaining, executor.config.lambda_overcut, executor.config.lambda_anchor),
    )
    scores = np.asarray([
        provisional_metrics_score(
            task, stitched_rollout.remaining_fraction[row, -1],
            stitched_rollout.overcut_fraction[row, -1], stitched_rollout.clearance_mm[row, -1],
            stitched_rollout.pulse_count[row, -1], beam.maximum_pulses,
        )
        for row in accepted_positions
    ], np.float32)
    identifiers = []
    for row, parent in zip(accepted_positions, retained_parents, strict=True):
        digest = sha256()
        digest.update(beam.candidate_ids[parent].encode())
        digest.update(beam.source_ids[parent].encode())
        digest.update(b"parallel_segment_stitch")
        digest.update(np.ascontiguousarray(stitched[row]).tobytes())
        digest.update(np.ascontiguousarray(beam.linearization_path[parent]).tobytes())
        identifiers.append(digest.hexdigest())
    next_beam = SegmentBeam(
        stitched[accepted_positions], beam.origin_actions[retained_parents],
        beam.action_mask[retained_parents], beam.linearization_path[retained_parents],
        tuple(beam.source_ids[parent] for parent in retained_parents),
        tuple(beam.anchor_ids[parent] for parent in retained_parents), tuple(identifiers),
        beam.anchor_slots[retained_parents],
    )
    return ExactSegmentBeam(
        next_beam, stitched_rollout.current_sdf[accepted_positions, -1],
        lengths[retained_parents], np.asarray(final_cost[:, 0], np.float32), scores,
        np.full(len(retained_parents), -1, np.int32), np.ones(len(retained_parents), bool), int(np.max(segments)),
    ), stats
