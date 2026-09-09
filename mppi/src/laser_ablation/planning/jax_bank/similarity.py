"""Fingerprint-gated SDF comparison and non-averaged routed-tail extraction."""

from __future__ import annotations

import numpy as np

from laser_ablation.core.sdf import SDFState
from laser_ablation.planning.jax_bank.bank import PlanBank
from laser_ablation.planning.jax_bank.comparison_library import ComparisonROILibrary
from laser_ablation.planning.jax_bank.contracts import (
    MatchingTail,
    MatchingTailBank,
    StaticTaskTensors,
    linearization_path_hash,
    trajectory_hash,
)


MATCHING_BATCH_ROWS = 256
_ROUTED_ROI_KERNEL: object | None = None


def _jax_modules():
    """Import JAX only for the explicitly selected state-matching operation."""
    import jax
    import jax.numpy as jnp

    return jax, jnp


def reduced_geometry_state(current: np.ndarray) -> np.ndarray:
    """Return the edge-padded 2x2x2 anti-aliased current-geometry descriptor."""
    values = np.asarray(current, dtype=np.float32)
    if values.ndim < 3 or any(length <= 0 for length in values.shape[-3:]):
        raise ValueError("geometry state must end in a nonempty (x, y, z) SDF lattice")
    padding = [(0, 0)] * (values.ndim - 3) + [
        (0, values.shape[-3] % 2),
        (0, values.shape[-2] % 2),
        (0, values.shape[-1] % 2),
    ]
    padded = np.pad(values, padding, mode="edge")
    shape = padded.shape[:-3] + (
        padded.shape[-3] // 2, 2,
        padded.shape[-2] // 2, 2,
        padded.shape[-1] // 2, 2,
    )
    return np.asarray(
        padded.reshape(shape).mean(axis=(-5, -3, -1), dtype=np.float32), dtype=np.float32
    )


def dynamic_similarity(current: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """Compare normalized current-tissue SDF geometry by cosine similarity."""
    current_values = np.asarray(current, dtype=np.float32)
    stored_values = np.asarray(stored, dtype=np.float32)
    if current_values.ndim != 3 or stored_values.shape[1:] != current_values.shape:
        raise ValueError("SDF states must have shapes (x, y, z) and (batch, x, y, z)")
    if not np.all(np.isfinite(current_values)) or not np.all(np.isfinite(stored_values)):
        raise ValueError("state comparison requires finite dynamic channels")
    if not np.any(current_values) or np.any(
        np.all(stored_values == 0.0, axis=(1, 2, 3))
    ):
        raise ValueError("state comparison requires nonzero SDF norms")
    jax, jnp = _jax_modules()

    @jax.jit
    def compare(reference: object, candidates: object) -> object:
        reference = jnp.clip(reference, -1.0, 1.0)
        candidates = jnp.clip(candidates, -1.0, 1.0)
        numerator = jnp.sum(candidates * reference[None, ...], axis=(1, 2, 3))
        reference_norm = jnp.sum(reference * reference)
        candidate_norm = jnp.sum(candidates * candidates, axis=(1, 2, 3))
        return numerator / (jnp.sqrt(reference_norm * candidate_norm) + jnp.float32(1e-8))

    values = compare(jnp.asarray(current_values), jnp.asarray(stored_values))
    return np.asarray(values)


def roi_cosine_similarity(
    left_roi: object, right_roi: object, roi_mask: object,
) -> tuple[object, object]:
    """Return masked ROI cosine values and finite nonzero-norm validity flags."""
    _, jnp = _jax_modules()
    left = jnp.asarray(left_roi)
    right = jnp.asarray(right_roi)
    mask = jnp.asarray(roi_mask, dtype=jnp.bool_)
    left, right, mask = jnp.broadcast_arrays(left, right, mask)
    selected_finite = jnp.where(mask, jnp.isfinite(left) & jnp.isfinite(right), True)
    left_values = jnp.where(mask, left, jnp.zeros((), dtype=left.dtype))
    right_values = jnp.where(mask, right, jnp.zeros((), dtype=right.dtype))
    numerator = jnp.sum(left_values * right_values, axis=-1)
    left_norm = jnp.sum(left_values * left_values, axis=-1)
    right_norm = jnp.sum(right_values * right_values, axis=-1)
    valid_norm = (
        jnp.all(selected_finite, axis=-1)
        & (left_norm > jnp.asarray(0.0, dtype=left_norm.dtype))
        & (right_norm > jnp.asarray(0.0, dtype=right_norm.dtype))
    )
    cosine = jnp.where(
        valid_norm,
        numerator / jnp.sqrt(left_norm * right_norm),
        jnp.zeros((), dtype=numerator.dtype),
    )
    return cosine, valid_norm


def _routed_roi_kernel() -> object:
    """Compile the one fixed-row routed gather and masked cosine reduction."""
    global _ROUTED_ROI_KERNEL
    if _ROUTED_ROI_KERNEL is None:
        jax, jnp = _jax_modules()

        @jax.jit
        def compare(
            left_flat: object, right_flat: object, indices: object, roi_mask: object,
        ) -> tuple[object, object]:
            left_values = left_flat[indices]
            right_values = jnp.take_along_axis(right_flat, indices, axis=1)
            return roi_cosine_similarity(left_values, right_values, roi_mask)

        _ROUTED_ROI_KERNEL = compare
    return _ROUTED_ROI_KERNEL


def _routed_roi_values(
    left_sdf: np.ndarray,
    right_sdf: np.ndarray,
    routes: np.ndarray,
    library: ComparisonROILibrary,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather route-local SDF rows before the one shared cosine reduction."""
    left = np.asarray(left_sdf, dtype=np.float32)
    right = np.asarray(right_sdf, dtype=np.float32)
    path = np.asarray(routes, dtype=np.int32)
    if left.shape != library.state_shape or right.ndim != 4 or right.shape[1:] != left.shape:
        raise ValueError("routed ROI comparison requires matching current-SDF lattice shapes")
    if path.shape != (len(right), 2):
        raise ValueError("routed ROI comparison requires one [seed, step] route per state")
    seeds, steps = path[:, 0], path[:, 1]
    if (
        np.any(seeds < 0) or np.any(seeds >= library.seed_count)
        or np.any(steps < 0) or np.any(steps >= library.maximum_pulses)
    ):
        raise ValueError("routed ROI comparison received an invalid comparison route")
    if np.any(~library.step_mask[seeds, steps]):
        raise ValueError("routed ROI comparison received a masked comparison route")
    if not 0 < len(right) <= MATCHING_BATCH_ROWS:
        raise ValueError("routed ROI comparison requires one nonempty fixed-size batch")
    indices = library.roi_indices[seeds, steps]
    mask = library.roi_mask[seeds, steps]
    count = len(right)
    if count < MATCHING_BATCH_ROWS:
        padding = MATCHING_BATCH_ROWS - count
        right = np.concatenate((right, np.repeat(right[:1], padding, axis=0)))
        indices = np.concatenate((indices, np.repeat(indices[:1], padding, axis=0)))
        mask = np.concatenate((mask, np.repeat(mask[:1], padding, axis=0)))
    _, jnp = _jax_modules()
    cosine, valid_norm = _routed_roi_kernel()(
        jnp.asarray(left.reshape(-1)), jnp.asarray(right.reshape(MATCHING_BATCH_ROWS, -1)),
        jnp.asarray(indices), jnp.asarray(mask),
    )
    return np.asarray(cosine[:count], dtype=np.float32), np.asarray(valid_norm[:count], dtype=bool)


def matching_tail_bank(
    bank: PlanBank,
    current_task: StaticTaskTensors,
    current_sdf_state: SDFState,
    library: ComparisonROILibrary,
    minimum_similarity: float,
    active_trajectory_id: str | None = None,
    active_prefix: int | None = None,
) -> MatchingTailBank:
    """Return every ROI-compatible nonterminal tail without blending trajectories."""
    if bank.task.fingerprint != current_task.fingerprint:
        raise ValueError("state comparison requires an identical static task fingerprint")
    if library.task_fingerprint != current_task.fingerprint:
        raise ValueError("state comparison requires a library for the identical static task")
    if not -1.0 <= minimum_similarity <= 1.0:
        raise ValueError("ROI match cosine threshold must lie in [-1, 1]")
    if (active_trajectory_id is None) != (active_prefix is None):
        raise ValueError("active trajectory and prefix must be supplied together")
    if active_trajectory_id is not None:
        active = bank.trajectory(active_trajectory_id)
        if active_prefix is None or not 0 <= active_prefix < len(active.actions):
            raise ValueError("active prefix must select a remaining active action")
    current = np.asarray(current_sdf_state.tensor[SDFState.CURRENT_TISSUE], dtype=np.float32)
    candidates: list[tuple[object, int, int]] = []
    for trajectory in bank.trajectories:
        path = np.asarray(trajectory.linearization_path, dtype=np.int32)
        seeds, steps = path[:, 0], path[:, 1]
        usable = (
            (seeds >= 0) & (seeds < library.seed_count)
            & (steps >= 0) & (steps < library.maximum_pulses)
        )
        safe_seed = np.clip(seeds, 0, library.seed_count - 1)
        safe_step = np.clip(steps, 0, library.maximum_pulses - 1)
        usable &= library.step_mask[safe_seed, safe_step]
        prefixes = (
            (int(active_prefix),)
            if trajectory.trajectory_id == active_trajectory_id else range(len(trajectory.actions))
        )
        for prefix in prefixes:
            if not usable[prefix]:
                continue
            invalid = np.flatnonzero(~usable[prefix:])
            stop = prefix + (int(invalid[0]) if len(invalid) else len(usable) - prefix)
            candidates.append((trajectory, prefix, stop))
    selected: dict[str, tuple[object, int, int, float]] = {}
    for start in range(0, len(candidates), MATCHING_BATCH_ROWS):
        batch = candidates[start:start + MATCHING_BATCH_ROWS]
        padded = batch + [batch[0]] * (MATCHING_BATCH_ROWS - len(batch))
        states = np.stack(
            [trajectory.current_sdf[prefix] for trajectory, prefix, _ in padded]
        ).astype(np.float32, copy=False)
        routes = np.stack(
            [trajectory.linearization_path[prefix] for trajectory, prefix, _ in padded]
        ).astype(np.int32, copy=False)
        values, valid_norm = _routed_roi_values(current, states, routes, library)
        values, valid_norm = values[:len(batch)], valid_norm[:len(batch)]
        if not np.all(valid_norm):
            raise ValueError("ROI state comparison requires finite nonzero selected geometry norms")
        for (trajectory, prefix, stop), similarity in zip(batch, values, strict=True):
            if similarity < minimum_similarity:
                continue
            previous = selected.get(trajectory.trajectory_id)
            if previous is None or prefix > previous[1]:
                selected[trajectory.trajectory_id] = (trajectory, prefix, stop, float(similarity))
    tails: list[MatchingTail] = []
    for trajectory, prefix, stop, similarity in selected.values():
        tail_actions = trajectory.actions[prefix:stop]
        tail_path = trajectory.linearization_path[prefix:stop]
        identifier = trajectory_hash(
            tail_actions, trajectory.current_sdf[prefix], tail_path, trajectory.source_id
        )
        tails.append(
            MatchingTail(
                tail_id=identifier,
                matched_state=bank.state_tensor(trajectory, prefix),
                actions=tail_actions.copy(),
                linearization_path=tail_path.copy(),
                source_id=trajectory.source_id,
                states=bank.state_tensors(trajectory, prefix)[:len(tail_actions) + 1],
                similarity=similarity,
                trajectory_ids=(trajectory.trajectory_id,),
                source_records=({
                    "trajectory_id": trajectory.trajectory_id,
                    "source_id": trajectory.source_id,
                    "linearization_path_hash": linearization_path_hash(tail_path),
                    "origins": trajectory.origins,
                    "prefix": prefix,
                    "similarity": similarity,
                },),
                score=trajectory.score,
                remaining_pulses=len(tail_actions),
                remaining_energy_j=float(np.sum(tail_actions[:, 4])),
            )
        )
    return MatchingTailBank(
        bank.task.fingerprint,
        tuple(sorted(tails, key=lambda tail: (-tail.similarity, tail.score, tail.tail_id))),
    )
