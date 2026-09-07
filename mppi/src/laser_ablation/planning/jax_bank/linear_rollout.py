"""Batch-parallel sparse local-affine rollouts for a stored linearization library."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from laser_ablation.planning.jax_bank.contracts import (
    QuickRolloutBatch,
    StaticTaskTensors,
)
from laser_ablation.planning.jax_bank.jax_transition import FLAG_NAMES
from laser_ablation.planning.jax_bank.linear_contracts import (
    LinearizationLibrary,
    LinearizedActionBatch,
)
from laser_ablation.planning.jax_bank.rollout import screening_mask


LINEAR_FLAG_NAMES = FLAG_NAMES + (
    "irreversibility", "linearization_support", "linearization_trust",
)
_LINEAR_EXECUTORS: dict[tuple[str, str, int], Callable[..., object]] = {}


def _build_executor(
    task: StaticTaskTensors,
    library: LinearizationLibrary,
    devices: tuple[object, ...],
) -> Callable[..., object]:
    """Compile one sparse-affine executor for fixed task and library semantics."""
    import jax
    import jax.numpy as jnp

    nominal_states = jnp.asarray(library.nominal_states)
    nominal_actions = jnp.asarray(library.nominal_actions)
    step_mask = jnp.asarray(library.step_mask)
    roi_indices = jnp.asarray(library.roi_indices)
    roi_mask = jnp.asarray(library.roi_mask)
    state_gain = jnp.asarray(library.state_gain)
    action_jacobian = jnp.asarray(library.action_jacobian)
    support_lower = jnp.asarray(library.action_support_lower)
    support_upper = jnp.asarray(library.action_support_upper)
    initial_tissue = jnp.asarray(task.initial_tissue)
    target = jnp.asarray(task.target_mask)
    constraint = jnp.asarray(task.constraint_mask)
    clearance_field = jnp.asarray(task.physical_clearance_mm, dtype=jnp.float32)
    lower = jnp.asarray(task.lower_bounds, dtype=jnp.float32)
    upper = jnp.asarray(task.upper_bounds, dtype=jnp.float32)
    target_count = jnp.float32(task.initial_target_voxels)
    healthy_count = jnp.float32(task.initial_healthy_voxels)
    voxel_count = int(np.prod(task.shape))
    seed_count = library.seed_count
    maximum_pulses = library.maximum_pulses

    def one_plan(initial: object, actions: object, action_mask: object, path: object):
        """Apply a nominal-local transition selected independently for every action."""
        initial_remaining = jnp.count_nonzero((initial <= 0.0) & target)
        initial_overcut = jnp.count_nonzero(initial_tissue & (initial > 0.0) & ~target)

        def advance(carry: tuple[object, ...], inputs: tuple[object, object, object]):
            current, minimum_clearance, pulse_count, energy, completed = carry
            action, requested, route = inputs
            seed_raw, step_raw = route
            path_valid = (
                (seed_raw >= 0) & (seed_raw < seed_count)
                & (step_raw >= 0) & (step_raw < maximum_pulses)
            )
            seed = jnp.clip(seed_raw, 0, seed_count - 1)
            step = jnp.clip(step_raw, 0, maximum_pulses - 1)
            path_valid = path_valid & step_mask[seed, step]
            active = requested & ~completed
            nominal_current = nominal_states[seed, step]
            nominal_next = nominal_states[seed, step + 1]
            nominal_action = nominal_actions[seed, step]
            indices = roi_indices[seed, step]
            in_roi = roi_mask[seed, step]
            safe_indices = jnp.where(in_roi, indices, voxel_count)
            current_flat = current.reshape(-1)
            nominal_current_flat = nominal_current.reshape(-1)
            nominal_next_flat = nominal_next.reshape(-1)
            extended_current = jnp.concatenate((current_flat, jnp.zeros(1, jnp.float32)))
            current_roi = extended_current[safe_indices]
            nominal_current_roi = jnp.concatenate((
                nominal_current_flat, jnp.zeros(1, jnp.float32),
            ))[safe_indices]
            nominal_next_roi = jnp.concatenate((
                nominal_next_flat, jnp.zeros(1, jnp.float32),
            ))[safe_indices]
            delta_action = action - nominal_action
            local_values = (
                nominal_next_roi
                + state_gain[seed, step] * (current_roi - nominal_current_roi)
                + jnp.sum(action_jacobian[seed, step] * delta_action[None], axis=1)
            )
            candidate = extended_current.at[safe_indices].set(
                jnp.where(in_roi, local_values, extended_current[safe_indices])
            )
            membership = jnp.zeros(voxel_count + 1, dtype=jnp.bool_).at[safe_indices].set(in_roi)
            local_state = jnp.where(membership, candidate, extended_current)[:-1].reshape(task.shape)
            support_violation = (
                ~path_valid
                | jnp.any(action < support_lower[seed, step])
                | jnp.any(action > support_upper[seed, step])
            )
            apply = active & ~support_violation
            next_state = jnp.where(apply, local_state, current)
            was_tissue = current <= 0.0
            is_tissue = next_state <= 0.0
            removal = initial_tissue & was_tissue & ~is_tissue
            removed = initial_tissue & ~is_tissue
            has_change = jnp.any(next_state != current)
            newly_removed = jnp.any(removal)
            clearance = jnp.min(jnp.where(
                removal, clearance_field, jnp.asarray(jnp.inf, dtype=jnp.float32)
            ))
            flags = jnp.array((
                active & (jnp.any(action < lower) | jnp.any(action > upper)),
                active & ~has_change,
                active & ~newly_removed,
                active & jnp.any(removal & constraint),
                active & (clearance < task.hard_margin_mm),
                active & (~jnp.all(jnp.isfinite(next_state)) | ~jnp.all(jnp.isfinite(action))),
                active & jnp.any(initial_tissue & (current > 0.0) & (next_state <= 0.0)),
                active & support_violation,
                jnp.bool_(False),
            ))
            remaining = jnp.count_nonzero(is_tissue & target) / target_count
            overcut = jnp.count_nonzero(removed & ~target) / target_count
            next_carry = (
                next_state, jnp.minimum(minimum_clearance, clearance),
                pulse_count + active.astype(jnp.int32),
                energy + jnp.where(active, action[4], 0.0),
                completed | (remaining <= task.completion_remaining_fraction),
            )
            return next_carry, (
                next_state, flags, remaining, overcut, next_carry[1],
                next_carry[2], next_carry[3],
            )

        _, outputs = jax.lax.scan(
            advance,
            (
                initial, jnp.float32(jnp.inf), jnp.int32(0), jnp.float32(0.0),
                initial_remaining / target_count <= task.completion_remaining_fraction,
            ),
            (actions, action_mask, path),
        )
        states, flags, remaining, overcut, clearance, pulse_count, energy = outputs
        remaining_voxels = jnp.rint(remaining * target_count).astype(jnp.int32)
        overcut_voxels = jnp.rint(overcut * target_count).astype(jnp.int32)
        all_remaining = jnp.concatenate((initial_remaining[None], remaining_voxels))
        all_overcut = jnp.concatenate((initial_overcut[None], overcut_voxels))
        return (
            jnp.concatenate((initial[None], states)), all_remaining, all_overcut,
            all_remaining / target_count, all_overcut / target_count,
            jnp.where(healthy_count > 0.0, all_overcut / healthy_count, jnp.nan),
            jnp.concatenate((jnp.asarray([jnp.inf], jnp.float32), clearance)),
            jnp.concatenate((jnp.asarray([0], jnp.int32), pulse_count)),
            jnp.concatenate((jnp.asarray([0.0], jnp.float32), energy)), flags,
        )

    vectorized = jax.vmap(one_plan, in_axes=(0, 0, 0, 0))
    if len(devices) == 1:
        return jax.jit(vectorized)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(devices), ("plan",))
    return jax.jit(
        vectorized,
        in_shardings=(
            NamedSharding(mesh, PartitionSpec("plan", None, None, None)),
            NamedSharding(mesh, PartitionSpec("plan", None, None)),
            NamedSharding(mesh, PartitionSpec("plan", None)),
            NamedSharding(mesh, PartitionSpec("plan", None, None)),
        ),
    )


def rollout_linearized_batch(
    task: StaticTaskTensors,
    library: LinearizationLibrary,
    batch: LinearizedActionBatch,
    devices: tuple[object, ...],
) -> QuickRolloutBatch:
    """Roll a fixed library batch and treat unsupported actions as hard-infeasible."""
    if task.fingerprint != library.task_fingerprint:
        raise ValueError("linear rollout task and library fingerprints disagree")
    if not devices:
        raise ValueError("linear rollout requires explicit JAX devices")
    if batch.initial_current_sdf.shape[1:] != task.shape:
        raise ValueError("linear rollout initial SDF shape disagrees with the task")
    original_count = batch.actions.shape[0]
    padding = (-original_count) % len(devices)
    if padding:
        batch = LinearizedActionBatch(
            actions=np.concatenate((batch.actions, np.zeros((padding,) + batch.actions.shape[1:], np.float32))),
            action_mask=np.concatenate((batch.action_mask, np.zeros((padding,) + batch.action_mask.shape[1:], bool))),
            linearization_path=np.concatenate((
                batch.linearization_path,
                np.full((padding,) + batch.linearization_path.shape[1:], -1, np.int32),
            )),
            initial_current_sdf=np.concatenate((
                batch.initial_current_sdf,
                np.zeros((padding,) + task.shape, np.float32),
            )),
            source_ids=batch.source_ids + tuple(f"linear_device_padding:{index}" for index in range(padding)),
        )
    key = (task.fingerprint, library.library_fingerprint, len(devices))
    executor = _LINEAR_EXECUTORS.get(key)
    if executor is None:
        _LINEAR_EXECUTORS.clear()
        executor = _build_executor(task, library, devices)
        _LINEAR_EXECUTORS[key] = executor
    import jax.numpy as jnp

    values = executor(
        jnp.asarray(batch.initial_current_sdf), jnp.asarray(batch.actions),
        jnp.asarray(batch.action_mask), jnp.asarray(batch.linearization_path),
    )
    (
        states, remaining_voxels, overcut_voxels, remaining, overcut,
        healthy_overcut, clearance, pulse_count, energy, flags,
    ) = values
    flag_values = np.asarray(flags[:original_count])
    flag_arrays = {
        name: flag_values[..., index] for index, name in enumerate(LINEAR_FLAG_NAMES)
    }
    constraint_feasible = ~np.any(flag_values, axis=(1, 2))
    arrays = [np.asarray(value[:original_count]) for value in (
        states, remaining_voxels, overcut_voxels, remaining, overcut,
        healthy_overcut, clearance, pulse_count, energy,
    )]
    return QuickRolloutBatch(
        current_sdf=arrays[0], remaining_voxels=arrays[1], overcut_voxels=arrays[2],
        remaining_fraction=arrays[3], overcut_fraction=arrays[4],
        healthy_overcut_fraction=arrays[5], clearance_mm=arrays[6],
        pulse_count=arrays[7], energy_j=arrays[8], flags=flag_arrays,
        constraint_feasible=constraint_feasible,
        accepted=screening_mask(arrays[3][:, -1], arrays[4][:, -1], ~constraint_feasible),
    )


def rollout_linearized_in_batches(
    task: StaticTaskTensors,
    library: LinearizationLibrary,
    batch: LinearizedActionBatch,
    batch_size: int,
    devices: tuple[object, ...],
) -> QuickRolloutBatch:
    """Evaluate local-affine plans in deterministic memory-bounded sub-batches."""
    if batch_size <= 0:
        raise ValueError("linear rollout batch_size must be positive")
    parts = []
    for start in range(0, batch.actions.shape[0], batch_size):
        stop = min(start + batch_size, batch.actions.shape[0])
        parts.append(rollout_linearized_batch(
            task, library,
            LinearizedActionBatch(
                batch.actions[start:stop], batch.action_mask[start:stop],
                batch.linearization_path[start:stop], batch.initial_current_sdf[start:stop],
                batch.source_ids[start:stop],
            ),
            devices,
        ))
    fields = (
        "current_sdf", "remaining_voxels", "overcut_voxels", "remaining_fraction",
        "overcut_fraction", "healthy_overcut_fraction", "clearance_mm", "pulse_count",
        "energy_j", "constraint_feasible", "accepted",
    )
    values = {name: np.concatenate([getattr(part, name) for part in parts]) for name in fields}
    values["flags"] = {
        name: np.concatenate([part.flags[name] for part in parts]) for name in LINEAR_FLAG_NAMES
    }
    return QuickRolloutBatch(**values)
