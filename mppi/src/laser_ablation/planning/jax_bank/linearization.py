"""Build sparse local affine SDF dynamics around nominal global seed plans."""

from __future__ import annotations

import numpy as np

from laser_ablation.planning.jax_bank.contracts import PaddedActionBatch, StaticTaskTensors
from laser_ablation.planning.jax_bank.jax_transition import build_transition
from laser_ablation.planning.jax_bank.linear_contracts import LinearizationLibrary, linearization_fingerprint
from laser_ablation.planning.jax_bank.similarity import reduced_geometry_state


def _parallel_rows(function: object, devices: tuple[object, ...], ranks: tuple[int, ...]):
    """JIT one row-wise vectorized primitive with the leading row axis sharded."""
    import jax

    vectorized = jax.vmap(function)
    if len(devices) == 1:
        return jax.jit(vectorized)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(devices), ("row",))
    return jax.jit(
        vectorized,
        in_shardings=tuple(
            NamedSharding(mesh, PartitionSpec("row", *([None] * (rank - 1))))
            for rank in ranks
        ),
    )


def _pad_rows(values: tuple[np.ndarray, ...], device_count: int) -> tuple[tuple[np.ndarray, ...], int]:
    """Pad a row-parallel JAX call without adding an active nominal trajectory."""
    count = values[0].shape[0]
    if any(value.shape[0] != count for value in values) or count <= 0:
        raise ValueError("parallel rows require one nonempty shared leading dimension")
    padding = (-count) % device_count
    if not padding:
        return values, count
    return tuple(np.concatenate((
        value, np.zeros((padding,) + value.shape[1:], dtype=value.dtype),
    )) for value in values), count


def _nominal_rollout(
    task: StaticTaskTensors,
    batch: PaddedActionBatch,
    devices: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Roll every seed once, retaining only actions before nominal completion."""
    import jax
    import jax.numpy as jnp

    pulse = build_transition(task, freeze_contact_state=True)
    target = jnp.asarray(task.target_mask)
    target_count = jnp.float32(task.initial_target_voxels)
    initial = jnp.asarray(task.initial_current_sdf, dtype=jnp.float32)

    def one_seed(actions: object, action_mask: object) -> tuple[object, object]:
        initial_completed = (
            jnp.count_nonzero((initial <= 0.0) & target) / target_count
            <= task.completion_remaining_fraction
        )

        def advance(carry: tuple[object, object], inputs: tuple[object, object]):
            state, completed = carry
            action, requested = inputs
            active = requested & ~completed
            next_state, _, _, remaining, _ = pulse(state, action, active)
            return (next_state, completed | (remaining <= task.completion_remaining_fraction)), (
                next_state, active,
            )

        _, outputs = jax.lax.scan(advance, (initial, initial_completed), (actions, action_mask))
        states, active = outputs
        return jnp.concatenate((initial[None], states)), active

    padded, count = _pad_rows((batch.actions, batch.action_mask), len(devices))
    executor = _parallel_rows(one_seed, devices, (3, 2))
    states, step_mask = executor(jnp.asarray(padded[0]), jnp.asarray(padded[1]))
    return np.asarray(states[:count]), np.asarray(step_mask[:count], dtype=bool)


def _local_derivatives(
    task: StaticTaskTensors,
    states: np.ndarray,
    actions: np.ndarray,
    roi_indices: np.ndarray,
    roi_mask: np.ndarray,
    devices: tuple[object, ...],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate only gathered ROI outputs, never a full-state action Jacobian."""
    if not len(states):
        roi = int(roi_indices.shape[-1])
        return (
            np.zeros((0, roi), dtype=np.float32),
            np.zeros((0, roi, 5), dtype=np.float32),
        )
    import jax
    import jax.numpy as jnp

    pulse = build_transition(task, freeze_contact_state=True)

    voxel_count = int(np.prod(task.shape))

    def one_step(
        state: object, action: object, indices: object, in_roi: object,
    ) -> tuple[object, object]:
        safe_indices = jnp.where(in_roi, indices, voxel_count)

        def gathered(candidate_state: object, candidate_action: object) -> object:
            next_state = pulse(candidate_state, candidate_action, jnp.bool_(True))[0]
            values = jnp.concatenate((next_state.reshape(-1), jnp.zeros(1, jnp.float32)))[safe_indices]
            return jnp.where(in_roi, values, 0.0)

        tangent = jnp.zeros(voxel_count + 1, dtype=state.dtype).at[safe_indices].set(
            in_roi.astype(state.dtype)
        )[:-1].reshape(task.shape)
        _, state_gain = jax.jvp(
            lambda candidate_state: gathered(candidate_state, action),
            (state,), (tangent,),
        )
        action_jacobian = jax.jacfwd(
            lambda candidate_action: gathered(state, candidate_action)
        )(action)
        return state_gain, action_jacobian

    executor = _parallel_rows(one_step, devices, (4, 2, 2, 2))
    gains: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        print(
            f"linearization derivatives rows={start}:{min(start + batch_size, len(states))}/{len(states)}",
            flush=True,
        )
        padded, count = _pad_rows((
            states[start:start + batch_size], actions[start:start + batch_size],
            roi_indices[start:start + batch_size], roi_mask[start:start + batch_size],
        ), len(devices))
        gain, jacobian = executor(*tuple(jnp.asarray(values) for values in padded))
        gains.append(np.asarray(gain[:count]))
        jacobians.append(np.asarray(jacobian[:count]))
    return np.concatenate(gains), np.concatenate(jacobians)


def _corner_successors(
    task: StaticTaskTensors,
    states: np.ndarray,
    corners: np.ndarray,
    devices: tuple[object, ...],
    batch_size: int,
) -> np.ndarray:
    """Evaluate all 32 support corners in parallel for each active nominal step."""
    if not len(states):
        return np.zeros((0, 32) + task.shape, dtype=np.float32)
    import jax
    import jax.numpy as jnp

    pulse = build_transition(task, freeze_contact_state=True)

    def one_step(state: object, actions: object) -> object:
        return jax.vmap(lambda action: pulse(state, action, jnp.bool_(True))[0])(actions)

    executor = _parallel_rows(one_step, devices, (4, 3))
    outputs: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        print(
            f"linearization support rows={start}:{min(start + batch_size, len(states))}/{len(states)}",
            flush=True,
        )
        padded, count = _pad_rows((states[start:start + batch_size], corners[start:start + batch_size]), len(devices))
        values = executor(jnp.asarray(padded[0]), jnp.asarray(padded[1]))
        outputs.append(np.asarray(values[:count]))
    return np.concatenate(outputs)


def _roi_layout(
    task: StaticTaskTensors,
    states: np.ndarray,
    nominal_next: np.ndarray,
    corner_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn corner responses into one padded changed-voxel ROI plus one-voxel halo."""
    if not len(states):
        return np.zeros((0, 1), np.int32), np.zeros((0, 1), bool)
    changed = np.any(corner_next != states[:, None], axis=1) | (nominal_next != states)
    padded = np.pad(changed, ((0, 0), (1, 1), (1, 1), (1, 1)))
    halo = np.zeros_like(changed)
    for x_shift in range(3):
        for y_shift in range(3):
            for z_shift in range(3):
                halo |= padded[
                    :, x_shift:x_shift + task.shape[0], y_shift:y_shift + task.shape[1],
                    z_shift:z_shift + task.shape[2],
                ]
    maximum = max(1, int(np.max(np.count_nonzero(halo.reshape(len(states), -1), axis=1))))
    indices = np.zeros((len(states), maximum), dtype=np.int32)
    mask = np.zeros((len(states), maximum), dtype=bool)
    for row in range(len(states)):
        selected = np.flatnonzero(halo[row].ravel())
        indices[row, :len(selected)] = selected
        mask[row, :len(selected)] = True
    return indices, mask


def linearize_seed_batch(
    task: StaticTaskTensors,
    batch: PaddedActionBatch,
    canonical_seed_ids: tuple[str, ...],
    kappa: np.ndarray,
    derivative_batch_size: int,
    devices: tuple[object, ...],
) -> LinearizationLibrary:
    """Construct a sparse local affine model for every unique nominal seed plan."""
    if not devices:
        raise ValueError("linearization requires explicit JAX devices")
    if derivative_batch_size <= 0:
        raise ValueError("derivative_batch_size must be positive")
    if len(canonical_seed_ids) != batch.actions.shape[0] or len(set(canonical_seed_ids)) != len(canonical_seed_ids):
        raise ValueError("canonical_seed_ids must identify unique batch rows")
    scales = np.asarray(kappa, dtype=np.float32)
    if scales.shape != (5,) or not np.all(np.isfinite(scales)) or np.any(scales < 0.0):
        raise ValueError("kappa must contain five finite nonnegative action scales")
    nominal_states, step_mask = _nominal_rollout(task, batch, devices)
    flat_states = nominal_states[:, :-1].reshape((-1,) + task.shape)
    flat_next = nominal_states[:, 1:].reshape((-1,) + task.shape)
    flat_actions = batch.actions.reshape(-1, 5)
    active_indices = np.flatnonzero(step_mask.ravel())
    active_states = flat_states[active_indices]
    active_actions = flat_actions[active_indices]
    action_span = np.asarray(task.upper_bounds - task.lower_bounds, dtype=np.float32)
    support = np.float32(3.0) * scales * action_span
    support_lower = np.clip(batch.actions - support, task.lower_bounds, task.upper_bounds)
    support_upper = np.clip(batch.actions + support, task.lower_bounds, task.upper_bounds)
    lower = support_lower.reshape(-1, 5)[active_indices]
    upper = support_upper.reshape(-1, 5)[active_indices]
    signs = np.array(np.meshgrid(*([(-1.0, 1.0)] * 5), indexing="ij"), dtype=np.float32)
    corners = np.where(signs.reshape(5, -1).T[None] > 0.0, upper[:, None], lower[:, None])
    corner_next = _corner_successors(
        task, active_states, corners, devices, derivative_batch_size
    )
    if not np.all(np.isfinite(corner_next)):
        raise ValueError("linearization support rollouts must be finite")
    active_roi_indices, active_roi_mask = _roi_layout(
        task, active_states, flat_next[active_indices], corner_next
    )
    gains, jacobians = _local_derivatives(
        task, active_states, active_actions, active_roi_indices, active_roi_mask,
        devices, derivative_batch_size,
    )
    if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(jacobians)):
        raise ValueError("linearization derivatives must be finite")
    maximum_roi = int(active_roi_indices.shape[-1])
    seeds, steps = batch.action_mask.shape
    roi_indices = np.zeros((seeds, steps, maximum_roi), dtype=np.int32)
    roi_mask = np.zeros((seeds, steps, maximum_roi), dtype=bool)
    state_gain = np.zeros((seeds, steps, maximum_roi), dtype=np.float32)
    action_jacobian = np.zeros((seeds, steps, maximum_roi, 5), dtype=np.float32)
    seed_indices, step_indices = np.unravel_index(active_indices, step_mask.shape)
    for row, (seed, step) in enumerate(zip(seed_indices, step_indices, strict=True)):
        roi_indices[seed, step] = active_roi_indices[row]
        roi_mask[seed, step] = active_roi_mask[row]
        state_gain[seed, step] = gains[row]
        action_jacobian[seed, step] = jacobians[row]
    descriptors = reduced_geometry_state(nominal_states)
    fingerprint = linearization_fingerprint(
        task.fingerprint, batch.actions, step_mask, nominal_states, roi_indices,
        roi_mask, state_gain, action_jacobian, support_lower, support_upper,
        canonical_seed_ids,
    )
    return LinearizationLibrary(
        nominal_actions=np.asarray(batch.actions, dtype=np.float32), step_mask=step_mask,
        nominal_states=nominal_states, nominal_descriptors=descriptors,
        roi_indices=roi_indices, roi_mask=roi_mask, state_gain=state_gain,
        action_jacobian=action_jacobian, action_support_lower=support_lower,
        action_support_upper=support_upper, canonical_seed_ids=canonical_seed_ids,
        task_fingerprint=task.fingerprint, library_fingerprint=fingerprint,
    )
