"""JIT-compiled, batch-parallel quick SDF rollouts and screening."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from laser_ablation.planning.jax_bank.contracts import (
    PaddedActionBatch,
    QuickRolloutBatch,
    StaticTaskTensors,
)
from laser_ablation.planning.jax_bank.jax_transition import FLAG_NAMES, build_transition


_EXECUTORS: dict[tuple[str, int], Callable[..., object]] = {}


def _build_executor(task: StaticTaskTensors, devices: tuple[object, ...]) -> Callable[..., object]:
    """Compile one static-geometry executor that accepts the changing current SDF."""
    import jax
    import jax.numpy as jnp

    pulse = build_transition(task)
    target = jnp.asarray(task.target_mask)
    initial_tissue = jnp.asarray(task.initial_tissue)
    target_denominator = jnp.float32(task.initial_target_voxels)
    healthy_denominator = jnp.float32(task.initial_healthy_voxels)

    def one_plan(initial: object, actions: object, action_mask: object):
        """Preserve every prefix state and cumulative quick metric for one plan."""
        initial_remaining = jnp.count_nonzero((initial <= 0.0) & target)
        initial_overcut = jnp.count_nonzero(initial_tissue & (initial > 0.0) & ~target)

        def advance(carry: tuple[object, ...], inputs: tuple[object, object]):
            current, minimum_clearance, pulse_count, energy, completed = carry
            action, active = inputs
            next_state, flags, clearance, remaining, overcut = pulse(
                current, action, active & ~completed
            )
            next_carry = (
                next_state,
                jnp.minimum(minimum_clearance, clearance),
                pulse_count + (active & ~completed).astype(jnp.int32),
                energy + jnp.where(active & ~completed, action[4], 0.0),
                completed | (remaining <= task.completion_remaining_fraction),
            )
            outputs = (
                next_state, flags, remaining, overcut,
                next_carry[1], next_carry[2], next_carry[3],
            )
            return next_carry, outputs

        _, outputs = jax.lax.scan(
            advance,
            (
                initial, jnp.float32(jnp.inf), jnp.int32(0), jnp.float32(0.0),
                initial_remaining / target_denominator <= task.completion_remaining_fraction,
            ),
            (actions, action_mask),
        )
        states, flags, remaining, overcut, clearance, pulse_count, energy = outputs
        remaining_voxels = jnp.rint(remaining * target_denominator).astype(jnp.int32)
        overcut_voxels = jnp.rint(overcut * target_denominator).astype(jnp.int32)
        all_remaining = jnp.concatenate((initial_remaining[None], remaining_voxels))
        all_overcut = jnp.concatenate((initial_overcut[None], overcut_voxels))
        return (
            jnp.concatenate((initial[None], states)),
            all_remaining,
            all_overcut,
            all_remaining / target_denominator,
            all_overcut / target_denominator,
            jnp.where(healthy_denominator > 0.0, all_overcut / healthy_denominator, jnp.nan),
            jnp.concatenate((jnp.asarray([jnp.inf], jnp.float32), clearance)),
            jnp.concatenate((jnp.asarray([0], jnp.int32), pulse_count)),
            jnp.concatenate((jnp.asarray([0.0], jnp.float32), energy)),
            flags,
        )

    vectorized = jax.vmap(one_plan, in_axes=(None, 0, 0))
    if len(devices) == 1:
        return jax.jit(vectorized)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(devices), ("plan",))
    replicated = NamedSharding(mesh, PartitionSpec())
    actions = NamedSharding(mesh, PartitionSpec("plan", None, None))
    masks = NamedSharding(mesh, PartitionSpec("plan", None))
    return jax.jit(vectorized, in_shardings=(replicated, actions, masks))


def rollout_batch(
    task: StaticTaskTensors,
    batch: PaddedActionBatch,
    devices: tuple[object, ...],
) -> QuickRolloutBatch:
    """Roll every padded complete plan with scan-over-time and vmap-over-plans."""
    if task.initial_target_voxels <= 0:
        raise ValueError("quick rollout requires initial target tissue")
    import jax.numpy as jnp

    if not devices:
        raise ValueError("quick rollout requires an explicit JAX device set")
    original_count = batch.actions.shape[0]
    remainder = original_count % len(devices)
    if remainder:
        padding = len(devices) - remainder
        batch = PaddedActionBatch(
            np.concatenate((batch.actions, np.zeros((padding,) + batch.actions.shape[1:], np.float32))),
            np.concatenate((batch.action_mask, np.zeros((padding,) + batch.action_mask.shape[1:], bool))),
            batch.source_ids + tuple(f"device_padding:{index}" for index in range(padding)),
        )
    executor_key = (task.fingerprint, len(devices))
    executor = _EXECUTORS.get(executor_key)
    if executor is None:
        executor = _build_executor(task, devices)
        _EXECUTORS[executor_key] = executor
    values = executor(
        jnp.asarray(task.initial_current_sdf, jnp.float32),
        jnp.asarray(batch.actions),
        jnp.asarray(batch.action_mask),
    )
    (
        states, remaining_voxels, overcut_voxels, remaining, overcut,
        healthy_overcut, clearance, pulse_count, energy, flags,
    ) = values
    flag_arrays = {
        name: np.asarray(flags[:original_count, ..., index])
        for index, name in enumerate(FLAG_NAMES)
    }
    hard_flag_indices = tuple(
        index for index, name in enumerate(FLAG_NAMES) if name != "no_positive_removal"
    )
    constraint_feasible = ~np.any(
        np.asarray(flags[:original_count])[..., hard_flag_indices], axis=(1, 2)
    )
    arrays = [np.asarray(value[:original_count]) for value in (
        states, remaining_voxels, overcut_voxels, remaining, overcut,
        healthy_overcut, clearance, pulse_count, energy,
    )]
    accepted = screening_mask(
        arrays[3][:, -1], arrays[4][:, -1], ~constraint_feasible
    )
    return QuickRolloutBatch(
        current_sdf=arrays[0],
        remaining_voxels=arrays[1],
        overcut_voxels=arrays[2],
        remaining_fraction=arrays[3],
        overcut_fraction=arrays[4],
        healthy_overcut_fraction=arrays[5],
        clearance_mm=arrays[6],
        pulse_count=arrays[7],
        energy_j=arrays[8],
        flags=flag_arrays,
        constraint_feasible=constraint_feasible,
        accepted=accepted,
    )


def rollout_in_batches(
    task: StaticTaskTensors,
    batch: PaddedActionBatch,
    batch_size: int,
    devices: tuple[object, ...],
) -> QuickRolloutBatch:
    """Evaluate every plan in deterministic CPU-safe JAX sub-batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    parts = []
    for start in range(0, batch.actions.shape[0], batch_size):
        stop = min(start + batch_size, batch.actions.shape[0])
        parts.append(
            rollout_batch(
                task,
                PaddedActionBatch(
                    batch.actions[start:stop], batch.action_mask[start:stop],
                    batch.source_ids[start:stop],
                ),
                devices,
            )
        )
    fields = (
        "current_sdf", "remaining_voxels", "overcut_voxels", "remaining_fraction",
        "overcut_fraction", "healthy_overcut_fraction", "clearance_mm", "pulse_count",
        "energy_j", "constraint_feasible", "accepted",
    )
    values = {name: np.concatenate([getattr(part, name) for part in parts]) for name in fields}
    values["flags"] = {
        name: np.concatenate([part.flags[name] for part in parts]) for name in FLAG_NAMES
    }
    return QuickRolloutBatch(**values)


def screening_mask(
    remaining_fraction: np.ndarray, overcut_fraction: np.ndarray, flags: np.ndarray
) -> np.ndarray:
    """Admit every hard-feasible row; terminal quality affects ranking, not admission."""
    remaining = np.asarray(remaining_fraction)
    overcut = np.asarray(overcut_fraction)
    hard_flags = np.asarray(flags, dtype=bool)
    if remaining.shape != overcut.shape or remaining.shape != hard_flags.shape:
        raise ValueError("screening values and hard flags must have identical shapes")
    return ~hard_flags
