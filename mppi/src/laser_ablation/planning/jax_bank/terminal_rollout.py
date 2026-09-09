"""Terminal-only exact SDF rollouts for bounded segment shooting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from laser_ablation.planning.jax_bank.contracts import StaticTaskTensors
from laser_ablation.planning.jax_bank.jax_transition import (
    FLAG_NAMES, ROLLOUT_DIAGNOSTIC_FLAG_NAMES, build_transition,
)


@dataclass(frozen=True)
class TerminalRolloutBatch:
    """Exact terminal geometry and compact metrics without intermediate SDF frames."""

    current_sdf: np.ndarray
    remaining_fraction: np.ndarray
    overcut_fraction: np.ndarray
    healthy_overcut_fraction: np.ndarray
    clearance_mm: np.ndarray
    pulse_count: np.ndarray
    energy_j: np.ndarray
    flags: dict[str, np.ndarray]
    constraint_feasible: np.ndarray
    accepted: np.ndarray


_EXECUTORS: dict[tuple[str, int, int, int], Callable[..., object]] = {}


def _build_terminal_executor(
    task: StaticTaskTensors,
    devices: tuple[object, ...],
    steps: int,
    rows: int,
) -> Callable[..., object]:
    """Compile one exact scan that emits only terminal state, metrics, and flags."""
    import jax
    import jax.numpy as jnp

    transition = build_transition(task, freeze_contact_state=False)
    target = jnp.asarray(task.target_mask)
    initial_tissue = jnp.asarray(task.initial_tissue)
    target_denominator = jnp.float32(task.initial_target_voxels)
    healthy_denominator = jnp.float32(task.initial_healthy_voxels)

    def one_plan(initial: object, actions: object, action_mask: object) -> tuple[object, ...]:
        initial_remaining = jnp.count_nonzero((initial <= 0.0) & target)
        initial_overcut = jnp.count_nonzero(initial_tissue & (initial > 0.0) & ~target)
        initial_completed = initial_remaining / target_denominator <= task.completion_remaining_fraction

        def advance(
            carry: tuple[object, ...], inputs: tuple[object, object],
        ) -> tuple[tuple[object, ...], object]:
            current, clearance, pulse_count, energy, completed, remaining, overcut, flags_so_far = carry
            action, requested = inputs
            active = requested & ~completed
            next_state, flags, step_clearance, next_remaining, next_overcut = transition(
                current, action, active,
            )
            next_completed = completed | (next_remaining <= task.completion_remaining_fraction)
            return (
                next_state,
                jnp.minimum(clearance, step_clearance),
                pulse_count + active.astype(jnp.int32),
                energy + jnp.where(active, action[4], 0.0),
                next_completed,
                next_remaining,
                next_overcut,
                jnp.logical_or(flags_so_far, flags),
            ), None

        final, _ = jax.lax.scan(
            advance,
            (
                initial, jnp.float32(jnp.inf), jnp.int32(0), jnp.float32(0.0),
                initial_completed, initial_remaining / target_denominator,
                initial_overcut / target_denominator,
                jnp.zeros(len(FLAG_NAMES), dtype=bool),
            ),
            (actions, action_mask),
            length=steps,
        )
        current, clearance, pulse_count, energy, _, remaining, overcut, flags = final
        healthy_overcut = jnp.where(
            healthy_denominator > 0.0,
            jnp.count_nonzero(initial_tissue & (current > 0.0) & ~target) / healthy_denominator,
            jnp.nan,
        )
        return (
            current, remaining, overcut, healthy_overcut,
            clearance, pulse_count, energy, flags,
        )

    vectorized = jax.vmap(one_plan, in_axes=(0, 0, 0))
    if len(devices) == 1:
        return jax.jit(vectorized)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    if rows % len(devices):
        raise ValueError("terminal rollout rows must divide evenly across devices")
    mesh = Mesh(np.asarray(devices), ("plan",))
    sharded = NamedSharding(mesh, PartitionSpec("plan", None, None, None))
    action_sharding = NamedSharding(mesh, PartitionSpec("plan", None, None))
    mask_sharding = NamedSharding(mesh, PartitionSpec("plan", None))
    return jax.jit(vectorized, in_shardings=(sharded, action_sharding, mask_sharding))


def rollout_terminal_in_batches(
    task: StaticTaskTensors,
    initial_sdf: np.ndarray,
    actions: np.ndarray,
    action_mask: np.ndarray,
    batch_size: int,
    devices: tuple[object, ...],
    initial_rows: np.ndarray | None = None,
) -> TerminalRolloutBatch:
    """Roll exact per-row initial states through fixed-width GPU batches without state traces."""
    import jax.numpy as jnp

    initial = np.asarray(initial_sdf, np.float32)
    action_values = np.asarray(actions, np.float32)
    mask = np.asarray(action_mask, bool)
    if not devices or batch_size <= 0 or batch_size % len(devices):
        raise ValueError("terminal batch size must be positive and divisible by the device count")
    if action_values.ndim != 3 or action_values.shape[-1] != 5:
        raise ValueError("terminal actions must have shape (row, step, 5)")
    row_mapping = None if initial_rows is None else np.asarray(initial_rows, np.int32)
    if mask.shape != action_values.shape[:2]:
        raise ValueError("terminal actions and masks must share their row dimension")
    if row_mapping is None:
        if initial.shape not in {task.shape, (len(action_values),) + task.shape}:
            raise ValueError("terminal states must be shared or align with every action row")
    elif (row_mapping.shape != (len(action_values),) or initial.ndim != 4
          or initial.shape[1:] != task.shape or np.any(row_mapping < 0)
          or np.any(row_mapping >= len(initial))):
        raise ValueError("terminal state rows must map every action row to one initial SDF")
    if len(action_values) <= 0 or not np.all(np.isfinite(initial)) or not np.all(np.isfinite(action_values)):
        raise ValueError("terminal rollout requires finite nonempty input arrays")
    steps = int(action_values.shape[1])
    key = (task.fingerprint, len(devices), steps, batch_size)
    executor = _EXECUTORS.get(key)
    if executor is None:
        executor = _build_terminal_executor(task, devices, steps, batch_size)
        _EXECUTORS[key] = executor

    parts: list[tuple[np.ndarray, ...]] = []
    batch_count = (len(action_values) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(action_values), batch_size), start=1):
        stop = min(start + batch_size, len(action_values))
        count = stop - start
        packed_initial = np.broadcast_to(task.initial_current_sdf, (batch_size,) + task.shape).copy()
        packed_actions = np.zeros((batch_size, steps, 5), np.float32)
        packed_mask = np.zeros((batch_size, steps), bool)
        if row_mapping is None and initial.shape == task.shape:
            packed_initial[:count] = initial
        elif row_mapping is None:
            packed_initial[:count] = initial[start:stop]
        else:
            packed_initial[:count] = initial[row_mapping[start:stop]]
        packed_actions[:count] = action_values[start:stop]
        packed_mask[:count] = mask[start:stop]
        values = executor(
            jnp.asarray(packed_initial), jnp.asarray(packed_actions), jnp.asarray(packed_mask),
        )
        parts.append(tuple(np.asarray(value[:count]) for value in values))
        del values
        print(f"rollout kind=terminal batch={batch_index}/{batch_count} rows=[{start},{stop})", flush=True)

    combined = tuple(np.concatenate([part[index] for part in parts]) for index in range(8))
    states, remaining, overcut, healthy, clearance, pulses, energy, flag_values = combined
    hard_indices = tuple(index for index, name in enumerate(FLAG_NAMES)
                         if name not in ROLLOUT_DIAGNOSTIC_FLAG_NAMES)
    feasible = ~np.any(flag_values[..., hard_indices], axis=1)
    accepted = feasible.copy()
    return TerminalRolloutBatch(
        states, remaining, overcut, healthy, clearance, pulses, energy,
        {name: flag_values[..., index] for index, name in enumerate(FLAG_NAMES)},
        feasible, accepted,
    )
