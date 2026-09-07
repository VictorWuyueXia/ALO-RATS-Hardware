"""Fixed-shape affine rollouts over an explicit recentered repair workspace."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from laser_ablation.planning.jax_bank.contracts import StaticTaskTensors
from laser_ablation.planning.jax_bank.jax_transition import FLAG_NAMES
from laser_ablation.planning.jax_bank.linear_contracts import DeviceLinearizationWorkspace


WORKSPACE_FLAG_NAMES = FLAG_NAMES + (
    "irreversibility", "linearization_support", "linearization_contact_guard",
    "linearization_trust",
)
WORKSPACE_ARRAY_NAMES = (
    "base_nominal_actions", "base_step_mask",
    "base_roi_indices", "base_roi_mask", "base_state_gain",
    "base_action_jacobian", "base_action_support_lower",
    "base_action_support_upper", "center_states", "center_actions",
    "center_action_mask", "anchor_routes", "anchor_valid", "refresh_mask",
    "overlay_roi_indices", "overlay_roi_mask", "overlay_state_gain",
    "overlay_action_jacobian", "overlay_action_support_lower",
    "overlay_action_support_upper",
)


def workspace_array_tuple(workspace: DeviceLinearizationWorkspace) -> tuple[object, ...]:
    """Freeze the one explicit argument order shared by installation and rollout."""
    return tuple(getattr(workspace, name) for name in WORKSPACE_ARRAY_NAMES)


def build_workspace_executor(
    task: StaticTaskTensors,
    workspace: DeviceLinearizationWorkspace,
    devices: tuple[object, ...],
    *,
    trace: bool,
) -> Callable[..., object]:
    """Compile either terminal screening or full traces from explicit workspace arrays."""
    import jax
    import jax.numpy as jnp
    from laser_ablation.planning.jax_bank.jax_transition import build_contact_query
    from laser_ablation.planning.jax_bank.similarity import roi_cosine_similarity

    if workspace.task_fingerprint != task.fingerprint or workspace.state_shape != task.shape:
        raise ValueError("workspace rollout requires matching static task geometry")
    if not devices:
        raise ValueError("workspace rollout requires explicit JAX devices")
    seeds = workspace.base_seed_capacity
    anchors = workspace.anchor_capacity
    pulses = workspace.maximum_pulses
    voxels = int(np.prod(task.shape))
    initial_tissue = jnp.asarray(task.initial_tissue)
    target = jnp.asarray(task.target_mask)
    constraint = jnp.asarray(task.constraint_mask)
    clearance_field = jnp.asarray(task.physical_clearance_mm, dtype=jnp.float32)
    lower = jnp.asarray(task.lower_bounds, dtype=jnp.float32)
    upper = jnp.asarray(task.upper_bounds, dtype=jnp.float32)
    target_count = jnp.float32(task.initial_target_voxels)
    healthy_count = jnp.float32(task.initial_healthy_voxels)
    contact_query = build_contact_query(task)
    maximum_contact_shift = jnp.float32(0.25 * task.spacing_mm)

    def one_plan(
        initial: object, actions: object, action_mask: object, path: object,
        anchor_slot: object, arrays: tuple[object, ...], trust_threshold: object,
    ) -> tuple[object, ...]:
        (
            _, base_step_mask, base_indices, base_roi_mask, base_gain,
            base_jacobian, base_lower, base_upper, center_states, center_actions,
            center_mask, anchor_routes, anchor_valid, refresh_mask, overlay_indices,
            overlay_roi_mask, overlay_gain, overlay_jacobian, overlay_lower,
            overlay_upper,
        ) = arrays
        raw_slot = anchor_slot
        slot_valid = (raw_slot >= 0) & (raw_slot < anchors)
        slot = jnp.clip(raw_slot, 0, anchors - 1)
        initial_remaining = jnp.count_nonzero((initial <= 0.0) & target)
        initial_overcut = jnp.count_nonzero(initial_tissue & (initial > 0.0) & ~target)

        def advance(
            carry: tuple[object, ...], inputs: tuple[object, object, object, object],
        ) -> tuple[tuple[object, ...], tuple[object, ...]]:
            current, minimum_clearance, pulse_count, energy, completed, failed, any_flags = carry
            action, requested, route, local_step = inputs
            raw_seed, raw_step = route
            route_valid = (
                (raw_seed >= 0) & (raw_seed < seeds)
                & (raw_step >= 0) & (raw_step < pulses)
            )
            seed = jnp.clip(raw_seed, 0, seeds - 1)
            step = jnp.clip(raw_step, 0, pulses - 1)
            route_valid = route_valid & base_step_mask[seed, step]
            anchored = slot_valid & anchor_valid[slot] & center_mask[slot, local_step]
            route_valid = route_valid & anchored & jnp.all(route == anchor_routes[slot, local_step])
            use_overlay = anchored & refresh_mask[slot, local_step]
            indices = jnp.where(use_overlay, overlay_indices[slot, local_step], base_indices[seed, step])
            in_roi = jnp.where(use_overlay, overlay_roi_mask[slot, local_step], base_roi_mask[seed, step])
            state_gain = jnp.where(use_overlay, overlay_gain[slot, local_step], base_gain[seed, step])
            jacobian = jnp.where(
                use_overlay, overlay_jacobian[slot, local_step], base_jacobian[seed, step],
            )
            support_lower = jnp.where(
                use_overlay, overlay_lower[slot, local_step], base_lower[seed, step],
            )
            support_upper = jnp.where(
                use_overlay, overlay_upper[slot, local_step], base_upper[seed, step],
            )
            safe = jnp.where(in_roi, indices, voxels)
            current_extended = jnp.concatenate((current.reshape(-1), jnp.zeros(1, jnp.float32)))
            center_current = jnp.concatenate((
                center_states[slot, local_step].reshape(-1), jnp.zeros(1, jnp.float32),
            ))[safe]
            center_next = jnp.concatenate((
                center_states[slot, local_step + 1].reshape(-1), jnp.zeros(1, jnp.float32),
            ))[safe]
            current_roi = current_extended[safe]
            cosine, valid_norm = roi_cosine_similarity(current_roi, center_current, in_roi)
            support_violation = (
                ~route_valid | jnp.any(action < support_lower) | jnp.any(action > support_upper)
            )
            active = requested & ~completed & ~failed

            def contact_diagnostics(_: object) -> tuple[object, ...]:
                _, child_contact, child_has, child_entry, child_interval = contact_query(current, action)
                _, center_contact, center_has, center_entry, center_interval = contact_query(
                    center_states[slot, local_step], action,
                )
                has_mismatch = child_has != center_has
                entry_mismatch = child_has & center_has & (child_entry != center_entry)
                interval_mismatch = (
                    child_has & center_has & ~child_entry & ~center_entry
                    & (child_interval != center_interval)
                )
                displacement = jnp.where(
                    child_has & center_has,
                    jnp.linalg.norm(child_contact - center_contact),
                    jnp.float32(0.0),
                )
                trusted = ~has_mismatch & (
                    ~child_has | (
                        ~entry_mismatch & ~interval_mismatch
                        & (displacement <= maximum_contact_shift)
                    )
                )
                return trusted, displacement, has_mismatch, entry_mismatch, interval_mismatch

            # ``-1`` disables rejection but still measures contact drift along
            # the unrestricted trace for counterfactual evidence.
            contact_values = jax.lax.cond(
                active & route_valid & valid_norm,
                contact_diagnostics,
                lambda _: (
                    jnp.bool_(True), jnp.float32(0.0), jnp.bool_(False),
                    jnp.bool_(False), jnp.bool_(False),
                ),
                operand=None,
            )
            (
                raw_contact_trusted, contact_displacement, has_mismatch,
                entry_mismatch, interval_mismatch,
            ) = contact_values
            contact_guard_failure = active & ~raw_contact_trusted
            contact_trusted = (trust_threshold <= -1.0) | raw_contact_trusted
            trust_violation = ~valid_norm | (cosine < trust_threshold) | ~contact_trusted
            local_values = (
                center_next + state_gain * (current_roi - center_current)
                + jnp.sum(jacobian * (action - center_actions[slot, local_step])[None], axis=1)
            )
            candidate = current_extended.at[safe].set(
                jnp.where(in_roi, local_values, current_extended[safe])
            )
            membership = jnp.zeros(voxels + 1, dtype=jnp.bool_).at[safe].set(in_roi)
            local_state = jnp.where(membership, candidate, current_extended)[:-1].reshape(task.shape)
            apply = active & ~support_violation & ~trust_violation
            next_state = jnp.where(apply, local_state, current)
            was_tissue = current <= 0.0
            is_tissue = next_state <= 0.0
            removal = initial_tissue & was_tissue & ~is_tissue
            removed = initial_tissue & ~is_tissue
            has_change = jnp.any(next_state != current)
            newly_removed = jnp.any(removal)
            clearance = jnp.min(jnp.where(
                removal, clearance_field, jnp.asarray(jnp.inf, dtype=jnp.float32),
            ))
            flags = jnp.asarray((
                active & (jnp.any(action < lower) | jnp.any(action > upper)),
                active & ~has_change,
                active & ~newly_removed,
                active & jnp.any(removal & constraint),
                active & (clearance < task.hard_margin_mm),
                active & (~jnp.all(jnp.isfinite(next_state)) | ~jnp.all(jnp.isfinite(action))),
            ), dtype=jnp.bool_)
            flags = jnp.concatenate((flags, jnp.asarray((
                active & jnp.any(initial_tissue & (current > 0.0) & (next_state <= 0.0)),
                active & support_violation,
                active & ~contact_trusted,
                active & trust_violation,
            ))))
            remaining = jnp.count_nonzero(is_tissue & target) / target_count
            overcut = jnp.count_nonzero(removed & ~target) / target_count
            next_carry = (
                next_state, jnp.minimum(minimum_clearance, clearance),
                pulse_count + active.astype(jnp.int32),
                energy + jnp.where(active, action[4], 0.0),
                completed | (remaining <= task.completion_remaining_fraction),
                failed | (active & (support_violation | trust_violation)),
                any_flags | flags,
            )
            if trace:
                return next_carry, (
                    next_state, flags, remaining, overcut, next_carry[1],
                    next_carry[2], next_carry[3], cosine, contact_displacement,
                    has_mismatch, entry_mismatch, interval_mismatch, contact_guard_failure,
                )
            return next_carry, None

        final, outputs = jax.lax.scan(
            advance,
            (
                initial, jnp.float32(jnp.inf), jnp.int32(0), jnp.float32(0.0),
                initial_remaining / target_count <= task.completion_remaining_fraction, jnp.bool_(False),
                jnp.zeros(len(WORKSPACE_FLAG_NAMES), dtype=jnp.bool_),
            ),
            (actions, action_mask, path, jnp.arange(pulses, dtype=jnp.int32)),
        )
        if not trace:
            final_state, minimum_clearance, final_pulses, final_energy, _, _, any_flags = final
            final_remaining = jnp.count_nonzero((final_state <= 0.0) & target) / target_count
            final_overcut = jnp.count_nonzero(initial_tissue & (final_state > 0.0) & ~target) / target_count
            return (
                final_remaining,
                final_overcut,
                jnp.where(healthy_count > 0.0, final_overcut * target_count / healthy_count, jnp.nan),
                minimum_clearance,
                final_pulses,
                final_energy,
                any_flags,
            )
        (
            states, flags, remaining, overcut, clearance, pulse_count, energy, cosine,
            contact_displacement, has_mismatch, entry_mismatch, interval_mismatch,
            contact_guard_failure,
        ) = outputs
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
            jnp.concatenate((jnp.asarray([0.0], jnp.float32), energy)), flags, cosine,
            contact_displacement, has_mismatch, entry_mismatch, interval_mismatch,
            contact_guard_failure,
        )

    vectorized = jax.vmap(one_plan, in_axes=(0, 0, 0, 0, 0, None, None))
    if len(devices) == 1:
        return jax.jit(vectorized)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(devices), ("plan",))
    row_shardings = (
        NamedSharding(mesh, PartitionSpec("plan", None, None, None)),
        NamedSharding(mesh, PartitionSpec("plan", None, None)),
        NamedSharding(mesh, PartitionSpec("plan", None)),
        NamedSharding(mesh, PartitionSpec("plan", None, None)),
        NamedSharding(mesh, PartitionSpec("plan")),
    )
    return jax.jit(vectorized, in_shardings=row_shardings + (None, None))
