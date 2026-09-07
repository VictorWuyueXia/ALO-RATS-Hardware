"""Observed-state centre scans and sparse overlays for one local-linear repair."""

from __future__ import annotations

import numpy as np
from time import perf_counter

from laser_ablation.planning.jax_bank.contracts import StaticTaskTensors
from laser_ablation.planning.jax_bank.jax_transition import build_transition
from laser_ablation.planning.jax_bank.linear_contracts import (
    DeviceLinearizationWorkspace,
    LinearizationLibrary,
)
from laser_ablation.planning.jax_bank.linearization import (
    _corner_successors,
    _local_derivatives,
    _pad_rows,
    _parallel_rows,
    _roi_layout,
)


ROI_INSTALL_HEADROOM = 1.05


def _center_rollout(
    task: StaticTaskTensors,
    actions: np.ndarray,
    requested: np.ndarray,
    anchor_valid: np.ndarray,
    devices: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Scan each tail from the observed SDF in parallel by anchor."""
    import jax
    import jax.numpy as jnp

    pulse = build_transition(task)
    target, initial = jnp.asarray(task.target_mask), jnp.asarray(task.initial_current_sdf, jnp.float32)
    target_count = jnp.float32(task.initial_target_voxels)

    def one_anchor(
        anchor_actions: object, anchor_requested: object, valid: object,
    ) -> tuple[object, object]:
        initially_complete = (
            jnp.count_nonzero((initial <= 0.0) & target) / target_count
            <= task.completion_remaining_fraction
        )

        def advance(carry: tuple[object, object], values: tuple[object, object]):
            current, completed = carry
            action, requested_step = values
            active = valid & requested_step & ~completed
            next_state, _, _, remaining, _ = pulse(current, action, active)
            return (next_state, completed | (remaining <= task.completion_remaining_fraction)), (
                next_state, active,
            )

        _, outputs = jax.lax.scan(
            advance, (initial, initially_complete), (anchor_actions, anchor_requested),
        )
        states, active = outputs
        return jnp.concatenate((initial[None], states)), active

    padded, count = _pad_rows((actions, requested, anchor_valid), len(devices))
    states, effective = _parallel_rows(one_anchor, devices, (3, 2, 1))(
        *tuple(jnp.asarray(values) for values in padded)
    )
    return np.asarray(states[:count]), np.asarray(effective[:count], dtype=bool)


def _refresh_predicate(
    centers: np.ndarray,
    actions: np.ndarray,
    active: np.ndarray,
    routes: np.ndarray,
    library: LinearizationLibrary,
    threshold: float,
) -> np.ndarray:
    """Mark only centre/local pairs outside a canonical base map's trust region."""
    import jax
    import jax.numpy as jnp
    from laser_ablation.planning.jax_bank.similarity import roi_cosine_similarity

    voxels = int(np.prod(library.state_shape))
    seed_count, pulse_count = library.seed_count, library.maximum_pulses
    states = jnp.asarray(library.nominal_states)
    step_mask = jnp.asarray(library.step_mask)
    indices, roi_mask = jnp.asarray(library.roi_indices), jnp.asarray(library.roi_mask)
    lower, upper = jnp.asarray(library.action_support_lower), jnp.asarray(library.action_support_upper)

    def one_step(state: object, action: object, requested: object, route: object):
        raw_seed, raw_step = route
        route_valid = (
            (raw_seed >= 0) & (raw_seed < seed_count)
            & (raw_step >= 0) & (raw_step < pulse_count)
        )
        seed, step = jnp.clip(raw_seed, 0, seed_count - 1), jnp.clip(raw_step, 0, pulse_count - 1)
        route_valid = route_valid & step_mask[seed, step]
        local_indices, local_mask = indices[seed, step], roi_mask[seed, step]
        safe = jnp.where(local_mask, local_indices, voxels)
        live = jnp.concatenate((state.reshape(-1), jnp.zeros(1, jnp.float32)))[safe]
        nominal = jnp.concatenate((states[seed, step].reshape(-1), jnp.zeros(1, jnp.float32)))[safe]
        cosine, valid_norm = roi_cosine_similarity(live, nominal, local_mask)
        support_escape = jnp.any(action < lower[seed, step]) | jnp.any(action > upper[seed, step])
        invalid = requested & (~route_valid | ~valid_norm)
        return requested & route_valid & valid_norm & ((cosine < jnp.float32(threshold)) | support_escape), invalid

    refresh, invalid = jax.jit(jax.vmap(jax.vmap(one_step)))(
        jnp.asarray(centers[:, :-1]), jnp.asarray(actions), jnp.asarray(active), jnp.asarray(routes),
    )
    if np.any(np.asarray(invalid)):
        raise ValueError("refresh construction requires a valid nonzero routed ROI cosine")
    return np.asarray(refresh, dtype=bool)


def build_recentered_workspace(
    task: StaticTaskTensors,
    prepared_anchors: object,
    library: LinearizationLibrary,
    config: object,
    devices: tuple[object, ...],
    *,
    maximum_pulses: int | None = None,
    roi_capacity: int | None = None,
    rollout_rows: int | None = None,
    force_refresh: bool = False,
) -> DeviceLinearizationWorkspace:
    """Build a fixed ten-seed/four-anchor repair workspace around observed geometry."""
    if task.fingerprint != library.task_fingerprint or not devices:
        raise ValueError("workspace requires a matching task, library, and explicit devices")
    pulses = library.maximum_pulses if maximum_pulses is None else int(maximum_pulses)
    roi = (
        min(int(np.prod(task.shape)), int(np.ceil(ROI_INSTALL_HEADROOM * library.maximum_roi_voxels)))
        if roi_capacity is None else int(roi_capacity)
    )
    rows = int(config.rollout_batch_size if rollout_rows is None else rollout_rows)
    if pulses < library.maximum_pulses or pulses <= 0 or rows <= 0:
        raise ValueError("workspace fixed capacities cannot shrink the installed base library")
    if roi < library.maximum_roi_voxels:
        raise ValueError("fresh_roi_capacity")
    if library.seed_count > 10 or len(prepared_anchors.tails) > 4:
        raise ValueError("workspace supports at most ten base seeds and four repair anchors")
    source_actions = np.asarray(prepared_anchors.origin_actions, dtype=np.float32)
    source_mask = np.asarray(prepared_anchors.action_mask, dtype=bool)
    source_routes = np.asarray(prepared_anchors.linearization_path, dtype=np.int32)
    if source_actions.shape[:2] != source_mask.shape or source_routes.shape != source_actions.shape[:2] + (2,):
        raise ValueError("prepared anchors must contain aligned actions, masks, and routes")
    if source_actions.shape[1] > pulses:
        raise ValueError("prepared anchor tail exceeds the fixed workspace pulse capacity")

    total_started = perf_counter()
    padding_started = perf_counter()
    shape = task.shape
    base_actions = np.zeros((10, pulses, 5), np.float32)
    base_mask = np.zeros((10, pulses), bool)
    base_states = np.zeros((10, pulses + 1) + shape, np.float32)
    base_indices = np.zeros((10, pulses, roi), np.int32)
    base_roi_mask = np.zeros((10, pulses, roi), bool)
    base_gain = np.zeros((10, pulses, roi), np.float32)
    base_jacobian = np.zeros((10, pulses, roi, 5), np.float32)
    base_lower = np.zeros((10, pulses, 5), np.float32)
    base_upper = np.zeros((10, pulses, 5), np.float32)
    seeds, base_pulses, base_roi = library.seed_count, library.maximum_pulses, library.maximum_roi_voxels
    base_actions[:seeds, :base_pulses] = library.nominal_actions
    base_mask[:seeds, :base_pulses] = library.step_mask
    base_states[:seeds, :base_pulses + 1] = library.nominal_states
    base_indices[:seeds, :base_pulses, :base_roi] = library.roi_indices
    base_roi_mask[:seeds, :base_pulses, :base_roi] = library.roi_mask
    base_gain[:seeds, :base_pulses, :base_roi] = library.state_gain
    base_jacobian[:seeds, :base_pulses, :base_roi] = library.action_jacobian
    base_lower[:seeds, :base_pulses] = library.action_support_lower
    base_upper[:seeds, :base_pulses] = library.action_support_upper

    center_actions = np.zeros((4, pulses, 5), np.float32)
    center_requested = np.zeros((4, pulses), bool)
    anchor_routes = np.full((4, pulses, 2), -1, np.int32)
    anchor_valid = np.zeros(4, bool)
    anchors, tail_pulses = source_actions.shape[:2]
    center_actions[:anchors, :tail_pulses] = source_actions
    center_requested[:anchors, :tail_pulses] = source_mask
    anchor_routes[:anchors, :tail_pulses] = source_routes
    anchor_valid[:anchors] = True
    padding_seconds = perf_counter() - padding_started
    centre_started = perf_counter()
    center_states, center_active = _center_rollout(
        task, center_actions, center_requested, anchor_valid, devices,
    )
    center_actions = np.where(center_active[..., None], center_actions, 0.0)
    anchor_routes = np.where(center_active[..., None], anchor_routes, -1)
    centre_seconds = perf_counter() - centre_started
    predicate_started = perf_counter()
    refresh = _refresh_predicate(
        center_states, center_actions, center_active, anchor_routes, library,
        float(config.roi_linearization_cosine_min),
    )
    if force_refresh:
        refresh = center_active.copy()
    predicate_seconds = perf_counter() - predicate_started

    overlay_started = perf_counter()
    overlay_indices = np.zeros((4, pulses, roi), np.int32)
    overlay_mask = np.zeros((4, pulses, roi), bool)
    overlay_gain = np.zeros((4, pulses, roi), np.float32)
    overlay_jacobian = np.zeros((4, pulses, roi, 5), np.float32)
    overlay_lower = np.zeros((4, pulses, 5), np.float32)
    overlay_upper = np.zeros((4, pulses, 5), np.float32)
    flat_refresh = np.flatnonzero(refresh.ravel())
    support_seconds = derivative_seconds = 0.0
    if len(flat_refresh):
        flat_states = center_states[:, :-1].reshape((-1,) + shape)[flat_refresh]
        flat_actions = center_actions.reshape(-1, 5)[flat_refresh]
        support = np.float32(3.0) * np.asarray(config.kappa, np.float32) * (
            task.upper_bounds - task.lower_bounds
        )
        fresh_lower = np.clip(flat_actions - support, task.lower_bounds, task.upper_bounds)
        fresh_upper = np.clip(flat_actions + support, task.lower_bounds, task.upper_bounds)
        signs = np.array(np.meshgrid(*([(-1.0, 1.0)] * 5), indexing="ij"), np.float32)
        corners = np.where(signs.reshape(5, -1).T[None] > 0.0, fresh_upper[:, None], fresh_lower[:, None])
        support_started = perf_counter()
        successors = _corner_successors(task, flat_states, corners, devices, rows)
        nominal_next = _corner_successors(task, flat_states, flat_actions[:, None], devices, rows)[:, 0]
        support_seconds = perf_counter() - support_started
        fresh_indices, fresh_mask = _roi_layout(task, flat_states, nominal_next, successors)
        if fresh_indices.shape[-1] > roi:
            raise ValueError(
                f"fresh_roi_capacity required={fresh_indices.shape[-1]} installed={roi}"
            )
        derivative_started = perf_counter()
        fresh_gain, fresh_jacobian = _local_derivatives(
            task, flat_states, flat_actions, fresh_indices, fresh_mask, devices, rows,
        )
        derivative_seconds = perf_counter() - derivative_started
        if not np.all(np.isfinite(fresh_gain)) or not np.all(np.isfinite(fresh_jacobian)):
            raise ValueError("fresh linearization derivatives must be finite")
        overlay_indices.reshape(-1, roi)[flat_refresh, :fresh_indices.shape[-1]] = fresh_indices
        overlay_mask.reshape(-1, roi)[flat_refresh, :fresh_mask.shape[-1]] = fresh_mask
        overlay_gain.reshape(-1, roi)[flat_refresh, :fresh_gain.shape[-1]] = fresh_gain
        overlay_jacobian.reshape(-1, roi, 5)[flat_refresh, :fresh_jacobian.shape[-2]] = fresh_jacobian
        overlay_lower.reshape(-1, 5)[flat_refresh] = fresh_lower
        overlay_upper.reshape(-1, 5)[flat_refresh] = fresh_upper

    overlay_seconds = perf_counter() - overlay_started
    phases = (
        ("fixed_shape_padding", padding_seconds),
        ("center_scan", centre_seconds),
        ("refresh_predicate", predicate_seconds),
        ("support_corners", support_seconds),
        ("derivative_refresh", derivative_seconds),
        ("overlay_assembly", overlay_seconds),
        ("workspace_total", perf_counter() - total_started),
    )
    return DeviceLinearizationWorkspace(
        base_actions, base_mask, base_states, base_indices, base_roi_mask, base_gain,
        base_jacobian, base_lower, base_upper, center_states, center_actions,
        center_active, anchor_routes, anchor_valid, refresh, overlay_indices,
        overlay_mask, overlay_gain, overlay_jacobian, overlay_lower, overlay_upper,
        task.fingerprint, library.library_fingerprint,
        float(config.roi_linearization_cosine_min), seeds, anchors, rows, phases,
    )
