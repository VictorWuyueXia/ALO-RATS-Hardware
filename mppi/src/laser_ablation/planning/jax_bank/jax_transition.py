"""Pure float32 SDF transition used only by the optional JAX proposal route."""

from __future__ import annotations

from typing import Callable

import numpy as np

from laser_ablation.planning.jax_bank.contracts import StaticTaskTensors


FLAG_NAMES = (
    "out_of_bounds",
    "no_contact",
    "no_positive_removal",
    "protected_overlap",
    "clearance_violation",
    "nonfinite",
)
ROLLOUT_DIAGNOSTIC_FLAG_NAMES = ("no_positive_removal",)


def _jax_modules():
    """Import JAX only when the optional planner is constructed."""
    import jax
    import jax.numpy as jnp

    return jax, jnp


def build_contact_query(
    task: StaticTaskTensors,
) -> Callable[[object, object], tuple[object, object, object, object, object]]:
    """Return the exact first-contact query shared by dynamics and rollout trust.

    The returned values are beam direction, interpolated contact point, contact
    existence, entry-inside-tissue status, and first crossing-interval index.
    It reads the complete SDF only along the action ray; it does not form a
    crater or mutate the state.
    """
    _, jnp = _jax_modules()
    x_axis = jnp.asarray(task.x_axis_mm, dtype=jnp.float32)
    y_axis = jnp.asarray(task.y_axis_mm, dtype=jnp.float32)
    z_axis = jnp.asarray(task.z_axis_mm, dtype=jnp.float32)
    lower = jnp.asarray(
        (
            task.x_axis_mm[0] - 0.5 * task.spacing_mm,
            task.y_axis_mm[0] - 0.5 * task.spacing_mm,
            task.z_axis_mm[0] - 0.5 * task.spacing_mm,
        ),
        dtype=jnp.float32,
    )
    upper = jnp.asarray(
        (
            task.x_axis_mm[-1] + 0.5 * task.spacing_mm,
            task.y_axis_mm[-1] + 0.5 * task.spacing_mm,
            task.z_axis_mm[-1] + 0.5 * task.spacing_mm,
        ),
        dtype=jnp.float32,
    )
    truncation = jnp.float32(task.truncation_mm)
    ray_step = jnp.float32(0.5 * task.spacing_mm)
    ray_length = float(np.linalg.norm(np.asarray((
        task.x_axis_mm[-1] - task.x_axis_mm[0] + task.spacing_mm,
        task.y_axis_mm[-1] - task.y_axis_mm[0] + task.spacing_mm,
        task.z_axis_mm[-1] - task.z_axis_mm[0] + task.spacing_mm,
    ))))
    distances = jnp.arange(0.0, ray_length + task.spacing_mm, ray_step, dtype=jnp.float32)

    def sample(field: object, points: object) -> object:
        axes = (x_axis, y_axis, z_axis)
        lower_indices, fractions = [], []
        for dimension, axis in enumerate(axes):
            index = jnp.searchsorted(axis, points[..., dimension], side="right") - 1
            index = jnp.clip(index, 0, field.shape[dimension] - 2)
            upper_index = index + 1
            lower_indices.append(index)
            fractions.append((points[..., dimension] - axis[index]) / (axis[upper_index] - axis[index]))
        result = jnp.zeros(points.shape[:-1], dtype=jnp.float32)
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    weight = (
                        (1.0 - fractions[0] if ix == 0 else fractions[0])
                        * (1.0 - fractions[1] if iy == 0 else fractions[1])
                        * (1.0 - fractions[2] if iz == 0 else fractions[2])
                    )
                    result = result + weight * field[
                        lower_indices[0] + ix, lower_indices[1] + iy, lower_indices[2] + iz,
                    ]
        return result

    def query(current_sdf: object, action: object) -> tuple[object, object, object, object, object]:
        beam = jnp.array((
            -jnp.sin(action[3]),
            jnp.sin(action[2]) * jnp.cos(action[3]),
            -jnp.cos(action[2]) * jnp.cos(action[3]),
        ), dtype=jnp.float32)
        beam = beam / jnp.linalg.norm(beam)
        reference = jnp.array((action[0], action[1], task.plane_z_mm), dtype=jnp.float32)
        entry = reference - ((upper[2] - reference[2]) / -beam[2]) * beam
        ray_points = entry[None, :] + distances[:, None] * beam[None, :]
        ray_inside = jnp.all((ray_points >= lower) & (ray_points <= upper), axis=1)
        ray_phi = jnp.where(
            ray_inside, sample(current_sdf * truncation, jnp.clip(ray_points, lower, upper)), truncation,
        )
        crossing = (
            ray_inside[:-1] & ray_inside[1:]
            & (ray_phi[:-1] >= 0.0) & (ray_phi[1:] <= 0.0)
        )
        entry_inside_tissue = ray_inside[0] & (ray_phi[0] <= 0.0)
        has_contact = entry_inside_tissue | jnp.any(crossing)
        crossing_index = jnp.argmax(crossing.astype(jnp.int32))
        denominator = ray_phi[crossing_index + 1] - ray_phi[crossing_index]
        fraction = jnp.where(denominator != 0.0, -ray_phi[crossing_index] / denominator, 0.0)
        crossed_contact = entry + (distances[crossing_index] + fraction * ray_step) * beam
        contact = jnp.where(entry_inside_tissue, entry, crossed_contact)
        return beam, contact, has_contact, entry_inside_tissue, crossing_index

    return query


def build_transition(
    task: StaticTaskTensors,
    freeze_contact_state: bool = False,
) -> Callable[[object, object, object], tuple[object, object, object, object, object]]:
    """Build one lattice-fixed SDF pulse transition with explicit safety flags.

    ``freeze_contact_state`` changes derivatives only: contact sampling treats the
    SDF values as constants while retaining its ray-coordinate action gradient.
    """
    jax, jnp = _jax_modules()
    x_axis = jnp.asarray(task.x_axis_mm, dtype=jnp.float32)
    y_axis = jnp.asarray(task.y_axis_mm, dtype=jnp.float32)
    z_axis = jnp.asarray(task.z_axis_mm, dtype=jnp.float32)
    grid = jnp.stack(jnp.meshgrid(x_axis, y_axis, z_axis, indexing="ij"), axis=-1)
    action_lower = jnp.asarray(task.lower_bounds, dtype=jnp.float32)
    action_upper = jnp.asarray(task.upper_bounds, dtype=jnp.float32)
    initial_tissue = jnp.asarray(task.initial_tissue)
    target = jnp.asarray(task.target_mask)
    constraint = jnp.asarray(task.constraint_mask)
    clearance_field = jnp.asarray(task.physical_clearance_mm, dtype=jnp.float32)
    truncation = jnp.float32(task.truncation_mm)
    hard_margin = jnp.float32(task.hard_margin_mm)
    power = jnp.float32(task.physics.super_gaussian_power)
    spot_squared = jnp.float32(task.physics.spot_size_mm**2)
    threshold = jnp.float32(task.physics.ablation_threshold)
    depth_scale = jnp.float32(task.physics.depth_scale_mm_per_j)
    initial_target_count = jnp.float32(task.initial_target_voxels)
    contact_query = build_contact_query(task)

    if freeze_contact_state:
        @jax.custom_jvp
        def radial_distance(squared: object) -> object:
            return jnp.sqrt(jnp.maximum(squared, 0.0))

        @radial_distance.defjvp
        def radial_distance_jvp(
            primals: tuple[object], tangents: tuple[object]
        ) -> tuple[object, object]:
            squared, = primals
            tangent, = tangents
            value = jnp.sqrt(jnp.maximum(squared, 0.0))
            denominator = jnp.where(value > 0.0, 2.0 * value, 1.0)
            return value, jnp.where(squared > 0.0, tangent / denominator, 0.0)
    else:
        def radial_distance(squared: object) -> object:
            return jnp.sqrt(squared)

    def pulse(current_sdf: object, action: object, active: object):
        """Apply one active pulse, or leave the state and cumulative quantities inert."""
        def apply(_: object):
            contact_sdf = (
                jax.lax.stop_gradient(current_sdf) if freeze_contact_state else current_sdf
            )
            beam, contact, has_contact, _, _ = contact_query(contact_sdf, action)

            offset = grid - contact
            axial = jnp.sum(offset * beam, axis=-1)
            radial_squared = jnp.maximum(jnp.sum(offset * offset, axis=-1) - axial**2, 0.0)
            fluence = action[4] * jnp.exp(-((radial_squared / (2.0 * spot_squared)) ** power))
            depth = depth_scale * jnp.maximum(fluence - threshold, 0.0)
            support_energy = jnp.maximum(action[4] / threshold, 1.0)
            support_radius = jnp.sqrt(
                2.0 * spot_squared * (jnp.log(support_energy) ** (1.0 / power))
            )
            crater_sdf = jnp.maximum(
                jnp.maximum(
                    -(axial + 0.5 * task.spacing_mm), axial - depth
                ),
                radial_distance(radial_squared) - support_radius,
            )
            crater_next = jnp.maximum(current_sdf * truncation, -crater_sdf) / truncation
            next_sdf = jnp.where(has_contact, crater_next, current_sdf)
            was_tissue = current_sdf <= 0.0
            is_tissue = next_sdf <= 0.0
            removal = initial_tissue & was_tissue & ~is_tissue
            removed = initial_tissue & ~is_tissue
            newly_removed = jnp.count_nonzero(removal) > 0
            protected_overlap = jnp.any(removal & constraint)
            clearance = jnp.min(
                jnp.where(removal, clearance_field, jnp.asarray(jnp.inf, dtype=jnp.float32))
            )
            out_of_bounds = jnp.any(action < action_lower) | jnp.any(action > action_upper)
            nonfinite = ~jnp.all(jnp.isfinite(next_sdf)) | ~jnp.all(jnp.isfinite(action))
            flags = jnp.array(
                (
                    out_of_bounds,
                    ~has_contact,
                    ~newly_removed,
                    protected_overlap,
                    clearance < hard_margin,
                    nonfinite,
                )
            )
            remaining_fraction = jnp.count_nonzero(is_tissue & target) / initial_target_count
            overcut_fraction = jnp.count_nonzero(removed & ~target) / initial_target_count
            return next_sdf.astype(jnp.float32), flags, clearance, remaining_fraction, overcut_fraction

        def inert(_: object):
            return (
                current_sdf,
                jnp.zeros(len(FLAG_NAMES), dtype=jnp.bool_),
                jnp.asarray(jnp.inf, dtype=jnp.float32),
                jnp.count_nonzero((current_sdf <= 0.0) & target) / initial_target_count,
                jnp.count_nonzero(initial_tissue & (current_sdf > 0.0) & ~target)
                / initial_target_count,
            )

        return jax.lax.cond(active, apply, inert, operand=None)

    return pulse
