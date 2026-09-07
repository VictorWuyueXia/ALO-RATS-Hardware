"""Vectorized MPPI tail cost, feasible softmax, and anchor update primitives."""

from __future__ import annotations


def tail_costs(
    sampled_actions: object,
    origin_actions: object,
    action_mask: object,
    remaining_fraction: object,
    healthy_overcut_fraction: object,
    lower_bounds: object,
    upper_bounds: object,
    weights: tuple[float, float, float],
) -> tuple[object, object]:
    """Evaluate completion, healthy overcut, and normalized origin deviation."""
    import jax.numpy as jnp

    active = action_mask[:, None, :, None]
    normalized_delta = (sampled_actions - origin_actions[:, None]) / (
        upper_bounds - lower_bounds
    )
    squared = jnp.sum(jnp.where(active, normalized_delta**2, 0.0), axis=(2, 3))
    lengths = jnp.sum(action_mask, axis=1, keepdims=True)
    deviation = squared / (jnp.float32(5.0) * lengths)
    components = jnp.stack((remaining_fraction, healthy_overcut_fraction, deviation), axis=-1)
    cost = jnp.sum(components * jnp.asarray(weights, dtype=jnp.float32), axis=-1)
    return cost, components


def path_integral_weights(cost: object, feasible: object, temperature: float) -> object:
    """Normalize exponentiated relative costs over each anchor's feasible samples."""
    import jax.numpy as jnp

    minimum = jnp.min(jnp.where(feasible, cost, jnp.inf), axis=1, keepdims=True)
    logits = jnp.where(feasible, -(cost - minimum) / jnp.float32(temperature), -jnp.inf)
    maximum = jnp.max(jnp.where(feasible, logits, -jnp.inf), axis=1, keepdims=True)
    exponent = jnp.where(feasible, jnp.exp(logits - maximum), 0.0)
    denominator = jnp.sum(exponent, axis=1, keepdims=True)
    return jnp.where(denominator > 0.0, exponent / denominator, 0.0)


def update_nominal(
    nominal_actions: object,
    perturbations: object,
    sample_weights: object,
    action_mask: object,
    lower_bounds: object,
    upper_bounds: object,
) -> object:
    """Apply a path-integral perturbation mean without making it selectable."""
    import jax.numpy as jnp

    correction = jnp.sum(sample_weights[:, :, None, None] * perturbations, axis=1)
    updated = jnp.clip(nominal_actions + correction, lower_bounds, upper_bounds)
    return jnp.where(action_mask[:, :, None], updated, 0.0)
