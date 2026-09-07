"""Pure JAX correlated sampling and projection for multi-anchor plan repair."""

from __future__ import annotations


def sample_correlated_actions(
    anchor_keys: object,
    nominal_actions: object,
    action_mask: object,
    perturb_mask: object,
    lower_bounds: object,
    upper_bounds: object,
    samples_per_anchor: int,
    beta: float,
    kappa: object,
) -> tuple[object, object]:
    """Generate a zero control sample and AR(1) perturbations for every anchor."""
    import jax
    import jax.numpy as jnp

    anchor_count, pulse_count, action_dimension = nominal_actions.shape
    sample_indices = jnp.arange(1, samples_per_anchor, dtype=jnp.uint32)

    def draw_anchor(key: object) -> object:
        sample_keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(sample_indices)
        draws = jax.vmap(
            lambda sample_key: jax.random.normal(
                sample_key, (pulse_count, action_dimension), dtype=jnp.float32
            )
        )(sample_keys)
        return jnp.transpose(draws, (1, 0, 2))

    white = jnp.transpose(jax.vmap(draw_anchor)(anchor_keys), (1, 0, 2, 3))
    lengths = jnp.sum(action_mask, axis=1, dtype=jnp.int32)
    pulse_index = jnp.arange(pulse_count, dtype=jnp.float32)[None, :, None]
    denominator = jnp.maximum(lengths[:, None, None] - 1, 1).astype(jnp.float32)
    pulse_scale = jnp.where(
        lengths[:, None, None] == 1,
        jnp.float32(0.25),
        jnp.float32(0.25) + jnp.float32(0.75) * pulse_index / denominator,
    )
    sigma = jnp.swapaxes(pulse_scale, 0, 1) * jnp.asarray(kappa, dtype=jnp.float32) * (
        upper_bounds - lower_bounds
    )
    innovation_scale = (
        jnp.sqrt(jnp.float32(1.0 - beta**2)) * sigma[:, :, None, :]
    )

    enabled = jnp.swapaxes(perturb_mask, 0, 1)

    def advance(previous: object, inputs: tuple[object, object, object]) -> tuple[object, object]:
        independent, scale, is_perturbed = inputs
        perturbation = jnp.float32(beta) * previous + scale * independent
        perturbation = jnp.where(
            is_perturbed[:, None, None], perturbation, jnp.float32(0.0)
        )
        return perturbation, perturbation

    initial = jnp.zeros((anchor_count, samples_per_anchor - 1, action_dimension), jnp.float32)
    _, random_perturbations = jax.lax.scan(
        advance, initial, (white, innovation_scale, enabled)
    )
    random_perturbations = jnp.transpose(random_perturbations, (1, 2, 0, 3))
    zero = jnp.zeros((anchor_count, 1, pulse_count, action_dimension), jnp.float32)
    perturbations = jnp.concatenate((zero, random_perturbations), axis=1)
    active = action_mask[:, None, :, None]
    perturbed = active & perturb_mask[:, None, :, None]
    perturbations = jnp.where(perturbed, perturbations, 0.0)
    projected = jnp.clip(
        nominal_actions[:, None, :, :] + perturbations,
        lower_bounds,
        upper_bounds,
    )
    return jnp.where(active, projected, 0.0), perturbations
