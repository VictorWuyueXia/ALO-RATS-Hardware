"""Differentiable, normalized metric expressions for development MPC.

The expressions in this module are proposal quantities only. Headline metrics
remain the output of :func:`laser_ablation.metrics.evaluate_ablation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class SymbolicAblationMetrics:
    """CasADi scalar expressions evaluated on one deterministic ROI."""

    remaining_fraction: ca.MX
    total_overcut_fraction: ca.MX
    target_bottom_overcut_fraction: ca.MX
    normal_damage_fraction: ca.MX


def tissue_probability(phi_mm: ca.MX, smoothing_mm: float) -> ca.MX:
    """Smooth negative-inside occupancy used only by the analytic NLP."""
    if not np.isfinite(smoothing_mm) or smoothing_mm <= 0.0:
        raise ValueError("occupancy smoothing must be finite and positive")
    return 0.5 * (1.0 - ca.tanh(phi_mm / float(smoothing_mm)))


def normalized_metrics(
    phi_mm: ca.MX,
    *,
    target_selector: np.ndarray,
    bottom_selector: np.ndarray,
    normal_selector: np.ndarray,
    voxel_volume_mm3: float,
    initial_target_volume_mm3: float,
    outside_remaining_volume_mm3: float,
    outside_bottom_volume_mm3: float,
    outside_normal_volume_mm3: float,
    smoothing_mm: float,
) -> SymbolicAblationMetrics:
    """Return Remaining and both official collateral components.

    Every outside-target initial-tissue voxel is labelled as exactly one of
    target-column bottom overcut or normal-tissue damage. This deliberately
    avoids the historical error of excluding target columns from collateral.
    """
    if initial_target_volume_mm3 <= 0.0 or voxel_volume_mm3 <= 0.0:
        raise ValueError("metric volumes must be positive")
    selectors = tuple(
        ca.DM(np.asarray(value, dtype=float).reshape(-1, 1))
        for value in (target_selector, bottom_selector, normal_selector)
    )
    if any(int(value.numel()) != int(phi_mm.numel()) for value in selectors):
        raise ValueError("metric selectors must match the ROI state size")
    occupancy = tissue_probability(phi_mm, smoothing_mm)
    removed = 1.0 - occupancy
    scale = float(voxel_volume_mm3 / initial_target_volume_mm3)
    remaining = (
        float(outside_remaining_volume_mm3 / initial_target_volume_mm3)
        + scale * ca.dot(selectors[0], occupancy)
    )
    bottom = (
        float(outside_bottom_volume_mm3 / initial_target_volume_mm3)
        + scale * ca.dot(selectors[1], removed)
    )
    normal = (
        float(outside_normal_volume_mm3 / initial_target_volume_mm3)
        + scale * ca.dot(selectors[2], removed)
    )
    return SymbolicAblationMetrics(remaining, bottom + normal, bottom, normal)


def normalized_sdf_tracking(
    phi_mm: ca.MX,
    reference_phi_mm: np.ndarray,
    weights: np.ndarray,
    d_max_mm: float,
) -> ca.MX:
    """Compute the bounded reference-SDF term specified for the MPC."""
    reference = ca.DM(np.asarray(reference_phi_mm, dtype=float).reshape(-1, 1))
    weight = ca.DM(np.asarray(weights, dtype=float).reshape(-1, 1))
    count = int(phi_mm.numel())
    if count <= 0 or int(reference.numel()) != count or int(weight.numel()) != count:
        raise ValueError("tracking arrays must have the same nonzero size")
    raw_weights = np.asarray(weights)
    if d_max_mm <= 0.0 or np.any(raw_weights < 0.0) or np.any(raw_weights > 1.0):
        raise ValueError("tracking normalization and weights are invalid")
    normalized = ca.fmin(ca.fmax(phi_mm / float(d_max_mm), -1.0), 1.0)
    normalized_reference = ca.fmin(
        ca.fmax(reference / float(d_max_mm), -1.0), 1.0
    )
    difference = weight * (normalized - normalized_reference)
    return ca.dot(difference, difference) / float(4 * count)
