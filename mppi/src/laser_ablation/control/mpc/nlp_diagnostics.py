"""Numerical decoding and size diagnostics for the beam energy NLP."""

from __future__ import annotations

import math
from typing import Mapping

import casadi as ca
import numpy as np

from laser_ablation.control.mpc.beam_geometry import BeamMPCROI
from laser_ablation.control.mpc.contracts import ConstraintValue


def unpack_diagnostics(
    values: tuple[ca.DM, ...], layout: Mapping[str, object], roi_size: int
) -> dict[str, object]:
    state_count = int(layout["state_count"])
    states = tuple(np.asarray(value, dtype=float).reshape(roi_size) for value in values[:state_count])
    names = tuple(layout["objective_names"])
    scalars = values[state_count : state_count + len(names)]
    objectives = {
        str(name): float(np.asarray(value).reshape(()))
        for name, value in zip(names, scalars, strict=True)
    }
    return {"predicted_states": states, "objective_terms": objectives}


def constraint_mapping(
    values: np.ndarray, layout: tuple[tuple[str, int], ...]
) -> dict[str, ConstraintValue]:
    result: dict[str, ConstraintValue] = {}
    cursor = 0
    for name, count in layout:
        segment = tuple(map(float, values[cursor : cursor + count]))
        result[name] = segment[0] if count == 1 else segment
        cursor += count
    if cursor != values.size:
        raise RuntimeError("constraint diagnostic layout is inconsistent")
    return result


def maximum_constraint_violation(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    below = np.maximum(lower - values, 0.0)
    above = np.maximum(values - upper, 0.0)
    return float(max(np.max(below, initial=0.0), np.max(above, initial=0.0)))


def ipopt_infeasibilities(stats: Mapping[str, object]) -> tuple[float, float]:
    iterations = stats.get("iterations")
    if not isinstance(iterations, Mapping):
        return math.nan, math.nan
    values = []
    for name in ("inf_pr", "inf_du"):
        raw = iterations.get(name)
        array = np.asarray((), dtype=float) if raw is None else np.asarray(raw, dtype=float).reshape(-1)
        values.append(float(array[-1]) if array.size else math.nan)
    return values[0], values[1]


def roi_numeric_bytes(roi: BeamMPCROI) -> int:
    arrays = (
        roi.indices, roi.points_mm, roi.flat_indices, roi.full_to_roi,
        roi.target_selector, roi.bottom_selector, roi.normal_selector, roi.tracking_weights,
    )
    return int(sum(array.nbytes for array in arrays))


__all__ = [
    "constraint_mapping", "ipopt_infeasibilities", "maximum_constraint_violation",
    "roi_numeric_bytes", "unpack_diagnostics",
]
