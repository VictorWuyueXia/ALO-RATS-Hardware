"""One authoritative definition of completion, collateral damage, and safety."""

import numpy as np
from scipy.ndimage import distance_transform_edt

from laser_ablation.core.metrics import AblationMetrics
from laser_ablation.core.state import VoxelState


_CLEARANCE_FIELDS: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}


def evaluate_ablation(state: VoxelState) -> AblationMetrics:
    """Evaluate exact Boolean occupancy against immutable 3-D task masks.

    Total overcut is all removed initial tissue outside the volumetric target.
    The target-bottom/normal split uses target XY support and is diagnostic only.
    """
    voxel_volume = state.voxel_volume_mm3
    target_volume = state.initial_target_volume_mm3
    if target_volume <= 0.0:
        raise ValueError("the exact metric requires a nonempty target")
    remaining = state.tissue & state.target_mask
    removed = state.initial_tissue & ~state.tissue
    outside_target = removed & ~state.target_mask
    target_columns = np.any(state.target_mask, axis=2)[:, :, None]
    target_bottom = outside_target & target_columns
    normal_damage = outside_target & ~target_columns

    remaining_volume = np.count_nonzero(remaining) * voxel_volume
    total_overcut = np.count_nonzero(outside_target) * voxel_volume
    bottom_volume = np.count_nonzero(target_bottom) * voxel_volume
    normal_volume = np.count_nonzero(normal_damage) * voxel_volume
    removed_volume = np.count_nonzero(removed) * voxel_volume
    violations = int(np.count_nonzero(state.constraint_mask & ~state.tissue))
    clearance = _minimum_clearance_mm(state, removed)

    return AblationMetrics(
        float(remaining_volume),
        100.0 * remaining_volume / target_volume,
        float(total_overcut),
        100.0 * total_overcut / target_volume,
        float(bottom_volume),
        100.0 * bottom_volume / target_volume,
        float(normal_volume),
        100.0 * normal_volume / target_volume,
        float(removed_volume),
        float(clearance),
        violations,
    )


def _minimum_clearance_mm(state: VoxelState, removed: np.ndarray) -> float:
    if not np.any(removed) or not np.any(state.constraint_mask):
        return float("inf")
    cache_key = id(state.constraint_mask)
    cached = _CLEARANCE_FIELDS.get(cache_key)
    if (
        cached is None
        or cached[0] is not state.constraint_mask
        or cached[1] != state.spacing_mm
    ):
        cached = (
            state.constraint_mask,
            state.spacing_mm,
            distance_transform_edt(~state.constraint_mask, sampling=state.spacing_mm),
        )
        _CLEARANCE_FIELDS[cache_key] = cached
    # Subtract the two voxel half-widths to convert centre distance to box gap.
    return float(np.min(cached[2][removed]) - state.spacing_mm)
