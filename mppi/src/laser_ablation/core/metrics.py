"""Exact terminal and prefix metrics reported by every method."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationMetrics:
    remaining_volume_mm3: float
    remaining_pct: float
    total_overcut_volume_mm3: float
    total_overcut_pct: float
    target_bottom_overcut_volume_mm3: float
    target_bottom_overcut_pct: float
    normal_damage_volume_mm3: float
    normal_damage_pct: float
    removed_volume_mm3: float
    minimum_clearance_mm: float
    hard_violations: int

    def is_complete(self, threshold_pct: float = 10.0) -> bool:
        return self.remaining_pct <= threshold_pct
