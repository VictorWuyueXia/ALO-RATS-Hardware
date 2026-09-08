"""Configured full-raster plan source shared by synthetic scenarios."""

from __future__ import annotations

from typing import Any, Mapping

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.state import VoxelState
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.interface import PlanCandidate


class ConfiguredRasterPlanGenerator:
    """Build four configured raw raster action sequences without feasibility screening."""

    def __init__(
        self, scenario_name: str, scenario: Mapping[str, Any], simulator: ExactVoxelSimulator
    ) -> None:
        self.scenario_name = scenario_name
        self.simulator = simulator
        self.grid_pitches_xy_mm = tuple(
            tuple(map(float, pair))
            for pair in scenario["candidate_grid_pitches_xy_mm"]
        )
        self.grid_shape = tuple(int(value) for value in scenario["candidate_grid_shape"])
        self.grid_center_xy_mm = tuple(
            map(float, scenario["candidate_grid_center_xy_mm"])
        )
        self.depth_repetitions = int(scenario["candidate_depth_repetitions"])
        self.energies_j = tuple(float(value) for value in scenario["candidate_energies_j"])
        if (
            len(self.grid_pitches_xy_mm) != 2
            or any(len(pair) != 2 for pair in self.grid_pitches_xy_mm)
            or len(self.energies_j) != 2
        ):
            raise ValueError("raster plan bank requires two XY pitch pairs and two energies")
        if (
            len(self.grid_shape) != 2 or len(self.grid_center_xy_mm) != 2
            or min(self.grid_shape) <= 0
            or any(not side % 2 for side in self.grid_shape)
            or self.depth_repetitions <= 0
        ):
            raise ValueError("raster shape must contain positive odd sides")
        if min(map(min, self.grid_pitches_xy_mm)) <= 0.0 or min(self.energies_j) <= 0.0:
            raise ValueError("candidate raster pitches and energies must be positive")
    def generate(self, state: VoxelState) -> tuple[PlanCandidate, ...]:
        """Return all configured raw rasters without voxel simulation or filtering."""
        candidates: list[PlanCandidate] = []
        x_offsets = tuple(index - self.grid_shape[0] // 2 for index in range(self.grid_shape[0]))
        y_offsets = tuple(index - self.grid_shape[1] // 2 for index in range(self.grid_shape[1]))
        for pitch_x_mm, pitch_y_mm in self.grid_pitches_xy_mm:
            for energy_j in self.energies_j:
                raster = tuple(
                    PhysicalAction(
                        self.grid_center_xy_mm[0] + x_index * pitch_x_mm,
                        self.grid_center_xy_mm[1] + y_index * pitch_y_mm,
                        0.0, 0.0, energy_j,
                    )
                    for _ in range(self.depth_repetitions)
                    for row, y_index in enumerate(y_offsets)
                    for x_index in (x_offsets if row % 2 == 0 else x_offsets[::-1])
                )
                candidate = PlanCandidate(
                    raster,
                    f"{self.scenario_name}_raster:px={pitch_x_mm:.6g}:py={pitch_y_mm:.6g}",
                    "vertical",
                    f"uniform:{energy_j:.6g}",
                    1.0,
                    1.0,
                    "serpentine",
                )
                candidates.append(candidate)
                print(
                    f"global candidate scenario={self.scenario_name} "
                    f"pitch=({pitch_x_mm:.3f},{pitch_y_mm:.3f}) "
                    f"energy={energy_j:.3f} configured_pulses={len(raster)}",
                    flush=True,
                )
        return tuple(candidates)


__all__ = ["ConfiguredRasterPlanGenerator"]
