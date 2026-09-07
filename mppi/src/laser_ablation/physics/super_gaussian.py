"""Canonical calibrated Super-Gaussian ablation profile."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsConfig:
    tissue_density: float = 1.0
    ablation_threshold: float = 1.939
    specific_enthalpy: float = 1.0 / 0.333885349
    spot_size_mm: float = 0.483187452
    super_gaussian_power: float = 12.72661
    min_energy_j: float = 2.0
    max_energy_j: float = 14.0

    @property
    def depth_scale_mm_per_j(self) -> float:
        return 1.0 / (self.tissue_density * self.specific_enthalpy)


def laser_axis(tilt_x_rad: float, tilt_y_rad: float) -> np.ndarray:
    """Unit propagation direction for the project's ordered x/y rotations."""
    direction = np.array(
        [
            -np.sin(tilt_y_rad),
            np.sin(tilt_x_rad) * np.cos(tilt_y_rad),
            -np.cos(tilt_x_rad) * np.cos(tilt_y_rad),
        ]
    )
    return direction / np.linalg.norm(direction)


def super_gaussian_depth(
    radial_squared_mm2: np.ndarray | float,
    energy_j: float,
    config: PhysicsConfig,
    response_scale: float = 1.0,
) -> np.ndarray:
    """Ablation depth along the beam at squared radial offset ``r^2``."""
    radial = np.asarray(radial_squared_mm2, dtype=float)
    fluence = energy_j * np.exp(
        -((radial / (2.0 * config.spot_size_mm**2)) ** config.super_gaussian_power)
    )
    return response_scale * config.depth_scale_mm_per_j * np.maximum(
        fluence - config.ablation_threshold, 0.0
    )
