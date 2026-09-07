"""Canonical five-dimensional physical laser action and its bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalAction:
    """One pulse ``[x, y, tilt_x, tilt_y, energy]`` in mm, rad, and joules."""

    x_mm: float
    y_mm: float
    tilt_x_rad: float
    tilt_y_rad: float
    energy_j: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x_mm, self.y_mm, self.tilt_x_rad, self.tilt_y_rad, self.energy_j],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "PhysicalAction":
        vector = np.asarray(values, dtype=float)
        if vector.shape != (5,) or not np.all(np.isfinite(vector)):
            raise ValueError("a physical action must contain five finite values")
        return cls(*map(float, vector))


@dataclass(frozen=True)
class PhysicalActionBounds:
    """Single source of truth for valid physical actions."""

    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    tilt_x_rad: tuple[float, float]
    tilt_y_rad: tuple[float, float]
    energy_j: tuple[float, float]

    @property
    def lower(self) -> np.ndarray:
        return np.array(
            [self.x_mm[0], self.y_mm[0], self.tilt_x_rad[0], self.tilt_y_rad[0], self.energy_j[0]],
            dtype=float,
        )

    @property
    def upper(self) -> np.ndarray:
        return np.array(
            [self.x_mm[1], self.y_mm[1], self.tilt_x_rad[1], self.tilt_y_rad[1], self.energy_j[1]],
            dtype=float,
        )

    def contains(self, action: PhysicalAction) -> bool:
        values = action.as_array()
        return bool(np.all(values >= self.lower) and np.all(values <= self.upper))

    @classmethod
    def from_mapping(cls, values: dict[str, list[float]]) -> "PhysicalActionBounds":
        return cls(**{key: tuple(map(float, item)) for key, item in values.items()})
