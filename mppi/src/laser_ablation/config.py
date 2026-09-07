"""Small YAML-to-dataclass configuration boundary."""

from pathlib import Path
from typing import Any

import yaml

from laser_ablation.core.actions import PhysicalActionBounds
from laser_ablation.physics.super_gaussian import PhysicsConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    return values


def physics_from_mapping(values: dict[str, Any]) -> PhysicsConfig:
    return PhysicsConfig(**values["physics"])


def bounds_from_mapping(values: dict[str, Any]) -> PhysicalActionBounds:
    bounds = dict(values["action_bounds"])
    physics = values["physics"]
    bounds["energy_j"] = [physics["min_energy_j"], physics["max_energy_j"]]
    return PhysicalActionBounds.from_mapping(bounds)
