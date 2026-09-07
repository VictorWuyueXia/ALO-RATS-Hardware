"""Configured scenario registry for the unified controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from laser_ablation.geometry.base import VoxelGeometry
from laser_ablation.geometry.scenarios.cavity import build as build_cavity
from laser_ablation.geometry.scenarios.cavity_sidewall import build as build_cavity_sidewall
from laser_ablation.geometry.scenarios.concave_volume import build as build_concave_volume
from laser_ablation.geometry.scenarios.flat_box import build as build_flat_box
from laser_ablation.geometry.scenarios.mature_rectangle import build as build_mature_rectangle
from laser_ablation.geometry.scenarios.oblique_surface import build as build_oblique_surface
from laser_ablation.geometry.scenarios.protected_boundary import build as build_protected_boundary
from laser_ablation.geometry.scenarios.repeat_depth import build as build_repeat_depth
from laser_ablation.geometry.scenarios.square import build as build_square
from laser_ablation.geometry.scenarios.stepped_depth import build as build_stepped_depth
from laser_ablation.geometry.scenarios.stepped_target import build as build_stepped_target
from laser_ablation.geometry.scenarios.two_pulse_wide import build as build_two_pulse_wide
from laser_ablation.geometry.scenarios.vertical_control import build as build_vertical_control


@dataclass(frozen=True)
class ScenarioBundle:
    """One controller-ready geometry and its auditable configuration record."""

    name: str
    geometry: VoxelGeometry
    configuration: Mapping[str, Any]


class ScenarioNotEnabled(RuntimeError):
    """Typed boundary status for a valid reserved scenario contract."""


def _brats(values: Mapping[str, Any]) -> VoxelGeometry:
    required = (
        "segmentation_path",
        "protection_model",
        "crop_margin_mm",
        "isotropic_scale_factor",
    )
    if any(name not in values for name in required):
        raise ValueError("BraTS scenario configuration is incomplete")
    raise ScenarioNotEnabled("BraTS unified planning is configured but not yet enabled")


SCENARIO_BUILDERS: dict[str, Callable[[Mapping[str, Any]], VoxelGeometry]] = {
    "square": build_square,
    "flat_box": build_flat_box,
    "cavity": build_cavity,
    "stepped_target": build_stepped_target,
    "vertical_control": build_vertical_control,
    "oblique_surface": build_oblique_surface,
    "concave_volume": build_concave_volume,
    "stepped_depth": build_stepped_depth,
    "cavity_sidewall": build_cavity_sidewall,
    "two_pulse_wide": build_two_pulse_wide,
    "repeat_depth": build_repeat_depth,
    "protected_boundary": build_protected_boundary,
    "mature_rectangle": build_mature_rectangle,
    "brats": _brats,
}


def build_scenario(name: str, values: Mapping[str, Any]) -> ScenarioBundle:
    """Build exactly the selected scenario without substituting geometry."""
    if name not in SCENARIO_BUILDERS:
        raise ValueError(f"unknown scenario builder: {name}")
    if not bool(values["enabled"]):
        if name == "brats":
            _brats(values)
        raise RuntimeError(f"scenario {name!r} is disabled")
    return ScenarioBundle(name, SCENARIO_BUILDERS[name](values), dict(values))


__all__ = ["SCENARIO_BUILDERS", "ScenarioBundle", "ScenarioNotEnabled", "build_scenario"]
