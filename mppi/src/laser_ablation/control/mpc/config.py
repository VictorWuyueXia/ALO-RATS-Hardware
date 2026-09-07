"""Validated development configuration for energy-only CasADi MPC.

This module deliberately has no CasADi import. Physical constants, action
bounds, and acceptance gates are read from the shared project YAML files
instead of being copied into the MPC configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from laser_ablation.config import bounds_from_mapping, load_yaml, physics_from_mapping


@dataclass(frozen=True)
class WeightSchedule:
    """One normalized set of dimensionless MPC objective weights."""

    remaining: float
    overcut: float
    tracking: float
    energy: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in asdict(self).values())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("MPC weights must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("remaining/overcut/tracking/energy weights must sum to one")

    def interpolate(self, other: "WeightSchedule", progress: float) -> "WeightSchedule":
        """Linearly interpolate two normalized schedules at progress in [0, 1]."""
        rho = float(progress)
        if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ValueError("treatment progress must be finite and lie in [0, 1]")
        return WeightSchedule(
            remaining=(1.0 - rho) * self.remaining + rho * other.remaining,
            overcut=(1.0 - rho) * self.overcut + rho * other.overcut,
            tracking=(1.0 - rho) * self.tracking + rho * other.tracking,
            energy=(1.0 - rho) * self.energy + rho * other.energy,
        )


@dataclass(frozen=True, kw_only=True)
class EnergyMPCConfig:
    """Fixed DEVELOPMENT settings for the energy-only local NLP.

    The completion and hard-margin values are injected from the shared planner
    config by :func:`load_energy_mpc_config`; they are not MPC tuning knobs.
    """

    completion_remaining_pct: float
    hard_margin_mm: float
    horizon: int = 5
    d_max_mm: float = 2.0
    roi_halo_mm: float = 0.2
    roi_closure_voxels: int = 1
    occupancy_smoothing_mm: float = 0.0001
    crater_inclusion_epsilon_mm: float = 0.0001
    tracking_base_weight: float = 0.25
    tracking_boundary_band_mm: float = 0.2
    minimum_target_removal_fraction: float = 1e-4
    energy_trust_region_j: float = 0.02
    terminal_remaining_weight: float = 1.0
    clearance_samples_per_axis: int = 7
    solver_tolerance: float = 1e-8
    solver_max_iterations: int = 300
    ipopt_print_level: int = 0
    weight_early: WeightSchedule = WeightSchedule(0.50, 0.15, 0.25, 0.10)
    weight_late: WeightSchedule = WeightSchedule(0.25, 0.40, 0.25, 0.10)

    def __post_init__(self) -> None:
        _positive_integer("horizon", self.horizon)
        _positive("d_max_mm", self.d_max_mm)
        _nonnegative("roi_halo_mm", self.roi_halo_mm)
        _nonnegative_integer("roi_closure_voxels", self.roi_closure_voxels)
        _positive("occupancy_smoothing_mm", self.occupancy_smoothing_mm)
        _nonnegative("crater_inclusion_epsilon_mm", self.crater_inclusion_epsilon_mm)
        _nonnegative("tracking_base_weight", self.tracking_base_weight)
        _positive("tracking_boundary_band_mm", self.tracking_boundary_band_mm)
        _nonnegative("minimum_target_removal_fraction", self.minimum_target_removal_fraction)
        _positive("energy_trust_region_j", self.energy_trust_region_j)
        if self.minimum_target_removal_fraction > 1.0:
            raise ValueError("minimum_target_removal_fraction cannot exceed one")
        _nonnegative("terminal_remaining_weight", self.terminal_remaining_weight)
        if self.horizon == 1 and not math.isclose(
            self.terminal_remaining_weight, 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "H=1 already includes post-action Remaining in the stage cost; "
                "terminal_remaining_weight must be zero to avoid double-counting"
            )
        _percentage("completion_remaining_pct", self.completion_remaining_pct)
        _nonnegative("hard_margin_mm", self.hard_margin_mm)
        _positive_integer("clearance_samples_per_axis", self.clearance_samples_per_axis)
        if self.clearance_samples_per_axis < 2:
            raise ValueError("clearance_samples_per_axis must be at least two")
        _positive("solver_tolerance", self.solver_tolerance)
        _positive_integer("solver_max_iterations", self.solver_max_iterations)
        _nonnegative_integer("ipopt_print_level", self.ipopt_print_level)
        if not isinstance(self.weight_early, WeightSchedule) or not isinstance(
            self.weight_late, WeightSchedule
        ):
            raise TypeError("weight_early and weight_late must be WeightSchedule values")

    def weights_at(self, progress: float) -> WeightSchedule:
        """Return the normalized early/late interpolation at treatment progress."""
        return self.weight_early.interpolate(self.weight_late, progress)


@dataclass(frozen=True)
class MPCFingerprints:
    """Semantic hashes of the shared physics and safety authorities."""

    physics: str
    safety: str

    def __post_init__(self) -> None:
        for name, value in (("physics", self.physics), ("safety", self.safety)):
            _validate_sha256(f"{name} fingerprint", value)


def load_energy_mpc_config(
    path: str | Path, *, repository_root: str | Path | None = None
) -> EnergyMPCConfig:
    """Load development settings and inject the shared acceptance gates."""
    experiment_path = Path(path).resolve()
    values = load_yaml(experiment_path)
    shared = _mapping(values, "shared_configs")
    tunable = dict(_mapping(values, "TUNABLE_DEVELOPMENT_HYPERPARAMETERS"))
    root = Path(repository_root).resolve() if repository_root else _find_root(experiment_path)
    planner_values = load_yaml(_shared_path(root, shared, "planner"))
    tunable["completion_remaining_pct"] = planner_values["completion_remaining_pct"]
    tunable["hard_margin_mm"] = planner_values["hard_margin_mm"]
    tunable["weight_early"] = WeightSchedule(**_mapping(tunable, "weight_early"))
    tunable["weight_late"] = WeightSchedule(**_mapping(tunable, "weight_late"))
    return EnergyMPCConfig(**tunable)


def fingerprints_from_shared_configs(
    physics_config: str | Path, planner_config: str | Path
) -> MPCFingerprints:
    """Hash only semantics that can change physics or terminal acceptance."""
    physics_values = load_yaml(physics_config)
    planner_values = load_yaml(planner_config)
    physics_payload = {
        "physics": asdict(physics_from_mapping(physics_values)),
        "action_bounds": asdict(bounds_from_mapping(physics_values)),
    }
    safety_payload = {
        "completion_remaining_pct": float(planner_values["completion_remaining_pct"]),
        "hard_margin_mm": float(planner_values["hard_margin_mm"]),
    }
    return MPCFingerprints(
        physics=_semantic_sha256("laser_ablation_mpc_physics_v1", physics_payload),
        safety=_semantic_sha256("laser_ablation_mpc_safety_v1", safety_payload),
    )


def energy_mpc_config_fingerprint(config: EnergyMPCConfig) -> str:
    """Return a deterministic semantic hash of all effective MPC settings."""
    return _semantic_sha256("laser_ablation_energy_mpc_config_v1", asdict(config))


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = values.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"configuration field {key!r} must be a mapping")
    return result


def _find_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"cannot locate repository root above {path}")


def _shared_path(root: Path, values: Mapping[str, Any], key: str) -> Path:
    raw = values.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"shared_configs.{key} must be a nonempty path")
    path = (root / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _semantic_sha256(namespace: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    namespace_bytes = namespace.encode("ascii") + bytes((0,))
    return hashlib.sha256(namespace_bytes + encoded).hexdigest()


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _percentage(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0:
        raise ValueError(f"{name} must be a finite percentage in [0, 100]")


__all__ = [
    "EnergyMPCConfig",
    "MPCFingerprints",
    "WeightSchedule",
    "energy_mpc_config_fingerprint",
    "fingerprints_from_shared_configs",
    "load_energy_mpc_config",
]
