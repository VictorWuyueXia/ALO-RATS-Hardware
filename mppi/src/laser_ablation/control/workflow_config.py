"""Strict configuration boundary for the unified paper-workflow controller."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from laser_ablation.config import load_yaml
from laser_ablation.planning.jax_bank.repair import MPPIRepairConfig


@dataclass(frozen=True)
class ComputeProfile:
    """One explicit JAX platform, device set, and deterministic rollout batch size."""

    name: str
    platform: str
    device_indices: tuple[int, ...]
    rollout_batch_size: int
    preallocate: bool

    def __post_init__(self) -> None:
        if self.platform not in {"cpu", "gpu"}:
            raise ValueError("compute platform must be cpu or gpu")
        if not self.device_indices or min(self.device_indices) < 0:
            raise ValueError("compute profile requires nonnegative device indices")
        if len(set(self.device_indices)) != len(self.device_indices):
            raise ValueError("compute device indices must be unique")
        if self.rollout_batch_size <= 0:
            raise ValueError("rollout batch size must be positive")
        if self.rollout_batch_size % len(self.device_indices):
            raise ValueError("rollout batch size must be divisible by device count")
        if not isinstance(self.preallocate, bool):
            raise TypeError("preallocate must be Boolean")

    def activate(self) -> tuple[object, ...]:
        """Select the requested JAX resources exactly, without backend fallback."""
        if "jax" in sys.modules:
            raise RuntimeError("compute profile must be activated before importing JAX")
        backend = "cpu" if self.platform == "cpu" else "cuda"
        if self.platform == "gpu":
            configured_devices = ",".join(map(str, self.device_indices))
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = configured_devices
            elif len(visible_devices.split(",")) != len(self.device_indices):
                raise RuntimeError(
                    f"compute profile {self.name!r} requires {len(self.device_indices)} "
                    f"visible GPU devices, received {visible_devices!r}"
                )
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(self.preallocate).lower()
        os.environ["JAX_PLATFORMS"] = backend
        import jax

        devices = tuple(jax.devices(backend))
        if len(devices) != len(self.device_indices):
            raise RuntimeError(
                f"compute profile {self.name!r} requires {len(self.device_indices)} "
                f"{self.platform} devices, found {len(devices)}"
            )
        return devices


@dataclass(frozen=True)
class UnifiedWorkflowConfig:
    """Fully resolved controller, compute, scenario, and shared-config settings."""

    experiment: str
    repository_root: Path
    compute: ComputeProfile
    scenario_name: str
    scenario: Mapping[str, Any]
    physics_path: Path
    planner_path: Path
    geometry_aware_global_path: Path
    energy_mpc_path: Path
    alignment_persistence: int
    periodic_repair_pulses: int
    maximum_pulses: int
    maximum_repairs_per_prefix: int
    mppi: MPPIRepairConfig

    def __post_init__(self) -> None:
        if min(
            self.alignment_persistence, self.periodic_repair_pulses, self.maximum_pulses,
            self.maximum_repairs_per_prefix,
        ) <= 0:
            raise ValueError("controller pulse and repair limits must be positive")


def load_unified_workflow_config(
    config_path: str | Path,
    compute_override: str | None = None,
    scenario_override: str | None = None,
) -> UnifiedWorkflowConfig:
    """Resolve the workflow with CLI precedence limited to profile selection."""
    path = Path(config_path).resolve()
    root = next(
        parent for parent in (path.parent, *path.parents)
        if (parent / "pyproject.toml").is_file()
    )
    values = load_yaml(path)
    compute_values = load_yaml(root / _string(values, "compute_config"))
    scenario_values = load_yaml(root / _string(values, "scenario_config"))
    compute_name = compute_override or _string(compute_values, "default_profile")
    scenario_name = scenario_override or _string(scenario_values, "default_scenario")
    profiles = _mapping(compute_values, "profiles")
    scenarios = _mapping(scenario_values, "scenarios")
    if compute_name not in profiles:
        raise ValueError(f"unknown compute profile: {compute_name}")
    if scenario_name not in scenarios:
        raise ValueError(f"unknown scenario: {scenario_name}")
    profile = _mapping(profiles, compute_name)
    scenario = dict(_mapping(scenarios, scenario_name))
    _validate_scenario(scenario_name, scenario)
    shared = _mapping(values, "shared_configs")
    physics_path = (root / _string(shared, "physics")).resolve()
    planner_path = (root / _string(shared, "planner")).resolve()
    geometry_aware_global_path = (root / _string(shared, "geometry_aware_global")).resolve()
    geometry_shared = _mapping(load_yaml(geometry_aware_global_path), "shared_configs")
    geometry_physics_path = (root / _string(geometry_shared, "physics")).resolve()
    geometry_planner_path = (root / _string(geometry_shared, "planner")).resolve()
    if geometry_physics_path != physics_path or geometry_planner_path != planner_path:
        raise ValueError(
            "geometry-aware global physics and planner paths must match the active exact authority"
        )
    mppi = dict(_mapping(values, "mppi"))
    mppi["kappa"] = tuple(float(value) for value in mppi["kappa"])
    return UnifiedWorkflowConfig(
        experiment=_string(values, "experiment"),
        repository_root=root,
        compute=ComputeProfile(
            compute_name,
            _string(profile, "platform"),
            tuple(int(index) for index in profile["device_indices"]),
            int(profile["rollout_batch_size"]),
            profile["preallocate"],
        ),
        scenario_name=scenario_name,
        scenario=scenario,
        physics_path=physics_path,
        planner_path=planner_path,
        geometry_aware_global_path=geometry_aware_global_path,
        energy_mpc_path=root / _string(shared, "energy_mpc"),
        alignment_persistence=int(values["alignment_persistence"]),
        periodic_repair_pulses=int(values["periodic_repair_pulses"]),
        maximum_pulses=int(values["maximum_pulses"]),
        maximum_repairs_per_prefix=int(values["maximum_repairs_per_prefix"]),
        mppi=MPPIRepairConfig(
            rollout_batch_size=int(profile["rollout_batch_size"]),
            **mppi,
        ),
    )


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = values.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"configuration field {key!r} must be a mapping")
    return result


def _string(values: Mapping[str, Any], key: str) -> str:
    result = values.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"configuration field {key!r} must be a nonempty string")
    return result


def _validate_scenario(name: str, values: Mapping[str, Any]) -> None:
    required = {
        "enabled", "kind", "voxel_spacing_mm", "candidate_grid_pitches_xy_mm",
        "candidate_grid_shape", "candidate_grid_center_xy_mm",
        "candidate_depth_repetitions", "candidate_energies_j",
        "physics_noise",
    }
    if name == "square":
        required.update({
            "workspace_xy_mm", "xy_voxels", "z_axis_mm", "target_half_width_mm",
        })
    if name == "brats":
        required = {
            "enabled", "kind", "segmentation_path", "protection_model",
            "crop_margin_mm", "isotropic_scale_factor", "physics_noise",
        }
    missing = required - set(values)
    if missing:
        raise ValueError(f"scenario {name!r} omits {sorted(missing)}")
    noise = _mapping(values, "physics_noise")
    if not isinstance(noise.get("enabled"), bool):
        raise ValueError("physics_noise.enabled must be Boolean")
    if float(noise.get("response_scale_standard_deviation", 0.0)) <= 0.0:
        raise ValueError("physics response noise standard deviation must be positive")
    bounds = noise.get("response_scale_bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("physics response scale bounds must contain two values")
    if not 0.0 < float(bounds[0]) < float(bounds[1]):
        raise ValueError("physics response scale bounds must be positive and increasing")
