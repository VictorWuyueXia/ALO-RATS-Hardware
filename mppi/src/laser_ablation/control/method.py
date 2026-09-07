"""Resolve the existing MPPI method without creating a synthetic experiment scenario."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from laser_ablation.config import bounds_from_mapping, load_yaml, physics_from_mapping
from laser_ablation.control.factory import build_planning_components
from laser_ablation.control.workflow_config import ComputeProfile
from laser_ablation.planning.jax_bank.repair import MPPIRepairConfig


@dataclass(frozen=True)
class ResolvedMethod:
    physics: object
    bounds: object
    mppi: MPPIRepairConfig
    global_settings: dict
    completion_remaining_pct: float
    hard_margin_mm: float
    maximum_pulses: int
    periodic_repair_pulses: int
    compute_profile: str
    source_hashes: dict[str, str]
    source_values: dict

    def components(self, case_name, raster_settings, devices, random_seed):
        """Only task-specific seed settings differ from the original method authority."""
        return build_planning_components(
            physics=self.physics, bounds=self.bounds, mppi=self.mppi, devices=devices,
            random_seed=random_seed, maximum_pulses=self.maximum_pulses,
            completion_remaining_pct=self.completion_remaining_pct,
            hard_margin_mm=self.hard_margin_mm, raster_name=case_name,
            raster_settings=raster_settings, global_settings=self.global_settings,
        )


def activate_compute_profile(controller_path: Path) -> tuple[object, ...]:
    """Activate the controller's default compute profile before importing JAX."""
    path = Path(controller_path).resolve()
    root = path.parent.parent
    controller = load_yaml(path)
    compute_values = load_yaml(root / controller["compute_config"])
    profile_name = compute_values["default_profile"]
    profile = compute_values["profiles"][profile_name]
    return ComputeProfile(
        profile_name, profile["platform"], tuple(profile["device_indices"]),
        int(profile["rollout_batch_size"]), profile["preallocate"],
    ).activate()


def load_method(controller_path: Path, devices) -> ResolvedMethod:
    """Read one method authority and match supplied JAX devices to one compute profile."""
    path = Path(controller_path).resolve()
    root = path.parent.parent
    controller = load_yaml(path)
    paths = {name: root / controller["shared_configs"][name]
             for name in ("physics", "planner", "frozen_global")}
    paths.update(controller=path, compute=root / controller["compute_config"])
    values = {name: load_yaml(source) for name, source in paths.items()}
    for name in ("physics", "planner"):
        if (root / values["frozen_global"]["shared_configs"][name]).resolve() != paths[name].resolve():
            raise ValueError("Global source and controller require the same method authority")
    # Match the complete detected JAX device set to one declared compute authority.
    if not devices:
        raise ValueError("At least one JAX device is required")
    profiles = values["compute"]["profiles"]
    platforms = {device.platform for device in devices}
    if len(platforms) != 1:
        raise ValueError(f"JAX devices must share one platform, received {sorted(platforms)}")
    platform = platforms.pop()
    configured_platform = "gpu" if platform in {"cuda", "gpu"} else platform
    matches = [name for name, candidate in profiles.items()
               if candidate["platform"] == configured_platform
               and candidate["device_indices"] == list(range(len(devices)))]
    if len(matches) != 1:
        raise ValueError(
            f"No unique compute profile for {len(devices)} {configured_platform} device(s)")
    profile_name = matches[0]
    profile = profiles[profile_name]
    mppi_values = dict(controller["mppi"])
    mppi_values["kappa"] = tuple(mppi_values["kappa"])
    mppi = MPPIRepairConfig(**mppi_values, rollout_batch_size=profile["rollout_batch_size"])
    return ResolvedMethod(
        physics_from_mapping(values["physics"]), bounds_from_mapping(values["physics"]),
        mppi, values["frozen_global"]["geometry_aware_global"],
        float(values["planner"]["completion_remaining_pct"]),
        float(values["planner"]["hard_margin_mm"]), int(controller["maximum_pulses"]),
        int(controller["periodic_repair_pulses"]),
        profile_name,
        {str(source): sha256(source.read_bytes()).hexdigest() for source in paths.values()}, values,
    )
