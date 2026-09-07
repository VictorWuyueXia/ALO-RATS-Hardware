"""Resolve the existing MPPI method without creating a synthetic experiment scenario."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from laser_ablation.config import bounds_from_mapping, load_yaml, physics_from_mapping
from laser_ablation.control.factory import build_planning_components
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


def load_method(controller_path: Path) -> ResolvedMethod:
    """Read one explicit configuration authority and its declared shared files."""
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
    cpu = values["compute"]["profiles"]["cpu"]
    if cpu["platform"] != "cpu" or cpu["device_indices"] != [0]:
        raise ValueError("Robot simulation requires the explicit single-CPU compute profile")
    mppi_values = dict(controller["mppi"])
    mppi_values["kappa"] = tuple(mppi_values["kappa"])
    mppi = MPPIRepairConfig(**mppi_values, rollout_batch_size=cpu["rollout_batch_size"])
    return ResolvedMethod(
        physics_from_mapping(values["physics"]), bounds_from_mapping(values["physics"]),
        mppi, values["frozen_global"]["geometry_aware_global"],
        float(values["planner"]["completion_remaining_pct"]),
        float(values["planner"]["hard_margin_mm"]), int(controller["maximum_pulses"]),
        int(controller["periodic_repair_pulses"]),
        {str(source): sha256(source.read_bytes()).hexdigest() for source in paths.values()}, values,
    )
