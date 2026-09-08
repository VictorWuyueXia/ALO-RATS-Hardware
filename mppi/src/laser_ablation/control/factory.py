"""Construct the existing planner from resolved inputs, without a scenario or checkout."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from laser_ablation.control.session import ControllerSession
from laser_ablation.geometry.sdf import SDFObserver
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.terminal_verification import ExactPlanVerifier
from laser_ablation.planning.geometry_aware_bank import GeometryAwarePlanGenerator
from laser_ablation.planning.jax_bank.mppi_repairer import MPPIPlanRepairer
from laser_ablation.planning.jax_bank.planner import JaxPlanBankPlanner
from laser_ablation.planning.raster_bank import ConfiguredRasterPlanGenerator


@dataclass(frozen=True)
class PlanningComponents:
    planner: JaxPlanBankPlanner
    observer: SDFObserver
    raster_generator: ConfiguredRasterPlanGenerator
    geometry_generator: GeometryAwarePlanGenerator
    completion_remaining_pct: float
    maximum_pulses: int
    authority_id: str

    def session(self, periodic_repair_pulses, output_directory, random_seed):
        """Create an external-observation session without constructing an execution plant."""
        return ControllerSession(
            self.planner, self.observer, self.raster_generator,
            self.completion_remaining_pct, periodic_repair_pulses, self.maximum_pulses,
            output_directory, random_seed,
        )


def build_planning_components(*, physics, bounds, mppi, devices, random_seed,
                              maximum_pulses, completion_remaining_pct, hard_margin_mm,
                              raster_name, raster_settings, global_settings):
    """Reuse the unchanged numerical algorithms with caller-supplied method and seed inputs."""
    model = ExactVoxelSimulator(physics, bounds)
    verifier = ExactPlanVerifier(model, completion_remaining_pct, hard_margin_mm)
    observer = SDFObserver(hard_margin_mm=hard_margin_mm)
    repairer = MPPIPlanRepairer(mppi, devices, random_seed, maximum_pulses)
    geometry = GeometryAwarePlanGenerator(
        {"geometry_aware_global": dict(global_settings)}, physics, bounds, verifier,
    )
    planner = JaxPlanBankPlanner(
        maximum_pulses, physics, bounds, completion_remaining_pct / 100,
        repairer, devices, mppi.rollout_batch_size, geometry,
    )
    raster = ConfiguredRasterPlanGenerator(raster_name, raster_settings, model)
    authority = sha256(json.dumps({
        "physics": asdict(physics), "bounds": asdict(bounds), "mppi": asdict(mppi),
        "completion_remaining_pct": completion_remaining_pct,
        "hard_margin_mm": hard_margin_mm, "maximum_pulses": maximum_pulses,
        "global_settings": global_settings, "raster_settings": raster_settings,
    }, sort_keys=True).encode()).hexdigest()
    return PlanningComponents(planner, observer, raster, geometry,
                              completion_remaining_pct, maximum_pulses, authority)
