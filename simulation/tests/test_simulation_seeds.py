"""Verify raw-raster construction and the active global-seed contract."""

from unittest.mock import Mock

import jax
import pytest

from simulation.robot.robot_scene import ROOT
from simulation.oct.scan_adapter import designate_task
from simulation.simulation.simulation_cases import CASE_NAMES, simulation_case
from laser_ablation.control.method import load_method
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.interface import GlobalPlanningFailure
from laser_ablation.planning.jax_bank.global_seeds import RasterGlobalSeeds
from laser_ablation.planning.raster_bank import ConfiguredRasterPlanGenerator


@pytest.mark.parametrize("name", CASE_NAMES)
def test_configured_rasters_are_raw_four_parent_sources(name):
    """Every configured task supplies four unfiltered raster parents to MPPI."""
    case = simulation_case(name)
    state = designate_task(case.scan(), case.designation).state
    method = load_method(ROOT / "mppi/configs/controller.yaml", tuple(jax.devices()))
    seeds = ConfiguredRasterPlanGenerator(
        name, case.raster_settings, ExactVoxelSimulator(method.physics, method.bounds),
    ).generate(state)
    assert len(seeds) == 4
    assert all(seed.actions for seed in seeds)


def test_global_seed_bank_requires_exactly_four_rasters():
    """The initial path never reserves or duplicates parents to fill the bank."""
    raster_generator = Mock()
    raster_generator.generate.return_value = ()
    with pytest.raises(GlobalPlanningFailure, match="exactly four raw parents"):
        RasterGlobalSeeds.build(Mock(), raster_generator, None)


def test_geometry_source_is_not_requested_for_initial_raster_seeds():
    """The initial bank contains only four raw rasters and no geometry call."""
    case = simulation_case("centered_rectangle")
    state = designate_task(case.scan(), case.designation).state
    method = load_method(ROOT / "mppi/configs/controller.yaml", tuple(jax.devices()))
    raster = ConfiguredRasterPlanGenerator(
        case.name, case.raster_settings, ExactVoxelSimulator(method.physics, method.bounds),
    )
    geometry = Mock()
    seeds = RasterGlobalSeeds.build(state, raster, None)
    assert len(seeds.actions) == 4
    assert seeds.provenance["geometry_parent_count"] == 0
    geometry.generate.assert_not_called()
