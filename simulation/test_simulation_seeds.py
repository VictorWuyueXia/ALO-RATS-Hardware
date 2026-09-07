"""Verify raster feasibility and the variable hybrid seed-bank contract."""

from unittest.mock import Mock

import jax
import numpy as np
import pytest

from robot_scene import ROOT
from scan_adapter import designate_task
from simulation_cases import CASE_NAMES, simulation_case
from laser_ablation.control.method import load_method
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.interface import GlobalPlanningFailure
from laser_ablation.planning.global_3d.terminal_verification import ExactPlanVerifier
from laser_ablation.planning.jax_bank.global_seeds import HybridGlobalSeeds
from laser_ablation.planning.raster_bank import ConfiguredRasterPlanGenerator


@pytest.mark.parametrize("name", CASE_NAMES)
def test_exact_complete_rasters_for_configured_task(name):
    """Seed feasibility alone does not prove nonlinear MPPI, residual planning, or low overcut."""
    case = simulation_case(name)
    state = designate_task(case.scan(), case.designation).state
    method = load_method(ROOT / "mppi/configs/controller.yaml", tuple(jax.devices()))
    verifier = ExactPlanVerifier(ExactVoxelSimulator(method.physics, method.bounds),
                                 method.completion_remaining_pct, method.hard_margin_mm)
    seeds = ConfiguredRasterPlanGenerator(name, case.raster_settings, verifier).generate(state)
    assert seeds
    assert all(verifier.verify(state, seed.actions).valid for seed in seeds)


def test_hybrid_seed_bank_accepts_reduced_exact_raster_set():
    case = simulation_case("centered_rectangle")
    state = designate_task(case.scan(), case.designation).state
    method = load_method(ROOT / "mppi/configs/controller.yaml", tuple(jax.devices()))
    verifier = ExactPlanVerifier(ExactVoxelSimulator(method.physics, method.bounds),
                                 method.completion_remaining_pct, method.hard_margin_mm)
    raster_candidates = ConfiguredRasterPlanGenerator(
        case.name, case.raster_settings, verifier).generate(state)[:3]
    raster_generator = Mock()
    raster_generator.generate.return_value = raster_candidates
    frozen_generator = Mock()
    frozen_generator.propose_verified.side_effect = GlobalPlanningFailure("unavailable")

    seeds = HybridGlobalSeeds.build(state, raster_generator, frozen_generator)

    reserve_sources = seeds.source_ids[3:]
    assert seeds.raster_bootstrap_count == seeds.provenance["raster_bootstrap_count"] == 3
    assert len(seeds.actions) == 10 and len(reserve_sources) == 7
    assert all(source.startswith("reserve:raster:") for source in reserve_sources)
    assert all(np.array_equal(seeds.actions[index], seeds.actions[index % 3])
               for index in range(3, 10))


def test_hybrid_seed_bank_rejects_empty_raster_set():
    raster_generator = Mock()
    raster_generator.generate.return_value = ()

    with pytest.raises(GlobalPlanningFailure, match="at least one exact-complete raster plan"):
        HybridGlobalSeeds.build(Mock(), raster_generator, Mock())
