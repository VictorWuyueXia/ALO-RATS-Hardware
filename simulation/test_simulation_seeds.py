"""Verify the real four-raster source on every frozen operator-designated task."""

import jax
import pytest

from robot_scene import ROOT
from scan_adapter import designate_task
from simulation_cases import CASE_NAMES, simulation_case
from laser_ablation.control.method import load_method
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.terminal_verification import ExactPlanVerifier
from laser_ablation.planning.raster_bank import ConfiguredRasterPlanGenerator


@pytest.mark.parametrize("name", CASE_NAMES)
def test_four_exact_complete_rasters_for_frozen_task(name):
    """Seed feasibility alone does not prove nonlinear MPPI, residual planning, or low overcut."""
    case = simulation_case(name)
    state = designate_task(case.scan(), case.designation).state
    method = load_method(ROOT / "mppi/configs/controller.yaml", tuple(jax.devices()))
    verifier = ExactPlanVerifier(ExactVoxelSimulator(method.physics, method.bounds),
                                 method.completion_remaining_pct, method.hard_margin_mm)
    seeds = ConfiguredRasterPlanGenerator(name, case.raster_settings, verifier).generate(state)
    assert len(seeds) == 4
    assert all(verifier.verify(state, seed.actions).valid for seed in seeds)
