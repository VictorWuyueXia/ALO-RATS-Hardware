"""CasADi graph construction for tilted beam energy optimization."""

from __future__ import annotations

import math

import casadi as ca
import numpy as np

from laser_ablation.control.mpc.beam_geometry import BeamMPCROI
from laser_ablation.control.mpc.casadi_metrics import normalized_metrics, normalized_sdf_tracking
from laser_ablation.control.mpc.casadi_transition import CasadiBeamTransition
from laser_ablation.control.mpc.config import EnergyMPCConfig
from laser_ablation.control.mpc.contracts import EnergyMPCProblem
from laser_ablation.core.actions import PhysicalActionBounds
from laser_ablation.metrics import evaluate_ablation
from laser_ablation.physics.super_gaussian import PhysicsConfig


def build_energy_nlp_graph(
    problem: EnergyMPCProblem, roi: BeamMPCROI, physics: PhysicsConfig,
    bounds: PhysicalActionBounds, config: EnergyMPCConfig,
) -> dict[str, object]:
    """Construct one fixed-spatial-action, energy-only IPOPT graph."""
    horizon = problem.effective_horizon
    energies = ca.MX.sym("E", horizon)
    contact_fields = (
        problem.current_sdf_state.observation.tissue_sdf_mm,
        *problem.horizon_pre_action_tissue_sdf_mm[1:],
    )
    rollout = CasadiBeamTransition(
        problem.current_exact_voxel_state, problem.current_sdf_state,
        problem.horizon_actions, contact_fields, physics, roi,
        inclusion_epsilon_mm=config.crater_inclusion_epsilon_mm,
        clearance_samples_per_axis=config.clearance_samples_per_axis,
    ).rollout(energies)
    initial_phi = problem.current_sdf_state.observation.tissue_sdf_mm[tuple(roi.indices.T)]
    phi_steps: tuple[ca.MX, ...] = (
        ca.DM(np.asarray(initial_phi, dtype=float).reshape(-1, 1)), *rollout.phi_by_step,
    )
    metrics = tuple(
        normalized_metrics(
            phi, target_selector=roi.target_selector, bottom_selector=roi.bottom_selector,
            normal_selector=roi.normal_selector, voxel_volume_mm3=roi.voxel_volume_mm3,
            initial_target_volume_mm3=roi.initial_target_volume_mm3,
            outside_remaining_volume_mm3=roi.outside_remaining_volume_mm3,
            outside_bottom_volume_mm3=roi.outside_bottom_volume_mm3,
            outside_normal_volume_mm3=roi.outside_normal_volume_mm3,
            smoothing_mm=config.occupancy_smoothing_mm,
        ) for phi in phi_steps
    )
    exact_now = evaluate_ablation(problem.current_exact_voxel_state)
    progress = 1.0 - exact_now.remaining_volume_mm3 / problem.current_exact_voxel_state.initial_target_volume_mm3
    weights = config.weights_at(float(np.clip(progress, 0.0, 1.0)))
    energy_min, energy_max = bounds.energy_j
    energy_span = energy_max - energy_min
    remaining: list[ca.MX] = []
    overcut: list[ca.MX] = []
    tracking: list[ca.MX] = []
    energy_cost: list[ca.MX] = []
    target_progress: list[ca.MX] = []
    bottom_overcut: list[ca.MX] = []
    normal_damage: list[ca.MX] = []
    for step in range(horizon):
        before, after = metrics[step], metrics[step + 1]
        remaining.append(after.remaining_fraction)
        overcut.append(after.total_overcut_fraction - before.total_overcut_fraction)
        reference = problem.horizon_reference_tissue_sdf_mm[step][tuple(roi.indices.T)]
        tracking.append(normalized_sdf_tracking(
            phi_steps[step + 1], reference, roi.tracking_weights, config.d_max_mm
        ))
        nominal_energy = problem.horizon_actions[step].energy_j
        energy_cost.append(((energies[step] - nominal_energy) / energy_span) ** 2)
        target_progress.append(before.remaining_fraction - after.remaining_fraction)
        bottom_overcut.append(after.target_bottom_overcut_fraction - before.target_bottom_overcut_fraction)
        normal_damage.append(after.normal_damage_fraction - before.normal_damage_fraction)
    sums = tuple(ca.sum1(ca.vertcat(*values)) for values in (remaining, overcut, tracking, energy_cost))
    terminal = metrics[-1].remaining_fraction
    objective_terms = (
        weights.remaining * sums[0], weights.overcut * sums[1], weights.tracking * sums[2],
        weights.energy * sums[3], config.terminal_remaining_weight * terminal,
    )
    objective = ca.sum1(ca.vertcat(*objective_terms))
    diagnostic_terms = (
        ca.sum1(ca.vertcat(*bottom_overcut)), ca.sum1(ca.vertcat(*normal_damage)),
        metrics[-1].remaining_fraction, metrics[-1].total_overcut_fraction,
        metrics[-1].target_bottom_overcut_fraction, metrics[-1].normal_damage_fraction,
    )
    constraints: list[ca.MX] = []
    lower: list[float] = []
    upper: list[float] = []
    layout: list[tuple[str, int]] = []

    def add(name: str, values: tuple[ca.MX, ...] | list[ca.MX], low: float, high: float) -> None:
        constraints.extend(values)
        lower.extend([low] * len(values))
        upper.extend([high] * len(values))
        layout.append((name, len(values)))

    state = problem.current_exact_voxel_state
    contact_max = state.z_axis_mm[-1] - state.z_axis_mm[0] + state.spacing_mm
    add("contact_distance_mm", rollout.contact_distance_mm, 0.0, contact_max)
    add("contact_outside_phi_mm", rollout.contact_outside_phi_mm, 0.0, math.inf)
    add("contact_inside_phi_mm", rollout.contact_inside_phi_mm, -math.inf, 0.0)
    add("target_removal_fraction", target_progress, config.minimum_target_removal_fraction, math.inf)
    add("cumulative_clearance_mm", rollout.cumulative_clearance_mm, config.hard_margin_mm, math.inf)
    add("workspace_margin_mm", rollout.workspace_margin_mm, 0.0, math.inf)
    if problem.active_plan_index + horizon == len(problem.nominal_actions):
        add("terminal_remaining_fraction", [terminal], 0.0, config.completion_remaining_pct / 100.0)
    constraint_vector = ca.vertcat(*constraints)
    solver = ca.nlpsol("energy_mpc_ipopt", "ipopt", {
        "x": energies, "f": objective, "g": constraint_vector,
    }, {
        "print_time": False, "error_on_fail": False, "ipopt.print_level": config.ipopt_print_level,
        "ipopt.sb": "yes", "ipopt.max_iter": config.solver_max_iterations,
        "ipopt.tol": config.solver_tolerance, "ipopt.acceptable_tol": config.solver_tolerance,
        "ipopt.constr_viol_tol": config.solver_tolerance,
    })
    diagnostics = ca.Function(
        "energy_mpc_diagnostics", [energies], [*phi_steps, *objective_terms, objective, *diagnostic_terms]
    )
    graph_size = ca.Function("energy_mpc_graph_size", [energies], [objective, constraint_vector])
    return {
        "solver": solver, "diagnostics": diagnostics,
        "energy_initial": np.asarray([a.energy_j for a in problem.horizon_actions], dtype=float),
        "energy_lower": np.asarray([
            max(energy_min, action.energy_j - config.energy_trust_region_j)
            for action in problem.horizon_actions
        ]),
        "energy_upper": np.asarray([
            min(energy_max, action.energy_j + config.energy_trust_region_j)
            for action in problem.horizon_actions
        ]),
        "constraint_lower": np.asarray(lower), "constraint_upper": np.asarray(upper),
        "constraint_layout": tuple(layout),
        "diagnostic_layout": {
            "state_count": horizon + 1,
            "objective_names": (
                "stage_remaining", "incremental_total_overcut", "sdf_tracking", "energy",
                "terminal_remaining", "total", "incremental_target_bottom_overcut_fraction",
                "incremental_normal_damage_fraction", "predicted_terminal_remaining_fraction",
                "predicted_terminal_total_overcut_fraction", "predicted_terminal_target_bottom_overcut_fraction",
                "predicted_terminal_normal_damage_fraction",
            ),
        },
        "expression_nodes": ca.n_nodes(objective) + ca.n_nodes(constraint_vector),
        "expression_bytes": len(graph_size.serialize()),
    }


__all__ = ["build_energy_nlp_graph"]
