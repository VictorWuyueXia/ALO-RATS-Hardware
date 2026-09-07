"""Translate numerical energy candidates into exact-gated controller solutions."""

from __future__ import annotations

import math

import numpy as np

from laser_ablation.control.mpc.contracts import (
    EnergyMPCFailureReason, EnergyMPCProblem, EnergyMPCSolution,
)
from laser_ablation.control.mpc.energy_nlp import EnergyNLPCandidate
from laser_ablation.core.actions import PhysicalAction


def with_energy(action: PhysicalAction, energy: float) -> PhysicalAction:
    return PhysicalAction(
        action.x_mm, action.y_mm, action.tilt_x_rad, action.tilt_y_rad, float(energy)
    )


def candidate_contract_error(
    candidate: EnergyNLPCandidate, problem: EnergyMPCProblem
) -> str | None:
    if len(candidate.optimized_energies) != problem.effective_horizon:
        return "NLP returned an energy vector with the wrong effective horizon"
    if not all(np.isfinite(candidate.optimized_energies)):
        return "NLP returned a nonfinite energy"
    if candidate.predicted_states and len(candidate.predicted_states) != problem.effective_horizon + 1:
        return "NLP predicted-state trajectory must include phi_0 through phi_H"
    return None


def solution_from_failed_candidate(
    candidate: EnergyNLPCandidate,
    reason: EnergyMPCFailureReason | None = None,
    diagnostic_message: str | None = None,
) -> EnergyMPCSolution:
    objectives = dict(candidate.objective_terms)
    if diagnostic_message or candidate.diagnostic_message:
        objectives["failure_diagnostic_present"] = 1.0
    states = candidate.predicted_states
    if states and len(states) != len(candidate.optimized_energies) + 1:
        states = ()
    constraints = dict(candidate.constraint_values)
    constraints.update(candidate_diagnostic_constraints(candidate))
    return EnergyMPCSolution(
        success=False, optimized_energies=candidate.optimized_energies, first_action=None,
        predicted_states=states, objective_terms=objectives, constraint_values=constraints,
        ipopt_status=candidate.ipopt_status, iterations=candidate.iterations,
        solve_time_s=candidate.solve_time_s, exact_verified=False,
        exact_verification_metrics=None,
        failure_reason=reason or candidate.failure_reason or EnergyMPCFailureReason.SOLVER_FAILED,
    )


def typed_failure(reason: EnergyMPCFailureReason, message: str) -> EnergyMPCSolution:
    return EnergyMPCSolution(
        success=False,
        objective_terms={"failure_diagnostic_present": float(bool(message))},
        failure_reason=reason,
    )


def candidate_diagnostic_constraints(candidate: EnergyNLPCandidate) -> dict[str, float]:
    diagnostics = {
        "solver_roi_size": float(candidate.roi_size),
        "solver_symbolic_variable_count": float(candidate.symbolic_variable_count),
        "solver_symbolic_expression_nodes": float(candidate.symbolic_expression_nodes),
        "solver_symbolic_expression_bytes": float(candidate.symbolic_expression_bytes),
        "solver_numeric_roi_bytes": float(candidate.numeric_roi_bytes),
    }
    if math.isfinite(candidate.primal_infeasibility):
        diagnostics["solver_primal_infeasibility"] = float(candidate.primal_infeasibility)
    if math.isfinite(candidate.dual_infeasibility):
        diagnostics["solver_dual_infeasibility"] = float(candidate.dual_infeasibility)
    return diagnostics


__all__ = [
    "candidate_contract_error", "candidate_diagnostic_constraints",
    "solution_from_failed_candidate", "typed_failure", "with_energy",
]
