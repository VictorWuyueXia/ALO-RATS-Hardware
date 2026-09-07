"""Solve the tilted-beam energy NLP and return an unreleased proposal."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Mapping

import numpy as np

from laser_ablation.control.mpc.beam_geometry import BeamMPCROI, build_beam_roi
from laser_ablation.control.mpc.config import EnergyMPCConfig, MPCFingerprints
from laser_ablation.control.mpc.contracts import ConstraintValue, EnergyMPCFailureReason, EnergyMPCProblem
from laser_ablation.control.mpc.nlp_diagnostics import (
    constraint_mapping, ipopt_infeasibilities, maximum_constraint_violation,
    roi_numeric_bytes, unpack_diagnostics,
)
from laser_ablation.control.mpc.nlp_graph import build_energy_nlp_graph
from laser_ablation.core.actions import PhysicalAction, PhysicalActionBounds
from laser_ablation.physics.super_gaussian import PhysicsConfig


@dataclass(frozen=True)
class EnergyNLPCandidate:
    """Local IPOPT result before exact-authority acceptance."""

    success: bool
    optimized_energies: tuple[float, ...] = ()
    optimized_actions: tuple[PhysicalAction, ...] = ()
    predicted_states: tuple[np.ndarray, ...] = ()
    objective_terms: Mapping[str, float] = field(default_factory=dict)
    constraint_values: Mapping[str, ConstraintValue] = field(default_factory=dict)
    ipopt_status: str = "NOT_RUN"
    iterations: int = 0
    solve_time_s: float = 0.0
    primal_infeasibility: float = math.nan
    dual_infeasibility: float = math.nan
    failure_reason: EnergyMPCFailureReason | None = None
    diagnostic_message: str | None = None
    roi_size: int = 0
    symbolic_variable_count: int = 0
    symbolic_expression_nodes: int = 0
    symbolic_expression_bytes: int = 0
    numeric_roi_bytes: int = 0


class EnergyNLPSolver:
    """Optimize energy while preserving every nominal spatial action coordinate."""

    def __init__(self, physics: PhysicsConfig, bounds: PhysicalActionBounds,
                 config: EnergyMPCConfig, fingerprints: MPCFingerprints) -> None:
        self.physics = physics
        self.bounds = bounds
        self.config = config
        self.fingerprints = fingerprints

    def solve(self, problem: EnergyMPCProblem) -> EnergyNLPCandidate:
        mismatch = self._contract_mismatch(problem)
        if mismatch is not None:
            return self._failure(EnergyMPCFailureReason.SURROGATE_INFEASIBLE, mismatch)
        actions = problem.horizon_actions
        if any(not self.bounds.contains(action) for action in actions):
            return self._failure(
                EnergyMPCFailureReason.SURROGATE_INFEASIBLE,
                "nominal action lies outside shared physical bounds",
            )
        try:
            roi = build_beam_roi(
                problem.current_exact_voxel_state, problem.current_sdf_state, actions,
                self.physics, halo_mm=self.config.roi_halo_mm,
                closure_voxels=self.config.roi_closure_voxels,
                tracking_base_weight=self.config.tracking_base_weight,
                tracking_boundary_band_mm=self.config.tracking_boundary_band_mm,
            )
            graph = build_energy_nlp_graph(problem, roi, self.physics, self.bounds, self.config)
        except (ValueError, RuntimeError) as error:
            return self._failure(EnergyMPCFailureReason.SURROGATE_INFEASIBLE, str(error))
        started = perf_counter()
        try:
            result = graph["solver"](
                x0=graph["energy_initial"], lbx=graph["energy_lower"], ubx=graph["energy_upper"],
                lbg=graph["constraint_lower"], ubg=graph["constraint_upper"],
            )
        except RuntimeError as error:
            return self._failure(
                EnergyMPCFailureReason.SOLVER_FAILED, str(error), perf_counter() - started, roi, graph
            )
        solve_time = perf_counter() - started
        stats = graph["solver"].stats()
        status = str(stats.get("return_status", "UNKNOWN"))
        energies = np.asarray(result["x"], dtype=float).reshape(-1)
        constraints = np.asarray(result["g"], dtype=float).reshape(-1)
        decoded = unpack_diagnostics(
            graph["diagnostics"](energies), graph["diagnostic_layout"], roi.size
        )
        optimized_actions = tuple(
            PhysicalAction(action.x_mm, action.y_mm, action.tilt_x_rad, action.tilt_y_rad, float(energy))
            for action, energy in zip(actions, energies, strict=True)
        )
        primal, dual = ipopt_infeasibilities(stats)
        common = dict(
            optimized_energies=tuple(map(float, energies)), optimized_actions=optimized_actions,
            predicted_states=decoded["predicted_states"], objective_terms=decoded["objective_terms"],
            constraint_values=constraint_mapping(constraints, graph["constraint_layout"]),
            ipopt_status=status, iterations=int(stats.get("iter_count", 0)), solve_time_s=solve_time,
            primal_infeasibility=primal, dual_infeasibility=dual, roi_size=roi.size,
            symbolic_variable_count=len(actions), symbolic_expression_nodes=int(graph["expression_nodes"]),
            symbolic_expression_bytes=int(graph["expression_bytes"]), numeric_roi_bytes=roi_numeric_bytes(roi),
        )
        if not bool(stats.get("success", False)):
            return EnergyNLPCandidate(
                success=False, failure_reason=EnergyMPCFailureReason.SOLVER_FAILED,
                diagnostic_message=f"IPOPT did not report success: {status}", **common,
            )
        violation = maximum_constraint_violation(
            constraints, graph["constraint_lower"], graph["constraint_upper"]
        )
        if violation > self.config.solver_tolerance:
            return EnergyNLPCandidate(
                success=False, failure_reason=EnergyMPCFailureReason.SURROGATE_INFEASIBLE,
                diagnostic_message=f"surrogate constraint violation {violation:.6g}", **common,
            )
        return EnergyNLPCandidate(success=True, **common)

    def _contract_mismatch(self, problem: EnergyMPCProblem) -> str | None:
        if problem.physics_fingerprint != self.fingerprints.physics:
            return "problem physics fingerprint does not match shared authority"
        if problem.safety_fingerprint != self.fingerprints.safety:
            return "problem safety fingerprint does not match shared authority"
        if problem.horizon != self.config.horizon:
            return "problem horizon does not match preregistered MPC horizon"
        return None

    @staticmethod
    def _failure(reason: EnergyMPCFailureReason, message: str, solve_time_s: float = 0.0,
                 roi: BeamMPCROI | None = None, graph: Mapping[str, object] | None = None) -> EnergyNLPCandidate:
        return EnergyNLPCandidate(
            success=False, failure_reason=reason, diagnostic_message=message,
            solve_time_s=solve_time_s, roi_size=0 if roi is None else roi.size,
            symbolic_expression_nodes=0 if graph is None else int(graph["expression_nodes"]),
            symbolic_expression_bytes=0 if graph is None else int(graph["expression_bytes"]),
            numeric_roi_bytes=0 if roi is None else roi_numeric_bytes(roi),
        )


__all__ = ["EnergyNLPCandidate", "EnergyNLPSolver"]
