"""Constrained energy MPC with authoritative execution and a dormant exact gate."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from laser_ablation.control.mpc.contracts import EnergyMPCFailureReason, EnergyMPCProblem, EnergyMPCSolution
from laser_ablation.control.mpc.energy_nlp import EnergyNLPSolver
from laser_ablation.control.mpc.exact_acceptance import ExactAcceptanceResult, ExactEnergyMPCAcceptor
from laser_ablation.control.mpc.solution_translation import (
    candidate_contract_error, candidate_diagnostic_constraints,
    solution_from_failed_candidate, with_energy,
)
from laser_ablation.core.actions import PhysicalAction
from laser_ablation.core.sdf import SDFState
from laser_ablation.core.state import VoxelState
from laser_ablation.geometry.sdf import SDFObserver


class _AuthoritativeSimulator(Protocol):
    last_response_scale: float

    def step(self, state: VoxelState, action: PhysicalAction) -> VoxelState:
        ...


@dataclass(frozen=True)
class EnergyMPCStepResult:
    """One constrained solve, execution, and exact-state observation result."""

    solution: EnergyMPCSolution
    exact_acceptance: ExactAcceptanceResult | None
    active_plan_index: int
    next_plan_index: int
    next_voxel_state: VoxelState | None
    next_sdf_state: SDFState | None
    exact_execution_time_s: float = 0.0
    sdf_rebuild_time_s: float = 0.0
    physics_response_scale: float = 1.0

    @property
    def executed(self) -> bool:
        return self.next_voxel_state is not None

    def __post_init__(self) -> None:
        if self.active_plan_index < 0 or self.next_plan_index < 0:
            raise ValueError("plan indices must be nonnegative")
        if self.exact_execution_time_s < 0.0 or self.sdf_rebuild_time_s < 0.0:
            raise ValueError("step timings must be nonnegative")
        if not np.isfinite(self.physics_response_scale) or self.physics_response_scale <= 0.0:
            raise ValueError("physics response scale must be finite and positive")
        if self.executed and (
            not self.solution.success or self.next_sdf_state is None
            or self.next_plan_index != self.active_plan_index + 1
        ):
            raise ValueError("executed step must expose an accepted next exact/SDF state")
        if not self.executed and (
            self.next_sdf_state is not None or self.next_plan_index != self.active_plan_index
        ):
            raise ValueError("failed step must preserve state and plan index")


class EnergyMPCController:
    """Expose an energy action after the constrained local MPC solve."""

    def __init__(self, solver: EnergyNLPSolver, acceptor: ExactEnergyMPCAcceptor,
                 authoritative_simulator: _AuthoritativeSimulator, observer: SDFObserver) -> None:
        self.solver = solver
        self.acceptor = acceptor
        self.authoritative_simulator = authoritative_simulator
        self.observer = observer

    def solve(self, problem: EnergyMPCProblem) -> EnergyMPCSolution:
        solution, _ = self._solve_with_acceptance(problem)
        return solution

    def _solve_with_acceptance(
        self, problem: EnergyMPCProblem
    ) -> tuple[EnergyMPCSolution, ExactAcceptanceResult | None]:
        candidate = self.solver.solve(problem)
        if not candidate.success:
            return solution_from_failed_candidate(candidate), None
        malformed = candidate_contract_error(candidate, problem)
        if malformed is not None:
            return solution_from_failed_candidate(
                candidate, EnergyMPCFailureReason.SURROGATE_INFEASIBLE, malformed
            ), None
        optimized_prefix = tuple(
            with_energy(nominal, energy)
            for nominal, energy in zip(
                problem.horizon_actions, candidate.optimized_energies, strict=True
            )
        )
        acceptance = None
        # Dormant Level A/B exact release gate retained for possible reactivation.
        # tail_start = problem.active_plan_index + problem.effective_horizon
        # acceptance = self.acceptor.verify(
        #     problem.current_exact_voxel_state,
        #     optimized_prefix,
        #     problem.nominal_actions[tail_start:],
        # )
        constraints = dict(candidate.constraint_values)
        constraints.update(candidate_diagnostic_constraints(candidate))
        solution = EnergyMPCSolution(
            success=True, optimized_energies=candidate.optimized_energies,
            first_action=optimized_prefix[0], predicted_states=candidate.predicted_states,
            objective_terms=candidate.objective_terms, constraint_values=constraints,
            ipopt_status=candidate.ipopt_status, iterations=candidate.iterations,
            solve_time_s=candidate.solve_time_s, exact_verified=False,
            exact_verification_metrics=None,
        )
        solution.validate_against(problem)
        return solution, acceptance

    def step(self, problem: EnergyMPCProblem) -> EnergyMPCStepResult:
        """Solve constraints, execute one action, then observe its exact post-state once."""
        solution, acceptance = self._solve_with_acceptance(problem)
        if not solution.success:
            return EnergyMPCStepResult(
                solution, acceptance, problem.active_plan_index,
                problem.active_plan_index, None, None,
            )
        assert solution.first_action is not None
        started = perf_counter()
        next_voxel = self.authoritative_simulator.step(
            problem.current_exact_voxel_state, solution.first_action
        )
        execution_time = perf_counter() - started
        started = perf_counter()
        next_sdf = self.observer.observe(next_voxel)
        rebuild_time = perf_counter() - started
        source = next_sdf.observation.source_voxel_state
        if any(not np.array_equal(getattr(source, name), getattr(next_voxel, name))
               for name in ("tissue", "initial_tissue", "target_mask", "constraint_mask")):
            raise RuntimeError("observer did not rebuild from exact voxel truth")
        return EnergyMPCStepResult(
            solution, acceptance, problem.active_plan_index, problem.active_plan_index + 1,
            next_voxel, next_sdf, execution_time, rebuild_time,
            self.authoritative_simulator.last_response_scale,
        )


__all__ = ["EnergyMPCController", "EnergyMPCStepResult"]
