"""Geometry-aware complete-plan source for the active JAX plan-bank workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from laser_ablation.core.actions import PhysicalAction, PhysicalActionBounds
from laser_ablation.core.plan import PlanVerification
from laser_ablation.core.state import VoxelState
from laser_ablation.physics.super_gaussian import PhysicsConfig
from laser_ablation.planning.global_3d.exact_finalist import (
    _bind_verified_candidate_with_context,
    _make_exact_cache_entry,
    select_exact_global_finalist,
    verification_authority_identity,
    voxel_state_hash,
)
from laser_ablation.planning.global_3d.interface import (
    GlobalPlanningFailure,
    PlanCandidate,
    VerifiedGlobalPlan,
    canonical_action_sequence_hash,
)
from laser_ablation.planning.global_3d.terminal_verification import (
    ExactPlanVerifier,
    exact_terminal_key,
)


@dataclass(frozen=True)
class GeometryAwarePlanConfig:
    """Small deterministic factor set for geometry-aware complete plans."""

    anchor_count: int
    layout_families: tuple[str, ...]
    order_patterns: tuple[str, ...]
    depth_margin_mm: float
    refinement_position_scales_mm: tuple[float, ...]
    refinement_energy_scales_j: tuple[float, ...]
    refinement_passes_per_scale: int
    refinement_mode: str = "legacy_coordinate"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GeometryAwarePlanConfig":
        config = cls(
            anchor_count=int(values["anchor_count"]),
            layout_families=tuple(map(str, values["layout_families"])),
            order_patterns=tuple(map(str, values["order_patterns"])),
            depth_margin_mm=float(values["depth_margin_mm"]),
            refinement_position_scales_mm=tuple(
                map(float, values["refinement_position_scales_mm"])
            ),
            refinement_energy_scales_j=tuple(
                map(float, values["refinement_energy_scales_j"])
            ),
            refinement_passes_per_scale=int(values["refinement_passes_per_scale"]),
            refinement_mode=str(values.get("refinement_mode", "legacy_coordinate")),
        )
        if config.anchor_count <= 0 or config.refinement_passes_per_scale <= 0:
            raise ValueError("geometry-aware anchor count and refinement passes must be positive")
        if not config.layout_families or not config.order_patterns:
            raise ValueError("geometry-aware layout and order families cannot be empty")
        if set(config.layout_families) - {"principal_axis", "weighted_kmeans"}:
            raise ValueError("unknown geometry-aware layout family")
        if set(config.order_patterns) - {"forward", "reverse", "center_out"}:
            raise ValueError("unknown geometry-aware order pattern")
        if config.depth_margin_mm < 0.0:
            raise ValueError("depth margin must be nonnegative")
        if config.refinement_mode not in {"legacy_coordinate", "alternating_exact"}:
            raise ValueError("unknown geometry-aware refinement mode")
        if (
            len(config.refinement_position_scales_mm)
            != len(config.refinement_energy_scales_j)
            or not config.refinement_position_scales_mm
            or min(config.refinement_position_scales_mm) <= 0.0
            or min(config.refinement_energy_scales_j) <= 0.0
        ):
            raise ValueError("paired positive position/energy refinement scales are required")
        return config


@dataclass
class RefinementTrace:
    """Auditable move sequence for one exact whole-plan refinement."""

    mode: str
    accepted_spatial_moves: int = 0
    accepted_energy_moves: int = 0
    spatial_acceptance_events: list[tuple[int, int, int]] = field(
        default_factory=list
    )
    energy_acceptance_events: list[tuple[int, int, int]] = field(
        default_factory=list
    )
    cross_pulse_energy_events: list[tuple[int, int, int, tuple[int, ...]]] = field(
        default_factory=list
    )


class GeometryAwarePlanGenerator:
    """Construct a finite set of complete plans from the actual target geometry.

    The source is deliberately not a state-conditioned search planner. It builds
    a few target-conditioned layouts, assigns depth-derived vertical energies,
    and performs a fixed local whole-plan coordinate schedule. Every returned
    plan has been independently accepted by the existing exact verifier.
    """

    def __init__(
        self,
        scenario: Mapping[str, Any],
        physics: PhysicsConfig,
        bounds: PhysicalActionBounds,
        verifier: ExactPlanVerifier,
    ) -> None:
        if verifier.simulator.config != physics or verifier.simulator.bounds != bounds:
            raise ValueError("geometry source and exact verifier must share physics authority")
        self.config = GeometryAwarePlanConfig.from_mapping(
            scenario["geometry_aware_global"]
        )
        self.physics = physics
        self.bounds = bounds
        self.verifier = verifier
        self.exact_plan_evaluations = 0
        self.last_diagnostics: tuple[dict[str, object], ...] = ()
        self.last_refinement_traces: tuple[RefinementTrace, ...] = ()
        self.last_verified_candidates: tuple[VerifiedGlobalPlan, ...] = ()
        self.last_exact_finalist: VerifiedGlobalPlan | None = None
        self._verified_cache_key: tuple[str, str, str] | None = None
        self._verified_cache: tuple[VerifiedGlobalPlan, ...] = ()

    def generate_seed_plans(self, state: VoxelState) -> tuple[PlanCandidate, ...]:
        """Construct deterministic geometry-aware seeds without refinement."""
        points, weights = _target_columns(state)
        if points.size == 0:
            raise GlobalPlanningFailure("geometry-aware source received no target tissue")
        axis, centre = _principal_axis(points, weights)
        layouts = {
            "principal_axis": _principal_quantile_layout(
                points, weights, axis, centre, self.config.anchor_count
            ),
            "weighted_kmeans": _weighted_kmeans_layout(
                points, weights, self.config.anchor_count
            ),
        }
        seeds = []
        for layout_family in self.config.layout_families:
            anchors = layouts[layout_family]
            for order_pattern in self.config.order_patterns:
                ordered = _ordered_anchors(anchors, axis, centre, order_pattern)
                seeds.append(
                    PlanCandidate(
                        actions=self._base_actions(state, ordered),
                        layout_id=f"target_{layout_family}:k={len(anchors)}",
                        tilt_pattern="vertical",
                        energy_pattern="local_depth_fit",
                        repeat_depth_fraction=1.0,
                        pulse_fraction=1.0,
                        order_pattern=order_pattern,
                    )
                )
        return tuple(seeds)

    def _build_verified_candidates(
        self, state: VoxelState
    ) -> tuple[VerifiedGlobalPlan, ...]:
        """Build pure geometry-aware exact finalists and reuse them by authority."""
        authority = verification_authority_identity(self.verifier)
        state_digest = voxel_state_hash(state)
        cache_key = (
            state_digest,
            authority.digest,
            repr(self.config),
        )
        if authority.cache_reusable and self._verified_cache_key == cache_key:
            self.last_verified_candidates = self._verified_cache
            return self._verified_cache

        verified_candidates: list[VerifiedGlobalPlan] = []
        diagnostics: list[dict[str, object]] = []
        traces: list[RefinementTrace] = []
        for seed_index, seed in enumerate(self.generate_seed_plans(state)):
            refined, evaluation, trace = self.refine_seed(state, seed.actions)
            refined_candidate = PlanCandidate(
                actions=refined,
                layout_id=seed.layout_id,
                tilt_pattern=seed.tilt_pattern,
                energy_pattern=(
                    "local_depth_fit+fixed_multiscale_refinement"
                    if self.config.refinement_mode == "legacy_coordinate"
                    else "local_depth_fit+alternating_exact"
                ),
                repeat_depth_fraction=seed.repeat_depth_fraction,
                pulse_fraction=seed.pulse_fraction,
                order_pattern=seed.order_pattern,
            )
            fresh = _make_exact_cache_entry(
                state,
                refined_candidate,
                evaluation,
                self.verifier,
                authority=authority,
                state_digest=state_digest,
            )
            verified = _bind_verified_candidate_with_context(
                state,
                refined_candidate,
                self.verifier,
                cached=fresh,
                authority=authority,
                state_digest=state_digest,
                provenance=(
                    ("generator", type(self).__name__),
                    ("refinement_mode", self.config.refinement_mode),
                    ("seed_index", str(seed_index)),
                ),
            )
            strict = self.verifier.require_complete(verified.verification)
            metrics = verified.verification.terminal_metrics
            traces.append(trace)
            verified_candidates.append(verified)
            diagnostics.append(
                {
                    "layout_family": verified.layout_id,
                    "order_pattern": verified.order_pattern,
                    "base_anchor_count": len(seed.actions),
                    "refinement_mode": self.config.refinement_mode,
                    "valid": strict.valid,
                    "remaining_pct": metrics.remaining_pct,
                    "total_overcut_pct": metrics.total_overcut_pct,
                    "pulses": len(verified.executed_actions),
                    "energy_j": float(
                        sum(action.energy_j for action in verified.executed_actions)
                    ),
                    "planned_action_hash": verified.planned_action_hash,
                    "executed_prefix_hash": verified.executed_prefix_hash,
                    "verification_cache_hit": verified.verification_cache_hit,
                    "exact_replay_count": verified.exact_replay_count,
                    "rejection_reason": strict.rejection_reason,
                }
            )
            print(
                "geometry candidate "
                f"layout={verified.layout_id} order={verified.order_pattern} "
                f"pulses={len(verified.executed_actions)} "
                f"remaining={metrics.remaining_pct:.3f}% "
                f"overcut={metrics.total_overcut_pct:.3f}% "
                f"accepted={strict.valid}",
                flush=True,
            )

        records = tuple(verified_candidates)
        self.last_diagnostics = tuple(diagnostics)
        self.last_refinement_traces = tuple(traces)
        self.last_verified_candidates = records
        if authority.cache_reusable:
            self._verified_cache_key = cache_key
            self._verified_cache = records
        else:
            self._verified_cache_key = None
            self._verified_cache = ()
        return records

    def propose_verified(self, state: VoxelState) -> VerifiedGlobalPlan:
        """Return the standalone exact-authority Geometry-Aware Global plan."""
        if self.config.refinement_mode != "legacy_coordinate":
            raise GlobalPlanningFailure(
                "the proposed Global authority requires legacy_coordinate refinement"
            )
        selected = select_exact_global_finalist(
            self._build_verified_candidates(state)
        )
        self.last_exact_finalist = selected
        return selected

    def generate(self, state: VoxelState) -> tuple[PlanCandidate, ...]:
        """Preserve the historical exact-valid candidate interface for JAX."""
        accepted: list[PlanCandidate] = []
        seen: set[bytes] = set()
        for verified in self._build_verified_candidates(state):
            strict = self.verifier.require_complete(verified.verification)
            actions = verified.executed_actions
            if not strict.valid or not actions:
                continue
            key = np.asarray(
                [action.as_array() for action in actions],
                dtype=np.float64,
            ).tobytes()
            if key in seen:
                continue
            seen.add(key)
            accepted.append(
                PlanCandidate(
                    actions=actions,
                    layout_id=verified.layout_id,
                    tilt_pattern=verified.tilt_pattern,
                    energy_pattern=verified.energy_pattern,
                    repeat_depth_fraction=verified.repeat_depth_fraction,
                    pulse_fraction=verified.pulse_fraction,
                    order_pattern=verified.order_pattern,
                )
            )
        if not accepted:
            raise GlobalPlanningFailure("no exact-valid geometry-aware complete plan")
        return tuple(accepted)

    def _base_actions(
        self, state: VoxelState, anchors: np.ndarray
    ) -> tuple[PhysicalAction, ...]:
        actions = []
        for anchor in anchors:
            energy = self._depth_fit_energy(state, anchor)
            action = PhysicalAction(
                float(anchor[0]), float(anchor[1]), 0.0, 0.0, energy
            )
            if not self.bounds.contains(action):
                raise RuntimeError("geometry-derived base action violates physical bounds")
            actions.append(action)
        return tuple(actions)

    def _depth_fit_energy(self, state: VoxelState, anchor: np.ndarray) -> float:
        probe = PhysicalAction(
            float(anchor[0]),
            float(anchor[1]),
            0.0,
            0.0,
            self.physics.max_energy_j,
        )
        contact = self.verifier.simulator.first_contact(state, probe)
        if not contact.hit or contact.point_mm is None:
            raise GlobalPlanningFailure("geometry-derived anchor has no exact tissue contact")
        ix = int(np.argmin(np.abs(state.x_axis_mm - anchor[0])))
        iy = int(np.argmin(np.abs(state.y_axis_mm - anchor[1])))
        target_z = state.z_axis_mm[state.tissue[ix, iy] & state.target_mask[ix, iy]]
        if target_z.size == 0:
            raise GlobalPlanningFailure("geometry-derived anchor does not intersect target")
        required_depth = (
            float(contact.point_mm[2] - np.min(target_z))
            + 0.5 * state.spacing_mm
            + self.config.depth_margin_mm
        )
        energy = (
            self.physics.ablation_threshold
            + required_depth / self.physics.depth_scale_mm_per_j
        )
        return float(np.clip(
            energy,
            np.nextafter(self.bounds.energy_j[0], np.inf),
            self.bounds.energy_j[1],
        ))

    def refine_seed(
        self,
        state: VoxelState,
        actions: tuple[PhysicalAction, ...],
        mode: str | None = None,
    ) -> tuple[tuple[PhysicalAction, ...], PlanVerification, RefinementTrace]:
        """Refine one shared seed with the selected exact-evaluated schedule."""
        selected_mode = self.config.refinement_mode if mode is None else mode
        if selected_mode == "legacy_coordinate":
            return self._refine_legacy_coordinate(state, actions)
        if selected_mode == "alternating_exact":
            return self._refine_alternating_exact(state, actions)
        raise ValueError(f"unknown refinement mode: {selected_mode}")

    def _refine_legacy_coordinate(
        self,
        state: VoxelState,
        actions: tuple[PhysicalAction, ...],
    ) -> tuple[tuple[PhysicalAction, ...], PlanVerification, RefinementTrace]:
        values = np.asarray([action.as_array() for action in actions], dtype=float)
        incumbent = self._evaluate(state, values)
        trace = RefinementTrace(mode="legacy_coordinate")
        for position_scale, energy_scale in zip(
            self.config.refinement_position_scales_mm,
            self.config.refinement_energy_scales_j,
        ):
            for _ in range(self.config.refinement_passes_per_scale):
                changed = False
                for pulse in range(values.shape[0]):
                    best_values = values
                    best_verification = incumbent
                    best_key = self._terminal_key(incumbent)
                    for dx in (-position_scale, 0.0, position_scale):
                        for dy in (-position_scale, 0.0, position_scale):
                            for de in (-energy_scale, 0.0, energy_scale):
                                candidate = values.copy()
                                candidate[pulse, 0] = np.clip(
                                    candidate[pulse, 0] + dx, *self.bounds.x_mm
                                )
                                candidate[pulse, 1] = np.clip(
                                    candidate[pulse, 1] + dy, *self.bounds.y_mm
                                )
                                candidate[pulse, 4] = np.clip(
                                    candidate[pulse, 4] + de, *self.bounds.energy_j
                                )
                                verification = self._evaluate(state, candidate)
                                key = self._terminal_key(verification)
                                if key < best_key:
                                    best_key = key
                                    best_values = candidate
                                    best_verification = verification
                    if not np.array_equal(best_values, values):
                        values = best_values
                        incumbent = best_verification
                        changed = True
                if not changed:
                    break
        return (
            tuple(PhysicalAction.from_array(row) for row in values),
            incumbent,
            trace,
        )

    def _refine_alternating_exact(
        self,
        state: VoxelState,
        actions: tuple[PhysicalAction, ...],
    ) -> tuple[tuple[PhysicalAction, ...], PlanVerification, RefinementTrace]:
        """Multiscale alternating block-coordinate whole-plan refinement."""
        values = np.asarray([action.as_array() for action in actions], dtype=float)
        incumbent = self._evaluate(state, values)
        trace = RefinementTrace(mode="alternating_exact")
        for scale_index, (position_scale, energy_scale) in enumerate(zip(
            self.config.refinement_position_scales_mm,
            self.config.refinement_energy_scales_j,
        )):
            for pass_index in range(self.config.refinement_passes_per_scale):
                spatial_changed = False
                spatial_movers: set[int] = set()
                for pulse in range(values.shape[0]):
                    best_values = values
                    best_verification = incumbent
                    best_key = self._terminal_key(incumbent)
                    for dx in (-position_scale, 0.0, position_scale):
                        for dy in (-position_scale, 0.0, position_scale):
                            candidate = values.copy()
                            candidate[pulse, 0] = np.clip(
                                candidate[pulse, 0] + dx, *self.bounds.x_mm
                            )
                            candidate[pulse, 1] = np.clip(
                                candidate[pulse, 1] + dy, *self.bounds.y_mm
                            )
                            verification = self._evaluate(state, candidate)
                            key = self._terminal_key(verification)
                            if key < best_key:
                                best_key = key
                                best_values = candidate
                                best_verification = verification
                    if not np.array_equal(best_values, values):
                        values = best_values
                        incumbent = best_verification
                        spatial_changed = True
                        spatial_movers.add(pulse)
                        trace.accepted_spatial_moves += 1
                        trace.spatial_acceptance_events.append(
                            (scale_index, pass_index, pulse)
                        )

                energy_changed = False
                for pulse in range(values.shape[0]):
                    best_values = values
                    best_verification = incumbent
                    best_key = self._terminal_key(incumbent)
                    for de in (-energy_scale, 0.0, energy_scale):
                        candidate = values.copy()
                        candidate[pulse, 4] = np.clip(
                            candidate[pulse, 4] + de, *self.bounds.energy_j
                        )
                        verification = self._evaluate(state, candidate)
                        key = self._terminal_key(verification)
                        if key < best_key:
                            best_key = key
                            best_values = candidate
                            best_verification = verification
                    if not np.array_equal(best_values, values):
                        values = best_values
                        incumbent = best_verification
                        energy_changed = True
                        trace.accepted_energy_moves += 1
                        trace.energy_acceptance_events.append(
                            (scale_index, pass_index, pulse)
                        )
                        other_spatial_pulses = tuple(
                            sorted(index for index in spatial_movers if index != pulse)
                        )
                        if other_spatial_pulses:
                            trace.cross_pulse_energy_events.append(
                                (
                                    scale_index,
                                    pass_index,
                                    pulse,
                                    other_spatial_pulses,
                                )
                            )
                if not spatial_changed and not energy_changed:
                    break
        return (
            tuple(PhysicalAction.from_array(row) for row in values),
            incumbent,
            trace,
        )

    def _evaluate(self, state: VoxelState, values: np.ndarray) -> PlanVerification:
        self.exact_plan_evaluations += 1
        return self.verifier.evaluator.evaluate(
            state,
            tuple(PhysicalAction.from_array(row) for row in values),
        )

    def _terminal_key(
        self, verification: PlanVerification
    ) -> tuple[float, ...]:
        return exact_terminal_key(
            verification,
            self.verifier.completion_remaining_pct,
            self.verifier.hard_margin_mm,
        )


def _target_columns(state: VoxelState) -> tuple[np.ndarray, np.ndarray]:
    remaining = state.tissue & state.target_mask
    column_weights = np.count_nonzero(remaining, axis=2)
    indices = np.argwhere(column_weights > 0)
    points = np.column_stack(
        (state.x_axis_mm[indices[:, 0]], state.y_axis_mm[indices[:, 1]])
    ).astype(float)
    return points, column_weights[tuple(indices.T)].astype(float)


def _principal_axis(
    points: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centre = np.average(points, axis=0, weights=weights)
    centred = points - centre
    covariance = (centred * weights[:, None]).T @ centred / np.sum(weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    first_nonzero = int(np.flatnonzero(np.abs(axis) > 1.0e-12)[0])
    if axis[first_nonzero] < 0.0:
        axis = -axis
    return axis, centre


def _principal_quantile_layout(
    points: np.ndarray,
    weights: np.ndarray,
    axis: np.ndarray,
    centre: np.ndarray,
    count: int,
) -> np.ndarray:
    projection = (points - centre) @ axis
    order = np.argsort(projection, kind="stable")
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1]
    anchors = []
    for quantile in (np.arange(count, dtype=float) + 0.5) / count:
        source = points[order[int(np.searchsorted(cumulative, quantile * total))]]
        anchors.append(source)
    return np.asarray(anchors, dtype=float)


def _weighted_kmeans_layout(
    points: np.ndarray, weights: np.ndarray, count: int
) -> np.ndarray:
    centres = [np.average(points, axis=0, weights=weights)]
    minimum_distance = np.sum((points - centres[0]) ** 2, axis=1)
    while len(centres) < count:
        index = int(np.argmax(minimum_distance * weights))
        centres.append(points[index])
        minimum_distance = np.minimum(
            minimum_distance,
            np.sum((points - centres[-1]) ** 2, axis=1),
        )
    centres_array = np.asarray(centres, dtype=float)
    for _ in range(32):
        labels = np.argmin(
            np.sum(
                (points[:, None, :] - centres_array[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        updated = np.asarray(
            [
                np.average(points[labels == group], axis=0, weights=weights[labels == group])
                for group in range(count)
            ],
            dtype=float,
        )
        if np.allclose(updated, centres_array, rtol=0.0, atol=1.0e-12):
            break
        centres_array = updated
    return np.asarray(
        [
            points[int(np.argmin(np.sum((points - centre) ** 2, axis=1)))]
            for centre in centres_array
        ],
        dtype=float,
    )


def _ordered_anchors(
    anchors: np.ndarray,
    axis: np.ndarray,
    centre: np.ndarray,
    pattern: str,
) -> np.ndarray:
    projection = (anchors - centre) @ axis
    if pattern == "forward":
        order = np.argsort(projection, kind="stable")
    elif pattern == "reverse":
        order = np.argsort(-projection, kind="stable")
    elif pattern == "center_out":
        order = np.lexsort((projection, np.abs(projection)))
    else:
        raise ValueError(f"unknown anchor order: {pattern}")
    return anchors[order]


__all__ = [
    "GeometryAwarePlanConfig",
    "GeometryAwarePlanGenerator",
    "RefinementTrace",
]
