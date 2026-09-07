"""Hybrid exact-global seed assembly for the initial JAX MPPI population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_ablation.core.state import VoxelState
from laser_ablation.planning.global_3d.interface import (
    GlobalPlanningFailure,
    VerifiedGlobalPlan,
    canonical_action_sequence_hash,
)


@dataclass(frozen=True)
class HybridGlobalSeeds:
    """Ordered raster bootstraps plus exact-safe Frozen exploration records."""

    actions: tuple[np.ndarray, ...]
    source_ids: tuple[str, ...]
    raw_canonical_hashes: tuple[str, ...]
    raster_bootstrap_count: int
    nominal_actions: tuple[np.ndarray, ...]
    raw_to_nominal_seed: tuple[int, ...]
    frozen_reference: tuple[tuple[str, object], ...] | None
    frozen_candidates: tuple[tuple[tuple[str, object], ...], ...]
    frozen_construction_failure: tuple[tuple[str, str], ...] | None

    def __post_init__(self) -> None:
        if not 1 <= self.raster_bootstrap_count <= len(self.actions):
            raise ValueError("hybrid global seeds require at least one raster bootstrap")
        if len(self.actions) != len(self.source_ids) or len(self.actions) != len(self.raw_canonical_hashes):
            raise ValueError("hybrid action rows, source IDs, and canonical hashes must align")
        if len(self.actions) != 10:
            raise ValueError("hybrid seeds must fill exactly ten plan-bank lineages")
        if len(self.actions) != len(self.raw_to_nominal_seed):
            raise ValueError("hybrid raw-to-nominal mapping must align with action rows")
        if not self.nominal_actions:
            raise ValueError("hybrid seeds require at least one nominal action sequence")
        for actions in self.actions + self.nominal_actions:
            if actions.ndim != 2 or actions.shape[1] != 5 or not len(actions):
                raise ValueError("hybrid actions must have finite nonempty shape (pulse, 5)")
            if actions.dtype != np.float32 or not np.all(np.isfinite(actions)):
                raise ValueError("hybrid actions must be finite float32 arrays")
            if actions.flags.writeable:
                raise ValueError("hybrid action arrays must be immutable")
        for raw_index, nominal_index in enumerate(self.raw_to_nominal_seed):
            if not 0 <= nominal_index < len(self.nominal_actions):
                raise ValueError("hybrid nominal seed index lies outside the library")
            if not np.array_equal(self.actions[raw_index], self.nominal_actions[nominal_index]):
                raise ValueError("hybrid nominal seed does not match its raw source row")
        if not all(source_id for source_id in self.source_ids):
            raise ValueError("hybrid source IDs must be nonempty")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.raw_canonical_hashes
        ):
            raise ValueError("hybrid canonical action hashes must be SHA-256 digests")

    @classmethod
    def build(
        cls,
        voxel_state: VoxelState,
        raster_generator: object,
        frozen_generator: object,
    ) -> "HybridGlobalSeeds":
        """Create the exact-complete raster prefix and safe Frozen exploration suffix."""
        raster_candidates = tuple(raster_generator.generate(voxel_state))
        if not raster_candidates:
            raise GlobalPlanningFailure(
                "hybrid global source requires at least one exact-complete raster plan"
            )
        raster_bootstrap_count = len(raster_candidates)
        actions = [_action_rows(candidate.actions) for candidate in raster_candidates]
        source_ids = [
            (
                f"generated:raster:{index}:{candidate.layout_id}:"
                f"{candidate.tilt_pattern}:{candidate.energy_pattern}"
            )
            for index, candidate in enumerate(raster_candidates)
        ]
        canonical_hashes = [
            canonical_action_sequence_hash(candidate.actions)
            for candidate in raster_candidates
        ]
        reference: tuple[tuple[str, object], ...] | None = None
        candidate_provenance: list[tuple[tuple[str, object], ...]] = []
        construction_failure: tuple[tuple[str, str], ...] | None = None
        try:
            finalist = frozen_generator.propose_verified(voxel_state)
        except (GlobalPlanningFailure, ZeroDivisionError) as failure:
            construction_failure = (
                ("exception_type", type(failure).__name__),
                ("message", str(failure)),
            )
        else:
            records = tuple(frozen_generator.last_verified_candidates)
            if not records:
                raise RuntimeError("Frozen Global returned no verified candidate records")
            if not any(record.candidate_id == finalist.candidate_id for record in records):
                raise RuntimeError("Frozen Global finalist is absent from its verified records")
            for index, record in enumerate(records):
                exact_safe = bool(record.verification.valid and record.executed_actions)
                retained = exact_safe and len(actions) < 10
                raw_seed_index = len(actions) if retained else None
                provenance = _frozen_record_provenance(
                    record, raw_seed_index, record.candidate_id == finalist.candidate_id
                )
                candidate_provenance.append(provenance)
                if record.candidate_id == finalist.candidate_id:
                    reference = provenance
                if not retained:
                    continue
                actions.append(_action_rows(record.executed_actions))
                source_ids.append(
                    f"frozen:{index}:{record.candidate_id}:{record.executed_prefix_hash}"
                )
                canonical_hashes.append(record.executed_prefix_hash)
            if reference is None:
                raise RuntimeError("Frozen Global finalist provenance could not be recorded")
        reserve_index = 0
        while len(actions) < 10:
            raster_index = reserve_index % raster_bootstrap_count
            actions.append(actions[raster_index])
            source_ids.append(
                f"reserve:raster:{reserve_index}:{source_ids[raster_index]}"
            )
            canonical_hashes.append(canonical_hashes[raster_index])
            reserve_index += 1
        nominal_actions, raw_to_nominal_seed = _unique_nominal_actions(
            tuple(actions), tuple(canonical_hashes)
        )
        return cls(
            tuple(actions),
            tuple(source_ids),
            tuple(canonical_hashes),
            raster_bootstrap_count,
            nominal_actions,
            raw_to_nominal_seed,
            reference,
            tuple(candidate_provenance),
            construction_failure,
        )

    @property
    def provenance(self) -> dict[str, object]:
        """Return JSON-ready exact-source provenance without float32 hash conflation."""
        return {
            "schema": "laser-ablation/hybrid-global-seeds/v1",
            "raster_bootstrap_count": self.raster_bootstrap_count,
            "raw_seed_count": len(self.actions),
            "raster_reserve_count": sum(
                source_id.startswith("reserve:raster:") for source_id in self.source_ids
            ),
            "source_ids": list(self.source_ids),
            "raw_canonical_hashes": list(self.raw_canonical_hashes),
            "raw_to_nominal_seed": list(self.raw_to_nominal_seed),
            "frozen_reference": (
                None if self.frozen_reference is None else dict(self.frozen_reference)
            ),
            "frozen_candidates": [dict(record) for record in self.frozen_candidates],
            "frozen_source_construction_failed": (
                None
                if self.frozen_construction_failure is None
                else dict(self.frozen_construction_failure)
            ),
        }

    @property
    def nominal_canonical_hashes(self) -> tuple[str, ...]:
        """Return one pre-float32 identity for each unique nominal sequence."""
        values = ["" for _ in self.nominal_actions]
        for canonical_hash, nominal_index in zip(
            self.raw_canonical_hashes, self.raw_to_nominal_seed, strict=True
        ):
            if not values[nominal_index]:
                values[nominal_index] = canonical_hash
        return tuple(values)


def _action_rows(actions: tuple[object, ...]) -> np.ndarray:
    values = np.asarray([action.as_array() for action in actions], dtype=np.float32)
    if values.shape != (len(actions), 5) or not len(values):
        raise ValueError("global seed actions must have nonempty shape (pulse, 5)")
    values = np.ascontiguousarray(values)
    values.setflags(write=False)
    return values


def _frozen_record_provenance(
    record: VerifiedGlobalPlan, raw_seed_index: int | None, is_frozen_finalist: bool,
) -> tuple[tuple[str, object], ...]:
    metrics = record.verification.terminal_metrics
    exact_safe = bool(record.verification.valid and record.executed_actions)
    strict_complete = bool(
        exact_safe and metrics.remaining_pct <= record.completion_remaining_pct
    )
    return (
        ("candidate_id", record.candidate_id),
        ("planned_action_hash", record.planned_action_hash),
        ("executed_prefix_hash", record.executed_prefix_hash),
        ("state_hash", record.state_hash),
        ("verification_authority_hash", record.verification_authority_hash),
        ("exact_terminal_key", tuple(map(float, record.exact_terminal_key))),
        ("layout_id", record.layout_id),
        ("order_pattern", record.order_pattern),
        ("is_frozen_finalist", is_frozen_finalist),
        ("exact_safe", exact_safe),
        ("strict_complete", strict_complete),
        ("executed_pulses", len(record.executed_actions)),
        ("remaining_pct", float(metrics.remaining_pct)),
        ("total_overcut_pct", float(metrics.total_overcut_pct)),
        ("minimum_clearance_mm", float(metrics.minimum_clearance_mm)),
        ("hard_violations", int(metrics.hard_violations)),
        ("rejection_reason", record.verification.rejection_reason),
        ("raw_seed_index", raw_seed_index),
    )


def _unique_nominal_actions(
    actions: tuple[np.ndarray, ...], canonical_hashes: tuple[str, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    unique: list[np.ndarray] = []
    positions: dict[str, int] = {}
    raw_to_nominal: list[int] = []
    for action_row, key in zip(actions, canonical_hashes, strict=True):
        if key not in positions:
            positions[key] = len(unique)
            unique.append(action_row)
        elif not np.array_equal(action_row, unique[positions[key]]):
            raise RuntimeError("canonical action hash collision in hybrid seed library")
        raw_to_nominal.append(positions[key])
    return tuple(unique), tuple(raw_to_nominal)


__all__ = ["HybridGlobalSeeds"]
