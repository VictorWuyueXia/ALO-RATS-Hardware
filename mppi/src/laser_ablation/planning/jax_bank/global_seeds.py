"""Configured raster and optional geometry-aware sources for JAX MPPI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_ablation.core.state import VoxelState
from laser_ablation.planning.global_3d.interface import GlobalPlanningFailure, canonical_action_sequence_hash


@dataclass(frozen=True)
class RasterGlobalSeeds:
    """Four raster parents plus up to six geometry-aware replan parents."""

    actions: tuple[np.ndarray, ...]
    source_ids: tuple[str, ...]
    canonical_hashes: tuple[str, ...]
    geometry_requested: bool
    geometry_source_failure: str | None

    def __post_init__(self) -> None:
        if not 4 <= len(self.actions) <= 10:
            raise ValueError("global seeds require four through ten parents")
        if len(self.actions) != len(self.source_ids) or len(self.actions) != len(self.canonical_hashes):
            raise ValueError("raster actions, source IDs, and canonical hashes must align")
        if len(set(self.source_ids)) != len(self.source_ids) or not all(self.source_ids):
            raise ValueError("raster source IDs must be unique and nonempty")
        if not all(value.startswith("raster:") for value in self.source_ids[:4]):
            raise ValueError("the first four global seeds must be configured rasters")
        if not all(value.startswith("geometry-aware:") for value in self.source_ids[4:]):
            raise ValueError("global seeds after the four rasters must be geometry-aware")
        for actions in self.actions:
            if actions.ndim != 2 or actions.shape[1] != 5 or not len(actions):
                raise ValueError("raster actions must have finite nonempty shape (pulse, 5)")
            if actions.dtype != np.float32 or not np.all(np.isfinite(actions)):
                raise ValueError("raster actions must be finite float32 arrays")
            if actions.flags.writeable:
                raise ValueError("raster action arrays must be immutable")
        if any(len(value) != 64 for value in self.canonical_hashes):
            raise ValueError("raster canonical hashes must be SHA-256 digests")

    @classmethod
    def build(
        cls,
        voxel_state: VoxelState,
        raster_generator: object,
        geometry_generator: object | None,
    ) -> "RasterGlobalSeeds":
        """Materialize four rasters and optional exact-verified geometry-aware plans."""
        raster_candidates = tuple(raster_generator.generate(voxel_state))
        if len(raster_candidates) != 4:
            raise GlobalPlanningFailure("raster global source requires exactly four raw parents")
        candidates = list(raster_candidates)
        geometry_failure = None
        if geometry_generator is not None:
            try:
                candidates.extend(tuple(geometry_generator.generate(voxel_state))[:6])
            except GlobalPlanningFailure as failure:
                geometry_failure = str(failure)
        source_ids = [
            f"raster:{index}:{item.layout_id}:{item.tilt_pattern}:{item.energy_pattern}"
            for index, item in enumerate(candidates[:4])
        ]
        source_ids.extend(
            f"geometry-aware:{index}:{item.layout_id}:{item.tilt_pattern}:{item.energy_pattern}"
            for index, item in enumerate(candidates[4:])
        )
        return cls(
            tuple(_action_rows(item.actions) for item in candidates),
            tuple(source_ids),
            tuple(canonical_action_sequence_hash(item.actions) for item in candidates),
            geometry_generator is not None,
            geometry_failure,
        )

    @property
    def provenance(self) -> dict[str, object]:
        """Return JSON-ready raw-raster source identities."""
        return {
            "schema": "laser-ablation/global-seeds/v2",
            "raw_parent_count": len(self.actions),
            "raster_parent_count": 4,
            "geometry_parent_count": len(self.actions) - 4,
            "geometry_requested": self.geometry_requested,
            "geometry_source_failure": self.geometry_source_failure,
            "source_ids": list(self.source_ids),
            "canonical_hashes": list(self.canonical_hashes),
        }


def _action_rows(actions: tuple[object, ...]) -> np.ndarray:
    values = np.asarray([action.as_array() for action in actions], dtype=np.float32)
    if values.shape != (len(actions), 5) or not len(values):
        raise ValueError("raster seed actions must have nonempty shape (pulse, 5)")
    values = np.ascontiguousarray(values)
    values.setflags(write=False)
    return values


__all__ = ["RasterGlobalSeeds"]
