"""Fixed-record segment beams for bounded local-linear MPPI repair."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from laser_ablation.planning.jax_bank.contracts import ACTION_DIMENSION, StaticTaskTensors, action_hash, trajectory_hash
from laser_ablation.planning.jax_bank.linear_contracts import LinearizationLibrary
from laser_ablation.planning.jax_bank.mppi_records import provisional_metrics_score
from laser_ablation.planning.jax_bank.repair import FeasibleRepairTrajectory

if TYPE_CHECKING:
    from laser_ablation.planning.jax_bank.mppi_execution import MPPIExecutor


@dataclass(frozen=True)
class SegmentBeam:
    """Retained segment-repair parents with immutable nominal-tail provenance."""

    actions: np.ndarray
    origin_actions: np.ndarray
    action_mask: np.ndarray
    linearization_path: np.ndarray
    source_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    anchor_slots: np.ndarray | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        origin = np.asarray(self.origin_actions, dtype=np.float32)
        mask = np.asarray(self.action_mask, dtype=bool)
        path = np.asarray(self.linearization_path, dtype=np.int32)
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIMENSION:
            raise ValueError("segment beam actions must have shape (row, pulse, 5)")
        if origin.shape != actions.shape:
            raise ValueError("segment beam origin actions must align with actions")
        if mask.shape != actions.shape[:2]:
            raise ValueError("segment beam action mask must align with actions")
        if path.shape != actions.shape[:2] + (2,):
            raise ValueError("segment beam linearization path must have shape (row, pulse, 2)")
        count = actions.shape[0]
        slots = (
            np.arange(count, dtype=np.int32)
            if self.anchor_slots is None else np.asarray(self.anchor_slots, dtype=np.int32)
        )
        if count <= 0:
            raise ValueError("segment beam requires at least one parent")
        if any(len(values) != count for values in (self.source_ids, self.anchor_ids, self.candidate_ids)):
            raise ValueError("segment beam identifiers must contain one value per parent")
        if len(set(self.candidate_ids)) != count:
            raise ValueError("segment beam candidate identifiers must be unique")
        if slots.shape != (count,) or np.any(slots < 0):
            raise ValueError("segment beam anchor slots must contain one nonnegative slot per parent")
        if np.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("segment beam action masks must be prefix-contiguous")
        if np.any(path[mask] < 0):
            raise ValueError("active segment beam paths must be nonnegative")
        if np.any(path[~mask] != -1):
            raise ValueError("masked segment beam paths must equal [-1, -1]")
        if np.any(actions[~mask] != 0.0) or np.any(origin[~mask] != 0.0):
            raise ValueError("masked segment beam actions must be zero")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(origin)):
            raise ValueError("segment beam actions must be finite")
        for values in (actions, origin, mask, path, slots):
            values.setflags(write=False)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "origin_actions", origin)
        object.__setattr__(self, "action_mask", mask)
        object.__setattr__(self, "linearization_path", path)
        object.__setattr__(self, "anchor_slots", slots)
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "anchor_ids", tuple(self.anchor_ids))
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))

    @property
    def count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def maximum_pulses(self) -> int:
        return int(self.actions.shape[1])

    def record_bytes(self, state_shape: tuple[int, int, int]) -> int:
        voxels = int(np.prod(state_shape))
        if voxels <= 0:
            raise ValueError("segment beam state shape must contain voxels")
        pulses = self.maximum_pulses
        frames = pulses + 1
        beam_bytes = (2 * pulses * ACTION_DIMENSION * 4) + pulses + (2 * pulses * 4) + 4
        trace_bytes = (frames * voxels * 4) + (8 * frames * 4) + (9 * pulses) + 2
        return int(beam_bytes + trace_bytes)

    def segment_bounds(
        self, segment_index: int, segment_length_pulses: int, minimum_pulses: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if segment_index < 0 or segment_length_pulses <= 0 or minimum_pulses <= 0:
            raise ValueError("segment index must be nonnegative and segment lengths must be positive")
        if minimum_pulses > segment_length_pulses:
            raise ValueError("segment minimum cannot exceed the fixed segment length")
        lengths = np.count_nonzero(self.action_mask, axis=1).astype(np.int32)
        starts = np.minimum(np.int32(segment_index * segment_length_pulses), lengths)
        stops = np.minimum(starts + np.int32(segment_length_pulses), lengths)
        return starts, stops, stops - starts >= minimum_pulses

    def take(self, indices: np.ndarray) -> "SegmentBeam":
        """Copy a deterministic retained subset into a new immutable beam."""
        rows = np.asarray(indices, dtype=np.int32)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError("segment beam selection requires at least one row")
        if np.any(rows < 0) or np.any(rows >= self.count):
            raise ValueError("segment beam selection indices are out of range")
        return SegmentBeam(
            self.actions[rows].copy(), self.origin_actions[rows].copy(),
            self.action_mask[rows].copy(), self.linearization_path[rows].copy(),
            tuple(self.source_ids[index] for index in rows),
            tuple(self.anchor_ids[index] for index in rows),
            tuple(self.candidate_ids[index] for index in rows),
            self.anchor_slots[rows].copy(),
        )


def stream_linearized_segment(
    executor: "MPPIExecutor",
    seed: int,
    stage: int,
    beam: SegmentBeam,
    starts: np.ndarray,
    stops: np.ndarray,
    task: StaticTaskTensors,
    library: LinearizationLibrary,
    initial_current_sdf: np.ndarray,
    capacity_records: int,
    microbatch_size: int,
) -> tuple[SegmentBeam | None, tuple[FeasibleRepairTrajectory, ...], dict[str, object]]:
    """Sample one interval per parent and retain a global bounded hard-feasible reservoir."""
    if capacity_records <= 0 or microbatch_size <= 0:
        raise ValueError("segment capacity and microbatch size must be positive")
    if getattr(executor, "workspace", None) is None:
        raise RuntimeError("active segment repair requires an installed recentered workspace")
    starts, stops = np.asarray(starts, np.int32), np.asarray(stops, np.int32)
    lengths = np.count_nonzero(beam.action_mask, axis=1).astype(np.int32)
    if starts.shape != (beam.count,) or stops.shape != (beam.count,):
        raise ValueError("segment bounds must contain one start and stop per parent")
    if np.any(starts < 0) or np.any(stops <= starts) or np.any(stops > lengths):
        raise ValueError("segment streaming requires unfinished bounds inside each parent tail")
    initial = np.asarray(initial_current_sdf, np.float32)
    if initial.shape != task.shape or not np.all(np.isfinite(initial)):
        raise ValueError("segment rollout requires one finite observed SDF matching the task")

    samples_per_parent = executor.config.samples_per_anchor
    parent_chunk = max(1, microbatch_size // samples_per_parent)
    pulses = np.arange(beam.maximum_pulses, dtype=np.int32)[None]
    retained: list[tuple[tuple[float, float, str], tuple[object, ...]]] = []
    rejection_counts: dict[str, int] = {}
    sampling_seconds = rollout_seconds = ranking_seconds = 0.0
    child_count = hard_feasible_count = admissible_count = support_rejections = trust_rejections = 0
    contact_guard_rejections = 0
    summary_before = executor.workspace_summary_rows_evaluated
    trace_before = executor.workspace_full_trace_rows_fetched

    for parent_start in range(0, beam.count, parent_chunk):
        parent_rows = np.arange(parent_start, min(parent_start + parent_chunk, beam.count), dtype=np.int32)
        parent = beam.take(parent_rows)
        perturb_mask = (
            (pulses >= starts[parent_rows, None])
            & (pulses < stops[parent_rows, None]) & parent.action_mask
        )
        sampled_at = perf_counter()
        samples, _ = executor.sample(
            seed, stage, parent.candidate_ids, np.arange(parent.count, dtype=np.int32),
            parent.actions, parent.action_mask, perturb_mask, task,
        )
        sampling_seconds += perf_counter() - sampled_at
        actions = samples.reshape(-1, beam.maximum_pulses, ACTION_DIMENSION)
        origins = np.repeat(parent.origin_actions, samples_per_parent, axis=0)
        masks = np.repeat(parent.action_mask, samples_per_parent, axis=0)
        paths = np.repeat(parent.linearization_path, samples_per_parent, axis=0)
        slots = np.repeat(parent.anchor_slots, samples_per_parent, axis=0)
        parent_rows = np.repeat(np.arange(parent.count, dtype=np.int32), samples_per_parent)
        sample_rows = np.tile(np.arange(samples_per_parent, dtype=np.int32), parent.count)

        for child_start in range(0, len(actions), microbatch_size):
            child_stop = min(child_start + microbatch_size, len(actions))
            child_actions, child_origins = actions[child_start:child_stop], origins[child_start:child_stop]
            child_masks, child_paths = masks[child_start:child_stop], paths[child_start:child_stop]
            child_slots = slots[child_start:child_stop]
            child_parents, child_samples = parent_rows[child_start:child_stop], sample_rows[child_start:child_stop]
            diagnostics_ids = tuple(
                f"linear_segment:{parent.candidate_ids[row]}:{stage}:{int(sample)}"
                for row, sample in zip(child_parents, child_samples, strict=True)
            )
            rolled_at = perf_counter()
            initial_rows = np.broadcast_to(initial, (len(child_actions),) + task.shape).copy()
            summary = executor.rollout_workspace_summary(
                child_actions, child_origins, child_masks, child_paths, child_slots, initial_rows, task,
            )
            rollout_seconds += perf_counter() - rolled_at
            ranked_at = perf_counter()
            child_count += len(child_actions)
            hard_rows = np.flatnonzero(summary.constraint_feasible)
            hard_feasible_count += len(hard_rows)
            admissible_count += int(np.count_nonzero(summary.accepted))
            for name, values in summary.flags.items():
                rejection_counts[name] = rejection_counts.get(name, 0) + int(np.count_nonzero(values))
            support_rejections += int(np.count_nonzero(summary.flags["linearization_support"]))
            contact_guard_rejections += int(np.count_nonzero(
                summary.flags["linearization_contact_guard"]
            ))
            trust_rejections += int(np.count_nonzero(summary.flags["linearization_trust"]))
            candidates: list[tuple[tuple[float, float, str], int, str]] = []
            for row in hard_rows:
                parent_row = int(child_parents[row])
                digest = sha256()
                digest.update(parent.candidate_ids[parent_row].encode())
                digest.update(parent.source_ids[parent_row].encode())
                digest.update(diagnostics_ids[row].encode())
                digest.update(np.ascontiguousarray(child_actions[row]).tobytes())
                digest.update(np.ascontiguousarray(child_paths[row]).tobytes())
                digest.update(np.asarray(child_slots[row], dtype=np.int32).tobytes())
                identifier = digest.hexdigest()
                candidates.append((
                    (
                        float(summary.mppi_cost[row]),
                        provisional_metrics_score(
                            task,
                            float(summary.remaining_fraction[row]),
                            float(summary.overcut_fraction[row]),
                            float(summary.clearance_mm[row]),
                            int(summary.pulse_count[row]),
                            beam.maximum_pulses,
                        ),
                        identifier,
                    ),
                    int(row),
                    identifier,
                ))
            choices = [(key, False, index, "") for index, (key, _) in enumerate(retained)]
            choices.extend((key, True, row, identifier) for key, row, identifier in candidates)
            retained_next: list[tuple[tuple[float, float, str], tuple[object, ...]]] = []
            for key, new, row, identifier in sorted(choices, key=lambda value: value[0])[:capacity_records]:
                if not new:
                    retained_next.append(retained[row])
                    continue
                parent_row, length = int(child_parents[row]), int(np.count_nonzero(child_masks[row]))
                source_id = parent.source_ids[parent_row]
                retained_next.append((key, (
                    child_actions[row].copy(), child_origins[row].copy(), child_masks[row].copy(),
                    child_paths[row].copy(), source_id, parent.anchor_ids[parent_row], identifier,
                    np.int32(child_slots[row]), int(child_samples[row]), bool(summary.accepted[row]),
                    float(summary.mppi_cost[row]), tuple(map(float, summary.cost_components[row])),
                )))
            retained = retained_next
            ranking_seconds += perf_counter() - ranked_at

    record_bytes = beam.record_bytes(task.shape)
    stats: dict[str, object] = {
        "parent_count": beam.count, "child_count": child_count,
        "hard_feasible_count": hard_feasible_count, "admissible_count": admissible_count,
        "retained_count": len(retained), "capacity_records": capacity_records,
        "record_bytes": record_bytes, "used_record_bytes": len(retained) * record_bytes,
        "support_rejections": support_rejections,
        "contact_guard_rejections": contact_guard_rejections,
        "linearization_trust_rejections": trust_rejections,
        "rejection_counts": rejection_counts,
        "summary_rows_evaluated": executor.workspace_summary_rows_evaluated - summary_before,
        "full_trace_rows_fetched": executor.workspace_full_trace_rows_fetched - trace_before,
        "sampling_seconds": sampling_seconds, "rollout_seconds": rollout_seconds,
        "ranking_seconds": ranking_seconds,
    }
    if not retained:
        return None, (), stats
    records = tuple(values for _, values in retained)
    next_beam = SegmentBeam(
        np.stack([values[0] for values in records]), np.stack([values[1] for values in records]),
        np.stack([values[2] for values in records]), np.stack([values[3] for values in records]),
        tuple(values[4] for values in records), tuple(values[5] for values in records),
        tuple(values[6] for values in records), np.asarray([values[7] for values in records], np.int32),
    )
    trajectories: list[FeasibleRepairTrajectory] = []
    for trace_start in range(0, len(records), executor.workspace.rollout_rows):
        trace_records = records[trace_start:trace_start + executor.workspace.rollout_rows]
        traced_at = perf_counter()
        rollout = executor.rollout_workspace(
            np.stack([values[0] for values in trace_records]),
            np.stack([values[2] for values in trace_records]),
            np.stack([values[3] for values in trace_records]),
            np.asarray([values[7] for values in trace_records], np.int32),
            np.broadcast_to(initial, (len(trace_records),) + task.shape).copy(),
        )
        rollout_seconds += perf_counter() - traced_at
        for row, values in enumerate(trace_records):
            actions, _, mask, path, source_id, anchor_id, _, _, sample, admissible, cost, components = values
            length = int(np.count_nonzero(mask))
            if not rollout.constraint_feasible[row] or bool(rollout.accepted[row]) != admissible:
                raise RuntimeError("workspace terminal screening and retained trace disagree")
            trajectories.append(FeasibleRepairTrajectory(
                trajectory_id=trajectory_hash(actions[:length], initial, path[:length], source_id),
                action_hash=action_hash(actions[:length]), anchor_tail_id=anchor_id, source_id=source_id,
                iteration=stage, sample=sample, actions=actions[:length].copy(),
                linearization_path=path[:length].copy(), current_sdf=rollout.current_sdf[row, :length + 1].copy(),
                remaining_voxels=rollout.remaining_voxels[row, :length + 1].copy(),
                overcut_voxels=rollout.overcut_voxels[row, :length + 1].copy(),
                remaining_fraction=rollout.remaining_fraction[row, :length + 1].copy(),
                overcut_fraction=rollout.overcut_fraction[row, :length + 1].copy(),
                healthy_overcut_fraction=rollout.healthy_overcut_fraction[row, :length + 1].copy(),
                clearance_mm=rollout.clearance_mm[row, :length + 1].copy(),
                pulse_count=rollout.pulse_count[row, :length + 1].copy(),
                energy_j=rollout.energy_j[row, :length + 1].copy(),
                flags={name: flag_values[row, :length].copy() for name, flag_values in rollout.flags.items()},
                mppi_cost=cost, cost_components=components, admissible=admissible,
            ))
    stats["full_trace_rows_fetched"] = executor.workspace_full_trace_rows_fetched - trace_before
    stats["rollout_seconds"] = rollout_seconds
    return next_beam, tuple(trajectories), stats
