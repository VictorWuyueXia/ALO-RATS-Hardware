"""Auditable Operation-2 repair artifacts and exact-schema diagnostic tables."""

from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from laser_ablation.planning.jax_bank.contracts import STATE_SHARD_BYTES, MatchingTailBank, StaticTaskTensors
from laser_ablation.planning.jax_bank.repair import MPPIRepairConfig, RepairContext, RepairResult


SEGMENT_TRACE_COLUMNS = (
    "anchor_tail_id", "segment_index", "segment_start", "segment_stop", "tail_length",
    "parent_count", "child_count", "hard_feasible_count", "admissible_count", "retained_count",
    "retained_bytes", "beam_capacity_records", "beam_capacity_bytes", "compacted",
    "pre_compaction_count", "post_compaction_count", "support_rejections", "sampling_seconds",
    "rollout_seconds", "ranking_seconds", "elapsed_seconds",
)
CANDIDATE_COLUMNS = (
    "trajectory_id", "action_hash", "anchor_tail_id", "source_id", "library_fingerprint",
    "segment_index", "mppi_cost", "provisional_score", "rank", "hard_feasible", "admissible",
    "remaining_fraction", "overcut_fraction", "clearance_mm", "linearization_support",
)
REBASE_COLUMNS = (
    "trajectory_id", "anchor_tail_id", "segment_index", "linear_mppi_cost", "nonlinear_mppi_cost",
    "state_relative_error", "sdf_sign_disagreement", "linear_hard_feasible", "nonlinear_hard_feasible",
    "linear_admissible", "nonlinear_admissible", "linear_rank", "nonlinear_rank",
)

def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_table(
    path: Path, rows: Sequence[Mapping[str, object]], columns: tuple[str, ...]
) -> None:
    for row in rows:
        if set(row) != set(columns):
            missing = tuple(name for name in columns if name not in row)
            extra = tuple(name for name in row if name not in columns)
            raise ValueError(f"{path.name} schema mismatch missing={missing} extra={extra}")
    stream = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows({
        name: (json.dumps(row[name], sort_keys=True, default=_json_default)
               if isinstance(row[name], (dict, list, tuple)) else row[name])
        for name in columns
    } for row in rows)
    stream.close()


def _route_fields(item: object) -> dict[str, object]:
    path = np.ascontiguousarray(np.asarray(item.linearization_path, dtype=np.int32))
    fields: dict[str, object] = {"linearization_route_sha256": sha256(path.tobytes()).hexdigest()}
    for name in ("source_id", "route_hash"):
        value = getattr(item, name, None)
        if value is not None:
            fields[name] = value
    return fields


def _sample_payload(index: int, item: object) -> tuple[dict[str, np.ndarray], int]:
    prefix = f"sample_{index:06d}"
    arrays = {
        f"{prefix}_actions": item.actions,
        f"{prefix}_linearization_path": item.linearization_path,
        f"{prefix}_current_sdf": item.current_sdf.astype(np.float16),
        f"{prefix}_remaining_voxels": item.remaining_voxels,
        f"{prefix}_overcut_voxels": item.overcut_voxels,
        f"{prefix}_remaining_fraction": item.remaining_fraction,
        f"{prefix}_overcut_fraction": item.overcut_fraction,
        f"{prefix}_healthy_overcut_fraction": item.healthy_overcut_fraction,
        f"{prefix}_clearance_mm": item.clearance_mm,
        f"{prefix}_pulse_count": item.pulse_count,
        f"{prefix}_energy_j": item.energy_j,
    }
    arrays.update({f"{prefix}_flag_{name}": values for name, values in item.flags.items()})
    return arrays, sum(value.nbytes for value in arrays.values())


def _write_feasible_shards(directory: Path, result: RepairResult) -> list[dict[str, object]]:
    index_records: list[dict[str, object]] = []
    payload: dict[str, np.ndarray] = {}
    payload_bytes, shard, pending = 0, 0, []

    def flush() -> None:
        nonlocal payload, payload_bytes, shard, pending
        if not payload:
            return
        filename = f"feasible_states_{shard:04d}.npz"
        print(f"saving repair state shard={shard + 1} samples={len(pending)}", flush=True)
        np.savez_compressed(directory / filename, **payload)
        for record_index in pending:
            index_records[record_index]["shard"] = filename
        payload, payload_bytes, pending, shard = {}, 0, [], shard + 1

    for index, item in enumerate(result.feasible_trajectories):
        arrays, item_bytes = _sample_payload(index, item)
        if payload and payload_bytes + item_bytes > STATE_SHARD_BYTES:
            flush()
        index_records.append({
            "key": f"sample_{index:06d}", "trajectory_id": item.trajectory_id,
            "action_hash": item.action_hash, "anchor_tail_id": item.anchor_tail_id,
            "iteration": item.iteration, "sample": item.sample,
            "action_count": len(item.actions),
            "mppi_cost": item.mppi_cost, "cost_components": item.cost_components,
            "admissible": item.admissible, **_route_fields(item),
        })
        payload.update(arrays)
        payload_bytes += item_bytes
        pending.append(index)
    flush()
    return index_records


def _write_human_plots(
    directory: Path, segment_trace: Sequence[Mapping[str, object]],
    rebase_comparison: Sequence[Mapping[str, object]],
    memory_accounting: Mapping[str, object] | None,
    timing: Mapping[str, object],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    steps = np.arange(len(segment_trace))
    if segment_trace:
        children = np.asarray([row["child_count"] for row in segment_trace])
        retained = np.asarray([row["retained_count"] for row in segment_trace])
        used = np.asarray([row["retained_bytes"] for row in segment_trace]) / 1024**3
        capacity = np.asarray([row["beam_capacity_bytes"] for row in segment_trace]) / 1024**3
        axes[0].plot(steps, children, "o-", label="children evaluated")
        axes[0].plot(steps, retained, "o-", label="retained beam")
        axes[1].plot(steps, used, "o-", label="retained records")
        axes[1].plot(steps, capacity, "--", label="8-GiB-derived capacity")
        duration = axes[0].twinx()
        duration.plot(steps, [row["elapsed_seconds"] for row in segment_trace], "s-",
                      color="#785EF0", label="linear segment seconds")
        duration.set_ylabel("seconds")
        duration.legend(loc="lower right")
    if memory_accounting is not None:
        for name, label, color in (("sampled_baseline_per_device_mib", "sampled device baseline", "#6A6A6A"),
                                  ("sampled_peak_per_device_mib", "sampled device peak", "#B26A00")):
            values = memory_accounting.get(name)
            if isinstance(values, (list, tuple)) and values:
                axes[1].axhline(max(float(value) for value in values) / 1024.0,
                                    color=color, linestyle=":", label=label)
    axes[0].set(xlabel="segment event", ylabel="trajectory count")
    axes[1].set(xlabel="segment event", ylabel="GiB per GPU")
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(loc="best")
    elapsed = float(timing.get("repair_elapsed_seconds", 0.0))
    figure.suptitle(f"total evidence time: {elapsed:.3f} s")
    figure.savefig(directory / "segment_beam_compute_memory.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    if rebase_comparison:
        linear = np.asarray([row["linear_mppi_cost"] for row in rebase_comparison])
        nonlinear = np.asarray([row["nonlinear_mppi_cost"] for row in rebase_comparison])
        error = np.asarray([row["state_relative_error"] for row in rebase_comparison])
        axes[0].scatter(linear, nonlinear, s=20, alpha=0.8)
        limits = np.asarray((linear.min(), linear.max(), nonlinear.min(), nonlinear.max()))
        axes[0].plot([limits.min(), limits.max()], [limits.min(), limits.max()], "k--")
        axes[1].plot(np.arange(len(error)), error, "o-")
    axes[0].set(xlabel="linear terminal MPPI cost", ylabel="nonlinear terminal MPPI cost")
    axes[1].set(xlabel="comparison candidate", ylabel="relative SDF error")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(directory / "segment_beam_rebase_comparison.png", dpi=180)
    plt.close(figure)


def _write_manifest(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    records = {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest(),
            "audience": "machine" if "machine_readables" in path.parts else "human",
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    _write_json(manifest_path, {"artifact_count": len(records), "artifacts": records})


def validate_repair_artifact_manifest(directory: Path) -> None:
    directory = Path(directory)
    records = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["artifacts"]
    actual = {
        str(path.relative_to(directory)) for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(records):
        raise ValueError("repair artifact manifest file set mismatch")
    for name, record in records.items():
        path = directory / name
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"repair artifact manifest size mismatch for {name}")
        if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"repair artifact manifest hash mismatch for {name}")


def save_repair_artifacts(
    directory: Path, context: RepairContext, task: StaticTaskTensors,
    anchors: MatchingTailBank, config: MPPIRepairConfig, result: RepairResult, *,
    segment_trace: Sequence[Mapping[str, object]] = (),
    segment_candidates: Sequence[Mapping[str, object]] = (),
    rebase_comparison: Sequence[Mapping[str, object]] = (),
    memory_accounting: Mapping[str, object] | None = None,
    timing: Mapping[str, object] | None = None,
    retained_arrays: Mapping[str, np.ndarray] | None = None,
    interpretation_summary: str | None = None,
) -> Path:
    """Persist one repair, segment-beam diagnostics, and a SHA-256 file manifest."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite MPPI repair artifacts: {directory}")
    machine, human = directory / "machine_readables", directory / "human_readables"
    machine.mkdir(parents=True)
    human.mkdir()
    request = context.request
    _write_json(machine / "repair_request.json", {
        "task_fingerprint": task.fingerprint,
        "observed_current_sdf_sha256": sha256(np.ascontiguousarray(task.initial_current_sdf).tobytes()).hexdigest(),
        "active_trajectory_id": request.active_trajectory_id,
        "active_prefix": request.active_prefix, "active_similarity": context.active_similarity,
        "trigger": request.trigger.value,
    })
    _write_json(machine / "anchor_selection.json", {
        "matched": [{"tail_id": tail.tail_id, "similarity": tail.similarity,
                     "score": tail.score, "remaining_pulses": tail.remaining_pulses,
                     "trajectory_ids": tail.trajectory_ids} for tail in anchors.tails],
        "selected": result.diagnostics.selected_anchors,
        "dropped": result.diagnostics.dropped_anchors,
    })
    import jax

    _write_json(machine / "rng_provenance.json", {
        "random_seed": request.random_seed,
        "fold_in": "iteration, first 32 bits of ordered tail SHA-256, then sample index",
        "config": asdict(config), "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(), "jax_devices": [str(device) for device in jax.devices()],
    })
    _write_json(machine / "iteration_summary.json", [asdict(item) for item in result.diagnostics.iterations])
    _write_json(machine / "rejection_ledger.json", [
        {"iteration": item.iteration, **item.rejection_counts} for item in result.diagnostics.iterations
    ])
    _write_json(machine / "feasible_sample_index.json", _write_feasible_shards(machine, result))
    selected = result.selected
    _write_json(machine / "selected_repair.json", {
        "trajectory_id": selected.trajectory_id, "action_hash": selected.action_hash,
        "anchor_tail_id": selected.anchor_tail_id, "iteration": selected.iteration,
        "sample": selected.sample, "mppi_cost": selected.mppi_cost,
        "cost_components": selected.cost_components, "action_count": len(selected.actions),
        "linearization_path": selected.linearization_path, **_route_fields(selected),
    })
    write_table(machine / "segment_trace.csv", segment_trace, SEGMENT_TRACE_COLUMNS)
    write_table(machine / "segment_candidates.csv", segment_candidates, CANDIDATE_COLUMNS)
    write_table(machine / "rebase_comparison.csv", rebase_comparison, REBASE_COLUMNS)
    if memory_accounting is not None:
        _write_json(machine / "memory_accounting.json", dict(memory_accounting))
    values = {"repair_elapsed_seconds": result.diagnostics.elapsed_seconds}
    values.update(timing or {})
    _write_json(machine / "timing.json", values)
    if retained_arrays is not None:
        np.savez_compressed(machine / "retained_arrays.npz", **retained_arrays)
    _write_json(machine / "artifact_schema.json", {
        "schema_version": "operation2_segment_rebase_v1",
        "segment_trace_columns": SEGMENT_TRACE_COLUMNS,
        "segment_candidate_columns": CANDIDATE_COLUMNS,
        "rebase_comparison_columns": REBASE_COLUMNS,
        "state_trace_storage": {"feasible_states": "float16", "exact_round_trip": False},
    })
    (human / "interpretation_summary.md").write_text(
        interpretation_summary or "# Operation-2 segment-beam repair\n\n"
        "Nonlinear values are diagnostic only and never alter active linear selection.\n",
        encoding="utf-8",
    )
    if segment_trace or rebase_comparison:
        _write_human_plots(human, segment_trace, rebase_comparison, memory_accounting, values)
    _write_manifest(directory)
    return directory
