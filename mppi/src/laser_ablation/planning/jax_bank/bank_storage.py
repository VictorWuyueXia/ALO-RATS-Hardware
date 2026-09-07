"""Reconstructable static, action, dynamic-shard, and provenance plan-bank storage."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from laser_ablation.physics.super_gaussian import PhysicsConfig
from laser_ablation.planning.jax_bank.contracts import (
    BankTrajectory, STATE_SHARD_BYTES, StaticTaskTensors, linearization_path_hash,
)


class PlanBankStorage:
    """Persistence behavior mixed into the in-memory plan bank."""

    def save(self, directory: Path, provenance: dict[str, Any]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "static_task.npz",
            x_axis_mm=self.task.x_axis_mm, y_axis_mm=self.task.y_axis_mm,
            z_axis_mm=self.task.z_axis_mm, initial_tissue=self.task.initial_tissue,
            target_mask=self.task.target_mask, constraint_mask=self.task.constraint_mask,
            initial_current_sdf=self.task.initial_current_sdf,
            target_sdf=self.task.target_sdf, safe_sdf=self.task.safe_sdf,
            physical_clearance_mm=self.task.physical_clearance_mm,
            lower_bounds=self.task.lower_bounds, upper_bounds=self.task.upper_bounds,
        )
        metadata = {
            "plane_z_mm": self.task.plane_z_mm, "spacing_mm": self.task.spacing_mm,
            "truncation_mm": self.task.truncation_mm, "hard_margin_mm": self.task.hard_margin_mm,
            "completion_remaining_fraction": self.task.completion_remaining_fraction,
            "physics": asdict(self.task.physics), "fingerprint": self.task.fingerprint,
            "maximum_pulses": self.maximum_pulses,
        }
        (directory / "static_task.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        actions = {
            key: value for record in self.trajectories for key, value in (
                (f"actions_{record.trajectory_id}", record.actions),
                (f"action_mask_{record.trajectory_id}", np.ones(len(record.actions), dtype=bool)),
                (f"linearization_path_{record.trajectory_id}", record.linearization_path),
            )
        }
        np.savez_compressed(directory / "actions.npz", **actions)
        voxel_count = int(np.prod(self.task.shape))
        packed_bytes = (voxel_count + 7) // 8
        bytes_per_prefix = voxel_count * np.dtype(np.float16).itemsize + 3 * packed_bytes
        chunk_size = max(1, STATE_SHARD_BYTES // bytes_per_prefix)
        index: dict[str, Any] = {
            "task_fingerprint": self.task.fingerprint,
            "state_shard_bytes": STATE_SHARD_BYTES,
            "total_bank_capacity_bytes": STATE_SHARD_BYTES,
            "bytes_per_prefix": bytes_per_prefix,
            "prefixes_per_shard": chunk_size,
            "total_raw_state_bytes": sum(
                len(record.current_sdf) * bytes_per_prefix for record in self.trajectories
            ),
            "trajectories": {},
        }
        for record in self.trajectories:
            index["trajectories"][record.trajectory_id] = {
                "origins": record.origins, "source_id": record.source_id,
                "active": record.active,
                "linearization_path_hash": linearization_path_hash(record.linearization_path),
                "score": record.score,
                "prefix_metrics": {
                    "remaining_fraction": record.remaining_fraction.tolist(),
                    "overcut_fraction": record.overcut_fraction.tolist(),
                    "clearance_mm": record.clearance_mm.tolist(),
                    "pulse_count": record.pulse_count.tolist(),
                    "energy_j": record.energy_j.tolist(),
                },
                "flags": {name: values.astype(bool).tolist() for name, values in record.flags.items()},
                "shards": [],
            }
        shard_index = 0
        for record in self.trajectories:
            for start in range(0, len(record.current_sdf), chunk_size):
                stop = min(start + chunk_size, len(record.current_sdf))
                filename = f"states_{shard_index:04d}.npz"
                shard_index += 1
                np.savez_compressed(
                    directory / filename, trajectory_id=np.array(record.trajectory_id),
                    start=np.array(start, dtype=np.int32),
                    current_sdf=record.current_sdf[start:stop].astype(np.float16),
                    remaining=np.packbits(record.masks["remaining"][start:stop], axis=-1),
                    removed=np.packbits(record.masks["removed"][start:stop], axis=-1),
                    overcut=np.packbits(record.masks["overcut"][start:stop], axis=-1),
                )
                index["trajectories"][record.trajectory_id]["shards"].append(
                    {"file": filename, "start": start, "stop": stop}
                )
        for name, payload in (
            ("bank_index.json", index), ("rejection_ledger.json", self.rejection_ledger),
        ):
            (directory / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        import jax

        environment = {
            "provenance": provenance, "python": platform.python_version(),
            "platform": platform.platform(), "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "stored_linearization_routes": {
                record.trajectory_id: {
                    "source_id": record.source_id, "active": record.active,
                    "linearization_path_hash": linearization_path_hash(record.linearization_path),
                }
                for record in self.trajectories
            },
        }
        (directory / "provenance.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )
        if self.linearization_library is not None:
            from laser_ablation.planning.jax_bank.linear_contracts import (
                save_linearization_library,
            )

            save_linearization_library(directory / "linearization", self.linearization_library)
        if self.comparison_library is not None:
            from laser_ablation.planning.jax_bank.comparison_library import (
                save_comparison_roi_library,
            )

            save_comparison_roi_library(directory / "comparison_roi", self.comparison_library)
        manifest = {
            str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for filename, expected_hash in manifest.items():
            path = directory / filename
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"manifest hash mismatch for {filename}")
        metadata = json.loads((directory / "static_task.json").read_text(encoding="utf-8"))
        static = np.load(directory / "static_task.npz")
        task = StaticTaskTensors(
            x_axis_mm=static["x_axis_mm"], y_axis_mm=static["y_axis_mm"],
            z_axis_mm=static["z_axis_mm"], initial_tissue=static["initial_tissue"],
            target_mask=static["target_mask"], constraint_mask=static["constraint_mask"],
            initial_current_sdf=static["initial_current_sdf"], target_sdf=static["target_sdf"],
            safe_sdf=static["safe_sdf"], physical_clearance_mm=static["physical_clearance_mm"],
            lower_bounds=static["lower_bounds"], upper_bounds=static["upper_bounds"],
            plane_z_mm=float(metadata["plane_z_mm"]), spacing_mm=float(metadata["spacing_mm"]),
            truncation_mm=float(metadata["truncation_mm"]),
            hard_margin_mm=float(metadata["hard_margin_mm"]),
            completion_remaining_fraction=float(metadata["completion_remaining_fraction"]),
            physics=PhysicsConfig(**metadata["physics"]), fingerprint=metadata["fingerprint"],
        )
        bank = cls(task, int(metadata["maximum_pulses"]))
        index = json.loads((directory / "bank_index.json").read_text(encoding="utf-8"))
        if index["task_fingerprint"] != task.fingerprint:
            raise ValueError("bank index and static task fingerprints disagree")
        actions = np.load(directory / "actions.npz")
        width = task.shape[-1]
        for identifier, item in index["trajectories"].items():
            current_parts: list[np.ndarray] = []
            masks = {name: [] for name in ("remaining", "removed", "overcut")}
            for location in item["shards"]:
                shard = np.load(directory / location["file"])
                current_parts.append(shard["current_sdf"].astype(np.float32))
                for name in masks:
                    masks[name].append(np.unpackbits(shard[name], axis=-1, count=width).astype(bool))
            metrics = item["prefix_metrics"]
            path_key = f"linearization_path_{identifier}"
            if path_key not in actions:
                raise ValueError(f"stored trajectory omits linearization path: {identifier}")
            linearization_path = actions[path_key].astype(np.int32)
            if linearization_path_hash(linearization_path) != item["linearization_path_hash"]:
                raise ValueError(f"stored trajectory linearization path hash mismatch: {identifier}")
            bank._records[identifier] = BankTrajectory(
                trajectory_id=identifier, actions=actions[f"actions_{identifier}"].astype(np.float32),
                linearization_path=linearization_path, source_id=str(item["source_id"]),
                active=bool(item["active"]),
                current_sdf=np.concatenate(current_parts),
                masks={name: np.concatenate(parts) for name, parts in masks.items()},
                remaining_fraction=np.asarray(metrics["remaining_fraction"], dtype=np.float32),
                overcut_fraction=np.asarray(metrics["overcut_fraction"], dtype=np.float32),
                clearance_mm=np.asarray(metrics["clearance_mm"], dtype=np.float32),
                pulse_count=np.asarray(metrics["pulse_count"], dtype=np.int32),
                energy_j=np.asarray(metrics["energy_j"], dtype=np.float32),
                flags={name: np.asarray(values, dtype=bool) for name, values in item["flags"].items()},
                score=float(item["score"]), origins=tuple(item["origins"]),
            )
        bank.rejection_ledger = json.loads(
            (directory / "rejection_ledger.json").read_text(encoding="utf-8")
        )
        library_directory = directory / "linearization"
        library_manifest = "linearization/linearization_manifest.json"
        if library_manifest in manifest:
            from laser_ablation.planning.jax_bank.linear_contracts import (
                load_linearization_library,
            )

            bank.linearization_library = load_linearization_library(library_directory)
        comparison_manifest = "comparison_roi/comparison_roi_manifest.json"
        if comparison_manifest in manifest:
            from laser_ablation.planning.jax_bank.comparison_library import (
                load_comparison_roi_library,
            )

            bank.comparison_library = load_comparison_roi_library(
                directory / "comparison_roi"
            )
        return bank


__all__ = ["PlanBankStorage"]
