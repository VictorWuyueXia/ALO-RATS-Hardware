"""Append-only execution evidence with separate machine-readable subdirectories."""

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np

from laser_ablation.core.state import VoxelState


def json_value(value):
    """Convert numerical and dataclass values into strict JSON-compatible values."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_voxels(path, state):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, x_axis_mm=state.x_axis_mm, y_axis_mm=state.y_axis_mm,
                            z_axis_mm=state.z_axis_mm, tissue=state.tissue,
                            initial_tissue=state.initial_tissue, target_mask=state.target_mask,
                            constraint_mask=state.constraint_mask, spacing_mm=state.spacing_mm,
                            plane_z_mm=state.plane_z_mm)


def load_voxels(path):
    with np.load(path, allow_pickle=False) as values:
        fields = {name: values[name] for name in values.files}
        for name in ("spacing_mm", "plane_z_mm"):
            fields[name] = float(fields[name])
        return VoxelState(**fields)


class RunRecords:
    """Create requested evidence directories and append every state-changing event."""

    def __init__(self, output, directories=("motions", "scans", "observed", "truth")):
        self.output = Path(output)
        for name in directories:
            (self.output / name).mkdir()
        self.events = []

    def event(self, kind, **values):
        event = json_value({"event": kind, **values})
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.events.append(event)

    def motion(self, name, robot):
        values = robot.last_motion
        with (self.output / "motions" / f"{name}.npz").open("xb") as stream:
            np.savez_compressed(stream, **{key: np.asarray(value) for key, value in values.items()})
