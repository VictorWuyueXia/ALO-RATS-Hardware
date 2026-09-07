"""Uniform experiment output and provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np


def prepare_result_directory(root: str | Path, experiment: str) -> tuple[Path, Path]:
    result = Path(root) / experiment
    machine = result / "machine_readables"
    human = result / "human_readables"
    machine.mkdir(parents=True, exist_ok=True)
    human.mkdir(parents=True, exist_ok=True)
    return machine, human


def write_json(path: str | Path, values: Any) -> None:
    Path(path).write_text(
        json.dumps(values, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def provenance(
    config_path: str | Path, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    repository = _git_repository(path.parent)
    values: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "config_path": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compute_backend": "numpy_cpu",
        "jax_installed": util.find_spec("jax") is not None,
        "package_versions": {
            package: metadata.version(distribution)
            for package, distribution in (
                ("numpy", "numpy"),
                ("scipy", "scipy"),
                ("matplotlib", "matplotlib"),
                ("nibabel", "nibabel"),
                ("pyyaml", "PyYAML"),
            )
        },
        "git_commit": _git_output(repository, "rev-parse", "HEAD"),
        "git_worktree_status": _git_output(repository, "status", "--short"),
    }
    values.update(extra or {})
    return values


def write_artifact_manifest(result_directory: str | Path) -> None:
    """Hash every completed session artifact for integrity and discovery."""
    root = Path(result_directory)
    manifest_path = root / "machine_readables" / "artifact_manifest.json"
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "audience": (
                    "machine" if "machine_readables" in path.parts else "human"
                ),
            }
        )
    write_json(manifest_path, {"artifact_count": len(records), "artifacts": records})


def validate_artifact_manifest(result_directory: str | Path) -> None:
    """Reject an artifact bundle whose recorded files or hashes no longer match."""
    root = Path(result_directory)
    manifest_path = root / "machine_readables" / "artifact_manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"]
    expected = {item["path"]: item for item in records}
    actual = {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        raise ValueError("artifact manifest file set mismatch")
    for relative, item in expected.items():
        path = root / relative
        if (
            path.stat().st_size != item["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"artifact manifest hash mismatch for {relative}")


def _git_repository(start: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
