#!/usr/bin/env python3
"""Inject burst-coast-mpc conda env into Cursor agent sessions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ENV_NAME = "alo-rats-hardware"


def find_conda_env_prefix(name: str) -> Path | None:
    candidates: list[Path] = []

    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        home = Path(userprofile)
        candidates.extend(
            [
                home / ".conda" / "envs" / name,
                home / "anaconda3" / "envs" / name,
                home / "miniconda3" / "envs" / name,
            ]
        )

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(Path(conda_exe).resolve().parent.parent / "envs" / name)

    programdata = os.environ.get("ProgramData")
    if programdata:
        candidates.append(Path(programdata) / "anaconda3" / "envs" / name)

    for path in candidates:
        python_exe = path / ("python.exe" if os.name == "nt" else "bin/python")
        if python_exe.is_file():
            return path

    return None


def build_path(prefix: Path) -> str:
    if os.name == "nt":
        path_parts = [
            str(prefix / "Scripts"),
            str(prefix / "Library" / "bin"),
            str(prefix),
        ]
    else:
        path_parts = [str(prefix / "bin")]

    existing_path = os.environ.get("PATH", "")
    return os.pathsep.join([*path_parts, existing_path])


def main() -> None:
    sys.stdin.read()

    prefix = find_conda_env_prefix(ENV_NAME)
    if prefix is None:
        print(
            json.dumps(
                {
                    "additional_context": (
                        f"Conda env `{ENV_NAME}` was not found. "
                        "Run `conda env update -f environment.yml` from the repo root."
                    )
                }
            )
        )
        return

    print(
        json.dumps(
            {
                "env": {
                    "CONDA_DEFAULT_ENV": ENV_NAME,
                    "CONDA_PREFIX": str(prefix),
                    "PATH": build_path(prefix),
                },
                "additional_context": (
                    f"This project uses conda env `{ENV_NAME}`. "
                    "Shell commands inherit its PATH automatically."
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
