"""Offline routed-ROI and original-plan diagnostics for controller artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from laser_ablation.core.sdf import SDFState
from laser_ablation.planning.jax_bank.bank import PlanBank
from laser_ablation.planning.jax_bank.similarity import roi_cosine_similarity


def enrich_controller_trace(
    rows: Sequence[Mapping[str, object]], machine_directory: Path,
) -> list[dict[str, object]]:
    """Add diagnostic-only routed similarity, nominal energy, and plan-bank counts."""
    machine = Path(machine_directory)
    selected_directories: dict[str, Path] = {}
    for index_path in sorted(machine.rglob("bank_index.json")):
        directory = index_path.parent
        environment = json.loads(
            (directory / "provenance.json").read_text(encoding="utf-8")
        )
        selected = environment["provenance"]["selected_trajectory"]
        selected_directories[str(selected)] = directory
    requested = {str(row["trajectory_id"]) for row in rows}
    missing = requested - set(selected_directories)
    if missing:
        raise ValueError(f"selected trajectories omit persisted plan banks: {sorted(missing)}")

    loaded: dict[str, tuple[PlanBank, object]] = {}
    enriched: list[dict[str, object]] = []
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id not in loaded:
            bank = PlanBank.load(selected_directories[trajectory_id])
            loaded[trajectory_id] = (bank, bank.trajectory(trajectory_id))
        bank, trajectory = loaded[trajectory_id]
        library = bank.comparison_library
        if library is None:
            raise ValueError("controller diagnostics require a persisted ROI comparison library")
        prefix = int(row["active_prefix"])
        if not 0 <= prefix < len(trajectory.actions):
            raise ValueError("controller diagnostic prefix lies outside its selected plan")
        seed, nominal_step = map(int, trajectory.linearization_path[prefix])
        if (
            seed < 0 or nominal_step < 0
            or seed >= library.seed_count or nominal_step >= library.maximum_pulses
        ):
            raise ValueError("controller diagnostic action has no canonical global-plan route")
        cosine_value = float("nan")
        if library.step_mask[seed, nominal_step]:
            observation_path = machine / "observations" / f"prefix_{int(row['pulse']):03d}.npz"
            with np.load(observation_path, allow_pickle=False) as observation:
                observed = np.asarray(
                    observation["tensor"][SDFState.CURRENT_TISSUE], dtype=np.float32
                ).ravel()
            predicted = np.asarray(trajectory.current_sdf[prefix], dtype=np.float32).ravel()
            indices = library.roi_indices[seed, nominal_step]
            mask = library.roi_mask[seed, nominal_step]
            cosine, valid = roi_cosine_similarity(
                observed[indices][None], predicted[indices][None], mask[None]
            )
            cosine_value, valid_value = (
                float(np.asarray(cosine)[0]), bool(np.asarray(valid)[0])
            )
            if not valid_value:
                raise ValueError("controller diagnostic ROI cosine has an invalid geometry norm")
        values = dict(row)
        values.update({
            "roi_substate_similarity": cosine_value,
            "selected_global_plan_energy_j": float(
                library.nominal_energy_j[seed, nominal_step]
            ),
            "plan_bank_candidate_count": len(bank.trajectories),
            "plan_bank_saved_state_count": sum(
                len(candidate.current_sdf) for candidate in bank.trajectories
            ),
        })
        enriched.append(values)
    return enriched


__all__ = ["enrich_controller_trace"]
