"""Internal square-only cadence and MPPI-kappa tuning evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import yaml

from laser_ablation.control.workflow_config import UnifiedWorkflowConfig
from laser_ablation.experiments.artifacts import write_artifact_manifest, write_json
from laser_ablation.experiments.full_capability import run_full_capability


@dataclass(frozen=True)
class SquareTuningCandidate:
    """One immutable cadence and five-component kappa replacement."""

    label: str
    periodic_repair_pulses: int
    kappa_multiplier: float


@dataclass(frozen=True)
class SquareTuningRun:
    """Terminal metrics and deterministic ordering tuple for one reconstructable run."""

    label: str
    seed: int
    output_directory: str
    hard_violations: int
    completion_gate: bool
    remaining_pct: float
    overcut_pct: float
    cumulative_administered_energy_j: float
    median_control_cycle_time_s: float
    peak_gpu_memory_total_mb: float

    @property
    def ranking(self) -> tuple[object, ...]:
        return (
            int(self.hard_violations != 0),
            int(not self.completion_gate),
            self.remaining_pct,
            self.overcut_pct,
            self.median_control_cycle_time_s,
            self.peak_gpu_memory_total_mb,
            self.label,
        )


def run_operation4_square_tuning(
    config: UnifiedWorkflowConfig,
    devices: tuple[object, ...],
    output_directory: Path,
    tuning_seed: int = 20260901,
    confirmation_seeds: tuple[int, int, int] = (20260825, 20260826, 20260827),
    validation_output_directory: Path | None = None,
) -> tuple[SquareTuningCandidate | None, tuple[SquareTuningRun, ...]]:
    """Run the locked five-candidate square protocol and three-seed confirmation."""
    if config.scenario_name != "square":
        raise ValueError("Operation-4 tuning is restricted to the square scenario")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Operation-4 tuning artifacts: {output}")
    machine, human, debug = (
        output / "machine_readables",
        output / "human_readables",
        output / "debug_runs",
    )
    machine.mkdir(parents=True)
    human.mkdir()
    debug.mkdir()
    (debug / "README.md").write_text(
        "# Non-publication reconstruction runs\n\n"
        "Each child directory is one complete deterministic tuning or confirmation run.\n",
        encoding="utf-8",
    )
    baseline = SquareTuningCandidate("cadence_10_kappa_1.0", 10, 1.0)
    cadence_candidates = (
        baseline,
        SquareTuningCandidate("cadence_5_kappa_1.0", 5, 1.0),
        SquareTuningCandidate("cadence_20_kappa_1.0", 20, 1.0),
    )
    tuning_runs = [
        _run_candidate(config, devices, debug, candidate, tuning_seed)
        for candidate in cadence_candidates
    ]
    best_cadence = min(tuning_runs, key=lambda item: item.ranking).label
    chosen_cadence = next(
        candidate.periodic_repair_pulses for candidate in cadence_candidates
        if candidate.label == best_cadence
    )
    kappa_candidates = (
        SquareTuningCandidate(f"cadence_{chosen_cadence}_kappa_0.5", chosen_cadence, 0.5),
        SquareTuningCandidate(f"cadence_{chosen_cadence}_kappa_1.5", chosen_cadence, 1.5),
    )
    tuning_runs.extend(
        _run_candidate(config, devices, debug, candidate, tuning_seed)
        for candidate in kappa_candidates
    )
    candidate_by_label = {candidate.label: candidate for candidate in cadence_candidates + kappa_candidates}
    winner_run = min(tuning_runs, key=lambda item: item.ranking)
    winner = candidate_by_label[winner_run.label]
    confirmations: list[SquareTuningRun] = []
    if winner_run.hard_violations == 0:
        confirmations = [
            _run_candidate(config, devices, debug, winner, seed, prefix="confirmation")
            for seed in confirmation_seeds
        ]
    confirmed = bool(confirmations) and all(
        run.hard_violations == 0 and run.completion_gate for run in confirmations
    )
    _write_tuning_artifacts(
        config, output, winner, tuning_runs, confirmations, confirmed, tuning_seed,
    )
    if validation_output_directory is not None:
        validation = Path(validation_output_directory)
        if validation.exists():
            raise FileExistsError(f"refusing to overwrite Operation-4 validation artifacts: {validation}")
        baseline_run = next(run for run in tuning_runs if run.label == baseline.label)
        shutil.copytree(baseline_run.output_directory, validation)
        write_json(validation / "machine_readables/operation4_validation.json", {
            "success_flag": "M4_PERIODIC_EXACT_REPAIR_PASS",
            "baseline_candidate": asdict(baseline),
            "baseline_run": asdict(baseline_run),
            "tuning_root": str(output),
        })
        (validation / "human_readables/operation4_validation_summary.md").write_text(
            "# Operation-4 periodic exact-repair validation\n\n"
            "This bundle is the baseline candidate reused by the square tuning protocol. "
            "Its raw controller trace, periodic repair ledger, timing breakdown, bank trace, "
            "memory trace, and nested repair artifacts remain machine-readable.\n",
            encoding="utf-8",
        )
        write_artifact_manifest(validation)
    return (winner if confirmed else None), tuple(tuning_runs + confirmations)


def _run_candidate(
    config: UnifiedWorkflowConfig,
    devices: tuple[object, ...],
    debug_directory: Path,
    candidate: SquareTuningCandidate,
    seed: int,
    prefix: str = "candidate",
) -> SquareTuningRun:
    """Run one immutable replacement and reduce its trace to the locked objective tuple."""
    tuned = replace(
        config,
        periodic_repair_pulses=candidate.periodic_repair_pulses,
        mppi=replace(config.mppi, kappa=tuple(
            value * candidate.kappa_multiplier for value in config.mppi.kappa
        )),
    )
    location = debug_directory / f"{prefix}_{candidate.label}_seed_{seed}"
    print(f"Operation-4 {prefix} started: {candidate.label} seed={seed}", flush=True)
    result = run_full_capability(tuned, devices, location, seed)
    trace = result.step_records
    final = result.outcome.final_metrics
    run = SquareTuningRun(
        label=candidate.label,
        seed=seed,
        output_directory=str(location),
        hard_violations=int(final.hard_violations),
        completion_gate=result.outcome.stopped_reason == "completion_gate",
        remaining_pct=float(final.remaining_pct),
        overcut_pct=float(final.total_overcut_pct),
        cumulative_administered_energy_j=(
            0.0 if not trace else float(trace[-1]["cumulative_administered_energy_j"])
        ),
        median_control_cycle_time_s=(
            0.0 if not trace else float(np.median([
                float(row["control_cycle_time_s"]) for row in trace
            ]))
        ),
        peak_gpu_memory_total_mb=(
            0.0 if not trace else float(np.max([
                float(row["gpu_memory_total_mb"]) for row in trace
            ]))
        ),
    )
    print(
        f"Operation-4 {prefix} finished: {candidate.label} seed={seed} "
        f"hard={run.hard_violations} complete={run.completion_gate} "
        f"remaining={run.remaining_pct:.3f}% overcut={run.overcut_pct:.3f}%",
        flush=True,
    )
    return run


def _write_tuning_artifacts(
    config: UnifiedWorkflowConfig,
    output: Path,
    winner: SquareTuningCandidate,
    tuning_runs: list[SquareTuningRun],
    confirmations: list[SquareTuningRun],
    confirmed: bool,
    tuning_seed: int,
) -> None:
    """Write compact human summaries and fully reconstructable machine selection data."""
    machine, human = output / "machine_readables", output / "human_readables"
    ordered = sorted(tuning_runs, key=lambda item: item.ranking)
    for path, rows in (
        (machine / "candidate_table.csv", ordered),
        (machine / "confirmation_summary.csv", confirmations),
    ):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(asdict(rows[0])) if rows else tuple(asdict(ordered[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    selected = next(row for row in tuning_runs if row.label == winner.label)
    values = {
        "periodic_repair_pulses": winner.periodic_repair_pulses,
        "mppi": {"kappa": [
            value * winner.kappa_multiplier for value in config.mppi.kappa
        ]},
    }
    write_json(machine / "selection.json", {
        "tuning_seed": tuning_seed,
        "winner": asdict(winner),
        "winner_ranking": selected.ranking,
        "all_candidate_rankings": {run.label: run.ranking for run in tuning_runs},
        "confirmation_required": [20260825, 20260826, 20260827],
        "confirmed": confirmed,
    })
    (machine / "confirmed_yaml_values.yaml").write_text(
        yaml.safe_dump(values if confirmed else {}, sort_keys=True), encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    labels = [run.label.replace("cadence_", "N").replace("_kappa_", " K") for run in ordered]
    axis.scatter(
        [run.overcut_pct for run in ordered], [run.remaining_pct for run in ordered],
        c=["#2A6F97" if run.hard_violations == 0 else "#C44536" for run in ordered], s=32,
    )
    for label, run in zip(labels, ordered, strict=True):
        axis.annotate(label, (run.overcut_pct, run.remaining_pct), xytext=(3, 3), textcoords="offset points", fontsize=6)
    axis.set(xlabel="Final overcut (%)", ylabel="Final remaining target (%)")
    figure.savefig(human / "candidate_frontier.png", dpi=240)
    plt.close(figure)
    selected_trace = Path(selected.output_directory) / "human_readables/control_timeseries_square.png"
    shutil.copy2(selected_trace, human / "selected_control_timeseries_square.png")
    confirmation_summary = (
        "# Operation-4 square tuning confirmation\n\n"
        f"Selected candidate: `{winner.label}`. Confirmation status: `{confirmed}`.\n"
    )
    (human / "confirmation_summary.md").write_text(confirmation_summary, encoding="utf-8")
    (human / "interpretation_summary.md").write_text(
        "# Operation-4 square tuning\n\n"
        "The debug subtree contains all five tuning candidates and any required confirmation runs. "
        f"The deterministic selection candidate is `{winner.label}`; confirmation is `{confirmed}`. "
        "Only a safe, complete three-seed confirmation permits configuration replacement.\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output)


__all__ = ["SquareTuningCandidate", "SquareTuningRun", "run_operation4_square_tuning"]
