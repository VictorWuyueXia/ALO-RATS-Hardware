"""Composition root and artifacts for the unified paper-workflow controller."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from laser_ablation.config import bounds_from_mapping, load_yaml, physics_from_mapping
from laser_ablation.control.unified import UnifiedAblationController, UnifiedControllerResult
from laser_ablation.control.factory import build_planning_components
from laser_ablation.control.workflow_config import UnifiedWorkflowConfig
from laser_ablation.core.actions import PhysicalAction
from laser_ablation.experiments.artifacts import write_artifact_manifest, write_json
from laser_ablation.experiments.control_diagnostics import enrich_controller_trace
from laser_ablation.experiments.feedforward_control import FeedforwardResult, run_feedforward_control
from laser_ablation.geometry.scenarios import build_scenario
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.jax_bank.contracts import action_hash
from laser_ablation.visualization.control_timeseries import write_control_timeseries
from laser_ablation.visualization.voxel_state_views import write_voxel_state_views

def run_full_capability(
    config: UnifiedWorkflowConfig,
    devices: tuple[object, ...],
    output_directory: Path,
    random_seed: int,
) -> UnifiedControllerResult:
    """Build and run the selected synthetic proof-of-concept workflow."""
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite workflow artifacts: {output}")
    scenario = build_scenario(config.scenario_name, config.scenario)
    machine = output / "machine_readables"
    human = output / "human_readables"
    machine.mkdir(parents=True)
    human.mkdir()
    physics_values = load_yaml(config.physics_path)
    planner_values = load_yaml(config.planner_path)
    physics = physics_from_mapping(physics_values)
    bounds = bounds_from_mapping(physics_values)
    physics_seed, mppi_seed = (
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(random_seed).spawn(2)
    )
    noise = config.scenario["physics_noise"]
    execution_simulator = ExactVoxelSimulator(
        physics,
        bounds,
        float(noise["response_scale_standard_deviation"]) if noise["enabled"] else 0.0,
        tuple(map(float, noise["response_scale_bounds"])),
        physics_seed,
    )
    frozen_values = load_yaml(config.frozen_global_path)
    components = build_planning_components(
        physics=physics, bounds=bounds, mppi=config.mppi, devices=devices,
        random_seed=mppi_seed, maximum_pulses=config.maximum_pulses,
        completion_remaining_pct=float(planner_values["completion_remaining_pct"]),
        hard_margin_mm=float(planner_values["hard_margin_mm"]),
        raster_name=config.scenario_name, raster_settings=config.scenario,
        global_settings=frozen_values["geometry_aware_global"],
    )
    controller = UnifiedAblationController(
        planner=components.planner,
        simulator=execution_simulator,
        observer=components.observer,
        completion_remaining_pct=float(planner_values["completion_remaining_pct"]),
        periodic_repair_pulses=config.periodic_repair_pulses,
        maximum_pulses=config.maximum_pulses,
        device_identity=tuple(str(device) for device in devices),
    )
    initial_state = scenario.geometry.initial_state()
    started = perf_counter()
    result = controller.run(
        initial_state, components.raster_generator, components.frozen_generator, output, mppi_seed
    )
    elapsed = perf_counter() - started
    initial_archive = np.load(machine / "plan_banks/global_000/actions.npz")
    initial_actions = initial_archive[f"actions_{result.selected_trajectory_ids[0]}"]
    feedforward_simulator = ExactVoxelSimulator(
        physics,
        bounds,
        float(noise["response_scale_standard_deviation"]) if noise["enabled"] else 0.0,
        tuple(map(float, noise["response_scale_bounds"])),
        physics_seed,
    )
    feedforward = run_feedforward_control(
        initial_state,
        tuple(PhysicalAction.from_array(action) for action in initial_actions),
        feedforward_simulator,
    )
    _write_artifacts(
        config, result, feedforward, output, random_seed, physics_seed, mppi_seed, elapsed
    )
    return result

def _write_artifacts(
    config: UnifiedWorkflowConfig, result: UnifiedControllerResult,
    feedforward: FeedforwardResult, output: Path, seed: int,
    physics_seed: int, mppi_seed: int, elapsed_s: float,
) -> None:
    machine = output / "machine_readables"
    human = output / "human_readables"
    first_step_tissue = np.load(machine / "observations/prefix_001.npz")["exact_tissue"]
    write_voxel_state_views(result.outcome.final_state, first_step_tissue, human)
    effective = {
        "experiment": config.experiment,
        "compute": asdict(config.compute),
        "scenario_name": config.scenario_name,
        "scenario": dict(config.scenario),
        "periodic_repair_pulses": config.periodic_repair_pulses,
        "maximum_pulses": config.maximum_pulses,
        "mppi": asdict(config.mppi),
        "shared_configs": {
            "physics": str(config.physics_path), "planner": str(config.planner_path),
            "frozen_global": str(config.frozen_global_path),
        },
    }
    (machine / "effective_merged_config.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=True), encoding="utf-8"
    )
    write_json(machine / "environment_device.json", {
        "profile": config.compute.name, "platform": config.compute.platform,
        "devices": result.device_identity, "seed": seed,
        "physics_noise_seed": physics_seed, "jax_mppi_seed": mppi_seed,
    })
    rows = enrich_controller_trace(result.step_records, machine)
    if rows:
        with (machine / "controller_prefixes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plotted = [
            {
                "scenario": config.scenario_name, "pulse": row["pulse"],
                "trajectory_id": row["trajectory_id"],
                "parent_trajectory_id": row["parent_trajectory_id"],
                "remaining_pct": row["remaining_pct"], "overcut_pct": row["overcut_pct"],
                "control_cycle_time_s": row["control_cycle_time_s"],
                "gpu_memory_total_mb": row["gpu_memory_total_mb"],
                "roi_substate_similarity": row["roi_substate_similarity"],
                "plan_bank_candidate_count": row["plan_bank_candidate_count"],
                "plan_bank_saved_state_count": row["plan_bank_saved_state_count"],
                "administered_energy_j": row["administered_energy_j"],
                "selected_global_plan_energy_j": row["selected_global_plan_energy_j"],
            }
            for row in rows
        ]
        write_control_timeseries(plotted, human, machine)
        with (machine / "timing_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "pulse", "control_cycle_time_s", "exact_execution_time_s",
                "sdf_observation_time_s", "periodic_repair_time_s",
            ))
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
        with (machine / "bank_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "pulse", "plan_bank_candidate_count", "plan_bank_saved_state_count",
                "bank_serialized_dynamic_state_bytes",
            ))
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
        with (machine / "memory_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("pulse", "gpu_memory_total_mb"))
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    feedforward_rows = list(feedforward.step_records)
    with (machine / "feedforward_global_plan.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(feedforward_rows[0]))
        writer.writeheader()
        writer.writerows(feedforward_rows)
    write_json(machine / "feedforward_terminal_metrics.json", {
        **asdict(feedforward.final_metrics),
        "contact_misses": feedforward.contact_misses,
        "commanded_pulses": len(feedforward.step_records),
    })
    initial_bank = machine / "plan_banks" / "global_000"
    bank_index = json.loads((initial_bank / "bank_index.json").read_text(encoding="utf-8"))
    bank_provenance = json.loads(
        (initial_bank / "provenance.json").read_text(encoding="utf-8")
    )
    action_archive = np.load(initial_bank / "actions.npz")
    plan_lengths = sorted(
        int(action_archive[name].shape[0])
        for name in action_archive.files if name.startswith("actions_")
    )
    generated_lengths = bank_provenance["provenance"]["generated_candidate_lengths"]
    shard_payloads = [
        (item["stop"] - item["start"]) * bank_index["bytes_per_prefix"]
        for trajectory in bank_index["trajectories"].values()
        for item in trajectory["shards"]
    ]
    capability = {
        "device_count": len(result.device_identity),
        "all_eight_devices_used": len(result.device_identity) == 8,
        "gpu_preallocation_enabled": config.compute.preallocate,
        "generated_global_plan_lengths": generated_lengths,
        "all_generated_plans_over_200_pulses": min(generated_lengths) > 200,
        "surrogate_bank_plan_lengths": plan_lengths,
        "state_shard_limit_bytes": bank_index["state_shard_bytes"],
        "state_shard_limit_mib": bank_index["state_shard_bytes"] / 1024**2,
        "total_bank_capacity_bytes": bank_index["total_bank_capacity_bytes"],
        "total_raw_bank_state_bytes": bank_index["total_raw_state_bytes"],
        "total_raw_bank_within_capacity": (
            bank_index["total_raw_state_bytes"] <= bank_index["total_bank_capacity_bytes"]
        ),
        "largest_raw_shard_payload_bytes": max(shard_payloads),
        "all_raw_shards_within_limit": max(shard_payloads) <= bank_index["state_shard_bytes"],
        "timeseries_subplot_rows": 5,
        "compute_subplot_gpu_memory_axis": True,
        "roi_similarity_subplot": True,
        "plan_bank_count_axes": True,
        "serialized_dynamic_bank_axis": False,
        "original_global_plan_energy_axis": True,
        "exact_voxel_state_views": ("after_step_001", "final"),
        "periodic_repair_pulses": config.periodic_repair_pulses,
        "active_controller_direct_exact_execution": True,
        "periodic_repair_events": sum(
            bool(row["periodic_repair_due"]) for row in result.step_records
        ),
        "feedforward_global_plan_failed": (
            feedforward.contact_misses > 0
            or not feedforward.final_metrics.is_complete(10.0)
            or feedforward.final_metrics.hard_violations > 0
        ),
    }
    write_json(machine / "capability_validation.json", capability)
    action_array = np.asarray(
        [action.as_array() for action in result.outcome.executed_actions], dtype=np.float32
    ).reshape((-1, 5))
    selected_hash = action_hash(action_array) if len(action_array) else None
    write_json(machine / "exact_terminal_metrics.json", asdict(result.outcome.final_metrics))
    write_json(machine / "terminal_summary.json", {
        "stopped_reason": result.outcome.stopped_reason,
        "repair_count": result.repair_count, "replan_count": result.replan_count,
        "selected_trajectory_ids": result.selected_trajectory_ids,
        "selected_action_sha256": selected_hash, "elapsed_s": elapsed_s,
        "periodic_repair_pulses": config.periodic_repair_pulses,
        "cumulative_administered_energy_j": (
            0.0 if not result.step_records
            else float(result.step_records[-1]["cumulative_administered_energy_j"])
        ),
        "direct_exact_action_execution": True,
        "feedforward_contact_misses": feedforward.contact_misses,
        "feedforward_remaining_pct": feedforward.final_metrics.remaining_pct,
        "feedforward_overcut_pct": feedforward.final_metrics.total_overcut_pct,
    })
    summary = (
        "# Unified workflow interpretation\n\n"
        f"The `{config.scenario_name}` proof-of-concept stopped at "
        f"`{result.outcome.stopped_reason}` after {len(result.outcome.executed_actions)} pulses, "
        f"{result.repair_count} repairs, and {result.replan_count} global replans. "
        f"Exact Remaining is {result.outcome.final_metrics.remaining_pct:.3f}%. "
        f"Exact Overcut is {result.outcome.final_metrics.total_overcut_pct:.3f}%. "
        f"The {len(generated_lengths)} initial exact global candidates span {min(generated_lengths)}-{max(generated_lengths)} pulses; "
        f"the fixed-lineage plan bank retained {len(plan_lengths)} initial trajectories. "
        f"After every {config.periodic_repair_pulses} successful physical pulses, the controller "
        "attempted one routed-tail repair before directly executing the next selected plan action "
        "with the exact voxel simulator. "
        f"The matched-noise feedforward plan had {feedforward.contact_misses} contact misses "
        f"and terminal Remaining {feedforward.final_metrics.remaining_pct:.3f}%. "
        "The five-panel trace reports exact outcome, step time and process GPU memory, "
        "diagnostic observed-versus-planned ROI similarity, plan-bank contents as counts, and "
        "administered energy against the routed original global-plan energy. In the energy "
        "panel, asterisks mark active-plan switches and NaN-separated red dashes prevent "
        "different original plans from appearing as one continuous trajectory. "
        "The two 3-D voxel views show the exact state after pulse one and at termination: "
        "red is remaining target, blue is removed target, orange is overcut, translucent "
        "gray is exposed healthy tissue, and purple is the protected-tissue boundary. "
        "The total raw plan-bank state budget is capped at 256 MiB. "
        "This is methodological evidence, not clinical validation.\n"
    )
    (human / "interpretation_summary.md").write_text(summary, encoding="utf-8")
    write_artifact_manifest(output)
