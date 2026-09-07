"""Construct one simulation workflow around the installed, unchanged MPPI planner."""

from hashlib import sha256
from pathlib import Path

import jax
import numpy as np
import pybullet as p

from laser_ablation.control.interaction import DesignatedTask
from laser_ablation.control.method import load_method
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from laser_ablation.planning.global_3d.terminal_verification import ExactPlanVerifier
from collision_scene import link_pose
from robot_executor import RobotExecutor
from robot_scene import ROOT
from run_records import RunRecords, write_json
from scan_adapter import designate_task, observe_task
from simulated_plant import SimulatedPlant
from simulation_cases import RANDOM_SEED, simulation_case
from surface_scan import save_scan
from task_designation import save_designation
from workflow import run_workflow
from workflow_display import WorkflowDisplay


def execute_case(case, scan, designation, output, *, gui, wait_for_start, isolation_violations):
    """Share exactly the same causal loop between DIRECT checks and live GUI execution."""
    output = Path(output).resolve()
    records = RunRecords(output)
    # Use one CUDA device for baseline parity while retaining the declared CPU test profile.
    devices = tuple(jax.devices())
    if devices[0].platform in {"cuda", "gpu"}:
        devices = devices[:1]
    method = load_method(ROOT / "mppi/configs/controller.yaml", devices)
    components = method.components(case.name, case.raster_settings, devices, RANDOM_SEED)
    write_json(output / "method.json", {
        "inputs": method.source_values, "source_sha256": method.source_hashes,
        "compute_profile": method.compute_profile, "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
    })
    client = p.connect(p.GUI if gui else p.DIRECT)
    try:
        display = WorkflowDisplay(client, output / "gui_frames")
        verifier = ExactPlanVerifier(ExactVoxelSimulator(method.physics, method.bounds),
                                     method.completion_remaining_pct, method.hard_margin_mm)
        robot = RobotExecutor(client, designation.grid_bounds_mm, verifier, display.poll)
        task = designate_task(scan, designation)
        authority = sha256((components.authority_id + robot.calibration.identity).encode()).hexdigest()
        task = DesignatedTask(task.state, task.frame_id, authority)
        display.bind(robot, task)
        write_json(output / "registration.json", {
            "calibration": robot.calibration, "legacy_order_chain": robot.registration_chain,
            "asset_sha256": robot.model.asset_hashes, "collision_pairs": robot.scene.pairs,
            "collision_links": robot.scene.links, "calibration_id": robot.calibration.identity,
            "joint_limits_rad": robot.model.limits,
            "task_id": task.task_id, "method_authority": components.authority_id,
        })
        plant = SimulatedPlant(task.state, method.physics, method.bounds, robot.calibration,
                               task.frame_id, case.disturbed_pulse)
        scan_pose = np.linalg.inv(robot.calibration.world_from_base_m) @ link_pose(robot.model, "ee_helper_link")
        initial_scan = plant.observe(scan_pose)
        save_scan(initial_scan, output / "scans/prefix_000.npz")
        observed = observe_task(initial_scan, task, 0, None)
        session = components.session(method.periodic_repair_pulses, output, RANDOM_SEED)
        if wait_for_start:
            display.wait_for_start()
        run_workflow(session, task, observed, robot, plant, records, display)
        if gui:
            display.closeup()
            display.capture("final_tissue")
            display.overview()
            display.capture("final_robot")
            if wait_for_start:
                display.wait_for_close()
    finally:
        write_json(output / "isolation.json", {"violations": isolation_violations})
        if p.isConnected(client):
            p.disconnect(client)


def launch(case_name, output, *, gui, isolation_violations):
    """Load the nominal scan, obtain designation approval, then enter automatic simulation."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    case = simulation_case(case_name)
    scan = case.scan()
    write_json(output / "case.json", case.manifest())
    save_scan(scan, output / "input_scan.npz")
    designation = case.designation
    if gui:
        import matplotlib.pyplot as plt
        from run_task_designation import TaskEditor
        editor = TaskEditor(scan, output / "approved", designation)
        plt.show()
        if not editor.approved:
            raise RuntimeError("Operator cancelled task designation")
        designation = editor.designation
    save_designation(designation, output / "task.json")
    execute_case(case, scan, designation, output, gui=gui, wait_for_start=gui,
                 isolation_violations=isolation_violations)
