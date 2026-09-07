# ALO-RATS experiment operator runbook

## Purpose and readiness boundary

This document gives one operator sequence for four show-of-concept experiments on the Ubuntu robot workstation. It distinguishes commands that work in this checkout from commands that can exist only after missing hardware integrations are recovered and qualified.

Readiness labels used below:

- **AVAILABLE**: the entry point exists in this repository and can be run now after its stated preconditions pass.
- **PARTIAL**: useful preparation exists, but the named experiment has no end-to-end entry point.
- **MISSING**: no implementation is present; the text defines the expected integration and its acceptance gate.
- **UNVERIFIED SOURCE**: discovered in the side repository or old machine, but not verified on the present machine.

This repository is a research proof of concept. It does not authorize clinical use. Never connect or energize the laser until the laboratory's laser-safety owner has approved the optical enclosure, interlocks, emergency stop, beam path, phantom, and operating procedure. Do not infer a calibration, device state, or delivered pulse from a successful software import.

## Capability map

| Experiment | Current status | Current entry point | Blocking gaps |
| --- | --- | --- | --- |
| Nominal-geometry simulation | **AVAILABLE** | `simulation/run_simulation.py` | Physical calibration and devices are intentionally isolated and irrelevant. |
| OCT-geometry simulation | **PARTIAL** | OCT import/designation exists; no OCT-driven simulation launcher exists | A scan-driven simulated-plant launcher and acceptance validator are missing. |
| UR5e dry run with OCT, no laser | **AVAILABLE** | `scripts/run_hardware_dry_run.py` | Raw OCT acquisition and OCT-to-planning registration are external; only state capture and one optional safe-pose move exist. |
| Phantom resection with OCT rescans | **MISSING** | None | Live OCT, segmentation, measured registration, action-to-pose execution, laser control/interlocks, pulse receipts, and the physical closed loop are absent. |

## 1. Establish the Ubuntu workstation

Perform this once in a desktop login session so Matplotlib and PyBullet can open windows.

1. Put this repository at a stable path and enter it:

   ```bash
   cd /path/to/ALO-RATS-Hardware
   ```

2. Create and activate the declared environment:

   ```bash
   conda env create -f environment.yml
   conda activate alo-rats-hardware
   python -m pip install -e ./mppi -e .
   ```

   On a Windows development workstation in PowerShell, activate `alo-rats-hardware`, then use the native path syntax and PowerShell environment assignment:

   ```powershell
   python -m pip install -e .\mppi -e .
   $check_root = Join-Path $PWD ("outputs\checks-" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
   New-Item -ItemType Directory -Path $check_root | Out-Null
   python simulation\check_robot_scene.py --output-dir (Join-Path $check_root "robot_scene")
   python scripts\create_nominal_oct_fixture.py --output (Join-Path $check_root "nominal_processed_oct.npz")
   python -m pytest -q --basetemp (Join-Path $check_root "pytest") tests simulation/test_scan_adapter.py simulation/test_robot_executor.py
   ```

   The editable-install command is required even when the Conda environment already exists. Without it, imports of `alo_rats_hardware` and `laser_ablation` fail.

3. Confirm that Python is using the intended environment:

   ```bash
   which python
   python --version
   python -m pip show alo-rats-hardware laser-ablation-icra2027 pybullet ur-rtde
   ```

4. Run the no-device checks:

   ```bash
   mkdir -p outputs
   check_root=$(mktemp -d "$PWD/outputs/checks.XXXXXX")
   python simulation/check_robot_scene.py --output-dir "$check_root/robot_scene"
   python scripts/create_nominal_oct_fixture.py --output "$check_root/nominal_processed_oct.npz"
   python -m pytest -q --basetemp "$check_root/pytest" tests simulation/test_scan_adapter.py simulation/test_robot_executor.py
   ```

5. On an Ubuntu NVIDIA workstation, install and verify CUDA-enabled JAX before running MPPI:

   ```bash
   nvidia-smi
   python -m pip install --upgrade "jax[cuda13]"
   python -c "import jax; print('backend:', jax.default_backend()); print('devices:', jax.devices())"
   ```

   Require `backend: gpu` and the intended device count. The program selects the matching `1gpu`, `4gpu`, or `8gpu` MPPI profile automatically and fails if that device count is not configured. PyBullet uses OpenGL acceleration for the GUI when provided by the display driver, while its physics and inverse kinematics remain CPU computations.

6. Require all of the following before proceeding: the URDF report exists at `$check_root/robot_scene/robot_scene.json`, the fixture command succeeds, and pytest reports no failures. Preserve the entire unique check root. These checks establish software packaging only; they do not validate the robot, OCT, registration, or laser.

## 2. Inventory the collaborator's robot machine before repairing anything

Run this section on the connected Ubuntu workstation before installing or changing packages. Preserve the outputs with the experiment records.

1. Create a new audit directory:

   ```bash
   audit_dir="$PWD/machine-audit-$(date +%Y%m%d-%H%M%S)"
   mkdir "$audit_dir"
   ```

2. Record the operating system, network, USB devices, serial identities, services, and listening TCP endpoints:

   ```bash
   uname -a | tee "$audit_dir/uname.txt"
   cat /etc/os-release | tee "$audit_dir/os-release.txt"
   ip -br address | tee "$audit_dir/ip-address.txt"
   ip route | tee "$audit_dir/ip-route.txt"
   ip neigh | tee "$audit_dir/ip-neighbours.txt"
   lsusb | tee "$audit_dir/usb.txt"
   ls -l /dev/serial/by-id | tee "$audit_dir/serial-by-id.txt"
   systemctl list-units --type=service --all | tee "$audit_dir/services.txt"
   ss -lntp | tee "$audit_dir/listening-tcp.txt"
   ```

   If `/dev/serial/by-id` does not exist, record that exact failure; do not substitute an assumed device path.

3. Locate candidate source trees and setup files in the collaborator-controlled locations:

   ```bash
   find /home/"$USER" /opt /srv -type f \( -name '*.py' -o -name 'environment*.yml' -o -name 'requirements*.txt' -o -name 'pyproject.toml' \) | sort | tee "$audit_dir/source-files.txt"
   ```

4. Search those files for the interfaces named by the historical workflow:

   ```bash
   grep -RInE 'oct_raster_scan|start_oct_scan|module_oct_vol_scan|RTDEControlInterface|RTDEReceiveInterface|moveJ|servoJ|module_laser_utils|laser_control_pwm|planned_cut_execute|ndyag|PWM|socket.connect' /home/"$USER" /opt /srv | tee "$audit_dir/interface-hits.txt"
   ```

5. Inventory Python environments without activating unknown ones:

   ```bash
   command -v conda | tee "$audit_dir/conda-command.txt"
   conda env list | tee "$audit_dir/conda-environments.txt"
   python -m pip freeze | tee "$audit_dir/current-python-packages.txt"
   ```

6. Inspect, but do not execute, every hit that controls a robot, scanner, socket, GPIO pin, or laser. Record its repository path, commit hash, Python environment, imported packages, configuration path, target IP/port or serial ID, and normal operator entry point in `machine-audit-.../recovered-components.md`.

7. Identify whether OCT acquisition runs locally or on a separate OCT workstation, and whether PWM runs on a separate Raspberry Pi. For each external host, record its operator, address, transport, service name, startup command, shutdown command, and artifact-transfer path. A source-code reference or ping response is not proof that the service is correct.

8. Before reuse, copy each recovered component into a controlled integration branch, preserve its commit/source hash, and test it in isolation. Do not place an unknown historical module directly on the active Python path.

## 3. Experiment A - nominal-geometry simulation (**AVAILABLE**)

1. Complete Sections 1.1-1.6. No robot, OCT, or laser connection is required or permitted by the simulation isolation guard.

2. Create a unique output parent:

   ```bash
   run_dir="$PWD/outputs/nominal-$(date +%Y%m%d-%H%M%S)"
   ```

3. Launch the nominal centered-rectangle case:

   ```bash
   python simulation/run_simulation.py --case centered_rectangle --output-dir "$run_dir"
   ```

4. In the designation window, verify the orange target, red protected volume, depths, and 0.1 mm voxel spacing. Click **Approve / save** only when the task matches the intended nominal case.

5. Focus the PyBullet window. Press **Enter** to start automatic planning and virtual treatment. Press **C** for the magnified tissue view, **V** for the robot overview, or **Esc/Q** to abort before the next released virtual action.

6. After the controller stops, inspect both views, then press **Esc/Q** to close.

7. Accept the run only if the process exits successfully and `$run_dir/acceptance.json` has `"accepted": true`. Also retain `workflow_status.json`, `events.jsonl`, `method.json`, `registration.json`, `scans/`, `observed/`, `truth/`, `motions/`, and `gui_frames/`.

8. If planning or acceptance fails, do not switch to the smaller `compact_diagnostic` and call it a successful treatment. That case intentionally exercises the path without meeting acceptance. Preserve the failed output directory and inspect `acceptance.json` and `workflow_status.json`.

Missing for this experiment: nothing required for the nominal software demonstration. The simulated registration, scan, tissue response, and laser pulse remain synthetic and cannot qualify hardware.

## 4. Experiment B - simulation from OCT-scanned geometry (**PARTIAL**)

There is currently no honest end-to-end command for this experiment. The repository can validate a processed scan and collect task designation, but `run_simulation.py` accepts only frozen nominal cases.

### Use what exists now

1. Obtain a processed `.npz` scan that satisfies `docs/PROCESSED_OCT_CONTRACT.md`. It must be a complete registered Boolean occupancy volume with uniform 0.1 mm voxel centers, millimeter units, a unique scan identity, a planning-frame identity, and a finite rigid scan pose.

2. Validate the scan through the strict loader:

   ```bash
   python -c "from alo_rats_hardware.processed_oct import load_processed_volume, volume_summary; import sys; print(volume_summary(load_processed_volume(sys.argv[1])))" /path/to/processed_oct_volume.npz
   ```

3. Designate and preserve the task without running a simulation:

   ```bash
   designation_dir="$PWD/outputs/oct-designation-$(date +%Y%m%d-%H%M%S)"
   PYTHONPATH=simulation python simulation/run_task_designation.py --scan /path/to/processed_oct_volume.npz --output-dir "$designation_dir"
   ```

4. Inspect `scan.npz`, `task.json`, `initial_voxels.npz`, `identity.json`, and `designation.png`. Stop here; these files do not prove that an OCT-driven simulation ran.

### Implement and qualify the missing path

1. Add one scan-driven launcher that accepts only `--scan` and `--output-dir`, invokes the existing designation editor, initializes `SimulatedPlant` from the approved OCT occupancy, and reuses the existing MPPI, PyBullet robot, virtual pulse, rescan, and evidence loop.

2. Keep measured input separate from simulated evolution: archive the original processed OCT as `input_scan.npz`; label every subsequent scan provenance as synthetic; retain simulated truth only under `truth/`.

3. Add a scan-driven acceptance validator that checks immutable axes/frame/task masks, unique and causal scan identities, one execution receipt per virtual pulse, checked robot motion, protected-volume compliance, completion, and hardware isolation. Do not compare an OCT task against a nominal case manifest.

4. Add tests for an uneven surface, a cavity/overhang, a protected subsurface region, invalid spacing, changed frame, stale scan, and duplicate scan identity.

5. Require all scan-driven tests plus one GUI run to pass. The expected command after implementation is:

   ```bash
   python simulation/run_oct_simulation.py --scan /path/to/processed_oct_volume.npz --output-dir /new/output/directory
   ```

This command is an interface target, not a command available in the present checkout.

## 5. Experiment C - UR5e hardware dry run with processed OCT and no laser (**AVAILABLE**)

### Preconditions

1. Physically disconnect or lock out the laser device and its PWM/network controller. Confirm no laser service is running on the robot workstation. The current code has no laser import or command path.

2. Complete the Ubuntu inventory in Section 2 and the no-device checks in Section 1.

3. Confirm RTDE imports and reachability using the laboratory-verified robot address:

   ```bash
   python -c "import rtde_control, rtde_receive; print('RTDE imports OK')"
   ip route get ROBOT_IP
   ping -c 3 ROBOT_IP
   ```

4. At the pendant, verify remote-control/RTDE readiness, clear the workspace, set a conservative speed limit, test the emergency stop, and have one operator remain at the pendant.

5. Copy the site template and replace every value with measured, pendant-reviewed values:

   ```bash
   cp config/site.example.yaml config/site.yaml
   ```

   The example IP and joint pose are historical values, not a calibration. Record the approved six-joint pose in radians, speed in radians per second, and acceleration in radians per second squared.

6. Produce or transfer a registered processed OCT volume, verify its checksum after transfer, and run the loader command in Section 4.2. Raw images, point clouds, surface-only data, and the synthetic fixture are not hardware observations.

### State-only dry run

1. Start without the motion flag:

   ```bash
   run_dir="$PWD/outputs/hardware-dry-run-$(date +%Y%m%d-%H%M%S)"
   python scripts/run_hardware_dry_run.py --site config/site.yaml --scan /path/to/processed_oct_volume.npz --output-dir "$run_dir"
   ```

2. In the designation window, draw the target footprint, enter left/right depths and protected Z, inspect the 3-D masks, and click **Approve / save**.

3. The program connects to RTDE, records measured joints and TCP pose, writes evidence, and disconnects. It does not execute an MPPI action.

4. Require `hardware_dry_run.json` to contain `laser_control_present: false`, `motion_commanded: false`, finite six-element robot states, the intended scan identity/frame, and MPPI source hashes.

### Optional single safe-pose move

1. Perform this only after the state-only run passes and the exact pose is approved on the pendant.

2. Use a new output directory and add the sole motion opt-in:

   ```bash
   python scripts/run_hardware_dry_run.py --site config/site.yaml --scan /path/to/processed_oct_volume.npz --output-dir /new/output/directory --move-safe-pose
   ```

3. Observe the robot continuously. Require `motion_commanded: true` and verify the before/after joint and TCP states against the approved pose.

Missing for this experiment: live OCT triggering/segmentation and measured OCT-to-planning registration are external. No planned cutting trajectory is sent to the robot; only the optional fixed `moveJ` is implemented.

## 6. Experiment D - full phantom deployment with OCT rescans (**MISSING; do not run yet**)

The intended interaction is **see -> designate -> plan -> approve -> move -> pulse -> return to scan -> acquire/segment/register -> update -> continue or stop**. The current ALO-RATS `ControllerSession` is the sole control authority: it requires a fresh registered observation after every confirmed pulse and attempts plan repair every ten confirmed pulses.

### Recover and qualify the missing components

1. **OCT acquisition:** recover `oct.module_oct_vol_scan.oct_raster_scan` and its `start_oct_scan(...)` path, or document the external OCT workstation procedure found in Section 2. Recover vendor/serial dependencies and fixed scan settings from source rather than guessing them.

2. **OCT processing:** reproduce the historical B-scan/C-scan reconstruction and tissue/critical-structure segmentation. Add one adapter that emits exactly the processed-OCT contract, including a new scan ID and timestamp for every acquisition. Verify geometry against a dimensioned phantom before connecting it to control.

3. **Registration:** remeasure OCT pixel-to-millimeter scale, OCT-to-end-effector transform, scan pose, robot base relationship, laser axis, focal point, and standoff. Version the calibration artifacts and reject expired, mismatched, or unverified identities. Do not reuse any prior measurement as present calibration evidence.

4. **Robot action executor:** implement the physical counterpart of the checked simulation executor. It must transform each MPPI beam action into a calibrated robot pose, enforce joint/workspace/tilt/keep-out limits, preview the requested and solved pose, require operator release, record achieved joints/TCP/beam geometry, and return to the qualified scan pose.

5. **Laser control:** recover the TCP laser helper, PWM service, and Raspberry Pi GPIO code identified in Section 2. Replace implicit socket success with explicit arm/disarm state, bounded command identity, interlock state, timeout, completion acknowledgment, and an append-only execution receipt. First validate into a non-tissue beam dump with a power meter; derive energy-to-command calibration on the intended phantom and optical setup. Never copy recovered duty cycles or energy values as current commands.

6. **Closed-loop coordinator:** connect the existing controller to the qualified robot, laser, and OCT adapters. Preserve the ordering: prepare/approve pose; execute exactly one identified pulse; persist its receipt; return to scan pose; acquire a new scan; segment/register/validate it; call the controller update; then request another action. A pulse with an uncertain receipt or a missing/stale scan latches a stop.

7. **Evidence:** write separate human-readable run summaries/figures and machine-readable calibration, command, receipt, robot-state, scan, voxel, plan/repair, timing, and failure records. Never overwrite a run directory.

### Qualification ladder

1. Run all no-device tests and both simulation workflows.
2. Run the state-only and safe-pose dry runs with the laser physically disconnected.
3. Replay recorded OCT volumes through the coordinator with robot and laser adapters mocked; require causal one-pulse/one-scan identity checks.
4. Execute robot trajectories over an inert phantom with the laser disconnected; verify pose, collision clearance, scan return, and OCT repeatability.
5. Exercise the laser service into a beam dump without robot motion; verify interlocks, measured output, receipts, emergency stop, and failure latching.
6. Perform a single stationary pulse on disposable phantom material, then OCT rescan and quantify the crater. Stop and recalibrate if measured geometry or energy falls outside the laboratory-approved bound.
7. Perform one robot-positioned pulse, return to the scan pose, acquire a new OCT volume, and require the controller to consume it exactly once.
8. Only after all prior gates pass, authorize a bounded multi-pulse phantom run with an operator approving every pulse.

### Expected operator sequence after qualification

1. Start a new run directory and load the exact versioned site, registration, OCT, laser, physics, and controller identities.
2. Verify enclosure/interlocks/emergency stop, clear the workspace, place and secure the OCT-visible absorbing phantom, inspect the protective window, and keep the laser disarmed.
3. Move to the qualified scan pose; acquire, reconstruct, segment, register, and validate the initial OCT volume.
4. Designate target and protected geometry in the current UI; independently inspect the 3-D masks and approve the immutable task.
5. Build the initial plan; display the proposed first beam pose, energy, predicted removal, clearance, and planned return-to-scan motion.
6. Arm only for the released command, approve one pulse, execute it, disarm immediately, and persist the execution receipt and achieved robot state.
7. Return to the scan pose, inspect/clean the protective window according to the approved lab procedure, and acquire a new OCT volume.
8. Reject the volume if its frame, axes, coverage, validity, identity, timestamp, or registration check differs from the task contract. Otherwise update the controller and inspect remaining target, overcut, and protected-volume status.
9. Continue one pulse at a time. Allow the configured repair/replan logic to run only after the fresh observation has been accepted. Stop on completion, maximum pulse count, operator abort, interlock loss, uncertain execution, motion failure, OCT failure, stale/duplicate scan, registration failure, or protected-volume violation.
10. Disarm and lock out the laser, return the robot through an approved motion, save the final OCT scan, close device transports, and review the complete append-only evidence before calling the experiment complete.

## 7. Source and authority map

- Current install and boundaries: [`../README.md`](../README.md) and [`CURRENT_STATUS_AND_TRANSFER.md`](CURRENT_STATUS_AND_TRANSFER.md).
- Current OCT input contract: [`PROCESSED_OCT_CONTRACT.md`](PROCESSED_OCT_CONTRACT.md).
- Current dry-run details: [`HARDWARE_DRY_RUN.md`](HARDWARE_DRY_RUN.md).
- Nominal closed loop: `simulation/run_simulation.py`, `simulation/simulation_app.py`, and `simulation/workflow.py`.
- Hardware boundary: `src/alo_rats_hardware/hardware_dry_run.py` and `src/alo_rats_hardware/robot.py`.
- Controller observation contract: `mppi/src/laser_ablation/control/session.py` and `mppi/configs/controller.yaml`.
- Original hardware arrangement and coordinate-frame provenance: [`../../see-plan-cut/data/See_Plan_Cut.pdf`](../../see-plan-cut/data/See_Plan_Cut.pdf). It is used only to help locate apparatus, calibration assets, and device interfaces; it is not a control-algorithm authority or present-machine qualification.
