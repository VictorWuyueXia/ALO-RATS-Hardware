# ALO-RATS preparation and certified-operator runbook

This file retains the completed software, OCT, robot, and inert-rehearsal preparation history. For the post-boot laboratory procedure and powered demonstration, use [`02-OPERATOR_HANDOFF_REFERENCE.md`](02-OPERATOR_HANDOFF_REFERENCE.md).

## Scope and role boundary

Preparation 1–4 is completed by the repository maintainer with the laser and Raspberry Pi unpowered. Deployment 5 is completed by the certified laser operator after the maintainer leaves the controlled room.

These numbers identify operational phases. The scientific experiment number remains `2` or `3`. The current physical entry point does not support scientific experiments 1 or 4 from [`05-EXPERIMENT_PLAN.md`](05-EXPERIMENT_PLAN.md).

| Operational phase | Person | Laser/Pi state | Completion record |
| --- | --- | --- | --- |
| 1. Software and simulation | Repository maintainer | Unpowered | `PREPARATION_1_SOFTWARE_PASS` |
| 2. OCT records | Maintainer and OCT operator | Unpowered | `PREPARATION_2_OCT_DATA_READY` |
| 3. UR5e state/safe-pose dry run | Maintainer and robot operator | Unpowered | `PREPARATION_3_DRY_RUN_PASS` |
| 4. One-feedback physical rehearsal | Maintainer and robot operator | Unpowered | `INERT_ONE_FEEDBACK_LOOP_PASS` |
| 5. Laser qualification and physical run | Certified laser operator; maintainer available remotely | Operator-controlled | `LASER_INTERFACE_PASS`, then experiment completion flag |

The certified operator uses the familiar `see-plan-cut` physical procedure for the enclosure, interlocks, beam dump, Windows Lumedica application, laser power, and direct motion observation. ALO-RATS replaces the earlier planner/controller. Do not run a `see-plan-cut` robot or laser client concurrently.

The [handoff reference](02-OPERATOR_HANDOFF_REFERENCE.md) lists every missing value, its exact form and source, its recording path, and the responsible code module.

## Stop conditions

Keep the laser disarmed and stop after any operator abort, emergency stop, interlock loss, unexpected person/object, unexpected robot motion, unsafe pose, RTDE error, missing or duplicated OCT scan, Raspberry Pi timeout or malformed reply, uncertain pulse, incorrect focus/standoff, energy outside the measured table, or protected-voxel removal.

Never repeat an uncertain pulse. Secure the hardware and retain the output directory.

## Create one handoff directory

Run from `/home/rp/Documents/victor/ALO-RATS-Hardware` and keep this terminal open so `$handoff_dir` remains defined:

```bash
handoff_dir="$PWD/outputs/operator-handoff-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$handoff_dir/human_readables" \
  "$handoff_dir/machine_readables/oct" \
  "$handoff_dir/machine_readables/robot" \
  "$handoff_dir/machine_readables/laser/power_traces"
printf '%s\n' '# Operator handoff checklist' > "$handoff_dir/human_readables/HANDOFF_CHECKLIST.md"
printf '%s\n' "$handoff_dir"
```

Put instructions and conclusions under `human_readables/`. Put device replies, measurements, hashes, configurations, and run records under `machine_readables/`.

## Preparation 1 — software and simulation

```bash
cd /home/rp/Documents/victor/ALO-RATS-Hardware
/home/rp/anaconda3/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pybullet==3.2.7 pyvista==0.48.4
.venv/bin/python -m pip install -e ./mppi -e '.[robot,test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m pytest -q
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/check_robot_scene.py
```

Require zero test failures and the intended UR5e in `robot_scene.json`. The tests load every tracked simulation case with `mppi/configs/controller.yaml`, verify each four-raster input, and verify that Experiment 2 and 3 use their tested random seed and raster settings. Treatment completion and outcome quality belong to planner evaluation and do not block hardware deployment. Then write `PREPARATION_1_SOFTWARE_PASS` in `HANDOFF_CHECKLIST.md`.

The `centered_rectangle` case uses a centered rectangular target with dimensions $4\\times4\\times2$ mm. Its current planning bounds are read from `config/simulation_cases.yaml`; its target floor is 2 mm below the scan surface and its protected plane is 3 mm below the scan surface, leaving a 1 mm interval. The task-designation 3D view uses equal millimetre scale along X, Y, and Z.

`config/simulation_cases.yaml` defines the geometry and raster inputs for every runnable simulation case; every case belongs to `acceptance_cases`. Bounds use increasing `[lower, upper]` order, each target remains inside the XY bounds and above the lattice lower boundary, and `constraint_depth_mm - depth_mm` defines the vertical target-to-protected-plane interval. Simulation and physical execution both load `mppi/configs/controller.yaml`. Experiment 2 must use the tested `protected_boundary` random seed and raster settings; Experiment 3 must use the tested `response_disturbance` values. The coordinator rejects deployment-specific planner substitutions.

## Preparation 2 — OCT records and reboot remount

The known CIFS source is `//192.168.1.2/OCT_Data`; the Ubuntu target is `/mnt/OCT_Data`. Verify it after every reboot:

```bash
findmnt -no SOURCE,FSTYPE,TARGET /mnt/OCT_Data
timeout 5 find /mnt/OCT_Data -mindepth 1 -maxdepth 1 -printf '%f\n' | head
```

If `findmnt` prints nothing:

```bash
findmnt --fstab --target /mnt/OCT_Data -no SOURCE,FSTYPE,TARGET,OPTIONS
mount /mnt/OCT_Data
findmnt -no SOURCE,FSTYPE,TARGET /mnt/OCT_Data
```

`mount /mnt/OCT_Data` works without sudo only when an administrator configured an automount or the `user`/`users` option. If root permission is required, stop and ask the machine administrator to restore the mount. The CIFS credentials and `/etc/fstab` options are missing from this repository; never save the share password here or substitute an improvised mount path.

Save the non-secret mount source for remount troubleshooting; this is not an experiment value:

```bash
findmnt -no SOURCE,FSTYPE,TARGET /mnt/OCT_Data \
  | tee "$handoff_dir/machine_readables/oct/mount.txt"
```

ALO-RATS watches `/mnt/OCT_Data/victor` for new volume-scan child folders. The OCT program uses 256 B-scans per volume. Each B-scan is a 512 by 512, 16-bit TIFF. The fixed pixel spacings are 0.014 mm along an A-scan row, 0.028 mm between B-scans, and 0.01459 mm in depth. These values come from the current `see_plan_cut` volume-resection configuration and are already in `config/site.yaml`; do not enter them for each experiment.

The runtime decodes each 16-bit TIFF to an 8-bit image before surface extraction. `surface_threshold_u8: 50` is the configured median-surface signal check for this export. It was measured from the current processed scanner volume and remains a site setting.

One complete preparation volume is sufficient for this demonstration. In the Windows application, save one volume scan as a new child folder under `victor`. The required structure is:

```text
/mnt/OCT_Data/victor/
└── 20260909-HHMMSS/
    ├── BSCAN-SGL-001.tif
    ├── ...
    ├── BSCAN-SGL-256.tif
    ├── BSCAN-SGL-001.jpg ... BSCAN-SGL-256.jpg
    └── FileAttributes.txt
```

The current files directly inside `/mnt/OCT_Data/victor` contain one TIFF and one JPEG. `FileAttributes.txt` identifies that acquisition as `Scan Pattern: Vol`, but one TIFF is not a complete planner volume. In the Windows Lumedica application, use **Save Queue Images**, corresponding to the reference command `SaveQueueImages`. Do not use **Save Raw Queue Images** or `SaveRawQueueImages`: that produces 256 `BSCAN-RAW` files plus `BSCAN-BKGD.tif`, which this runtime does not process. The required processed files are `BSCAN-SGL-001.tif` through `BSCAN-SGL-256.tif`. JPEG files and `FileAttributes.txt` are ignored.

Set the exact new child path and verify its TIFF count:

```bash
scan_folder=/mnt/OCT_Data/victor/20260909-HHMMSS
find "$scan_folder" -maxdepth 1 -type f -name 'BSCAN-SGL-*.tif' | wc -l
```

The count must be `256`. Then run the OCT folder check:

```bash
oct_output="$PWD/outputs/preparation-2-oct-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/check_oct_folder.py \
  --site config/site.yaml \
  --folder "$scan_folder" \
  --output-dir "$oct_output"
```

The command prints `PREPARATION_2_OCT_FOLDER_PASS` and records the folder path, image geometry, file hashes, and scan identity automatically. Preparation 2 requires no manually entered OCT value. Write the success flag in `HANDOFF_CHECKLIST.md`.

Each `scan_id` is an automatic SHA-256 identity of one folder's ordered TIFF names and contents. The controller records it and rejects reuse of an old feedback volume; the operator never enters it. `registration.calibration_id` names the fixed robot/OCT/laser transforms in `config/site.yaml`. Physical execution includes it in the task identity and run record; it is not an OCT-folder name and is not entered per experiment.

## Preparation 3 — registration and UR5e dry run

Place the robot at the pose used for the initial OCT scan and leave it there. The command below records the joints and TCP pose directly; the operator does not copy either value from the pendant. It also requires the active TCP offset to be zero, which is the `tool0` geometry used by the fixed `tcp_from_oct_m` calibration.

If **Program**, **Installation**, and **Move** are grey, the pendant is in Remote Control. To inspect the named TCP entry, stop the loaded program, tap the **Remote** icon in the header, select **Local Control**, tap the **Automatic** profile icon, and select **Manual**. Enter the operational-mode password if requested. Open **Installation → General → TCP**. The selected TCP must have zero `X`, `Y`, `Z`, `RX`, `RY`, and `RZ`; this is normally named `tool0`. Return to **Automatic**, then **Remote Control**, before running an RTDE command. If an external mode selector controls the operational mode, use that selector; PolyScope cannot override it.

Acquire the OCT folder while the robot remains at the configured current-bench scan pose, where the OCT scanner covers the tissue area. Then register that Preparation 2 folder at the unchanged pose. The command requires the measured joints to match `scan_joint_pose_rad`, requires active TCP `tool0`, and applies the current-bench planning frame and fixed tool0-to-OCT calibration:

```bash
registration_output="$PWD/outputs/preparation-3-registration-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/check_oct_folder.py \
  --site config/site.yaml \
  --folder "$scan_folder" \
  --output-dir "$registration_output" \
  --register-current-pose
```

Require `PREPARATION_3_OCT_REGISTRATION_PASS`. The command writes `registered_processed_oct_volume.npz` and `registration_manifest.json` under `$registration_output/machine_readables/`. The manifest records the measured joints, measured TCP pose, zero active TCP offset, fixed reference planning transform, and TIFF identities.

Run the state-only path with one real registered processed-OCT `.npz`:

```bash
run_dir="$PWD/outputs/preparation-3-dry-run-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/run_hardware_dry_run.py \
  --site config/site.yaml \
  --scan "$registration_output/machine_readables/registered_processed_oct_volume.npz" \
  --output-dir "$run_dir"
```

Require `laser_control_present: false` and `motion_commanded: false` in `hardware_dry_run.json`. Use `--move-safe-pose` only after the robot operator approves that exact pose.

For the 2026-09-09 setup, the measured current-bench scan TCP is `[0.50600, -0.53981, 0.18070, 0.64042, 1.48282, -0.61867]` in metres and rotation-vector radians. The separately observed central laser-alignment TCP is `[0.43628, -0.55388, 0.25742, 0.73901, 1.44207, -0.78095]`. These are two distinct robot poses. Visual centering of the laser lens establishes lateral alignment; it does not establish laser focus or authorize emission.

Write `PREPARATION_3_DRY_RUN_PASS` after the state-only check.

## Preparation 4 — one-feedback physical deployment rehearsal

### Result and exact workflow

This qualification runs the same Experiment 2 coordinator, tracked MPPI controller, registered-OCT conversion, target designation, robot action calculation, checked RTDE motion, scan-pose return, feedback update, and next-action planning used for deployment. `--inert-one-loop` makes two bounded changes: it creates no Raspberry Pi laser client, and it stops before executing the second robot action.

The controller sequence is:

```text
new initial OCT folder
→ registered observation 0
→ operator designation
→ initial global plan
→ checked robot action 1
→ zero-energy inert receipt
→ scan-pose return
→ new feedback OCT folder
→ registered observation 1
→ controller feedback update
→ next-action planning
→ stop before robot action 2
```

The zero-energy inert receipt has `energy_provenance: INERT_REHEARSAL_ZERO_ENERGY`. It advances the controller interaction sequence so the real feedback path runs, but it cannot produce `EXPERIMENT_2_COMPLETE` and is never scientific ablation evidence. Initial planning evaluates the four tracked raster parents. Feedback uses the remaining active plan and periodic MPPI repair. Sequence exhaustion or a local repair that requires a new global plan stops the run with `ACTION_SEQUENCE_EXHAUSTED` or `REPAIR_REQUIRES_GLOBAL_REPLAN`; the deployment coordinator does not invoke the geometry-generated global-plan branch after initial planning.

The provisional `tcp_from_laser_m` in `config/site.yaml` is anchored so a zero-tilt action at the planning center and `plane_z_mm = 0.6` reproduces the visually observed central laser-alignment TCP. This is sufficient to evaluate unpowered robot motion around the physical tissue area. It is not an emission calibration. Before laser power is enabled, the laser operator must confirm or replace this transform using the focused spot, power-meter procedure, and beam-axis measurement.

### Physical and software preparation

1. Turn the Nd:YAG laser power supply off and remove its emission authority according to the laboratory procedure. The Raspberry Pi may remain off. Confirm that no laser-control service is running on this workstation.
2. Keep the phantom fixed in the physically designated tissue area. Confirm that the OCT share is mounted and writable from the Windows OCT computer:

```bash
mountpoint /mnt/OCT_Data
find /mnt/OCT_Data/victor -maxdepth 1 -type d -printf '%f\n' | sort | tail
```

3. Close every other RTDE, URScript, `see_plan_cut`, and robot-control process. Put PolyScope in **Automatic → Remote Control**, release the brakes, and require `RUNNING` with `NORMAL` safety mode.
4. Keep the pendant with the robot operator. Clear the complete swept volume between the central laser-alignment pose, scanning pose, and displayed planned pose. Use the pendant stop for an orderly stop and the emergency stop when continued motion presents immediate danger.
5. If the robot is at the central laser-alignment pose, return along the reviewed straight path:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/temporary_return_to_scanning_pose.py
```

Review the displayed translation and orientation. Type `RETURN TO SCANNING POSE` exactly. Stop if the path differs from the previously observed clear path.

6. Do not create either OCT folder before the coordinator starts. At startup, the coordinator records every existing folder and accepts only one new complete folder for each scan request. In Lumedica, use **Save Queue Images** so each new timestamp folder contains exactly `BSCAN-SGL-001.tif` through `BSCAN-SGL-256.tif`.

### Run the deployment coordinator without laser control

Run Experiment 2 with the same tracked experiment raster used in simulation and deployment:

```bash
cd /home/rp/Documents/victor/ALO-RATS-Hardware
run_dir="$PWD/outputs/experiment-2-inert-$(date +%Y%m%d-%H%M%S)"

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m alo_rats_hardware.hardware_experiment \
  --site config/site.yaml \
  --experiment 2 \
  --metadata config/experiment_2.yaml \
  --output-dir "$run_dir" \
  --inert-one-loop
```

Follow these interactions in order:

1. The coordinator connects to the UR5e and commands the configured scanning joints. With the robot already at the scanning pose, this is a near-zero joint correction.
2. At `Create the initial volume...`, create and save one new Lumedica volume folder. Wait until all 256 processed TIFF files exist, then press Enter in the terminal.
3. In the designation window, keep the centered default footprint for this rehearsal. Set **Left depth** to `2.0`, **Right depth** to `2.0`, and **Constraint depth** to `3.0`. The resulting footprint on the current 5.6 mm OCT lattice is approximately `1.83 x 1.83 mm`. Inspect the orange target and red protected volume, then click **Approve / save**. Do not drag the footprint during the retry; this reproduces the task that was checked offline against scan `aa6d21de9a66c48a6b7733cdbf0b076de0476742059b4e6557bad86073527f3f`.
4. Wait for `INITIAL_PLANNING_COMPLETE`. The planner loads the same `mppi/configs/controller.yaml`, physics, planner, one-GPU compute profile, random seed, and Experiment 2 raster used by simulation validation. Each configured 400-pulse raster is truncated to the controller's tracked 64-pulse execution limit before MPPI screening. Expect `32` terminal batches of `64` rows, at least one hard-feasible weighted child, and a printed `peak_gpu_memory_mib` value. A zero hard-feasible count stops before robot treatment motion.
5. Read the printed requested action, checked target tool0 TCP, checked target joints, and maximum joint change. A near-center, near-zero-tilt action should be a millimetre-scale variation around the observed central laser-alignment TCP `[0.43628, -0.55388, 0.25742, 0.73901, 1.44207, -0.78095]`. Do not approve a target displaced by centimetres from that pose. Confirm the swept volume is clear, then type the displayed `MOVE <command_id>` exactly. Any other text stops the run before motion.
6. Observe the complete motion. The coordinator records achieved joints, achieved TCP, beam intercept error, and beam-axis error. It applies the reference projects' `1e-3 m` position tolerance as `1.0 mm` and `1e-2 rad` orientation tolerance as `0.01 rad`. It creates no laser connection and records `energy_delivered_j: 0.0`.
7. The robot returns automatically to the configured scanning pose.
8. At `Create the feedback volume...`, create and save a second new Lumedica volume folder. This folder must result from a new acquisition; copying or reusing the initial folder is rejected. Wait for all 256 processed TIFF files, then press Enter.
9. The coordinator imports and saves the second OCT volume. Because this mode delivered zero energy, it updates the controller with the pre-motion tissue state and records the raw OCT difference in `inert_feedback_reconciliation`. It then performs normal next-action planning, records the next request, and prints `INERT_ONE_FEEDBACK_LOOP_PASS`. It stops before asking for or executing robot action 2. Power down the robot after inspecting its scanning pose.

### Validate the result

```bash
.venv/bin/python - <<PY
import json
from pathlib import Path

status = json.loads(Path("$run_dir/machine_readables/workflow_status.json").read_text())
assert status["completion_flag"] == "INERT_ONE_FEEDBACK_LOOP_PASS"
assert status["accepted"] is True
assert status["inert_one_loop"] is True
assert status["laser_control_present"] is False
assert status["confirmed_pulses"] == 1
assert status["observed_pulses"] == 1
print(status["completion_flag"])
PY
```

Require these machine-readable records:

- `execution_mode.json`: `inert_one_loop: true`, `laser_control_present: false`, and `maximum_feedback_loops: 1`.
- `jax_memory.json`: `after_initial_planning.cuda:0.peak_bytes_in_use` records the JAX allocator peak in bytes for this process.
- `events.jsonl`: two unique `oct_scan` events, one `request`, one `inert_execution` with zero delivered energy, one `observation`, and one `post_feedback_request`.
- `events.jsonl`: one `inert_feedback_reconciliation` event records raw added and removed OCT voxels. It is an OCT repeatability diagnostic, not an ablation measurement.
- `motions/pulse_001_laser.npz`: requested and achieved treatment-pose motion.
- `motions/prefix_001_scan.npz`: measured return to the scanning pose.
- `workflow_status.json`: the exact success checks above.

Write `INERT_ONE_FEEDBACK_LOOP_PASS` and the absolute `$run_dir` into `HANDOFF_CHECKLIST.md`. Any exception, rejected pose, unexpected path, reused scan identity, missing feedback request, protective stop, emergency stop, or nonzero laser communication is a failed qualification. Preserve the complete output directory and do not retry until the cause is understood.

The failed run `outputs/experiment-2-inert-20260909-204910` used `/mnt/OCT_Data/victor/20260909-205032` as its initial scan. Its scan manifest, OCT event, and initial-plan event all contain scan ID `aa6d21de9a66c48a6b7733cdbf0b076de0476742059b4e6557bad86073527f3f`. It ran on `cuda:0` with the `1gpu` profile. That run did not collect allocator statistics, so its historical peak GPU memory is unavailable. Offline replay of the same approved task after the 64-pulse correction produced 441 hard-feasible samples, one hard-feasible weighted child, a 64-action plan, 9.666 seconds planning time, and `398796800` peak bytes (`380.3 MiB`) at the final 64-row batch size.

The 20260909-211810 inert attempt reached its zero-energy treatment motion and returned to the scan pose. Its second raw OCT scan differed from the first by 3,621 added and 6,640 removed voxels, with surface differences from -2.6 to +2.2 mm. A powered experiment treats this condition as an invalid feedback observation. The inert rehearsal retains the pre-motion tissue state because no laser energy was delivered, while preserving both raw scan files and the discrepancy record for OCT repeatability work.

## Deployment 5 — certified-operator card

The operator does not edit the repository.

1. Apply the familiar laboratory power-up, enclosure, interlock, emergency-stop, window, focus, and beam-dump procedure. Keep emission disarmed.
2. Record the Raspberry Pi service identity and exact stopped reply using the [laser record schema](02-OPERATOR_HANDOFF_REFERENCE.md#raspberry-pi-laser-and-power-meter-records).
3. With laser power disabled, save the unedited status response:

```bash
.venv/bin/python - <<'PY' | tee "$handoff_dir/machine_readables/laser/pi_status.json"
import json
import socket

with socket.create_connection(("10.194.210.35", 8000), timeout=5) as connection:
    connection.sendall(json.dumps({"action": "status"}).encode("utf-8"))
    print(connection.recv(1024).decode("utf-8"))
PY
```

4. With robot motion disabled and the beam in the approved beam dump, collect five power-meter traces at each selected duty cycle under `machine_readables/laser/power_traces/`. Complete the energy table and independent-cutoff record defined in the handoff reference.
5. Send the five laser values to the repository maintainer. Wait for `CONFIGURATION_VALIDATED`; do not arm emission.
6. The maintainer changes only the laser block, copies one calibration identifier into the selected experiment metadata, sets `physical_execution_enabled: true`, and validates without contacting hardware:

```bash
.venv/bin/python -c "from alo_rats_hardware.site import load_site; s=load_site('config/site.yaml'); assert s.physical_execution_enabled; print(s.identity)"
```

7. After `LASER_INTERFACE_PASS`, run the preselected command supplied by the maintainer:

```bash
run_dir="$PWD/outputs/experiment-2-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m alo_rats_hardware.hardware_experiment \
  --site config/site.yaml \
  --experiment 2 \
  --metadata config/experiment_2.yaml \
  --output-dir "$run_dir"
```

The program interaction is fixed:

1. Verify stopped laser status and observe motion to the approved scan pose.
2. Create the initial Lumedica volume and press Enter.
3. Approve the displayed target and protected volume.
4. Inspect the planned pose/energy; type the displayed `MOVE <command_id>`.
5. Observe the motion and measured endpoint.
6. Recheck enclosure, interlock, window, focus, beam path, and cutoff; type `PULSE <command_id>`.
7. Observe the bounded pulse and return to scan pose.
8. Inspect/clean the window, create one new feedback volume, and press Enter.
9. Repeat only after that unique scan is accepted.

Accept scientific experiment 2 only with `EXPERIMENT_2_COMPLETE`, zero protected-voxel removal, zero hard violations, no uncertain receipt, and one later unique observation per pulse. Scientific experiment 3 additionally requires mismatch above measured OCT repeatability and a successful repair after pulse 10. Read acceptance from `machine_readables/workflow_status.json`, never from console output or screenshots.

## Handoff decision

Before the maintainer leaves, give the operator the absolute `$handoff_dir`, this Deployment 5 card, the reviewed configs with all non-laser values filled, the preselected command, the pendant-approved pose record, the five-value laser worksheet, the stop conditions, and the maintainer's remote contact.

The preparation package is ready after Preparations 1–4. The operator-only handoff is ready after the one-feedback laser-free rehearsal passes. Laser-powered execution is ready after that record plus `LASER_INTERFACE_PASS` and `CONFIGURATION_VALIDATED`.
