# ALO-RATS preparation and certified-operator runbook

## Scope and role boundary

Preparation 1–3 is completed by the repository maintainer with the laser and Raspberry Pi unpowered. Deployment 4 is completed by the certified laser operator after the maintainer leaves the controlled room.

These numbers identify operational phases. The scientific experiment number remains `2` or `3`. The current physical entry point does not support scientific experiments 1 or 4 from [`Experiment_Plan.md`](Experiment_Plan.md).

| Operational phase | Person | Laser/Pi state | Completion record |
| --- | --- | --- | --- |
| 1. Software and simulation | Repository maintainer | Unpowered | `PREPARATION_1_SOFTWARE_PASS` |
| 2. OCT records | Maintainer and OCT operator | Unpowered | `PREPARATION_2_OCT_DATA_READY` |
| 3. UR5e state/safe-pose dry run | Maintainer and robot operator | Unpowered | `PREPARATION_3_DRY_RUN_PASS` |
| 4. Laser qualification and physical run | Certified laser operator; maintainer available remotely | Operator-controlled | `LASER_INTERFACE_PASS`, then experiment completion flag |

The certified operator uses the familiar `see-plan-cut` physical procedure for the enclosure, interlocks, beam dump, Windows Lumedica application, laser power, and direct motion observation. ALO-RATS replaces the earlier planner/controller. Do not run a `see-plan-cut` robot or laser client concurrently.

The [handoff reference](OPERATOR_HANDOFF_REFERENCE.md) lists every missing value, its exact form and source, its recording path, and the responsible code module.

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
simulation_dir="$PWD/outputs/preparation-1-simulation-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python -m simulation.paths.run_simulation \
  --case compact_diagnostic --output-dir "$simulation_dir"
.venv/bin/python scripts/check_robot_scene.py
```

Require zero test failures, normal simulation termination, and the intended UR5e in `robot_scene.json`. The recorded 2026-09-08 test reference was `105 passed`. Then write `PREPARATION_1_SOFTWARE_PASS` in `HANDOFF_CHECKLIST.md`.

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

Record the non-secret mount identity:

```bash
findmnt -no SOURCE,FSTYPE,TARGET /mnt/OCT_Data \
  | tee "$handoff_dir/machine_readables/oct/mount.txt"
```

In the Windows Lumedica application, record the application version and active preset. Acquire five folders from one unchanged phantom at the same scan pose. Record all OCT values in the exact [OCT record schema](OPERATOR_HANDOFF_REFERENCE.md#oct-computer-and-scan-records).

The repository has no OCT-only live-folder executable. Preserve the five folders and write `PREPARATION_2_OCT_DATA_READY`; do not write `OCT_FOLDER_TO_VOLUME_PASS` until the missing qualification command is implemented and passes.

## Preparation 3 — registration and UR5e dry run

Record the pendant, active TCP, safety installation, poses, and measured transforms using the exact [robot record schema](OPERATOR_HANDOFF_REFERENCE.md#ur5e-and-registration-records). Fill every non-laser experiment value listed in the [minimal config table](OPERATOR_HANDOFF_REFERENCE.md#minimal-manual-config-values).

Run the state-only path with one real registered processed-OCT `.npz`:

```bash
run_dir="$PWD/outputs/preparation-3-dry-run-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/run_hardware_dry_run.py \
  --site config/site.yaml \
  --scan /absolute/path/to/registered_processed_oct_volume.npz \
  --output-dir "$run_dir"
```

Require `laser_control_present: false` and `motion_commanded: false` in `hardware_dry_run.json`. Use `--move-safe-pose` only after the robot operator approves that exact pose.

This command cannot execute an MPPI treatment pose and scan return. Write `PREPARATION_3_DRY_RUN_PASS` after the state/safe-pose check; do not write `ROBOT_ACTION_AND_RETURN_PASS` until the missing inert-action command is implemented and passes.

## Code work required before Deployment 4

The intended operator-only handoff requires two compact laser-free commands:

1. An OCT command that runs `OCTFolderAdapter` on recorded or newly mounted folders, writes its existing scan/hash records, and computes five-scan repeatability without importing `LaserPWMClient`.
2. An inert-action command that loads one registered observation and approved designation, creates one MPPI request, calls `UR5eConnection.execute_action`, returns to the scan pose, and writes motion evidence without importing `LaserPWMClient`.

Validate them with the recorded active-preset folders and pendant-observed inert motion. Deployment 4 remains planned until `OCT_FOLDER_TO_VOLUME_PASS` and `ROBOT_ACTION_AND_RETURN_PASS` exist. The [code map](OPERATOR_HANDOFF_REFERENCE.md#code-responsibility-and-readiness) identifies the implemented modules and remaining physical qualifications.

## Deployment 4 — certified-operator card

The operator does not edit the repository.

1. Apply the familiar laboratory power-up, enclosure, interlock, emergency-stop, window, focus, and beam-dump procedure. Keep emission disarmed.
2. Record the Raspberry Pi service identity and exact stopped reply using the [laser record schema](OPERATOR_HANDOFF_REFERENCE.md#raspberry-pi-laser-and-power-meter-records).
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

Before the maintainer leaves, give the operator the absolute `$handoff_dir`, this Deployment 4 card, the reviewed configs with all non-laser values filled, the preselected command, the pendant-approved pose record, the five-value laser worksheet, the stop conditions, and the maintainer's remote contact.

The preparation package is ready after Preparations 1–3. The operator-only handoff is ready after both laser-free qualification commands pass. Physical execution is ready after those records plus `LASER_INTERFACE_PASS` and `CONFIGURATION_VALIDATED`.
