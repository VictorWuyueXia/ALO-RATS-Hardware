# Experiment 3 boot-to-demonstration manual

## Present status and tomorrow's objective

**Experiment 3 deployment integration succeeded on 2026-09-09.** The demonstrated path includes CUDA MPPI planning, a checked UR5e treatment motion, return to the scanning pose, detection and parsing of new 256-frame Lumedica folders, and entry into the feedback workflow. Experiments 2 and 3 use this same equipment path; their tracked raster inputs differ.

This success means the planner is deployed far enough to command the apparatus. It does not assert that MPPI completes the scientific removal objective. No powered laser pulse has been qualified.

Tomorrow has two remaining equipment objectives:

1. The OCT technician must make a physical phantom interface visible. The present scanner output contains the same dominant rows with a phantom and with an empty bench. The software therefore cannot yet identify the tissue surface reliably.
2. The certified laser operator must verify the Raspberry Pi stopped-status reply, energy-to-duty mapping, and independent output cutoff. Then the team may run one powered pulse followed by one new OCT volume and feedback planning.

Do not operate another robot or laser program at the same time as ALO-RATS.

## People and stop conditions

The **robot operator** holds the pendant and watches every motion. The **OCT technician** operates Lumedica and confirms the physical interface. The **certified laser operator** controls the enclosure, interlocks, window, beam dump, focus, power meter, emission authority, and emergency shutdown.

Stop immediately after an emergency stop, protective stop, interlock loss, unexpected person or object, unexpected motion, uncertain beam path, incorrect focus, damaged or dirty window, malformed Raspberry Pi reply, lost network during an unverified watchdog test, reused OCT folder, or uncertain pulse. Never repeat a pulse whose outcome is uncertain.

## Record only these values

Create one directory after starting Ubuntu. The programs place scans, configurations, motions, replies, and timing below it automatically.

```bash
cd /home/rp/Documents/victor/ALO-RATS-Hardware
handoff_dir="$PWD/outputs/laser-demonstration-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$handoff_dir/laser"
printf '%s\n' "$handoff_dir"
```

Record only the following manually:

| Value | Where to obtain it | Where it is used |
| --- | --- | --- |
| `experiment_run_id` | Choose one unique label for tomorrow's run | Identifies `run_metadata.json` and the output record |
| `laser_safety_approval_id` | Certified operator's approval record | Prevents a powered run with an unapproved setup |
| Raspberry Pi stopped-status key and value | Exact JSON returned by `{"action":"status"}` after `stop` | Verifies stopped output before startup, after every pulse, after failure, and at shutdown |
| `energy_to_duty_cycle` | Existing certified calibration for this unchanged setup, or tomorrow's power-meter measurement | Converts each requested energy in joules into Raspberry Pi PWM duty cycle |
| `laser.calibration_id` | Identifier attached to that certified energy mapping | Connects every execution receipt to the mapping used |
| `watchdog_qualified` | Result of the independent-cutoff test | Prevents powered execution when loss of Ubuntu commands could leave output active |
| Active UR installation name and `tool0` confirmation | PolyScope | Confirms the safety installation and the zero TCP used by the fixed transforms |

Do not re-record robot serial number, firmware version, fixed joint poses, 4 by 4 transforms, OCT pixel spacing, scan dimensions, file pattern, planner settings, random seed, phantom formulation, phantom batch, or preparation time. These values either remain in the working configuration or do not affect this demonstration.

## 1. Boot the equipment

1. Leave laser emission disarmed and the laser power supply off.
2. Boot the computer or NUC that shares `//192.168.1.2/OCT_Data`. Wait for its login and file sharing to finish.
3. Boot the Windows Lumedica computer, start Lumedica, and connect to the scanner. Do not start a scan yet.
4. Boot the Raspberry Pi. Leave the laser power supply off. The installed PWM service must listen on `10.194.210.35:8000`; the reference checkout contains the client protocol but not the installed server startup definition. If the port remains closed, the laser technician must start the deployed service on the Pi.
5. Boot the UR5e. Release the brakes, load the established laboratory installation, and leave the robot idle.
6. Boot this Ubuntu computer and open a terminal.

## 2. Check Ubuntu, the GPU, and the OCT share

Run:

```bash
cd /home/rp/Documents/victor/ALO-RATS-Hardware
test -x .venv/bin/python
findmnt -no SOURCE,FSTYPE,TARGET /mnt/OCT_Data
timeout 5 find /mnt/OCT_Data/victor -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Required results:

- `.venv/bin/python` exists.
- `findmnt` reports `/mnt/OCT_Data` and the folder listing completes.
- JAX prints `gpu` and includes `cuda:0`. The deployment configuration intentionally uses one GPU.

If `findmnt` prints nothing, run:

```bash
mount /mnt/OCT_Data
findmnt -no SOURCE,FSTYPE,TARGET /mnt/OCT_Data
```

If mounting reports that it cannot connect to `192.168.1.2`, confirm that the share-host NUC/computer is powered, logged in, connected to the laboratory network, and exporting `OCT_Data`. Do not create a substitute local directory.

Run the software check once after boot:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m pytest -q
```

Require zero failures.

## 3. Make the OCT phantom interface visible

The operator uses **Volume Scan** and **Save Queue Images**. The saved folder must contain `BSCAN-SGL-001.tif` through `BSCAN-SGL-256.tif`. Do not use **Save Raw Queue Images**.

The OCT technician must first inspect any optical filter, beam blocker, reference-arm setting, focus control, and scan-position control that may suppress the sample return. The current empty-bench folder is `/mnt/OCT_Data/victor/20260909-213747`; it is a negative reference, not a geometric calibration.

Use this physical visibility test:

1. Put the phantom at the intended scanning location.
2. Observe a live B-scan in Lumedica.
3. Raise the phantom by 0.5 to 1.0 mm, without moving the robot or scanner.
4. Identify one continuous interface that moves approximately 34 to 69 image rows. The configured depth spacing is 0.01459 mm per row.
5. Return the phantom to its intended height and save one new Volume Scan under `/mnt/OCT_Data/victor`.

A 5 cm displacement is outside the approximately 7.47 mm reconstructed depth span and is not a valid visibility test. Do not proceed when only the fixed rows near 102 and 187 remain visible.

Set the new timestamp folder and validate it:

```bash
scan_folder=/mnt/OCT_Data/victor/YYYYMMDD-HHMMSS
find "$scan_folder" -maxdepth 1 -type f -name 'BSCAN-SGL-*.tif' | wc -l
oct_check="$handoff_dir/oct-check"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/check_oct_folder.py \
  --site config/site.yaml \
  --folder "$scan_folder" \
  --output-dir "$oct_check"
```

Require a count of `256`, `PREPARATION_2_OCT_FOLDER_PASS`, and the technician's moving-interface confirmation. The command-level pass alone is insufficient because the current intensity test also accepts the empty bench.

## 4. Prepare the UR5e

1. Remove tools, hands, and cables from the complete path between the OCT scanning pose and the laser treatment area.
2. On PolyScope, change temporarily to Local/Manual mode and open **Installation → General → TCP**.
3. Confirm that the selected TCP is `tool0` with `X`, `Y`, `Z`, `RX`, `RY`, and `RZ` all equal to zero. Record the active installation name and this confirmation in `$handoff_dir/robot-check.txt`.
4. Return PolyScope to **Automatic → Remote Control**. Require robot mode `RUNNING` and safety mode `NORMAL`.
5. Keep the pendant in the robot operator's hands.

The configured scanning joints already represent the established scanning pose. If the robot is at the central laser-alignment pose, use the reviewed slow return:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/temporary_return_to_scanning_pose.py
```

Read the printed displacement, inspect the entire path, and type `RETURN TO SCANNING POSE` exactly. Stop when the proposed path does not match the clear path observed on 2026-09-09.

## 5. Verify the Raspberry Pi with emission disabled

Send `stop` followed by `status`, still with laser emission disabled:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python - <<'PY' | tee "$handoff_dir/laser/stop-status.json"
import json
import socket

replies = {}
for name in ("stop", "status"):
    command = {"action": name}
    with socket.create_connection(("10.194.210.35", 8000), timeout=5) as connection:
        connection.sendall(json.dumps(command).encode("utf-8"))
        replies[name] = json.loads(connection.recv(1024).decode("utf-8"))
print(json.dumps(replies, indent=2))
PY
```

The certified operator identifies the exact field and value that mean stopped. Preserve spelling, capitalization, and JSON type. For example, JSON number `0` and JSON string `"0"` are different values.

A connection-refused or timeout error means the Pi service is unavailable. Keep emission disarmed and ask the laser technician to start or diagnose the installed service.

## 6. Supply the three laser qualification results

The certified operator may reuse an existing calibration only when it covers the same laser, Pi/PWM output, optical path, pulse duration of 1.5 seconds, frequency of 100 Hz, and requested energy range. Otherwise measure at least two increasing duty cycles with the power meter and approved beam dump.

For measured power $P(t)$ in watts at time $t$ in seconds, delivered energy $E$ in joules is

$$
E=\int P(t)\,dt.
$$

The first column of `energy_to_duty_cycle` is measured $E$; the second is the duty cycle that produced it. The rows must increase in both columns and bracket every requested energy. Experiment 3 requests 4 J or 8 J; the calibrated energy range must therefore include 4–8 J.

The independent-cutoff test uses the approved beam dump and power meter. While PWM output is active at the operator's approved test duty, interrupt the Ubuntu-to-Pi control connection. Output must stop through a mechanism independent of a later Ubuntu `stop` command. Keep the physical laser stop immediately available. If output continues, stop it physically, set `watchdog_qualified: false`, and do not run ALO-RATS with emission enabled.

Edit only these fields in `config/site.yaml`:

```yaml
physical_execution_enabled: true
laser:
  energy_to_duty_cycle:
    - [MEASURED_LOW_ENERGY_J, LOW_DUTY_PERCENT]
    - [MEASURED_HIGH_ENERGY_J, HIGH_DUTY_PERCENT]
  status_key: EXACT_JSON_FIELD_NAME
  stopped_value: EXACT_JSON_VALUE
  watchdog_qualified: true
  calibration_id: EXACT_CALIBRATION_IDENTIFIER
```

Edit only these values in `config/experiment_3.yaml`; retain its random seed and raster block:

```yaml
experiment_run_id: UNIQUE_RUN_IDENTIFIER
laser_safety_approval_id: EXACT_APPROVAL_IDENTIFIER
laser_calibration_id: SAME_EXACT_CALIBRATION_IDENTIFIER
```

Validate the configuration without contacting hardware:

```bash
.venv/bin/python -c "from alo_rats_hardware.site import load_site; s=load_site('config/site.yaml'); assert s.physical_execution_enabled; print('CONFIGURATION_VALIDATED', s.identity)"
```

## 7. Run the powered Experiment 3 demonstration

The certified operator now applies the established enclosure, interlock, window, focus, beam-path, and emission procedure. The initial OCT folder must be created after the coordinator requests it; existing folders are ignored.

```bash
cd /home/rp/Documents/victor/ALO-RATS-Hardware
run_dir="$PWD/outputs/experiment-3-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m alo_rats_hardware.hardware_experiment \
  --site config/site.yaml \
  --experiment 3 \
  --metadata config/experiment_3.yaml \
  --output-dir "$run_dir"
```

Follow the prompts in this exact order:

1. The program sends `stop`, verifies stopped status, and moves to the configured scanning pose.
2. At `Create the initial volume`, save one new Lumedica Volume Scan. Wait for all 256 TIFF files, then press Enter.
3. In the designation window, select **rectangle** and drag a centered 4 by 4 mm footprint over the intended tissue area. Enter `2.0` for **Left depth**, `2.0` for **Right depth**, and `3.0` for **Constraint depth**. The equal left and right values define a 2 mm flat-bottom removal target; the 3 mm protected boundary leaves a 1 mm vertical interval below it. Confirm that the orange target follows the intended tissue and the red protected volume remains below it, then click **Approve / save**. Stop if the interface shown in this window does not match the moving interface confirmed in Section 3.
4. Wait for `INITIAL_PLANNING_COMPLETE`. Require at least one hard-feasible weighted child and `gpu`/`cuda:0` in the saved method record.
5. Read the requested action, target TCP, target joints, and maximum joint change. Clear the complete path. Type the displayed `MOVE <command_id>` exactly.
6. Observe the complete robot motion. Stop from the pendant if it differs from the displayed motion.
7. Recheck the enclosure, interlocks, window, focus, beam path, beam dump, and physical cutoff. Type the displayed `PULSE <command_id>` exactly.
8. The program emits one calibrated 1.5-second pulse, sends `stop`, verifies stopped status, and returns to the scanning pose.
9. Inspect and clean the OCT window according to the laboratory procedure. At `Create the feedback volume`, save one new 256-frame Volume Scan and press Enter. Never copy or reuse the initial folder.
10. The program parses the new folder, updates the observation, and performs feedback planning.

For the one-pulse apparatus demonstration, stop before a second motion: when the next `MOVE <command_id>` prompt appears, type `STOP`. The program records an operator abort and sends `stop` again. This intentional stop demonstrates one powered motion-pulse-return-scan-feedback cycle; it does not produce the scientific `EXPERIMENT_3_COMPLETE` flag.

Continue beyond the second motion only when the certified operator intends to run Experiment 3 to its scientific completion condition. Planner completion quality is separate from the apparatus-integration result.

## 8. Verify records and shut down

Inspect the final status:

```bash
.venv/bin/python - <<PY
import json
from pathlib import Path

path = Path("$run_dir/machine_readables/workflow_status.json")
print(json.dumps(json.loads(path.read_text()), indent=2))
PY
```

For a one-pulse demonstration, require one confirmed pulse, one later unique OCT scan, a saved treatment motion, a saved scan-pose return, a laser response record, and no `laser_stop_failure`. An intentional `STOP` produces `accepted: false` because scientific completion was not requested. A full scientific run requires `EXPERIMENT_3_COMPLETE`.

Shutdown order:

1. Send and verify `stop` again using the command in Section 5.
2. Disarm emission, turn off the laser power supply, and apply the laboratory key/interlock procedure.
3. Stop the robot program. Move or power down the UR5e according to the robot operator's established procedure.
4. Close Lumedica after confirming that the final folder contains all 256 TIFF files.
5. Shut down the Raspberry Pi normally.
6. Shut down the OCT share host after Ubuntu no longer needs `/mnt/OCT_Data`.
7. Preserve `$run_dir`, `$handoff_dir/laser/stop-status.json`, the OCT folders, and the laser calibration identifier. These contain every execution value needed for review.
