# ALO-RATS operator runbook for OCT, UR5e, and laser experiments

## Purpose and readiness

This runbook defines the exact operator sequence for experiment 2, protected-structure safety, and experiment 3, controlled model mismatch and recovery. The implementation structure, formulas, milestone validation, and acceptance flags are in [`OCT_LASER_EXPERIMENT_2_3_PLAN.md`](OCT_LASER_EXPERIMENT_2_3_PLAN.md).

Readiness on 2026-09-08:

| Capability | Software state | Physical state |
| --- | --- | --- |
| Processed OCT designation | Implemented | Existing dry run available |
| Mounted-folder OCT reconstruction | Implemented and offline-tested | Active preset and recorded folders require qualification |
| MPPI action to UR5e motion | Implemented and offline-tested | Inert motion and laboratory obstacle exclusions require qualification |
| Raspberry Pi PWM client | Implemented and loopback-tested | Live replies, watchdog, and energy table require qualification |
| Pulse/OCT feedback coordinator | Implemented and import-tested | Complete replay and one-cycle qualification remain pending |
| Experiments 2 and 3 | Entry point implemented | Blocked until every preceding success flag exists |

The Lumedica Windows application owns acquisition. Ubuntu reads a mounted folder. Do not run the historical scanner serial code. `config/site.yaml` carries the fixed Experiment 2 geometry transferred from `see_plan_cut`; `physical_execution_enabled: false` prevents emission.

Current host observation: `/mnt/OCT_Data` is mounted read-only from `//192.168.1.2/OCT_Data`, but directory and `stat` requests did not return within five seconds during the 2026-09-08 audit. Restore responsive access before OCT qualification.

## 1. Immediate stop conditions

Stop the run and keep the laser disarmed after any of these events:

- operator abort, emergency stop, enclosure/interlock loss, or an unexpected person/object in the workspace;
- unexpected robot motion, RTDE error, rejected UR safety check, excessive joint change, or endpoint beam error;
- missing, partial, unchanged, duplicated, ambiguous, or misregistered OCT scan;
- Raspberry Pi timeout, malformed reply, missing stopped status, or uncertainty after PWM may have started;
- incorrect focus/standoff, contaminated protective window, unexpected output, or energy outside the measured table;
- protected-voxel removal or another ALO-RATS hard-constraint violation.

Never repeat an uncertain pulse. Secure the hardware and retain the failure record.

## 2. Establish and verify the Ubuntu environment

Use the repository `.venv`; Conda and sudo are not required.

```bash
cd /home/rp/Documents/victor/ALO-RATS-Hardware
/home/rp/anaconda3/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pybullet==3.2.7 pyvista==0.48.4
.venv/bin/python -m pip install -e ./mppi -e '.[robot,test]'
```

On this machine, preload the system C++ runtime for PyBullet/OpenCV and disable unrelated global pytest plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m pytest -q
```

The current reference result is `105 passed`. Require zero failures. `environment.yml`, `pyproject.toml`, and `mppi/pyproject.toml` hold the portable dependency constraints.

For experiment execution, request an outside-sandbox run and verify the expected first CUDA device:

```bash
nvidia-smi
.venv/bin/python -m pip install --upgrade "jax[cuda12-pip]==0.4.38"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Require backend `gpu` and device `0`. The coordinator selects the baseline `1gpu` profile and records the detected device list.

## 3. Fixed reference parameters and live-interface audit

`config/site.yaml` contains the fixed parameters recovered from the prior Experiment 2 execution. The source chain is `see_plan_cut/planned_cut_execute.py`, `see_plan_cut/utils/UR5Controller.py`, and `see_plan_cut/robots/urdf/ur5e_fixed.urdf`:

- robot address, safe/scan pose, motion speed, and acceleration;
- the planning-frame-to-base transform from the recorded Experiment 2 scan pose and tissue-frame transform;
- TCP-to-OCT and TCP-to-laser transforms from the fixed URDF mount geometry;
- OCT image count, image shape, spacing, Raspberry Pi address/port, and the 100 Hz, 1.5 s historical pulse setting.

The following values are absent from both downloaded reference repositories. They must be read from the active equipment or its contemporaneous record before any physical laser experiment. They are unnecessary for the laser-free dry run.

| Value | Exact acquisition operation | Configuration destination |
| --- | --- | --- |
| Raspberry Pi stopped-state field and value | With laser power disabled, send `status`, retain the full JSON reply, then identify the field whose value means output is stopped. | `laser.status_key`, `laser.stopped_value` |
| Independent pulse cutoff | Interrupt Pi communication immediately after a nonemitting or beam-dump qualification command and observe the independent cutoff. | `laser.watchdog_qualified` |
| Duty-to-energy table | Record five beam-dump power-meter traces at each selected duty cycle and integrate each trace. For a trace with power $P(t)$ in watts at time $t$ in seconds, emitted energy is $$E = \int P(t)\,dt,$$ where $E$ is joules. | `laser.energy_to_duty_cycle`, `laser.calibration_id` |
| OCT repeatability | Acquire five unchanged registered volumes and calculate the occupied-voxel exclusive-or volume for every repeat pair. Use the largest value in cubic millimetres. | `oct_repeatability_xor_mm3` in experiment metadata |
| Experiment identity | Read the phantom label, formulation, batch, preparation record, operators, and laser safety approval from the experiment record. | `config/experiment_2.yaml` and `config/experiment_3.yaml` |

Create a unique audit directory and perform the following operations in order:

```bash
audit_dir="$PWD/outputs/live-interface-audit-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$audit_dir"
findmnt /mnt/OCT_Data | tee "$audit_dir/oct_mount.txt"
```

1. Restore responsive access to `/mnt/OCT_Data`. Record mount options, free space, Windows host, Lumedica preset, file pattern, B-scan count, image shape, and one untouched folder hash manifest in `$audit_dir`.
2. In the Windows Lumedica application, create one acquisition. On Ubuntu, confirm exactly one new complete folder and its contiguous B-scan indices.
3. On the UR pendant, record firmware, remote-control state, active TCP definition, safety planes, and the approved joint/error limits. The fixed geometry in `config/site.yaml` assumes the legacy URDF definition that `tool0` is the all-zero TCP frame.
4. On the Raspberry Pi, record the deployed service name and revision. With laser power disabled, run this nonemitting status query and save its exact reply:

```bash
.venv/bin/python - <<'PY' | tee "$audit_dir/pi_status.json"
import json
import socket

with socket.create_connection(("10.194.210.35", 8000), timeout=5) as connection:
    connection.sendall(json.dumps({"action": "status"}).encode("utf-8"))
    print(connection.recv(1024).decode("utf-8"))
PY
```

5. With the beam terminated in a beam dump and robot motion disabled, collect five power-meter traces for each selected duty cycle. The legacy analysis program expects `Power_<duty>/time_<seconds>.txt` folders under its `input_dir`; run it after placing the traces there:

```bash
.venv/bin/python ../see_plan_cut/analysis/laser_power_repetability.py
```

Copy two or more strictly increasing `mean_energy_J,duty_cycle` pairs from `per_instance_summary.csv` into `laser.energy_to_duty_cycle`.
6. Perform the independent-cutoff test. Set `watchdog_qualified: true` only after the cutoff stops output without a later Ubuntu command.
7. Acquire five unchanged registered OCT volumes, calculate the repeatability value, and fill the experiment metadata identities from the experiment record.

Write `LIVE_INTERFACES_IDENTIFIED` only when every value has direct evidence.

## 4. Prepare strict configuration

```bash
cp config/site.example.yaml config/site.yaml
cp config/experiment_2.example.yaml config/experiment_2.yaml
cp config/experiment_3.example.yaml config/experiment_3.yaml
```

Do not replace the transferred fixed geometry. Set `status_key` and `stopped_value` from the saved live reply. Fill at least two increasing energy/duty measurements. Set `watchdog_qualified: true` only after cutoff testing. Keep `physical_execution_enabled: false` through OCT, robot, and disabled-laser qualifications.

Each experiment metadata file must name the run, phantom, exact formulation, batch, preparation time, operators, safety approval, matching calibration identifiers, positive five-scan repeatability, random seed, and planner raster settings. Placeholder text or zero repeatability is rejected.

## 5. Qualify the OCT folder adapter

The adapter snapshots existing subdirectories when the program starts. Therefore start the program before asking the Windows operator to create the next scan folder.

Use at least three copied recorded scan folders: an unchanged phantom repeat, a pre-cut scan, and a post-cut scan. Confirm:

- one stable complete folder is accepted;
- numeric image indices are unique and contiguous;
- count and grayscale shape match the configuration;
- `scan_id` and occupancy repeat exactly on repeated conversion;
- the registered surface covers every planning column;
- incomplete, noncontiguous, wrong-shape, ambiguous, and low-surface-intensity fixtures fail.

The adapter writes the processed scan and a per-file hash manifest under `machine_readables/scans/`; source folders remain unchanged. Record `OCT_FOLDER_TO_VOLUME_PASS` after this validation.

## 6. Qualify robot motion with laser disconnected

The physical executor checks UR pose safety, inverse kinematics, UR joint safety, configured total joint change, final scan/safe-pose joint error, measured action endpoint, and beam intercept/axis error. It does not contain a calibrated collision model of the OCT head, fixture, or other laboratory objects. Configure UR installation safety planes and inspect every inert path.

First use the existing non-emitting dry run:

```bash
run_dir="$PWD/outputs/hardware-dry-run-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/run_hardware_dry_run.py \
  --site config/site.yaml \
  --scan /absolute/path/to/registered_processed_oct_volume.npz \
  --output-dir "$run_dir"
```

Use `--move-safe-pose` only after pendant approval of that exact pose. Then exercise scan, treatment, and return trajectories over an inert phantom with the laser physically disconnected. Verify measured joint/TCP records and registration fiducials. Record `ROBOT_ACTION_AND_RETURN_PASS` only after all approved trajectories satisfy the measured tolerances.

## 7. Qualify the Raspberry Pi PWM interface

1. Retain the passing loopback test result.
2. With laser power disabled, test `stop`, `status`, `start` at zero duty, `set_pwm`, `stop`, and `status` against the deployed service.
3. Force a connection loss after `set_pwm`; require the independent cutoff to stop output and require the client to mark the result uncertain without retry.
4. With no robot motion, direct the beam into a beam dump and measure energy for every configured duty value.
5. Confirm duty does not exceed 99%, frequency does not exceed the historical 290 Hz client limit, and requested energy never extrapolates beyond the table.

Record `LASER_BOUNDED_PULSE_PASS` when stopped-state verification and power-meter tolerance both pass.

## 8. Qualify one complete cycle

Run a recorded-folder replay with fake RTDE and PWM endpoints and inject failure at each phase. Record `COORDINATOR_REPLAY_PASS` after every failure terminates with a usable status record.

Then perform one qualified scan/motion/return/feedback-scan cycle with the laser disconnected. After review, perform one approved pulse into a phantom. Require one request, one completed receipt, one later unique scan, and one controller update. Record `ONE_PHYSICAL_CYCLE_PASS`.

After all prior flags, set `physical_execution_enabled: true`. The strict validator will still reject placeholders, unmatched calibration identifiers, an empty energy table, or an unqualified watchdog.

## 9. Experiment 2 command and exact operator interaction

```bash
run_dir="$PWD/outputs/experiment-2-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m alo_rats_hardware.hardware_experiment \
  --site config/site.yaml \
  --experiment 2 \
  --metadata config/experiment_2.yaml \
  --output-dir "$run_dir"
```

Before this command, confirm the enclosure, interlock, emergency stop, beam dump state, fixed scan pose, mounted OCT path, and the zero-output Pi status. The command performs this exact interaction sequence:

1. The program connects RTDE, verifies stopped laser status, and moves to the approved scan pose.
2. The operator creates the initial volume in the Windows application and presses Enter.
3. The adapter accepts the new folder. In the designation window, drag each target footprint, set target depth and constraint depth in millimetres, inspect orange target and red protected voxels, then click **Approve / save**.
4. MPPI creates one request. Verify its `command_id`, planned intercept, tilt, and energy against the displayed task. Type the exact `MOVE <command_id>` text.
5. The executor moves, measures the endpoint, predicts removal from the achieved beam, and rejects no tissue removal or protected removal.
6. Confirm the measured endpoint, unchanged protected mask, clear enclosure, and enabled independent cutoff. Type the exact `PULSE <command_id>` text.
7. The client executes the calibrated PWM sequence, verifies stopped state, and records the receipt.
8. The robot returns to the scan pose. The operator inspects/cleans the window, creates a new volume, and presses Enter.
9. The adapter requires a new content hash. `ControllerSession.update()` consumes the new observation exactly once.
10. The cycle continues until `completion_gate`. The existing controller repairs its plan after every ten confirmed pulses. Stop immediately under any condition in Section 1; never type either approval string for an uncertain state.

## 10. Experiment 2 acceptance

Use the protected-boundary phantom and unchanged nominal method. Accept only when:

- terminal reason is `completion_gate`;
- remaining target satisfies the configured completion threshold;
- protected voxels removed and hard violations both equal zero;
- every confirmed pulse has one completed receipt and one later unique observation;
- no receipt has uncertain status.

Require `EXPERIMENT_2_COMPLETE` in `machine_readables/workflow_status.json`. Review all evidence before preparing experiment 3.

## 11. Experiment 3 acceptance

Use a documented changed phantom formulation while retaining the experiment-2 controller, physics, calibrations, registration, and acquisition preset. Define a target that requires at least ten pulses. Run with `--experiment 3` and `config/experiment_3.yaml`.

Accept only when experiment-2 safety conditions remain satisfied, at least one predicted/observed tissue exclusive-or volume exceeds the measured five-scan OCT repeatability, and a periodic repair event is recorded after confirmed pulse 10. Require `EXPERIMENT_3_COMPLETE` in `machine_readables/workflow_status.json`.

## 12. Artifact review

`human_readables/` contains the designation figure and the operator-facing `interpretation_summary.md` written after the run. `machine_readables/` contains method/configuration identities, scan manifests and volumes, task masks, observations, motion states, PWM requests/replies, receipts, controller planning/repair records, events, mismatch metrics, and terminal status.

Do not declare completion from screenshots or console output. The experiment completion flag and its supporting machine-readable records define the result.

## Reference source map

- OCT protocol/processing: `../../hybrid_arm_mirror/oct/Serial commands for Lumedica OCT software.pdf`, the Python files under `../../hybrid_arm_mirror/oct/`, and root-level `../../hybrid_arm_mirror/oct_calib.py`, `laser_oct_world.py`, and `test_oct.py`.
- OCT/camera prototypes excluded from this direct registration chain: `../../hybrid_arm_mirror/unit_tests/oct_rgb_calibration_implementation.py` and `oct_rgb_calibration_test.py`.
- Raspberry Pi laser deployment files: `../../see_plan_cut/ndyag_laser_control/`.
- Historical integrated sequencing: `../../see_plan_cut/planned_cut_execute.py`.
- ALO-RATS physical entry point: `../src/alo_rats_hardware/hardware_experiment.py`.
