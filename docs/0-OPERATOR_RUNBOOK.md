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

The Lumedica Windows application owns acquisition. Ubuntu reads a mounted folder. Do not run the historical scanner serial code. `config/site.example.yaml` has `physical_execution_enabled: false` and contains deliberate placeholders, so it cannot emit a pulse.

Current host observation: `/mnt/OCT_Data` is mounted read-only from `//192.168.1.2/OCT_Data`, but directory and `stat` requests did not return within five seconds during the 2026-09-08 audit. Restore responsive access before OCT qualification. `config/site.yaml` is currently absent.

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

The current reference result is `103 passed`. Require zero failures. `environment.yml`, `pyproject.toml`, and `mppi/pyproject.toml` hold the portable dependency constraints.

For experiment execution, request an outside-sandbox run and verify the expected eight GPUs:

```bash
nvidia-smi
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Require backend `gpu` and eight devices. The coordinator records the detected devices and selects the matching existing MPPI compute profile.

## 3. Complete the live-interface audit

Create a unique audit directory under `outputs/` and record:

1. Restore responsive access to `/mnt/OCT_Data`; then record mount options, available space, Windows host, Lumedica preset, file pattern, B-scan count, image shape, and one untouched folder hash manifest.
2. One acquisition in the Windows application that creates exactly one new complete folder visible on Ubuntu.
3. UR5e address, firmware, remote-control state, active TCP definition, safety planes, scan joint pose, safe joint pose, maximum approved total joint change, and maximum accepted final joint error.
4. Raspberry Pi address, port, service name, service startup/shutdown process, and source/version actually running.
5. With laser power disabled, exact JSON replies to `stop` and `status`. The downloaded `ndyag_laser_control` folder contains no TCP listener, so inspect the running service rather than inferring replies.
6. An independent hardware pulse cutoff. Pulse duration controlled solely by a later Ubuntu `stop` command is unacceptable.
7. Measured planning/base, OCT/TCP, and laser/TCP transforms; laser standoff; five-scan OCT repeatability; and power-meter energy/duty table. Record hashes, dates, equipment identifiers, and operators.

Write `LIVE_INTERFACES_IDENTIFIED` only when every value has direct evidence.

## 4. Prepare strict configuration

```bash
cp config/site.example.yaml config/site.yaml
cp config/experiment_2.example.yaml config/experiment_2.yaml
cp config/experiment_3.example.yaml config/experiment_3.yaml
```

Replace every `REPLACE_...` value. Set OCT pixel spacing, axis order/signs, file pattern, count, shape, threshold, and planning bounds from the locked acquisition preset. Set `status_key` and `stopped_value` from the live reply. Fill at least two increasing energy/duty measurements. Set `watchdog_qualified: true` only after cutoff testing. Keep `physical_execution_enabled: false` through OCT, robot, and disabled-laser qualifications.

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
  --scan /absolute/path/to/processed_oct_volume.npz \
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

## 9. Standard experiment command and pulse cycle

```bash
run_dir="$PWD/outputs/experiment-2-$(date +%Y%m%d-%H%M%S)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m alo_rats_hardware.hardware_experiment \
  --site config/site.yaml \
  --experiment 2 \
  --metadata config/experiment_2.yaml \
  --output-dir "$run_dir"
```

The initial cycle is:

1. The program connects RTDE, verifies stopped laser status, and moves to the approved scan pose.
2. The operator creates the initial volume in the Windows application and presses Enter.
3. The adapter accepts the new folder; the operator designates and approves target and protected boundary.
4. MPPI creates one request. The operator verifies the displayed identity and types the exact `MOVE <command_id>` text.
5. The executor moves, measures the endpoint, predicts removal from the achieved beam, and rejects no tissue removal or protected removal.
6. The operator completes all hardware checks and types the exact `PULSE <command_id>` text.
7. The client executes the calibrated PWM sequence, verifies stopped state, and records the receipt.
8. The robot returns to the scan pose. The operator inspects/cleans the window, creates a new volume, and presses Enter.
9. The adapter requires a new content hash. `ControllerSession.update()` consumes the new observation exactly once.
10. The cycle continues; the existing controller repairs its plan after every ten confirmed pulses.

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
