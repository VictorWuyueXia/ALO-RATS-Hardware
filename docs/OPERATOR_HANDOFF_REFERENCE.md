# Operator handoff values and code reference

## Minimal manual config values

Do not copy `config/site.example.yaml` or either experiment example over the working files. That would erase transferred values.

Before leaving, the repository maintainer fills these fields in the selected `config/experiment_2.yaml` or `config/experiment_3.yaml`:

| Key | Required form | Exact source |
| --- | --- | --- |
| `experiment_run_id` | Unique text, for example `E2_20260909_A` | Laboratory run schedule |
| `phantom_id` | Label text, for example `P-20260909-03` | Phantom label |
| `phantom_formula` | Exact formulation and revision | Preparation sheet |
| `phantom_batch_id` | Batch label | Preparation sheet |
| `phantom_preparation_time` | ISO 8601 time with zone | Preparation sheet |
| `operator_ids` | List of laboratory identifiers | Operator roster |
| `laser_safety_approval_id` | Approval/permit identifier | Laboratory safety record |
| `registration_calibration_id` | Exact copy of `site.registration.calibration_id` | Reviewed site file |
| `oct_repeatability_xor_mm3` | Positive decimal in mm³ | Maximum pairwise value from five scans |

Keep the supplied random seed and raster settings unchanged. `laser_calibration_id` remains blank until Deployment 4.

The current files contain a blocking mismatch: `site.registration.calibration_id` is `see_plan_cut_2025_08_29_tool0_oct_laser_geometry`, while both experiment files contain `OCT_UR5E_REG_2026-09-08`. Replace all three with one identifier belonging to a reviewed current calibration. Neither existing name proves validity.

During Deployment 4, the operator measures five laser values and edits no file. The repository maintainer changes only these fields:

| Site key | Required form | Exact source |
| --- | --- | --- |
| `laser.status_key` | JSON field name, for example `"status"` | Unedited stopped reply |
| `laser.stopped_value` | Exact JSON value, for example `"stopped"` or `0` | Same reply; preserve type and capitalization |
| `laser.energy_to_duty_cycle` | Two or more increasing `[energy_j, duty_cycle_pct]` rows | Mean power-meter result at each duty |
| `laser.watchdog_qualified` | `true` after the independent cutoff passes | Cutoff record |
| `laser.calibration_id` | Unique calibration identifier | Energy worksheet |

The maintainer copies `laser.calibration_id` to the selected experiment's `laser_calibration_id`, then changes `physical_execution_enabled` from `false` to `true` after every qualification record passes. No planner field is changed during Deployment 4.

## OCT computer and scan records

Write `$handoff_dir/machine_readables/oct/acquisition.json`:

```json
{
  "windows_host": "192.168.1.2",
  "lumedica_application_version": "VALUE_FROM_ABOUT_DIALOG",
  "preset_name": "VALUE_FROM_ACTIVE_PRESET",
  "export_folder": "EXACT_NEW_FOLDER_NAME",
  "file_pattern": "*.tif",
  "b_scan_count": 128,
  "image_shape_px": [512, 512],
  "pixel_spacing_lateral_scan_depth_mm": [0.028, 0.11173, 0.01459]
}
```

The count, shape, and spacing are transferred values. Verify them from the active preset/export metadata and one untouched folder; update the corresponding `oct` fields in `config/site.yaml` if measured values differ.

Record the untouched folder:

```bash
find "/mnt/OCT_Data/EXACT_NEW_FOLDER_NAME" -maxdepth 1 -type f -name '*.tif' -print0 \
  | sort -z | xargs -0 sha256sum \
  | tee "$handoff_dir/machine_readables/oct/folder_manifest.sha256"
```

Acquire five unchanged-phantom folders at the same scan pose and preset. The future OCT qualification command writes `repeatability.csv` with:

```text
scan_id_a,scan_id_b,xor_voxels,xor_volume_mm3
```

`xor_volume_mm3` is the volume represented by voxels occupied in exactly one member of a scan pair. The required `oct_repeatability_xor_mm3` is the largest pairwise value.

## UR5e and registration records

Write `$handoff_dir/machine_readables/robot/installation.json`:

```json
{
  "robot_ip": "192.168.1.103",
  "controller_serial": "VALUE_FROM_PENDANT",
  "firmware_version": "VALUE_FROM_PENDANT",
  "installation_name": "VALUE_FROM_PENDANT",
  "remote_control_enabled": true,
  "active_tcp_m_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "safe_joint_pose_rad": [1.610071, -1.041262, 1.274265, -3.389604, -1.140393, 1.535890],
  "scan_joint_pose_rad": [1.610071, -1.041262, 1.274265, -3.389604, -1.140393, 1.535890],
  "safety_planes_and_limits": "EXACT_ACTIVE_INSTALLATION_SUMMARY"
}
```

The all-zero active TCP is required by the transferred transform chain. The addresses and poses are transferred values; compare them with the active pendant.

Write `$handoff_dir/machine_readables/robot/registration.json` with `calibration_id`, the measured 4×4 `base_from_planning_m`, `tcp_from_oct_m`, and `tcp_from_laser_m` transforms, `laser_standoff_m`, measurement method, timestamp, and responsible person. Copy the reviewed values to the matching `registration` fields in `config/site.yaml`.

Also record firmware, remote-control state, active installation, TCP, safety planes, approved joint limits, actual joints, and actual TCP at the scan and safe poses. The physical executor does not model external laboratory objects; direct observation and the active UR safety installation must qualify each inert path.

## Raspberry Pi, laser, and power-meter records

Write `$handoff_dir/machine_readables/laser/service.json`:

```json
{
  "service_name": "EXACT_SYSTEM_SERVICE_NAME",
  "revision": "GIT_COMMIT_OR_DEPLOYMENT_ID",
  "host": "10.194.210.35",
  "port": 8000
}
```

Keep the exact status reply unmodified in `laser/pi_status.json`. Identify the field and value that mean output is stopped.

Store each raw trace under `laser/power_traces/`. Write `laser/energy_calibration.csv`:

```text
calibration_id,duty_cycle_pct,repeat_index,trace_file,integrated_energy_j,peak_power_w,pulse_duration_s
```

For measured power $P(t)$ in watts at time $t$ in seconds, emitted energy $E$ in joules is

$$
E=\int P(t)\,dt.
$$

Use five traces per selected duty cycle. Use the mean energy at each duty for `energy_to_duty_cycle`. Energy and duty values must both increase; the client rejects extrapolation.

Write `laser/watchdog.json` with `test_time`, `command`, `communication_interruption`, `measured_cutoff_time_s`, `stopped_evidence`, and `passed`. Set `watchdog_qualified: true` only after the independent device stops output without a later Ubuntu command.

The live status schema, cutoff behavior, service identity, and energy table are missing. The host, port, 100 Hz frequency, and 1.5 s duration are transferred historical values and require operator confirmation.

## Code responsibility and readiness

| File/module | Workflow responsibility | Ready evidence | Remaining work |
| --- | --- | --- | --- |
| `simulation/simulation/`, `simulation/robot/` | Synthetic observation, PyBullet motion, virtual pulse, reobservation | Offline tests and nominal run | No physical-device role |
| `mppi/src/laser_ablation/` | MPPI planning, prediction, controller update, repair | Baseline config and offline tests | Verify Linux JAX GPU before a timed run |
| `src/alo_rats_hardware/site.py` | Strict site schema and physical-enable rejection | Offline validated | Measured site values required |
| `src/alo_rats_hardware/oct_folder.py` | New-folder detection, parallel B-scan processing, registration, scan/hash records | Synthetic-folder tests | OCT-only entry point and active-preset qualification required |
| `src/alo_rats_hardware/surface_scan.py`, `src/alo_rats_hardware/scan_adapter.py` | Processed-volume contract and 0.1 mm controller lattice | Offline tests | Current registration required |
| `src/alo_rats_hardware/designation.py`, `src/alo_rats_hardware/task_designation.py` | Target/protected-volume designation | Offline UI available | Verify physical task geometry |
| `src/alo_rats_hardware/robot.py`, `src/alo_rats_hardware/beam.py` | RTDE, IK, checked `moveJ`, beam pose, scan return | Fake-RTDE and geometry tests | Laser-free action runner and inert qualification required |
| `src/alo_rats_hardware/laser.py` | JSON PWM client, energy interpolation, pulse, stopped verification | Loopback-tested | Live schema, watchdog, and energy table required |
| `src/alo_rats_hardware/hardware_experiment.py` | Experiment 2/3 motion → pulse → OCT → controller update | Import/component tests | Replay and one live cycle unqualified; experiments 1/4 unsupported |
| `src/alo_rats_hardware/records.py` | Events, motions, receipts, terminal status | Used by tested components | Review first complete live record |

The package does not trigger raw OCT acquisition, implement the Raspberry Pi listener, model external laboratory obstacles, or provide a physical entry point for scientific experiment 1 or 4.

The current Windows workspace also does not contain the historical `planned_cut_execute.py` or `ndyag_laser_control/` files cited by older notes; its sibling `see-plan-cut` checkout is a paper website. Treat the certified operator's separate working copy as apparatus provenance and inspect its deployed revision before relying on it.
