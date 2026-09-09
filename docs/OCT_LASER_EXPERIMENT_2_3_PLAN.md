# OCT and laser integration plan for experiments 2 and 3

## Goal and present result

The physical workflow is:

```text
Windows Lumedica application -> mounted B-scan folder -> registered occupancy
-> operator target and protected-boundary designation -> unchanged ALO-RATS MPPI
-> checked UR5e beam pose -> Raspberry Pi PWM pulse -> return to scan pose
-> new OCT folder -> registered occupancy update -> continuation or periodic repair
```

Experiment 2 measures target removal with zero protected-voxel removal. Experiment 3 uses a documented changed phantom formulation, measures disagreement between predicted and OCT-observed tissue, and verifies periodic plan repair after ten confirmed pulses.

The software interfaces and coordinator are implemented, and their component tests pass. Coordinator replay and physical qualification remain pending. Physical experiments remain blocked until the live-interface, calibration, inert-motion, and bounded-pulse milestones below have direct evidence. This is a show-of-concept research system; the laboratory laser-safety procedure, enclosure, interlocks, emergency stop, beam dump, and trained operators govern physical use.

## Correct reference ownership and recovered behavior

The two reference trees are read-only.

### OCT: `hybrid_arm_mirror`

- `oct/module_oct_vol_scan.py` calls the historical local scanner trigger.
- `oct/oct_serial.py` uses 115200 baud and sends `SetSampleName`, `StartCapture VolumeScan`, and `SaveQueueImages`. The supplied Lumedica serial-command PDF describes the Windows application command interface and processed TIFF output.
- `oct/module_oct_folder_viz.py` and `oct/module_oct_fine.py` read ordered B-scans, filter image intensity, select a surface row, apply pixel spacing and transform chains, and save reconstructed data.
- `oct_calib.py` estimates an end-effector/OCT hand-eye transform from multiple robot poses and manually selected planar marker axes. `laser_oct_world.py` acquires at selected robot waypoints and logs the measured TCP pose with each OCT filename. Root-level `test_oct.py` uses those pose records to transform and stitch scans.
- `oct/module_oct_vol_truncate.py` and root-level `test_oct.py` contain historical OCT/TCP transform examples. Their axis and translation conventions differ, so they cannot serve as current calibration.
- `unit_tests/oct_rgb_calibration_implementation.py` and `oct_rgb_calibration_test.py` are prototypes for OCT-to-monocular-camera projection. The present workflow registers OCT directly to the planning/robot chain and does not add that camera frame.
- The active setup uses the Lumedica application on the Windows OCT computer and exposes completed scan folders through a mounted Ubuntu path. ALO-RATS therefore imports folders and does not issue serial commands.

### Raspberry Pi laser: `see_plan_cut/ndyag_laser_control`

- `laser_control_pwm.py`, `laser_pwm_client.py`, and `code_4_pi/laser_pwm_client2.py` send one JSON document per TCP connection. Their action names are `start`, `set_pwm`, `stop`, and `status`; their historical endpoint is `10.194.210.35:8000`.
- The clients start at zero duty cycle, wait 0.2 seconds, set duty cycle and frequency, wait for the pulse duration, stop, and request status.
- `code_4_pi/laser_control_pi.py` contains direct Raspberry Pi GPIO/PWM logic, but its attribute and constructor names are inconsistent.
- No file under the downloaded deployed-code folder calls `bind`, `listen`, or `accept`. The listening service implementation, exact response JSON, and independent pulse watchdog must be inspected on the live Raspberry Pi.
- `see_plan_cut/planned_cut_execute.py` shows the historical robot-move/PWM-pulse sequence and retries exceptions. ALO-RATS records an uncertain pulse and terminates; it never repeats that pulse automatically.

Historical file values identify protocol shape and processing intent. Every address, transform, pixel scale, response field, and PWM calibration value must be measured on the active equipment.

The current Ubuntu host reports `/mnt/OCT_Data` as a read-only CIFS mount from `//192.168.1.2/OCT_Data`. A five-second directory/stat check did not return, so folder contents and the active preset are not yet qualified. The local `config/site.yaml` contains transferred geometry and unresolved laser placeholders; it is not a qualified physical configuration.

## Fixed design

1. The Windows operator performs OCT acquisition. `OCTFolderAdapter` snapshots the current mounted folders at program start, accepts exactly one later stable complete folder, and leaves the source folder unchanged.
2. Every feedback scan uses one qualified fixed scan joint pose and stores the measured TCP transform. The historical multi-waypoint scan/stitch procedure remains available as calibration evidence but is not used during these repeatability-sensitive experiments.
3. `ControllerSession` remains the planning and state-update implementation. The MPPI source and `mppi/configs/controller.yaml` are unchanged.
4. Each confirmed pulse has one `ActionRequest`, one completed `ExecutionReceipt`, and one later unique `ControllerObservation`.
5. Every robot motion and pulse requires the operator to type the displayed command identifier.
6. TCP requests retain the deployed action names and one-connection-per-JSON exchange. No retry follows any command after emission may have begun.
7. The OCT processor implements the height-field tissue geometry used by experiments 2 and 3: it fills occupancy below the segmented surface. Cavities and overhangs require a later volumetric segmentation method.
8. The protected structure is the immutable protected boundary saved by the designation editor.
9. Independent file decoding/filtering uses threads. Robot motion, emission, and acquisition remain sequential because they change one specimen.
10. Code remains compact and task-oriented: 40–300 lines per code file, no optional fallback configuration, no thin wrapper functions, and no ignored exceptions. New code is added only for a required hardware boundary.

## Implemented structure and code budget

### Classes and objects

| Object | File | Exact responsibility |
| --- | --- | --- |
| `OCTFolderAdapter` | `src/alo_rats_hardware/oct_folder.py` | Identify one stable new B-scan folder, validate/hash its ordered images, reconstruct/register a surface, and emit one `VolumetricScan`. |
| `LaserPWMClient` | `src/alo_rats_hardware/laser.py` | Send the deployed JSON action vocabulary, convert calibrated energy to duty cycle, verify stopped status, and create pulse evidence. |
| `SiteConfiguration` | `src/alo_rats_hardware/site.py` | Validate the exact robot, registration, OCT-share, and laser configuration. |
| `UR5eConnection` | `src/alo_rats_hardware/robot.py` | Read RTDE state, solve and check one beam pose, execute `moveJ`, measure the endpoint, and return to the scan pose. |
| `BeamCalibration` | `src/alo_rats_hardware/beam.py` | Convert requested and achieved beam geometry through the planning/base transform and measured standoff. |
| `RunRecords` | `src/alo_rats_hardware/records.py` | Write append-only events and task-specific machine-readable arrays and JSON. |

No new cross-step data class is present. Existing `VolumetricScan`, `DesignatedTask`, `ActionRequest`, `ExecutionReceipt`, and `ControllerObservation` objects define the step boundaries.

### Functions and methods

| Function or method | Parameters | Exact responsibility |
| --- | --- | --- |
| `OCTFolderAdapter.__init__` | `site`, `records` | Validate the mounted root and snapshot existing directories. |
| `OCTFolderAdapter.acquire` | `command_id`, `scan_pose_base_m` | Wait for one stable new folder, reconstruct and register it, save scan/manifest evidence, and return the scan. |
| `LaserPWMClient.__init__` | `site`, `records` | Bind the validated endpoint, calibration table, response schema, and event record. |
| `LaserPWMClient._send_command` | `command` | Perform one TCP request/response exchange and record both JSON mappings. This is the single protocol helper. |
| `LaserPWMClient.stop` | none | Send `stop`, request `status`, and require the configured stopped field/value. |
| `LaserPWMClient.execute` | `request`, `achieved_action` | Interpolate duty cycle, execute the bounded command sequence, and return a completed receipt after stopped status. |
| `SiteConfiguration.__post_init__` | dataclass fields | Reject missing/extra keys, invalid shapes/limits, placeholders in physical mode, and unqualified calibration/watchdog data. |
| `load_site` | `path` | Parse one strict YAML site file. |
| `UR5eConnection.connect` | none | Open RTDE control and receive channels without motion. |
| `UR5eConnection.snapshot` | none | Read finite six-joint and six-value TCP state. |
| `UR5eConnection.move_to_safe_pose` | none | Execute the configured approved safe pose. |
| `UR5eConnection.return_to_scan_pose` | none | Execute the configured scan pose and return its measured TCP transform. |
| `UR5eConnection.execute_action` | `request`, `task_state` | Transform the beam request, check UR safety limits and joint change, move, measure, and validate beam error. |
| `UR5eConnection._move_approved_joints` | `joints`, `name` | Share the substantive safety/motion logic used by safe-pose and scan-pose operations. |
| `UR5eConnection.close` | none | Stop the RTDE script and disconnect both channels. |
| `run_experiment` | `site_path`, `experiment_number`, `metadata_path`, `output` | Execute the one-pulse/one-scan state machine and write terminal evidence. |
| `main` | none | Parse only `--site`, `--experiment`, `--metadata`, and `--output-dir`. |

No additional helper, wrapper, class, data class, or command-line parameter is planned for experiments 2 and 3.

### Independent configuration values

`config/site.yaml` contains exactly:

- `physical_execution_enabled`;
- `robot`: `ip`, `safe_joint_pose_rad`, `scan_joint_pose_rad`, `move_speed_rad_s`, `move_acceleration_rad_s2`, `maximum_joint_delta_rad`, `maximum_joint_error_rad`;
- `registration`: `calibration_id`, `base_from_planning_m`, `tcp_from_oct_m`, `tcp_from_laser_m`, `laser_standoff_m`, `maximum_intercept_error_mm`, `maximum_axis_error_rad`;
- `oct`: `shared_root`, `file_pattern`, `expected_b_scans`, `image_shape_px`, `pixel_spacing_lateral_scan_depth_mm`, `axis_order`, `axis_signs`, `surface_margin_px`, `surface_threshold_u8`, `planning_volume_bounds_mm`, `planning_frame_id`, `stable_observations`, `poll_interval_s`, `scan_timeout_s`;
- `laser`: `host`, `port`, `timeout_s`, `frequency_hz`, `pulse_duration_s`, `startup_delay_s`, `energy_to_duty_cycle`, `status_key`, `stopped_value`, `watchdog_qualified`, `calibration_id`.

The experiment metadata contains exactly `experiment_number`, `experiment_run_id`, `phantom_id`, `phantom_formula`, `phantom_batch_id`, `phantom_preparation_time`, `operator_ids`, `laser_safety_approval_id`, `registration_calibration_id`, `laser_calibration_id`, `oct_repeatability_xor_mm3`, `random_seed`, and `raster_settings`. `raster_settings` contains the five existing planner inputs shown in the example files. Placeholder identifiers and a non-positive repeatability measurement prevent execution.

## Governing conversions and measurements

For a scan acquired at the measured robot pose, the OCT-to-planning transform is

$$
{}^{P}T_{OCT}=\left({}^{B}T_P\right)^{-1}{}^{B}T_{TCP}{}^{TCP}T_{OCT}.
$$

$P$ is the ALO-RATS planning frame, $B$ is the UR5e base frame, $TCP$ is the tool-center-point frame, and $OCT$ is the reconstructed scanner frame. ${}^{B}T_P$, ${}^{B}T_{TCP}$, and ${}^{TCP}T_{OCT}$ are measured 4-by-4 rigid transforms. Pixel positions are first converted to millimeters with the preset-specific lateral, scan, and depth spacing, then transformed with ${}^{P}T_{OCT}$ and resampled onto the fixed 0.1 mm planning lattice.

The fixed-duration PWM calibration is

$$
D(E)=\operatorname{interp}\left(E;\{(E_i,D_i)\}_{i=1}^{n}\right).
$$

$E$ is requested energy in joules, $E_i$ is power-meter-measured energy in joules, $D_i$ is duty cycle in percent, and `interp` is piecewise-linear interpolation between adjacent measured pairs. Values outside the measured energy interval are rejected. The run reports calibrated commanded energy unless a pulse-specific power-meter measurement is added separately.

The experiment-3 discrepancy after pulse $k$ is

$$
V_{\mathrm{xor},k}=s^3\left|T_{\mathrm{pred},k}\mathbin{\triangle}T_{\mathrm{obs},k}\right|.
$$

$s=0.1$ mm is the planning voxel spacing, $T_{\mathrm{pred},k}$ is predicted Boolean tissue occupancy, $T_{\mathrm{obs},k}$ is OCT-observed Boolean tissue occupancy, $\triangle$ is exclusive-or, and vertical bars count voxels. `oct_repeatability_xor_mm3` is the maximum pairwise value measured from five unchanged-phantom scans at the same pose and preset.

Experiment 2 reports

$$
V_{\mathrm{remaining},k}=s^3|G\cap T_{\mathrm{obs},k}|,
\qquad
V_{\mathrm{protected},k}=s^3|C\cap(T_0\setminus T_{\mathrm{obs},k})|.
$$

$G$ is the immutable target mask, $C$ is the immutable protected mask, and $T_0$ is initial tissue occupancy. Completion requires $V_{\mathrm{protected},k}=0$ for every $k$.

## Milestones, validation, and success flags

### Stage 0 — live interface identification: pending

Restore responsive access to `/mnt/OCT_Data`, then record one untouched scan folder, file pattern/count/shape, Windows preset, mount details, robot identity/TCP/safety configuration, and exact Raspberry Pi `stop` and `status` replies with laser power disabled. Verify an independent hardware pulse cutoff. Measure transforms, standoff, five-scan OCT repeatability, and the power-meter energy table.

Validation: save hashes, dates, equipment identifiers, operator identity, and observed replies. No emission occurs.

Success flag: `LIVE_INTERFACES_IDENTIFIED`. Missing independent cutoff or unknown stopped reply prevents physical mode.

### Stage 1 — folder-to-volume adapter: implemented; recorded-scan qualification pending

The adapter checks a contiguous numeric B-scan sequence, stable file size/time signatures, image count/shape, intensity threshold, registration, coverage, and source content hashes. It converts files in parallel and writes a processed `.npz` plus manifest. The manifest preserves source paths and per-file hashes; source B-scans remain on the mounted share.

Validation: process at least three copied recorded folders twice and require identical `scan_id` and occupancy. Tests must reject incomplete folders, noncontiguous names, changed shape, ambiguous new folders, and insufficient surface evidence.

Success flag: `OCT_FOLDER_TO_VOLUME_PASS`.

### Stage 2 — robot action and return: implemented; inert qualification pending

The physical executor uses the measured laser/TCP transform, UR inverse kinematics, configured maximum joint change, measured scan/safe-pose joint error, UR safety-limit checks, measured endpoint, and beam intercept/axis tolerances. The simulation executor retains URDF self/environment collision checks. The physical path does not contain a calibrated model of the OCT head, fixture, or laboratory obstacles; the UR installation safety planes and inert-phantom trajectory qualification must represent those exclusions before use.

Validation: replay planned actions in PyBullet, then execute scan/treatment/return motions over an inert phantom with the laser disconnected. Measure endpoint and fiducial errors and inspect every path.

Success flag: `ROBOT_ACTION_AND_RETURN_PASS`.

### Stage 3 — bounded PWM pulse: implemented; live qualification pending

The client validates energy without extrapolation and sends `start(0)`, `set_pwm`, `stop`, and `status`. It accepts completion only when the audited response field equals the audited stopped value. Any ambiguity after PWM may begin creates an uncertain outcome and terminates the experiment.

Validation: use the local protocol test, then the live Raspberry Pi with laser power disabled, then a stationary beam dump and power meter. Force transport failures and confirm that no automatic retry occurs.

Success flag: `LASER_BOUNDED_PULSE_PASS`.

### Stage 4 — coordinator and evidence: implemented; replay and live-cycle qualification pending

The coordinator preserves request, motion approval, measured motion, predicted-removal check, pulse approval, receipt, return, new scan, observation, and controller update. It uses all visible JAX devices. Human-readable figures and machine-readable source/configuration/state records are separated.

Validation: component tests use fake RTDE and a loopback PWM service. Next, replay the complete coordinator with a recorded OCT sequence and device test doubles, and inject a failure at every phase. Then execute one inert motion/OCT cycle and one approved bounded phantom pulse/OCT cycle.

Success flags: `COORDINATOR_REPLAY_PASS`, followed by `ONE_PHYSICAL_CYCLE_PASS`.

### Stage 5 — experiments 2 and 3: pending physical work

Run experiment 2 first. Accept only `completion_gate`, zero protected removal, zero hard violations, no uncertain receipt, and matching receipt/observation counts. Then run experiment 3 with unchanged controller/physics/calibration, a documented changed phantom, a target requiring at least ten pulses, at least one discrepancy above OCT repeatability, and a recorded repair after pulse 10.

Success flags: `EXPERIMENT_2_COMPLETE` and `EXPERIMENT_3_COMPLETE` in their respective `machine_readables/workflow_status.json` files.

## Required execution order for today

1. Complete Stage 0 and copy the examples to local `config/site.yaml` and experiment metadata files.
2. Complete Stage 1 using recorded Lumedica folders.
3. Complete Stage 2 with laser power disconnected.
4. Complete Stage 3 with power disabled, then beam dump and power meter.
5. Complete Stage 4 replay and one-cycle qualifications.
6. Execute experiment 2 and review its artifacts before preparing experiment 3.
7. Execute experiment 3 and compare predicted/observed discrepancy around the pulse-10 repair.

No later stage compensates for a missing earlier success flag.
