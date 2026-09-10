# Current status and transfer handoff

## Present boundary

ALO-RATS-Hardware now contains two related workflows around the unchanged MPPI method:

```text
synthetic observation -> designation -> MPPI -> checked PyBullet UR5e motion
-> virtual pulse -> synthetic reobservation -> controller update

Windows Lumedica application -> mounted B-scan folder -> registered observation
-> designation -> MPPI -> checked physical UR5e motion -> Raspberry Pi PWM pulse
-> return -> new mounted B-scan folder -> controller update
```

The second workflow is implemented and disabled by default. It contains transferred fixed Experiment 2 geometry but has no live-device qualification. Software presence does not authorize physical motion or emission.

## Repository contents

- `mppi/`: copied MPPI runtime and exact method configuration; numerical source/configuration remain unchanged.
- `assets/ur5e/`: collaborator-supplied UR5e URDF and referenced meshes.
- `simulation/`: PyBullet-only workflow with virtual ablation and explicit non-local device isolation.
- `src/alo_rats_hardware/oct_folder.py` and `scripts/check_oct_folder.py`: stable mounted-folder import, parallel B-scan processing, height-field reconstruction, registration, processed-volume output, and one-folder preparation check.
- `src/alo_rats_hardware/robot.py`: RTDE state, safety/IK checks, action motion, endpoint measurement, and scan-pose return.
- `src/alo_rats_hardware/laser.py`: one-JSON-per-connection Raspberry Pi PWM client with energy calibration and stopped-state verification.
- `src/alo_rats_hardware/hardware_experiment.py`: experiment 2/3 operator-approved coordinator and terminal evidence.
- `config/site.yaml`: transferred fixed Experiment 2 geometry with physical execution disabled; live Raspberry Pi response, energy, watchdog, and experiment-record values remain unqualified.
- `config/site.example.yaml`, `config/experiment_2.example.yaml`, and `config/experiment_3.example.yaml`: portable templates.

The repository excludes raw OCT drivers, the Raspberry Pi TCP listener, live laser calibration measurements, hardware results, datasets, and old Git history.

## Validation completed on 2026-09-08

- `105 passed` with global pytest plugins disabled and the system C++ runtime preloaded.
- The supplied URDF loads with all required mesh assets.
- Processed-OCT fixtures satisfy the strict interchange loader.
- A generated B-scan folder converts deterministically to registered occupancy.
- The loopback PWM server observes `start`, `set_pwm`, `stop`, and `status` in order and verifies interpolation.
- Fake RTDE interfaces verify inverse-kinematics, safety, motion, measured-endpoint, and beam-error logic.
- The physical entry point imports without contacting hardware.

These results establish offline software behavior. They do not establish the active OCT preset, physical registration, laboratory obstacle clearance, live Raspberry Pi replies/watchdog, focus, energy delivery, or tissue outcome.

The host reports `/mnt/OCT_Data` as a read-only CIFS mount from `//192.168.1.2/OCT_Data`. Directory and `stat` requests did not return within five seconds during the current audit, so no scan folder was inspected. `config/site.yaml` now exists with transferred geometry and unresolved laser placeholders; it is not a qualified physical configuration.

## Destination computer preparation

```bash
cd /path/to/ALO-RATS-Hardware
/home/rp/anaconda3/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pybullet==3.2.7 pyvista==0.48.4
.venv/bin/python -m pip install -e ./mppi -e '.[robot,test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m pytest -q
```

The current host uses Python 3.12.4, NumPy 1.26.4, JAX 0.4.38, SciPy 1.13.1, OpenCV 4.11.0.86, and PyBullet 3.2.7. `environment.yml` and both package metadata files preserve the supported constraints. Verify RTDE separately:

```bash
.venv/bin/python -c "import rtde_control, rtde_receive; print('RTDE OK')"
```

GPU execution must occur outside the sandbox. Require JAX backend `gpu` with device `0` before a physical experiment; the coordinator selects the baseline `1gpu` profile and records the detected device list.

## Current use

### Optional planner evaluation

```bash
suite_root=$(mktemp -d /tmp/alo-rats-simulation-suite.XXXXXX)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m simulation.paths.check_simulation \
  --output-dir "$suite_root/run"
```

This expensive command evaluates completion across the tracked simulation case matrix with the same MPPI method configuration used by physical deployment. It is not a hardware-deployment prerequisite. The simulation isolation layer rejects RTDE, OCT, laser modules, and non-local socket connections.

### Processed-OCT robot dry run

```bash
.venv/bin/python scripts/run_hardware_dry_run.py \
  --site config/site.yaml \
  --scan /path/to/processed_oct_volume.npz \
  --output-dir outputs/dry_run_001
```

The input follows [`04-PROCESSED_OCT_CONTRACT.md`](04-PROCESSED_OCT_CONTRACT.md). The command records RTDE state and sends no motion unless the operator supplies `--move-safe-pose`. It does not import or call the laser client.

### Physical experiment entry point

```bash
.venv/bin/python -m alo_rats_hardware.hardware_experiment \
  --site config/site.yaml \
  --experiment 2 \
  --metadata config/experiment_2.yaml \
  --output-dir outputs/experiment_2_001
```

This command fails before hardware connection when physical execution is disabled or any required calibration/metadata value is missing. Follow [`01-OPERATOR_RUNBOOK.md`](01-OPERATOR_RUNBOOK.md); do not enable it before every earlier success flag is recorded.

## Correct interface provenance

### OCT

`hybrid_arm_mirror` is the OCT reference tree. Its `oct/` serial files document `SetSampleName`, `StartCapture VolumeScan`, and `SaveQueueImages`; its folder-processing files document B-scan ordering, filtering, surface extraction, pixel scaling, and transform intent. Root-level `oct_calib.py`, `laser_oct_world.py`, and `test_oct.py` add hand-eye estimation, scan-pose/filename recording, and multi-scan stitching evidence. The active apparatus uses the Windows Lumedica application and mounted Ubuntu share, so ALO-RATS uses the folder output and sends no scanner command. Historical transforms conflict and are excluded from the example calibration.

### Laser

`see_plan_cut/ndyag_laser_control/` is the Raspberry Pi deployment-code reference. Its client files use one JSON request per TCP connection, actions `start`, `set_pwm`, `stop`, and `status`, and historical endpoint `10.194.210.35:8000`. The downloaded folder contains direct GPIO code and TCP clients but no listener implementation. The live service's response schema and independent watchdog therefore remain required audit inputs.

### Robot

The physical executor uses the same ur-rtde control/receive interfaces as the collaborator workflow. It checks the requested pose and IK joints against the active UR safety configuration, limits total joint change, verifies final scan/safe-pose joint error, executes `moveJ`, and validates the measured beam endpoint. External laboratory objects are not represented in the physical executor; active UR safety planes and inert-motion qualification must cover them.

## Remaining work before experiments

1. Record `LIVE_INTERFACES_IDENTIFIED` from the mounted share, live Pi service, UR installation, watchdog, and calibration audit.
2. Save one complete 256-image Lumedica volume under `/mnt/OCT_Data/victor`, run the folder check, and record `PREPARATION_2_OCT_FOLDER_PASS`.
3. Execute inert scan/treatment/return paths and record `ROBOT_ACTION_AND_RETURN_PASS`.
4. Validate the Pi with power disabled, then measure a beam-dump energy table and record `LASER_BOUNDED_PULSE_PASS`.
5. Complete failure-injected replay and one physical cycle; record `COORDINATOR_REPLAY_PASS` and `ONE_PHYSICAL_CYCLE_PASS`.
6. Run experiment 2, review its complete evidence, then run experiment 3.

The detailed acceptance conditions are in [`01-OPERATOR_RUNBOOK.md`](01-OPERATOR_RUNBOOK.md) and [`05-EXPERIMENT_PLAN.md`](05-EXPERIMENT_PLAN.md).

## Reference source map

- OCT: [`../../hybrid_arm_mirror/oct/`](../../hybrid_arm_mirror/oct/).
- Raspberry Pi laser: [`../../see_plan_cut/ndyag_laser_control/`](../../see_plan_cut/ndyag_laser_control/).
- Historical integrated robot/laser sequence: [`../../see_plan_cut/planned_cut_execute.py`](../../see_plan_cut/planned_cut_execute.py).

These sources define historical behavior; they do not qualify the active apparatus.
