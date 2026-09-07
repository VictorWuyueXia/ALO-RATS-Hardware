# Current status and transfer handoff

## Purpose and present boundary

ALO-RATS-Hardware is a standalone delivery workspace for the ALO-RATS MPPI planner-controller and the collaborator's UR5e support model. It keeps the MPPI numerical implementation unchanged and supplies a clean robot/OCT-facing boundary around it.

The implemented and tested boundaries are:

```text
nominal scan -> designation -> unchanged MPPI -> PyBullet UR5e motion
                                  -> virtual pulse -> synthetic volume -> replanning

processed registered OCT volume -> designation -> MPPI method identity
                                  -> UR5e RTDE state -> evidence record
```

The second path is deliberately a robot-motion-only dry run. It cannot trigger an OCT acquisition, command a laser, or execute an MPPI action on hardware.

## What is in this repository

- `mppi/`: exact copied MPPI runtime source and configuration authority.
- `assets/ur5e/`: collaborator-supplied UR5e URDF and its referenced mesh assets.
- `simulation/`: PyBullet-only MPPI workflow, virtual ablation, and interaction preview. Its isolation guard blocks RTDE, OCT, laser, and all non-local socket access.
- `src/alo_rats_hardware/`: strict processed-OCT importer, task-designation UI, RTDE state/safe-motion client, and dry-run entry point.
- `config/site.example.yaml`: the only device-specific configuration template. `config/site.yaml` is local and Git-ignored.

The repository intentionally excludes the old Git histories, datasets, artifacts, raw OCT acquisition software, Nd:YAG/PWM control code, and hardware results.

## Validation completed on the development machine

- `92 passed` on the detected development-machine backend, including CPU/1-GPU/4-GPU/8-GPU profile-selection contracts.
- The supplied URDF loaded in headless PyBullet with all 15 required URDF/mesh assets.
- The nominal processed-OCT fixture passed the strict volumetric interchange loader.
- The hardware command-line entry point imports without loading RTDE until an explicit connection request.

These results establish software packaging and no-device behavior only. They do not establish physical registration, focus, safe motion, OCT acquisition, laser delivery, or tissue outcome.

## Prepare the destination computer

1. Clone or unpack this repository in a user-writable directory.
2. Create the declared environment and install both editable packages:

   ```bash
   cd /path/to/ALO-RATS-Hardware
   conda env create -f environment.yml
   conda activate alo-rats-hardware
   python -m pip install -e ./mppi -e .
   ```

   On Windows PowerShell, use `python -m pip install -e .\mppi -e .`; the editable installation is required even when the Conda environment already exists.

3. Run the no-device checks from a desktop login session:

   ```bash
   mkdir -p outputs
   check_root=$(mktemp -d "$PWD/outputs/checks.XXXXXX")
   python simulation/check_robot_scene.py --output-dir "$check_root/robot_scene"
   python scripts/create_nominal_oct_fixture.py --output "$check_root/nominal_processed_oct.npz"
   python -m pytest -q --basetemp "$check_root/pytest" tests simulation/test_scan_adapter.py simulation/test_robot_executor.py
   ```

   In Windows PowerShell, create `$check_root` under `outputs`, pass `(Join-Path $check_root "robot_scene")` to `simulation\check_robot_scene.py --output-dir`, and pass `(Join-Path $check_root "pytest")` to pytest with `--basetemp`. The full PowerShell block is in the repository README and experiment runbook.

4. For the UR5e dry run, copy `config/site.example.yaml` to `config/site.yaml`, then replace every value with laboratory-verified values. The copied endpoint and joint pose are historical values from the collaborator script, not a calibration certificate.
5. Verify the hardware machine can import the RTDE modules:

   ```bash
   python -c "import rtde_control, rtde_receive; print('RTDE OK')"
   ```

6. Preserve the environment inventory and unavailable-device evidence before any integration work:

   ```bash
   python -m pip freeze > hardware-python-packages.txt
   python -c "from oct.module_oct_vol_scan import oct_raster_scan; print('OCT trigger OK')"
   ```

The final OCT command is expected to fail with this repository alone. A successful import identifies that the missing scanner trigger has been recovered from the original hardware setup.

## How to use the current deliverable

### Nominal PyBullet MPPI demonstration

```bash
demo_root=$(mktemp -d /tmp/alo-rats-simulation.XXXXXX)
python simulation/run_simulation.py \
  --case compact_diagnostic --output-dir "$demo_root/run"
```

Approve the displayed geometry, then focus the PyBullet window and press Enter. JAX automatically uses its detected default backend and the matching configured MPPI compute profile; `method.json` records both. This runs one process: designation, MPPI planning, checked URDF motion, virtual ablation, synthetic volumetric observation, and replanning. The compact diagnostic intentionally fails its frozen treatment-acceptance gate, so a nonzero exit is expected after `acceptance.json` is written.

### Processed-OCT plus robot dry run

The input must meet [the processed-OCT contract](PROCESSED_OCT_CONTRACT.md): a fully segmented, registered 0.1 mm Boolean occupancy volume in `.npz` format. Raw OCT image folders, point clouds, and surface-only observations are rejected.

```bash
python scripts/run_hardware_dry_run.py \
  --site config/site.yaml \
  --scan /path/to/processed_oct_volume.npz \
  --output-dir outputs/dry_run_001
```

The operator uses the mouse-driven window to designate target and protected geometry. After approval, the program records the method identity, opens RTDE, records measured joints/TCP, writes evidence, and disconnects. It sends no motion by default. `--move-safe-pose` is an explicit opt-in that sends only the reviewed six-joint safe pose from `config/site.yaml`.

## Device-interface evidence and compatibility

### UR5e

The collaborator repository uses `rtde_control.RTDEControlInterface` and `rtde_receive.RTDEReceiveInterface` for `moveJ`, `servoJ`, joint state, and TCP state. This repository uses the same RTDE interfaces only for inspection and the opt-in safe `moveJ`. It should be compatible once `ur-rtde`, network routing, robot firmware, endpoint, and the approved pose are verified on the destination machine.

### OCT

The collaborator calibration script calls `oct.module_oct_vol_scan.oct_raster_scan().start_oct_scan(...)`. That module is absent from the checked-in robot repository. Its own comments state that OCT scans are acquired and processed on another OCT computer, then transferred to the robot-side workflow. The checkout also contains a Lumedica serial-command PDF and offline Open3D/OpenCV-based utilities, but not a complete usable scanner driver.

Therefore this delivery is compatible with the **processed OCT output**, not automatically with the scanner-control software. Recover the missing trigger module and its Lumedica/serial configuration, then implement a reviewed acquisition-and-segmentation adapter that writes the declared volume contract.

### Laser

The collaborator script calls a laser helper that connects by TCP socket to a separate PWM service. The old checkout also contains Raspberry Pi GPIO PWM code. That implies an additional reachable PWM server and likely a Raspberry Pi-specific software environment. This repository intentionally excludes all of those modules. It cannot fire a laser and is not compatible with the laser device without a new, separately reviewed safety implementation.

## Work still required before physical cutting

1. Recover and validate live OCT triggering and the raw-scan segmentation pipeline.
2. Measure and version OCT-to-robot/planning registration, including scan pose and treatment-frame authority.
3. Add a hardware executor that maps an MPPI action to a calibrated beam pose, performs collision/keep-out preflight, and records achieved robot state.
4. Validate focus/standoff, optical-axis convention, and physical energy-to-PWM response on suitable phantoms.
5. Add reviewed laser interlocks, emergency-stop behavior, arm/disarm state, bounded pulse commands, and an execution receipt.
6. Reacquire, segment, register, and validate OCT after every physical pulse before allowing the controller session to advance.
7. Complete phantom and tissue validation before any live cutting demonstration.

No current software test or simulation result authorizes those physical actions.
