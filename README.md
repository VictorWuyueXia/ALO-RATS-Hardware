# ALO-RATS-Hardware

This is a clean, history-free runtime workspace for integrating the ALO-RATS MPPI planner-controller with the collaborator's UR5e/PyBullet support model. Its purpose is to establish a reproducible route from a registered OCT observation and user-designated target to unchanged MPPI planning, UR5e-compatible motion geometry, and eventually reobservation/replanning.

It contains the **unchanged MPPI source and method configuration**, the supplied UR5e URDF/meshes, processed-OCT volume interchange, mouse-driven target designation, PyBullet simulation, and a robot-motion-only hardware dry run. It contains no Nd:YAG/PWM/laser-device code, no laser firing path, no raw OCT driver, no results, and no old Git history.

Read [the current situation and transfer handoff](docs/CURRENT_STATUS_AND_TRANSFER.md) before preparing a new computer. In particular, the OCT scanner trigger is not present in the collaborator checkout, and the package currently accepts a processed, registered OCT volume rather than raw scanner data.

## Install

```bash
cd /path/to/ALO-RATS-Hardware
conda env create -f environment.yml
conda activate alo-rats-hardware
python -m pip install -e ./mppi -e .
```

The environment uses PyBullet from Conda-forge and installs `ur-rtde` from PyPI. The MPPI package declares JAX and CasADi itself. Use a desktop login session for the Matplotlib and PyBullet windows.

## First machine check

```bash
mkdir -p outputs
check_root=$(mktemp -d "$PWD/outputs/checks.XXXXXX")
python simulation/check_robot_scene.py --output-dir "$check_root/robot_scene"
python scripts/create_nominal_oct_fixture.py --output "$check_root/nominal_processed_oct.npz"
python -m pytest -q --basetemp "$check_root/pytest" tests simulation/test_scan_adapter.py simulation/test_robot_executor.py
```

The scene command produces `$check_root/robot_scene/robot_scene.json`; it loads the supplied URDF in headless PyBullet and does not import RTDE or connect to any device. The fixture is synthetic and exists only to test the OCT/designation UI. A unique check root preserves prior evidence and gives pytest a known writable temporary directory.

On Windows development workstations, use PowerShell syntax after activating `alo-rats-hardware`:

```powershell
python -m pip install -e .\mppi -e .
$check_root = Join-Path $PWD ("outputs\checks-" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
New-Item -ItemType Directory -Path $check_root | Out-Null
python simulation\check_robot_scene.py --output-dir (Join-Path $check_root "robot_scene")
python scripts\create_nominal_oct_fixture.py --output (Join-Path $check_root "nominal_processed_oct.npz")
python -m pytest -q --basetemp (Join-Path $check_root "pytest") tests simulation/test_scan_adapter.py simulation/test_robot_executor.py
```

The editable-install command is required even when the Conda environment exists.

On an Ubuntu NVIDIA workstation, install the CUDA-enabled JAX wheel and verify the devices exposed to the program:

```bash
nvidia-smi
python -m pip install --upgrade "jax[cuda13]"
python -c "import jax; print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

The backend must print `gpu` before starting an experiment. The simulation uses exactly the first CUDA device and the baseline `1gpu` MPPI profile; a CPU-only JAX installation uses the declared `cpu` test profile. Native Windows JAX does not support NVIDIA CUDA, so PowerShell runs use the CPU backend; use the Ubuntu robot workstation or WSL2 for baseline-profile execution. PyBullet may use the graphics GPU for OpenGL display, but its rigid-body and inverse-kinematics computations remain CPU-side.

The repository snapshot passed 92 no-device tests on the development machine. The checks validate the processed-OCT contract, URDF asset closure, PyBullet interaction, controller/session interfaces, automatic compute-profile selection, and explicit simulation isolation. They do not validate an OCT scanner, a UR5e connection, laser focus, or tissue cutting.

## Robot and processed-OCT dry run

Copy `config/site.example.yaml` to `config/site.yaml` and replace all values with the laboratory's verified robot endpoint and safe joint pose. The bundled values reproduce the previous robot script's endpoint and home pose; they are not a calibration certificate.

After the OCT workstation has produced a registered `segmented_occupancy_volume` `.npz` file, launch:

```bash
python scripts/run_hardware_dry_run.py \
  --site config/site.yaml \
  --scan /path/to/processed_oct_volume.npz \
  --output-dir outputs/dry_run_001
```

The window shows the processed OCT upper envelope. Drag target footprints, set the left/right depths and protected Z, then click **Approve / save**. The program then resolves the exact MPPI configuration, connects through RTDE, reads the robot joint/TCP state, writes `hardware_dry_run.json`, and disconnects. It sends no robot motion by default and cannot control the laser.

Only after validating the output and physical workspace may an operator add `--move-safe-pose`. That one flag invokes `moveJ` only to the `safe_joint_pose_rad` explicitly placed in `config/site.yaml`; it still cannot fire a laser or execute an MPPI action.

See [the hardware dry-run procedure](docs/HARDWARE_DRY_RUN.md) and [the processed-OCT contract](docs/PROCESSED_OCT_CONTRACT.md) before connecting hardware.

## Simulation

The existing integrated demonstration remains separate from the hardware path:

```bash
demo_root=$(mktemp -d /tmp/alo-rats-simulation.XXXXXX)
python simulation/run_simulation.py \
  --case compact_diagnostic --output-dir "$demo_root/run"
```

It runs one process: designation → baseline MPPI authority (10 anchors, 128 samples per anchor, and seed `20260902`) → checked URDF motion → virtual pulse → synthetic volume observation → replanning. The root `config/simulation_cases.yaml` file is the sole authority for simulation geometry defaults and global-planner raster energy seeds. The comparable centered-rectangle and response-disturbance cases use the baseline 4/8-J raster seeds; geometrically distinct cases retain their exact-validated seed energies. `method.json` records the selected compute profile, JAX backend, and device list. It is a simulation-only application: `simulation/simulation_isolation.py` rejects RTDE, OCT, laser modules, and all non-local socket connections. The compact diagnostic is not an acceptance-quality treatment result; inspect its `acceptance.json`.

For a lightweight robot-interaction preview that does not run MPPI, use:

```bash
python simulation/run_robot_preview.py
```

This preview remains a nominal PyBullet demonstration only. It does not connect to a robot, OCT scanner, or laser.

## Contents

- `mppi/`: MPPI runtime source and its exact configurations.
- `assets/ur5e/`: supplied `ur5e_fixed.urdf` and only its referenced meshes.
- `simulation/`: PyBullet-only integrated MPPI and robot workflow.
- `src/alo_rats_hardware/`: processed-OCT, task UI, RTDE dry-run boundary.
- `config/`: site-specific endpoint and safe-pose template.
- `docs/`: operational contracts and provenance.

## Provenance

The MPPI runtime snapshot was copied from `laser_ablation` commit `0645dfc47f4669cd799230ba0e01b569ad230d90`; robot assets and simulation support were copied from `see_plan_cut` commit `37fab8136a299d03a417b3beee8a664d15395ce0`, plus the uncommitted integration modules developed in that checkout. This repository deliberately begins with an independent Git history.
