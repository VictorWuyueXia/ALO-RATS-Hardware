# ALO-RATS-Hardware

This is a clean, history-free runtime workspace for integrating the ALO-RATS MPPI planner-controller with the collaborator's UR5e, Lumedica OCT, Raspberry Pi PWM, and PyBullet support model. It provides a reproducible route from a registered OCT observation and user-designated target through unchanged MPPI planning, checked UR5e motion, one bounded pulse, and OCT reobservation/replanning.

It contains the **unchanged MPPI source and method configuration**, supplied UR5e URDF/meshes, processed-OCT interchange, mounted-folder reconstruction, mouse-driven designation, PyBullet simulation, RTDE execution, a disabled-by-default TCP PWM client, and the experiment 2/3 coordinator. It contains no scanner driver, Raspberry Pi listener service, calibration results, experiment results, or old Git history.

Read [the current situation and transfer handoff](docs/CURRENT_STATUS_AND_TRANSFER.md) before preparing a new computer. The active Lumedica acquisition runs in its Windows application and exposes scan folders through a mounted Ubuntu share. The OCT/laser implementation and remaining physical qualifications are in [the experiment plan](docs/OCT_LASER_EXPERIMENT_2_3_PLAN.md), with the exact [operator runbook](docs/0-OPERATOR_RUNBOOK.md).

## Install

```bash
cd /path/to/ALO-RATS-Hardware
/home/rp/anaconda3/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pybullet==3.2.7 pyvista==0.48.4
.venv/bin/python -m pip install -e ./mppi -e '.[robot,test]'
```

The remote Ubuntu server uses `.venv`; Conda and sudo are unavailable. `environment.yml` remains the cross-machine version record. The MPPI package declares JAX and CasADi itself. Use a desktop login session for the Matplotlib and PyBullet windows.

## First machine check

```bash
mkdir -p outputs
check_root=$(mktemp -d "$PWD/outputs/checks.XXXXXX")
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m simulation.paths.check_robot_scene --output-dir "$check_root/robot_scene"
.venv/bin/python scripts/create_nominal_oct_fixture.py --output "$check_root/nominal_processed_oct.npz"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m pytest -q --basetemp "$check_root/pytest"
```

The scene command produces `$check_root/robot_scene/robot_scene.json`; it loads the supplied URDF in headless PyBullet and does not import RTDE or connect to any device. The fixture is synthetic and exists only to test the OCT/designation UI. A unique check root preserves prior evidence and gives pytest a known writable temporary directory.

On Windows development workstations, use PowerShell syntax after activating `alo-rats-hardware`:

```powershell
python -m pip install -e .\mppi -e .
$check_root = Join-Path $PWD ("outputs\checks-" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
New-Item -ItemType Directory -Path $check_root | Out-Null
python -m simulation.paths.check_robot_scene --output-dir (Join-Path $check_root "robot_scene")
python scripts\create_nominal_oct_fixture.py --output (Join-Path $check_root "nominal_processed_oct.npz")
python -m pytest -q --basetemp (Join-Path $check_root "pytest") tests simulation/tests
```

The editable-install command is required even when the Conda environment exists.

On an Ubuntu NVIDIA workstation, install the CUDA-enabled JAX wheel and verify the devices exposed to the program:

```bash
nvidia-smi
.venv/bin/python -m pip install --upgrade "jax[cuda13]"
.venv/bin/python -c "import jax; print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

The backend must print `gpu` before starting an experiment. The simulation uses exactly the first CUDA device and the baseline `1gpu` MPPI profile; a CPU-only JAX installation uses the declared `cpu` test profile. Native Windows JAX does not support NVIDIA CUDA, so PowerShell runs use the CPU backend; use the Ubuntu robot workstation or WSL2 for baseline-profile execution. PyBullet may use the graphics GPU for OpenGL display, but its rigid-body and inverse-kinematics computations remain CPU-side.

The repository snapshot passed 103 no-device tests on the development machine. The checks validate the processed-OCT contract, mounted-folder conversion, PWM exchange, RTDE action checks, URDF asset closure, PyBullet interaction, controller/session interfaces, automatic compute-profile selection, and simulation isolation. They do not qualify the active scanner preset, robot installation, Raspberry Pi service, pulse cutoff, laser energy, or tissue outcome.

## Robot and processed-OCT dry run

Copy `config/site.example.yaml` to `config/site.yaml` and replace all values with the laboratory's verified robot endpoint and safe joint pose. The bundled values reproduce the previous robot script's endpoint and home pose; they are not a calibration certificate.

After the OCT workstation has produced a registered `segmented_occupancy_volume` `.npz` file, launch:

```bash
.venv/bin/python scripts/run_hardware_dry_run.py \
  --site config/site.yaml \
  --scan /path/to/processed_oct_volume.npz \
  --output-dir outputs/dry_run_001
```

The window shows the processed OCT upper envelope. Drag target footprints, set the left/right depths and protected Z, then click **Approve / save**. The program resolves the exact MPPI configuration, connects through RTDE, reads the robot joint/TCP state, writes `hardware_dry_run.json`, and disconnects. It sends no robot motion by default and never imports the PWM client.

Only after validating the output and physical workspace may an operator add `--move-safe-pose`. That flag invokes `moveJ` only to the reviewed `safe_joint_pose_rad` in `config/site.yaml`.

See [the hardware dry-run procedure](docs/HARDWARE_DRY_RUN.md) and [the processed-OCT contract](docs/PROCESSED_OCT_CONTRACT.md) before connecting hardware.

## Physical experiment coordinator

The physical entry point is implemented but rejects the example configuration. Complete every qualification flag in the operator runbook, replace every placeholder, and enable physical execution before invoking:

```bash
.venv/bin/python -m alo_rats_hardware.hardware_experiment \
  --site config/site.yaml \
  --experiment 2 \
  --metadata config/experiment_2.yaml \
  --output-dir outputs/experiment_2_001
```

Each cycle requires exact typed approval for motion and emission, one verified PWM receipt, return to the scan pose, and one new OCT folder before controller update. Unknown PWM outcome terminates the run without retry.

## Simulation

The existing integrated demonstration remains separate from the hardware path:

```bash
demo_root=$(mktemp -d /tmp/alo-rats-simulation.XXXXXX)
python -m simulation.paths.run_simulation \
  --case compact_diagnostic --output-dir "$demo_root/run"
```

It runs one process: designation → baseline MPPI method (10 anchors, 128 samples per anchor, and seed `20260902`) → checked URDF motion → virtual pulse → synthetic volume observation → replanning. The root `config/simulation_cases.yaml` file defines simulation geometry defaults and global-planner raster energy seeds. The comparable centered-rectangle and response-disturbance cases use the baseline 4/8-J raster seeds; geometrically distinct cases retain their exact-validated seed energies. `method.json` records the selected compute profile, JAX backend, and device list. It is a simulation-only application: `simulation/simulation/simulation_isolation.py` rejects RTDE, OCT, laser modules, and all non-local socket connections. The compact diagnostic is not an acceptance-quality treatment result; inspect its `acceptance.json`.

For a lightweight robot-interaction preview that does not run MPPI, use:

```bash
python -m simulation.paths.run_robot_preview
```

This preview remains a nominal PyBullet demonstration only. It does not connect to a robot, OCT scanner, or laser.

## Contents

- `mppi/`: MPPI runtime source and its exact configurations.
- `assets/ur5e/`: supplied `ur5e_fixed.urdf` and only its referenced meshes.
- `simulation/`: isolated PyBullet package; see [`simulation/README.md`](simulation/README.md) for robot, OCT, workflow, visualization, runnable-path, and test boundaries.
- `src/alo_rats_hardware/`: OCT-folder conversion, task UI, RTDE execution, PWM client, records, and coordinator.
- `config/`: strict site and experiment metadata templates.
- `docs/`: operational contracts and provenance.

## Provenance

The MPPI runtime snapshot was copied from `laser_ablation` commit `0645dfc47f4669cd799230ba0e01b569ad230d90`; robot assets and simulation support were copied from `see_plan_cut` commit `37fab8136a299d03a417b3beee8a664d15395ce0`, plus the uncommitted integration modules developed in that checkout. This repository deliberately begins with an independent Git history.
