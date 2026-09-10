# Simulation package map

`simulation/` is an isolated package. It contains no RTDE, scanner, Raspberry Pi, or non-local socket path.

| Scope | Contents | Boundary |
| --- | --- | --- |
| `robot/` | URDF loading, collision checks, PyBullet motion, and robot preview geometry | Uses only the supplied URDF and the caller-owned PyBullet client. |
| `oct/` | Synthetic OCT surface/volume interchange, fixtures, and task designation | Converts local synthetic scan data to the controller lattice. |
| `simulation/` | Cases, virtual plant, workflow, isolation guard, acceptance validation, and preview cut model | Owns virtual state transitions and never chooses a physical backend. |
| `visualization/` | PyBullet and Matplotlib display objects | Displays simulation state and does not change physical devices. |
| `paths/` | Executable simulation, robot-check, preview, designation, and acceptance commands | Calls the scoped modules through `python -m`. |
| `tests/` | Regression and negative-path tests | Imports only package-qualified simulation modules. |

Run the primary paths from the repository root:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python -m simulation.paths.check_robot_scene --output-dir /absolute/new/robot_scene
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python -m simulation.paths.check_simulation --output-dir /absolute/new/suite
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python -m simulation.paths.run_simulation --case centered_rectangle --output-dir /absolute/new/run
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python -m simulation.paths.run_robot_preview
```

Run the simulation tests with:

```bash
python -m pytest -q simulation/tests
```
