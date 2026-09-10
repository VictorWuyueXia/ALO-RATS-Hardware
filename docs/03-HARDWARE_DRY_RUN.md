# Hardware dry-run procedure

## Scope

This procedure verifies the path:

```text
processed OCT volume -> user designation -> MPPI method identity -> UR5e RTDE state -> evidence record
```

This dry-run command never imports the PWM client and never executes an MPPI action. `--move-safe-pose` is its sole motion option and calls UR RTDE `moveJ` with the laboratory-approved `safe_joint_pose_rad` from the local site file. The separate physical experiment coordinator remains disabled until the operator-runbook qualifications are complete.

## Preconditions

1. The UR5e is in the operator-approved state, its workspace is clear, and its controller accepts RTDE connections from the workstation.
2. `config/site.yaml` exists and was reviewed against the pendant/workspace. The example endpoint and pose are inherited from the collaborator script and must not be treated as verified for a new setup.
3. The OCT acquisition and segmentation workstation has produced the complete registered volume described in [04-PROCESSED_OCT_CONTRACT.md](04-PROCESSED_OCT_CONTRACT.md).
4. The volume is in the calibrated planning frame. This package validates the declared frame identifier and rigid scan pose but does not estimate OCT-to-robot calibration.

## Operator sequence

1. Run `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python scripts/check_robot_scene.py` and inspect `robot_scene.json` in the timestamped output directory printed by the command.
2. Run the Preparation 3 registration command in [`01-OPERATOR_RUNBOOK.md`](01-OPERATOR_RUNBOOK.md). It converts the validated `BSCAN-SGL-*.tif` folder acquired at the configured current-bench tool0 scan pose and writes the registered volume.
3. Run `scripts/run_hardware_dry_run.py` with that volume and `config/site.yaml`, without `--move-safe-pose`.
4. Use the mouse to drag one or more target footprints. Set depth below the registered initial surface and set the protected Z floor. Click **Approve / save** only after the displayed target and protected masks match the intended geometry.
5. Inspect the resulting `scan.npz`, `task.json`, `initial_voxels.npz`, `identity.json`, `designation.png`, and `hardware_dry_run.json`. The latter must say `laser_control_present: false` and `motion_commanded: false`.
6. If a safe robot-motion check is authorized, repeat in a new output directory with `--move-safe-pose`, observe the robot throughout its one approved joint move, and verify the before/after RTDE states in `hardware_dry_run.json`.

## Failure policy

Any invalid OCT lattice, geometry, RTDE connection, or robot reply raises an error. No missing measurement is synthesized and no alternate site configuration is used. A failed run does not infer whether a robot or scanner action occurred; inspect hardware state before retrying.

## Work outside this dry run

ALO-RATS now contains a mounted-folder adapter, beam action executor, PWM client, and experiment coordinator, but none is exercised by this command. Physical deployment still requires active-preset folder validation, focus/action calibration, laboratory obstacle exclusions, the independent pulse cutoff, energy calibration, and every success flag in [`01-OPERATOR_RUNBOOK.md`](01-OPERATOR_RUNBOOK.md).
