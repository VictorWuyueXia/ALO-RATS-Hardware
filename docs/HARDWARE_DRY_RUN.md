# Hardware dry-run procedure

## Scope

This procedure verifies the path:

```text
processed OCT volume -> user designation -> MPPI method identity -> UR5e RTDE state -> evidence record
```

There is no laser command, PWM import, or plan-action execution in this repository. `--move-safe-pose` is the sole motion command and calls UR RTDE `moveJ` with the laboratory-approved `safe_joint_pose_rad` from the local site file.

## Preconditions

1. The UR5e is in the operator-approved state, its workspace is clear, and its controller accepts RTDE connections from the workstation.
2. `config/site.yaml` exists and was reviewed against the pendant/workspace. The example endpoint and pose are inherited from the collaborator script and must not be treated as verified for a new setup.
3. The OCT acquisition and segmentation workstation has produced the complete registered volume described in [PROCESSED_OCT_CONTRACT.md](PROCESSED_OCT_CONTRACT.md).
4. The volume is in the calibrated planning frame. This package validates the declared frame identifier and rigid scan pose but does not estimate OCT-to-robot calibration.

## Operator sequence

1. Run `python scripts/check_robot_scene.py` and inspect `robot_scene.json` in the timestamped output directory printed by the command.
2. Copy the processed OCT volume onto the robot workstation. Do not substitute a raw image folder, `.pcd`, or unregistered point cloud.
3. Run `scripts/run_hardware_dry_run.py` without `--move-safe-pose`.
4. Use the mouse to drag one or more target footprints. Set depth below the registered initial surface and set the protected Z floor. Click **Approve / save** only after the displayed target and protected masks match the intended geometry.
5. Inspect the resulting `scan.npz`, `task.json`, `initial_voxels.npz`, `identity.json`, `designation.png`, and `hardware_dry_run.json`. The latter must say `laser_control_present: false` and `motion_commanded: false`.
6. If a safe robot-motion check is authorized, repeat in a new output directory with `--move-safe-pose`, observe the robot throughout its one approved joint move, and verify the before/after RTDE states in `hardware_dry_run.json`.

## Failure policy

Any invalid OCT lattice, geometry, RTDE connection, or robot reply raises an error. No missing measurement is synthesized and no alternate site configuration is used. A failed run does not infer whether a robot or scanner action occurred; inspect hardware state before retrying.

## Deferred hardware work

Physical deployment still requires measured OCT-to-robot registration, a scanner acquisition/segmentation adapter, beam focus and action-to-pose calibration, collision/keep-out validation, a laser safety interlock, and a separately reviewed MPPI-action executor. Those components are intentionally absent from this dry run.
