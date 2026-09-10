# Experiment plan

## Configuration identity

All simulation validation cases and physical experiments use the tracked `mppi/configs/controller.yaml` method and its referenced physics, planner, geometry-aware global planner, and compute configuration. Experiment 2 uses the random seed and raster settings validated by `protected_boundary`; Experiment 3 uses those validated by `response_disturbance`. Hardware records add measured identities, calibration values, OCT observations, and operator-designated target geometry without substituting planner values. The implementation remains compact: no reduced diagnostic case, deployment-specific controller, fallback configuration, or thin configuration wrapper is permitted.

Hardware deployment requires successful configuration loading, action production, robot safety checks, OCT reconstruction, laser protocol checks, and fail-closed interaction behavior. Whether the MPPI method completes a simulated or physical target within its outcome limits is recorded as an experiment result and does not block deployment of the integration.

## Experiment 1 — square-well reproduction performance baseline

Repeat the 6 mm by 6 mm by 2 mm phantom resection under the same conditions.

Compare:

- Previous feedforward planner.
- Previous OCT-feedback MPC.
- ALO-RATS.

Compare planning time, robot-motion time, OCT acquisition/processing time, and total elapsed time.

## Experiment 2 — subsurface protected-structure safety baseline

Repeat the embedded-structure experiment.

Measure:

- Protected voxels removed: required to be zero.
- Remaining target volume, overcut.
- Pulse count, total calibrated commanded energy, and pulse-specific measured energy when a power meter is present.
- Completion time.

## Experiment 3 — controlled model-mismatch and recovery

Use the square target with a documented phantom formulation whose ablation response differs from the nominal linear time-invariant model.

This answers: Can ALO-RATS measure a difference between predicted and observed removal, incorporate OCT feedback, repair the active plan, and finish with zero removed protected voxels?

Report per-pulse trajectories of:

- Predicted versus observed removed volume.
- Remaining target and cumulative overcut.
- State discrepancy at each OCT update.
- Repair trigger and repair success.
- Total energy and controller wall time.
- Failure or safety-stop events.

## Experiment 4 (optional) — geometrically difficult 3-D boundary

If time remains, use an oblique initial surface with a nonuniform-depth target and a nearby protected boundary, such as a stepped-depth or concave target.

Demonstrate that the geometry-aware global plan and exact 3-D voxel evaluation matter physically, and if the OCT visibility and laser line-of-sight permit tunnel or undercut geometry.

The execution sequence and physical qualification requirements for experiments 2 and 3 are defined in [`01-OPERATOR_RUNBOOK.md`](01-OPERATOR_RUNBOOK.md) and [`02-OPERATOR_HANDOFF_REFERENCE.md`](02-OPERATOR_HANDOFF_REFERENCE.md).
