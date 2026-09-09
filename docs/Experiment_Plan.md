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

The OCT, UR5e, and laser integration work required for experiments 2 and 3 is defined in [`OCT_LASER_EXPERIMENT_2_3_PLAN.md`](OCT_LASER_EXPERIMENT_2_3_PLAN.md).
