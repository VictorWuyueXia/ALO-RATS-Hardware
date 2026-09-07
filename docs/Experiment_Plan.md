## Square-well reproduction — performance baseline
Repeat the 6x6x2 mm^3 phantom resection under same conditions.
Compare:
- Previous feedforward planner.
- Previous OCT-feedback MPC.
- ALO-RATS.
Especially compare on time cost.

## Subsurface protected-structure experiment — safety baseline
Repeat the embedded-structure experiment.
Measure:
- Protected voxels removed: required to be zero.
- Remaining target volume, overcut.
- Steps and total delivered energy.
- Completion time.

## Controlled model-mismatch and recovery
Use the square target, but introduce a physical mismatch between the nominal LTI model and actual ablation response. i.e. Different phantom formula

This answers: Can ALO-RATS recognize that actual removal differs from predicted removal and recover through OCT feedback and plan repair without causing collateral damage?

Report per-pulse trajectories of:
- Predicted versus observed removed volume.
- Remaining target and cumulative overcut.
- State discrepancy at each OCT update.
- Repair trigger and repair success.
- Total energy and controller wall time.
- Failure or safety-stop events.

## (Optional) Geometrically difficult 3-D boundary
If have time, use an oblique/tilted initial surface with a nonuniform-depth target and a nearby protected boundary. i.e. A stepped-depth or concave target.

Demonstrate that the geometry-aware global plan and exact 3-D voxel evaluation matter physically, and if the OCT visibility and laser line-of-sight permit tunnel or undercut geometry.