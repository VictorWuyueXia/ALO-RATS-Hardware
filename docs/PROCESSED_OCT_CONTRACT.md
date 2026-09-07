# Processed-OCT volume contract

The hardware workflow accepts one `.npz` written by `alo_rats_hardware.surface_scan.save_scan`. It is a processed interchange artifact, not raw scanner output.

## Required representation

`metadata.representation` must be `segmented_occupancy_volume`, with `metadata.units: mm`. The archive must contain:

- `x_axis_mm`, `y_axis_mm`, `z_axis_mm`: increasing voxel-centre axes, each uniformly spaced by exactly 0.1 mm.
- `tissue`: Boolean `(len(x), len(y), len(z))` segmented occupancy.
- `valid`: Boolean array with the same shape and every element true.
- `scan_pose_base_m`: finite proper 4×4 rigid transform of the acquisition pose.
- metadata including nonempty `scan_id`, `frame_id`, `provenance`, and finite `timestamp_s`.

The volume must be registered to the intended planning/treatment frame before it is supplied. The initial volume needs occupied tissue in every XY column selected for designation. Subsequent observations for an MPPI session must use exactly the same axes and frame; cavities and overhangs are retained as occupancy rather than flattened into a height field.

## Deliberately unsupported inputs

The workflow rejects raw `.jpg`/NIfTI scanner folders, point-cloud files, incomplete masks, interpolated axes, surface-only scans, and arbitrary voxel resolutions. The collaborator repository documents a separate Lumedica acquisition application and a missing `oct.module_oct_vol_scan` trigger module; no substitute driver exists in the checked-in code. A hardware-qualified scanner adapter must produce this explicit artifact rather than being guessed from file names or point-cloud conventions.

## Minimal generation check

For installation/UI testing only:

```bash
mkdir -p outputs
fixture_dir=$(mktemp -d "$PWD/outputs/nominal-fixture.XXXXXX")
python scripts/create_nominal_oct_fixture.py --output "$fixture_dir/nominal_processed_oct.npz"
```

The generated file is simulated and must never be represented as an OCT measurement.
