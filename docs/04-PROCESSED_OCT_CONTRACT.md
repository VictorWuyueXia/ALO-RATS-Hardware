# Processed-OCT volume contract

The dry-run workflow accepts one `.npz` written by `alo_rats_hardware.surface_scan.save_scan`. The physical experiment coordinator creates the same artifact from a completed mounted Lumedica B-scan folder before designation or controller update.

## Required representation

`metadata.representation` must be `segmented_occupancy_volume`, with `metadata.units: mm`. The archive must contain:

- `x_axis_mm`, `y_axis_mm`, `z_axis_mm`: increasing voxel-centre axes, each uniformly spaced by exactly 0.1 mm.
- `tissue`: Boolean `(len(x), len(y), len(z))` segmented occupancy.
- `valid`: Boolean array with the same shape and every element true.
- `scan_pose_base_m`: finite proper 4×4 rigid transform of the acquisition pose.
- metadata including nonempty `scan_id`, `frame_id`, `provenance`, and finite `timestamp_s`.

The volume must be registered to the intended planning/treatment frame before it is supplied. The initial volume needs occupied tissue in every XY column selected for designation. Subsequent observations for an MPPI session must use exactly the same axes and frame. Externally prepared occupancy may retain cavities and overhangs. The implemented mounted-folder adapter creates a complete height field below one segmented surface, so that adapter does not represent cavities or overhangs.

## Input boundaries

The direct `.npz` loader rejects image folders, point-cloud files, incomplete validity masks, interpolated axes, surface-only scans, and arbitrary voxel resolutions. The physical coordinator accepts only the configured image pattern inside a new completed folder under the mounted share. Its adapter validates count, numeric order, shape, stable file signatures, intensity threshold, transform chain, and planning coverage before emitting this contract. Point clouds and NIfTI input remain unsupported.

The active acquisition remains the Lumedica application on the Windows computer. Historical serial-trigger code in `hybrid_arm_mirror` is reference material and is not called by ALO-RATS.

## Minimal generation check

For installation/UI testing only:

```bash
mkdir -p outputs
fixture_dir=$(mktemp -d "$PWD/outputs/nominal-fixture.XXXXXX")
python scripts/create_nominal_oct_fixture.py --output "$fixture_dir/nominal_processed_oct.npz"
```

The generated file is simulated and must never be represented as an OCT measurement.
