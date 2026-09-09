"""Analytic, registration, immutability, and operator-designation acceptance checks."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from laser_ablation.geometry.sdf import SDFObserver
from simulation.oct.scan_adapter import SPACING_MM, designate_task, export_surface, export_volume, observe_task
from simulation.oct.scan_fixtures import nominal_designation, nominal_scan, reexpress_scan
from simulation.oct.surface_scan import VolumetricScan, load_scan, save_scan
from simulation.oct.task_designation import TargetRegion, load_designation


@pytest.mark.parametrize("tilted", [False, True])
def test_analytic_scan_volume_and_sdf(tilted):
    scan = nominal_scan(tilted=tilted)
    task = designate_task(scan, nominal_designation())
    state = task.state
    expected_shape = tuple(round((upper - lower) / SPACING_MM)
                           for lower, upper in nominal_designation().grid_bounds_mm)
    assert state.grid_shape == expected_shape
    x, y = np.meshgrid(state.x_axis_mm, state.y_axis_mm, indexing="ij")
    expected = 0.06 * x - 0.04 * y if tilted else np.zeros_like(x)
    assert np.max(np.abs(scan.surface_on(state.x_axis_mm, state.y_axis_mm) - expected)) < 1e-8
    exported = export_surface(state)[:, 2].reshape(x.shape)
    assert np.max(np.abs(exported - expected)) <= SPACING_MM
    assert abs(state.initial_target_volume_mm3 - 2.56) <= 0.4
    observed = observe_task(scan, task, 0, None)
    sdf = SDFObserver().observe(observed.state)
    assert sdf.tensor.shape == (6, *state.grid_shape)
    assert not np.shares_memory(observed.state.tissue, state.tissue)
    assert not observed.state.target_mask.flags.writeable


def test_default_constraint_plane_tracks_the_initial_surface_one_mm_below_target():
    """The default 3 mm constraint plane leaves a 1 mm interval below the 2 mm target floor."""
    from simulation.simulation.simulation_cases import simulation_case

    case = simulation_case("centered_rectangle")
    state = designate_task(case.scan(), case.designation).state
    assert case.designation.grid_bounds_mm == ((-3.0, 3.0), (-3.0, 3.0), (-3.0, 0.0))
    assert case.designation.regions[0].half_size_xy_mm == (2.0, 2.0)
    assert case.designation.regions[0].depth_mm == 2.0
    assert case.designation.constraint_depth_mm == 3.0
    assert state.target_mask[:, :, :10].sum() == 0
    assert state.constraint_mask[:, :, 0].all()
    assert not state.constraint_mask[:, :, 1:].any()
    target_floor_mm = state.z_axis_mm[10] - SPACING_MM / 2
    constraint_surface_mm = state.z_axis_mm[0] - SPACING_MM / 2
    assert np.isclose(target_floor_mm - constraint_surface_mm, 1.0)


def test_exact_surface_export_reconstruction_and_immutable_masks():
    scan = nominal_scan()
    task = designate_task(scan, nominal_designation())
    state = task.state.copy()
    state.tissue[20:30, 20:30, 23:] = False
    points = export_surface(state)
    after_scan = replace(scan, points_mm=points, valid=np.ones(len(points), dtype=bool),
                         scan_id="after_pulse", timestamp_s=1)
    observed = observe_task(after_scan, task, 1, "pulse_0")
    assert np.array_equal(observed.state.tissue, state.tissue)
    for name in ("initial_tissue", "target_mask", "constraint_mask"):
        assert np.array_equal(getattr(observed.state, name), getattr(task.state, name))
    assert observed.task_id == task.task_id
    state.tissue[:] = False
    assert observed.state.tissue.any()


def test_registered_volumetric_observation_preserves_cavity_and_io(tmp_path):
    """A complete processed volume preserves arbitrary occupancy without a height-field assumption."""
    task = designate_task(nominal_scan(), nominal_designation())
    state = task.state.copy()
    state.tissue[20:30, 20:30, 5:8] = False
    scan = export_volume(state, "volume_after_pulse", 2.0, task.frame_id, np.eye(4))
    path = tmp_path / "volume.npz"
    save_scan(scan, path)
    reloaded = load_scan(path)
    observed = observe_task(reloaded, task, 1, "pulse_1")
    sdf = SDFObserver().observe(observed.state)
    assert isinstance(reloaded, VolumetricScan)
    assert np.array_equal(observed.state.tissue, state.tissue)
    assert sdf.tensor.shape == (6, *state.grid_shape)
    assert not np.shares_memory(observed.state.tissue, state.tissue)
    assert not reloaded.tissue.flags.writeable and not reloaded.valid.flags.writeable


def test_volumetric_initial_designation_matches_surface_task():
    surface_task = designate_task(nominal_scan(), nominal_designation())
    volume = export_volume(surface_task.state, "initial_volume", 0.0,
                           surface_task.frame_id, np.eye(4))
    volume_task = designate_task(volume, nominal_designation())
    for name in ("tissue", "initial_tissue", "target_mask", "constraint_mask"):
        assert np.array_equal(getattr(volume_task.state, name), getattr(surface_task.state, name))


@pytest.mark.parametrize("failure", ["axes", "coverage", "dtype", "outside_initial"])
def test_volumetric_observation_rejects_invalid_contract(failure):
    task = designate_task(nominal_scan(), nominal_designation())
    volume = export_volume(task.state, "volume", 1.0, task.frame_id, np.eye(4))
    if failure == "axes":
        changed = replace(volume, x_axis_mm=volume.x_axis_mm + 0.01)
        with pytest.raises(ValueError, match="axes"):
            observe_task(changed, task, 1, "pulse")
    elif failure == "coverage":
        valid = volume.valid.copy()
        valid[0, 0, 0] = False
        with pytest.raises(ValueError, match="coverage"):
            replace(volume, valid=valid)
    elif failure == "dtype":
        with pytest.raises(ValueError, match="Boolean"):
            replace(volume, tissue=volume.tissue.astype(np.uint8))
    else:
        tissue = volume.tissue.copy()
        tissue[~task.state.initial_tissue] = True
        changed = replace(volume, tissue=tissue)
        with pytest.raises(ValueError, match="outside initial"):
            observe_task(changed, task, 1, "pulse")


@pytest.mark.parametrize("tilted", [False, True])
def test_registration_round_trip_preserves_task(tilted):
    scan = nominal_scan(tilted=tilted)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.15, 0.4]).as_matrix()
    transform[:3, 3] = [20, -15, 4]
    pose = np.eye(4)
    pose[:3, 3] = [0.2, -0.7, 0.4]
    changed = reexpress_scan(scan, transform, pose)
    first = designate_task(scan, nominal_designation())
    second = designate_task(changed, nominal_designation())
    assert np.max(np.abs(scan.registered_points() - changed.registered_points())) < 1e-10
    assert first.task_id == second.task_id


def test_ellipse_step_and_disconnected_masks():
    scan = nominal_scan()
    base = nominal_designation()
    ellipse = designate_task(scan, nominal_designation("ellipse"))
    assert abs(ellipse.state.initial_target_volume_mm3 - np.pi * 1 * 0.8 * 0.8) < 0.3
    stepped = replace(base, regions=(replace(base.regions[0], right_depth_mm=1.2),))
    state = designate_task(scan, stepped).state
    left = state.target_mask[state.x_axis_mm < 0].sum()
    right = state.target_mask[state.x_axis_mm >= 0].sum()
    assert right == 1.5 * left
    regions = tuple(TargetRegion("ellipse", (x, 0), (0.35, 0.5), 0.5, None)
                    for x in (-1.0, 1.0))
    separated = designate_task(scan, replace(base, regions=regions)).state
    assert not separated.target_mask[np.abs(separated.x_axis_mm) < 0.5].any()
    assert separated.target_mask[separated.x_axis_mm < 0].any()
    assert separated.target_mask[separated.x_axis_mm > 0].any()


@pytest.mark.parametrize("failure", ["units", "coverage", "scale", "reflection", "cavity", "nan"])
def test_reject_unsupported_scan(failure):
    scan = nominal_scan()
    values = {}
    if failure == "units":
        values["units"] = "m"
    elif failure == "coverage":
        valid = scan.valid.copy()
        valid[0] = False
        values["valid"] = valid
    elif failure in {"scale", "reflection"}:
        transform = np.eye(4)
        transform[0, 0] = 2 if failure == "scale" else -1
        values["planning_from_scan_mm"] = transform
    elif failure == "cavity":
        values["representation"] = "volumetric_cavity"
    else:
        points = scan.points_mm.copy()
        points[0, 0] = np.nan
        values["points_mm"] = points
    with pytest.raises(ValueError):
        replace(scan, **values)


def test_reject_missing_coverage_target_overlap_and_hidden_residual():
    scan, designation = nominal_scan(), nominal_designation()
    short = replace(scan, points_mm=scan.points_mm[scan.points_mm[:, 0] < 0],
                    valid=np.ones((scan.points_mm[:, 0] < 0).sum(), dtype=bool))
    with pytest.raises(ValueError, match="coverage"):
        designate_task(short, designation)
    with pytest.raises(ValueError, match="disjoint"):
        designate_task(scan, replace(designation, constraint_depth_mm=0.5))
    state = designate_task(scan, designation).state.copy()
    state.tissue[10, 10, 5] = False
    with pytest.raises(ValueError, match="cavity"):
        export_surface(state)
    with pytest.raises(ValueError, match="lattice"):
        replace(designation, regions=(replace(designation.regions[0], center_xy_mm=(2.5, 0)),))


def test_scan_io_and_interactive_designation_parity(tmp_path):
    from simulation.paths.run_task_designation import TaskEditor

    scan = nominal_scan(tilted=True)
    path = tmp_path / "input.npz"
    save_scan(scan, path)
    reloaded = load_scan(path)
    assert np.array_equal(reloaded.points_mm, scan.points_mm)
    editor = TaskEditor(reloaded, tmp_path / "approved", nominal_designation())
    editor.select(SimpleNamespace(xdata=-0.8, ydata=-0.6),
                  SimpleNamespace(xdata=0.8, ydata=0.6))
    editor.shape.set_active(1)
    editor.right.set_val("1.1")
    expected = editor.task.task_id
    editor.save(None)
    replayed = designate_task(load_scan(tmp_path / "approved/scan.npz"),
                               load_designation(tmp_path / "approved/task.json"))
    assert replayed.task_id == expected
    assert editor.approved
    assert (tmp_path / "approved/designation.png").is_file()


def test_window_resize_commits_text_without_mouse_coordinates(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import ResizeEvent
    from simulation.paths.run_task_designation import TaskEditor

    editor = TaskEditor(nominal_scan(), tmp_path / "approved", nominal_designation())
    try:
        editor.depth.begin_typing()
        editor.depth.text_disp.set_text("0.6")
        event = ResizeEvent("resize_event", editor.figure.canvas)
        editor.figure.canvas.callbacks.process("resize_event", event)
        assert not editor.depth.capturekeystrokes
        assert editor.task is not None and editor.designation.regions[0].depth_mm == 0.6
    finally:
        plt.close(editor.figure)


def test_geometry_text_changes_redraw_3d_voxels(tmp_path):
    import matplotlib.pyplot as plt
    from simulation.paths.run_task_designation import TaskEditor

    editor = TaskEditor(nominal_scan(), tmp_path / "approved", nominal_designation())
    try:
        prior_protected_voxels = int(editor.task.state.constraint_mask.sum())
        editor.figure.canvas.draw = Mock(wraps=editor.figure.canvas.draw)
        editor.protection.text_disp.set_text("1.5")
        editor.protection._observers.process("change", "1.5")
        assert int(editor.task.state.constraint_mask.sum()) > prior_protected_voxels
        assert editor.figure.canvas.draw.called
    finally:
        plt.close(editor.figure)


@pytest.mark.parametrize("text", ["not-a-depth", "-1"])
def test_invalid_widget_edit_cannot_save_previous_task(tmp_path, text):
    import matplotlib.pyplot as plt
    from simulation.paths.run_task_designation import TaskEditor

    editor = TaskEditor(nominal_scan(), tmp_path / "invalid", nominal_designation())
    try:
        with pytest.raises(ValueError):
            editor.depth.set_val(text)
        assert editor.task is None
        with pytest.raises(ValueError):
            editor.save(None)
        assert not editor.output.exists()
    finally:
        plt.close(editor.figure)
