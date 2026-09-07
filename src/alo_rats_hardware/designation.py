"""Volume-derived editable target definition and mouse-driven operator approval."""

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.widgets import Button, RadioButtons, RectangleSelector, TextBox
import numpy as np

from .scan_adapter import designate_task, observe_task
from .surface_scan import save_scan
from .task_designation import TargetRegion, TaskDesignation, save_designation


def default_designation(scan):
    """Derive the immutable controller lattice directly from the registered OCT lattice."""
    axes = (scan.x_axis_mm, scan.y_axis_mm, scan.z_axis_mm)
    spacings = tuple(np.diff(axis) for axis in axes)
    if any(not np.allclose(spacing, spacing[0], atol=1e-10, rtol=0)
           or not np.isclose(spacing[0], 0.1) for spacing in spacings):
        raise ValueError("Hardware controller requires a uniform 0.1 mm processed-OCT lattice")
    bounds = tuple((float(axis[0] - 0.05), float(axis[-1] + 0.05)) for axis in axes)
    surface = scan.surface_on(scan.x_axis_mm, scan.y_axis_mm)
    center = tuple(np.mean(axis) for axis in axes[:2])
    size = tuple((axis[-1] - axis[0]) / 6 for axis in axes[:2])
    depth = min(0.8, float(surface.max() - bounds[2][0] - 0.1))
    return TaskDesignation((TargetRegion("rectangle", center, size, depth, None),), bounds,
                           bounds[2][0] + 0.5, bounds[2][1], scan.frame_id,
                           "processed_oct_registration_pending_hardware_calibration")


class DesignationTextBox(TextBox):
    """Commit text edits on resize without coupling window events to mouse selection."""

    def _resize(self, event):
        self.stop_typing()


class VolumeTaskEditor:
    """Allow human designation while preserving the exact OCT occupancy used by MPPI."""

    def __init__(self, scan, output, designation):
        self.scan, self.output, self.designation = scan, Path(output), designation
        self.regions, self.task, self.approved = list(designation.regions), None, False
        self.figure = plt.figure(figsize=(12, 7))
        self.ax = self.figure.add_axes([0.06, 0.29, 0.39, 0.63])
        self.volume_ax = self.figure.add_axes([0.50, 0.29, 0.46, 0.63], projection="3d")
        x, y = np.meshgrid(scan.x_axis_mm, scan.y_axis_mm, indexing="ij")
        self.ax.pcolormesh(x, y, scan.surface_on(scan.x_axis_mm, scan.y_axis_mm), shading="auto", cmap="viridis")
        self.ax.set(xlabel="Planning X [mm]", ylabel="Planning Y [mm]", title="Drag active target footprint")
        self.ax.set_aspect("equal")
        self.patches, self.message = [], self.figure.text(0.06, 0.025, "", fontsize=9)
        region = self.regions[-1]
        self.shape = RadioButtons(self.figure.add_axes([0.06, 0.075, 0.12, 0.12]),
                                  ("rectangle", "ellipse"), active=int(region.shape == "ellipse"))
        self.depth = DesignationTextBox(self.figure.add_axes([0.25, 0.12, 0.10, 0.05]), "", initial=str(region.depth_mm))
        self.right = DesignationTextBox(self.figure.add_axes([0.39, 0.12, 0.10, 0.05]), "", initial=str(region.depth_mm))
        self.protection = DesignationTextBox(self.figure.add_axes([0.54, 0.12, 0.10, 0.05]), "", initial=str(designation.protected_floor_mm))
        for field, label in ((self.depth, "Left depth [mm]"), (self.right, "Right depth [mm]"),
                             (self.protection, "Protected Z [mm]")):
            field.ax.set_title(label, fontsize=9, pad=6)
            field.on_submit(self.edit)
        self.add = Button(self.figure.add_axes([0.70, 0.12, 0.11, 0.05]), "Add region")
        self.approve = Button(self.figure.add_axes([0.83, 0.12, 0.11, 0.05]), "Approve / save")
        self.selector = RectangleSelector(self.ax, self.select, useblit=True, button=[1],
                                          minspanx=0.1, minspany=0.1, spancoords="data")
        self.shape.on_clicked(self.edit)
        self.add.on_clicked(self.add_region)
        self.approve.on_clicked(self.save)
        self.refresh()

    def select(self, start, end):
        """Convert one drag rectangle into the active geometric target region."""
        x0, x1 = sorted((start.xdata, end.xdata))
        y0, y1 = sorted((start.ydata, end.ydata))
        self.regions[-1] = TargetRegion(self.shape.value_selected, ((x0 + x1) / 2, (y0 + y1) / 2),
                                        ((x1 - x0) / 2, (y1 - y0) / 2), float(self.depth.text), float(self.right.text))
        self.refresh()

    def edit(self, event):
        """Rebuild the target after every numerical or shape edit."""
        self.regions[-1] = replace(self.regions[-1], shape=self.shape.value_selected,
                                   depth_mm=float(self.depth.text), right_depth_mm=float(self.right.text))
        self.refresh()

    def add_region(self, event):
        """Append a target component without changing earlier approved geometry."""
        self.regions.append(self.regions[-1])
        self.refresh()

    def refresh(self):
        """Reject invalid geometry before an operator can save it."""
        self.task = None
        self.volume_ax.clear()
        for patch in self.patches:
            patch.remove()
        self.patches = []
        for index, region in enumerate(self.regions):
            center, size = np.asarray(region.center_xy_mm), np.asarray(region.half_size_xy_mm)
            color = "orange" if index == len(self.regions) - 1 else "limegreen"
            patch = (Rectangle(center - size, *size * 2, fill=False, color=color)
                     if region.shape == "rectangle" else Ellipse(center, *size * 2, fill=False, color=color))
            self.ax.add_patch(patch)
            self.patches.append(patch)
        try:
            self.designation = replace(self.designation, regions=tuple(self.regions),
                                       protected_floor_mm=float(self.protection.text))
            self.task = designate_task(self.scan, self.designation)
        except ValueError as error:
            self.message.set_text(f"INVALID TASK — cannot approve: {error}")
        else:
            state = self.task.state
            for mask, color, label in ((state.target_mask, "orange", "Target"),
                                       (state.constraint_mask, "red", "Protected")):
                indices = np.argwhere(mask)[::max(1, int(mask.sum()) // 2000)]
                xyz = np.column_stack([axis[indices[:, i]] for i, axis in enumerate(
                    (state.x_axis_mm, state.y_axis_mm, state.z_axis_mm))])
                self.volume_ax.scatter(*xyz.T, s=2, c=color, alpha=0.3, label=label)
            self.volume_ax.legend(loc="upper right")
            self.message.set_text(f"Target {state.initial_target_volume_mm3:.3f} mm³; input is exact 0.1 mm occupancy.")
        self.volume_ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]", title="Processed OCT target and protected volume")
        # Paint the approved geometry before Qt enters its blocking event loop.
        self.figure.canvas.draw()

    def save(self, event):
        """Persist the approved OCT, task, and immutable voxel masks for the controller session."""
        self.edit(None)
        if self.task is None:
            raise ValueError("Invalid designation cannot be approved")
        self.output.mkdir(parents=True, exist_ok=False)
        save_scan(self.scan, self.output / "scan.npz")
        save_designation(self.designation, self.output / "task.json")
        observation = observe_task(self.scan, self.task, 0, None)
        np.savez_compressed(self.output / "initial_voxels.npz", tissue=observation.state.tissue,
                            target=self.task.state.target_mask, protected=self.task.state.constraint_mask)
        identity = {"task_id": self.task.task_id, "scan_id": self.scan.scan_id,
                    "source": self.scan.provenance, "units": "mm"}
        (self.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        self.figure.savefig(self.output / "designation.png", dpi=150)
        self.approved = True
        plt.close(self.figure)


def approve_volume_task(scan, output):
    """Open the sole human-input UI and return only after a valid task is saved."""
    editor = VolumeTaskEditor(scan, output, default_designation(scan))
    plt.show()
    if not editor.approved:
        raise RuntimeError("Task designation was cancelled; no controller input was created")
    return editor.task
