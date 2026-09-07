"""Approve scan-derived geometry without planning, robot motion, or hardware access."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.widgets import Button, RadioButtons, RectangleSelector, TextBox
import numpy as np

from scan_adapter import designate_task, observe_task
from scan_fixtures import nominal_designation
from surface_scan import load_scan, save_scan
from task_designation import TargetRegion, save_designation


class DesignationTextBox(TextBox):
    """Commit edits on resize without interpreting a window event as a mouse event."""

    def _resize(self, event):
        self.stop_typing()


class TaskEditor:
    """Drag footprints, edit vertical depths, and approve the same masks used by tests."""

    def __init__(self, scan, output, designation):
        self.scan, self.output = scan, Path(output)
        self.designation = designation
        self.regions = list(self.designation.regions)
        self.task = None
        self.approved = False
        self.figure = plt.figure(figsize=(12, 7))
        self.ax = self.figure.add_axes([0.06, 0.29, 0.39, 0.63])
        self.volume_ax = self.figure.add_axes([0.50, 0.29, 0.46, 0.63], projection="3d")
        points = scan.registered_points()
        self.ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=5, cmap="viridis")
        self.ax.set(xlabel="Planning X [mm]", ylabel="Planning Y [mm]",
                    title="Drag the active footprint; Add creates another")
        self.ax.set_aspect("equal")
        self.patches = []
        self.message = self.figure.text(0.06, 0.025, "", fontsize=9)
        self.shape = RadioButtons(self.figure.add_axes([0.06, 0.075, 0.12, 0.12]),
                                  ("rectangle", "ellipse"), active=int(self.regions[-1].shape == "ellipse"))
        region = self.regions[-1]
        right_depth = region.depth_mm if region.right_depth_mm is None else region.right_depth_mm
        self.depth = DesignationTextBox(self.figure.add_axes([0.25, 0.12, 0.10, 0.05]), "", initial=str(region.depth_mm))
        self.right = DesignationTextBox(self.figure.add_axes([0.39, 0.12, 0.10, 0.05]), "", initial=str(right_depth))
        self.protection = DesignationTextBox(self.figure.add_axes([0.54, 0.12, 0.10, 0.05]), "", initial=str(designation.protected_floor_mm))
        for field, label in ((self.depth, "Left depth [mm]"), (self.right, "Right depth [mm]"),
                              (self.protection, "Protected Z [mm]")):
            field.ax.set_title(label, fontsize=9, pad=6)
        self.add = Button(self.figure.add_axes([0.70, 0.12, 0.11, 0.05]), "Add region")
        self.approve = Button(self.figure.add_axes([0.83, 0.12, 0.11, 0.05]), "Approve / save")
        self.selector = RectangleSelector(self.ax, self.select, useblit=True, button=[1],
                                           minspanx=0.1, minspany=0.1, spancoords="data")
        self.shape.on_clicked(self.edit_depth)
        for field in (self.depth, self.right, self.protection):
            field.on_submit(self.edit_depth)
        self.add.on_clicked(self.add_region)
        self.approve.on_clicked(self.save)
        self.refresh()

    def select(self, start, end):
        self.task = None
        x0, x1 = sorted((start.xdata, end.xdata))
        y0, y1 = sorted((start.ydata, end.ydata))
        self.regions[-1] = TargetRegion(
            self.shape.value_selected, ((x0 + x1) / 2, (y0 + y1) / 2),
            ((x1 - x0) / 2, (y1 - y0) / 2), float(self.depth.text), float(self.right.text),
        )
        self.refresh()

    def edit_depth(self, event):
        # Invalidate approval before parsing, including edits rejected by a widget callback.
        self.task = None
        self.regions[-1] = replace(
            self.regions[-1], shape=self.shape.value_selected,
            depth_mm=float(self.depth.text), right_depth_mm=float(self.right.text),
        )
        self.refresh()

    def add_region(self, event):
        self.regions.append(self.regions[-1])
        self.refresh()
        self.message.set_text("Drag to place the new active region; approval preserves the union, not its bounding box.")

    def refresh(self):
        """A rejected edit invalidates approval; no previous valid task is substituted."""
        self.task = None
        self.volume_ax.clear()
        for patch in self.patches:
            patch.remove()
        self.patches = []
        for index, region in enumerate(self.regions):
            center, size = np.array(region.center_xy_mm), np.array(region.half_size_xy_mm)
            color = "orange" if index == len(self.regions) - 1 else "limegreen"
            patch = (Rectangle(center - size, *size * 2, fill=False, color=color)
                     if region.shape == "rectangle" else
                     Ellipse(center, *size * 2, fill=False, color=color))
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
            points = self.scan.registered_points()
            self.volume_ax.scatter(*points[::4].T, s=1, c="gray", alpha=0.2)
            for mask, color, label in ((state.target_mask, "orange", "Target"),
                                       (state.constraint_mask, "red", "Protected")):
                indices = np.argwhere(mask)
                indices = indices[::max(1, len(indices) // 2000)]
                xyz = np.column_stack([axis[indices[:, i]] for i, axis in enumerate(
                    (state.x_axis_mm, state.y_axis_mm, state.z_axis_mm))])
                self.volume_ax.scatter(*xyz.T, s=2, c=color, alpha=0.3, label=label)
            self.volume_ax.legend(loc="upper right")
            self.message.set_text(f"Target {state.initial_target_volume_mm3:.3f} mm³; spacing 0.1 mm. "
                                  "Depths follow planning Z; unequal left/right depths create a stepped floor.")
        self.volume_ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]",
                           title="Registered scan + designated target/protection")
        # Paint the approved geometry before Qt enters its blocking event loop.
        self.figure.canvas.draw()

    def save(self, event):
        """Persist the exact approved input and voxel/SDF-ready observation identity."""
        self.edit_depth(None)
        if self.task is None:
            raise ValueError("Invalid designation cannot be approved")
        self.output.mkdir(parents=True, exist_ok=False)
        save_scan(self.scan, self.output / "scan.npz")
        save_designation(self.designation, self.output / "task.json")
        observation = observe_task(self.scan, self.task, 0, None)
        np.savez_compressed(self.output / "initial_voxels.npz", tissue=observation.state.tissue,
                            target=self.task.state.target_mask, protected=self.task.state.constraint_mask)
        (self.output / "identity.json").write_text(json.dumps({
            "task_id": self.task.task_id, "scan_id": self.scan.scan_id,
            "source": self.scan.provenance, "units": "mm",
        }, indent=2) + "\n")
        self.figure.savefig(self.output / "designation.png", dpi=150)
        self.approved = True
        plt.close(self.figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    scan = load_scan(args.scan)
    editor = TaskEditor(scan, args.output_dir, replace(nominal_designation(), frame_id=scan.frame_id))
    plt.show()
    if not editor.approved:
        raise SystemExit("Task designation cancelled; no task was approved.")
