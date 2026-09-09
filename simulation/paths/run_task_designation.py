"""Approve scan-derived geometry without planning, robot motion, or hardware access."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.widgets import Button, RadioButtons, RectangleSelector, TextBox
import numpy as np

from simulation.oct.scan_adapter import SPACING_MM, designate_task, observe_task
from simulation.oct.scan_fixtures import nominal_designation
from simulation.oct.surface_scan import load_scan, save_scan
from simulation.oct.task_designation import TargetRegion, save_designation


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
        self.figure = plt.figure(figsize=(15, 9))
        self.ax = self.figure.add_axes([0.06, 0.27, 0.39, 0.66])
        self.volume_ax = self.figure.add_axes([0.50, 0.27, 0.46, 0.66], projection="3d")
        points = scan.registered_points()
        self.ax.scatter(points[:, 0], points[:, 1], c="#4c1d95", s=16, alpha=0.9)
        self.ax.set(xlabel="Planning X [mm]", ylabel="Planning Y [mm]")
        self.ax.set_title("Drag the active footprint; Add creates another")
        self.ax.xaxis.label.set_size(14)
        self.ax.yaxis.label.set_size(14)
        self.ax.title.set_size(16)
        self.ax.tick_params(labelsize=12)
        self.ax.set_aspect("equal")
        self.patches = []
        self.message = self.figure.text(0.06, 0.025, "", fontsize=12)
        self.shape = RadioButtons(self.figure.add_axes([0.06, 0.065, 0.14, 0.14]),
                                  ("rectangle", "ellipse"), active=int(self.regions[-1].shape == "ellipse"))
        for label in self.shape.labels:
            label.set_fontsize(12)
        region = self.regions[-1]
        right_depth = region.depth_mm if region.right_depth_mm is None else region.right_depth_mm
        self.depth = DesignationTextBox(self.figure.add_axes([0.25, 0.10, 0.11, 0.06]), "", initial=str(region.depth_mm))
        self.right = DesignationTextBox(self.figure.add_axes([0.40, 0.10, 0.11, 0.06]), "", initial=str(right_depth))
        self.protection = DesignationTextBox(self.figure.add_axes([0.55, 0.10, 0.11, 0.06]), "", initial=str(designation.protected_floor_mm))
        for field, label in ((self.depth, "Left depth [mm]"), (self.right, "Right depth [mm]"),
                              (self.protection, "Protected Z [mm]")):
            field.ax.set_title(label, fontsize=12, pad=8)
            field.text_disp.set_fontsize(13)
        self.add = Button(self.figure.add_axes([0.71, 0.10, 0.12, 0.06]), "Add region")
        self.approve = Button(self.figure.add_axes([0.85, 0.10, 0.12, 0.06]), "Approve / save")
        self.add.label.set_fontsize(12)
        self.approve.label.set_fontsize(12)
        self.selector = RectangleSelector(self.ax, self.select, useblit=True, button=[1],
                                           minspanx=SPACING_MM, minspany=SPACING_MM, spancoords="data")
        self.shape.on_clicked(self.edit_depth)
        for field in (self.depth, self.right, self.protection):
            field.on_text_change(self.edit_depth)
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
        """Rebuild the 3D task preview as each geometry field changes."""
        self.task = None
        edit_texts = (self.depth.text.strip(), self.right.text.strip(), self.protection.text.strip())
        if any(text in {"", "+", "-", ".", "+.", "-."} for text in edit_texts):
            self.message.set_text("INVALID TASK — finish entering all geometry values before approval")
            self.figure.canvas.draw_idle()
            return
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
            color = "#f59e0b" if index == len(self.regions) - 1 else "#22c55e"
            patch = (Rectangle(center - size, *size * 2, fill=False, color=color, linewidth=2.5)
                     if region.shape == "rectangle" else
                     Ellipse(center, *size * 2, fill=False, color=color, linewidth=2.5))
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
            self.volume_ax.scatter(*points[::4].T, s=8, c="#64748b", alpha=0.32, label="Scan surface")
            for mask, color, label in ((state.target_mask, "#f59e0b", "Target"),
                                       (state.constraint_mask, "#ef4444", "Protected")):
                indices = np.argwhere(mask)
                indices = indices[::max(1, len(indices) // 2000)]
                xyz = np.column_stack([axis[indices[:, i]] for i, axis in enumerate(
                    (state.x_axis_mm, state.y_axis_mm, state.z_axis_mm))])
                self.volume_ax.scatter(*xyz.T, s=12, c=color, alpha=0.72, label=label)
            self.volume_ax.legend(loc="upper right", fontsize=12, markerscale=1.5)
            self.message.set_text(f"Target {state.initial_target_volume_mm3:.3f} mm³; spacing {SPACING_MM:g} mm. "
                                  "Depths follow planning Z; unequal left/right depths create a stepped floor.")
        self.volume_ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]",
                           title="Registered scan + designated target/protection")
        self.volume_ax.set_xlabel("X [mm]", fontsize=14, labelpad=10)
        self.volume_ax.set_ylabel("Y [mm]", fontsize=14, labelpad=10)
        self.volume_ax.set_zlabel("Z [mm]", fontsize=14, labelpad=10)
        self.volume_ax.set_title("Registered scan + designated target/protection", fontsize=16, pad=16)
        self.volume_ax.tick_params(labelsize=11)
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
