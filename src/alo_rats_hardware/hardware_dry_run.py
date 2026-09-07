"""Processed OCT designation, MPPI-method identity check, and robot-motion-only hardware test."""

import argparse
import json
from pathlib import Path
from time import time

from laser_ablation.control.method import load_method

from .designation import approve_volume_task
from .processed_oct import load_processed_volume, volume_summary
from .robot import UR5eConnection
from .site import load_site


ROOT = Path(__file__).resolve().parents[2]
MPPI_CONTROLLER = ROOT / "mppi/configs/controller.yaml"


def run(site_path, scan_path, output, move_safe_pose):
    """Run the no-laser hardware procedure and save evidence before closing robot transport."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    site, scan = load_site(site_path), load_processed_volume(scan_path)
    task = approve_volume_task(scan, output)
    method = load_method(MPPI_CONTROLLER)
    robot = UR5eConnection(site)
    try:
        robot.connect()
        before = robot.snapshot()
        if move_safe_pose:
            robot.move_to_safe_pose()
        after = robot.snapshot()
    finally:
        robot.close()
    (output / "hardware_dry_run.json").write_text(json.dumps({
        "timestamp_s": time(), "laser_control_present": False, "motion_commanded": bool(move_safe_pose),
        "robot_before": before, "robot_after": after, "processed_oct": volume_summary(scan),
        "task_id": task.task_id, "mppi_method_source_hashes": method.source_hashes,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    """Expose the three required inputs without a simulation/hardware selector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--move-safe-pose", action="store_true")
    args = parser.parse_args()
    run(args.site, args.scan, args.output_dir, args.move_safe_pose)


if __name__ == "__main__":
    main()
