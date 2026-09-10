"""Validate one demonstration OCT volume folder and record its content identity."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from alo_rats_hardware.oct_folder import read_oct_folder, register_oct_folder
from alo_rats_hardware.robot import UR5eConnection
from alo_rats_hardware.site import load_site
from alo_rats_hardware.surface_scan import save_scan


def main():
    """Check one saved volume without robot motion or laser communication."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--folder", required=True,
                        help="One timestamp folder containing one complete OCT volume")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--register-current-pose", action="store_true",
                        help="Register this folder at the robot's current tool0 TCP pose")
    args = parser.parse_args()

    site = load_site(args.site)
    folder = Path(args.folder).resolve()
    print(f"Reading OCT volume folder {folder}", flush=True)
    files, surface_rows, hashes, scan_id = read_oct_folder(folder, site.oct)

    output = Path(args.output_dir).resolve()
    human = output / "human_readables"
    machine = output / "machine_readables"
    machine.mkdir(parents=True, exist_ok=False)
    human.mkdir()
    evidence = {
        "source_folder": str(folder),
        "scan_id": scan_id,
        "file_pattern": site.oct["file_pattern"],
        "b_scan_count": len(files),
        "image_shape_px": site.oct["image_shape_px"].tolist(),
        "pixel_spacing_lateral_scan_depth_mm":
            site.oct["pixel_spacing_lateral_scan_depth_mm"].tolist(),
        "surface_row_min": int(surface_rows.min()),
        "surface_row_max": int(surface_rows.max()),
    }
    if args.register_current_pose:
        robot = UR5eConnection(site)
        robot.connect()
        robot_state = robot.snapshot()
        tcp_offset = np.asarray(robot.control.getTCPOffset(), dtype=float)
        robot.close()
        if not np.allclose(tcp_offset, np.zeros(6), atol=1e-10, rtol=0):
            raise ValueError("OCT registration requires active TCP tool0 with zero offset")
        if np.max(np.abs(np.asarray(robot_state["actual_joints_rad"])
                         - site.robot["scan_joint_pose_rad"])) > site.robot["maximum_joint_error_rad"]:
            raise ValueError("OCT registration requires the configured reference scan joint pose")
        tcp = np.asarray(robot_state["actual_tcp_pose_m_rad"], dtype=float)
        scan_pose = np.eye(4)
        scan_pose[:3, :3] = Rotation.from_rotvec(tcp[3:]).as_matrix()
        scan_pose[:3, 3] = tcp[:3]
        scan, manifest = register_oct_folder(folder, site, scan_pose)
        save_scan(scan, machine / "registered_processed_oct_volume.npz")
        manifest["actual_joints_rad"] = robot_state["actual_joints_rad"]
        manifest["actual_tcp_pose_m_rad"] = robot_state["actual_tcp_pose_m_rad"]
        manifest["active_tcp_offset_m_rad"] = tcp_offset
        (machine / "registration_manifest.json").write_text(json.dumps({
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in manifest.items()
        }, indent=2) + "\n", encoding="utf-8")
        evidence["registered_processed_oct"] = "registered_processed_oct_volume.npz"
    (machine / "folder_check.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    (machine / "folder_manifest.sha256").write_text(
        "\n".join(f"{item['sha256']}  {item['name']}" for item in hashes) + "\n",
        encoding="utf-8",
    )
    (human / "interpretation_summary.md").write_text(
        "# OCT folder check\n\n"
        f"The folder passed the configured {len(files)}-slice image and surface checks. "
        f"Its automatic scan identity is `{scan_id}`."
        + (" The volume was registered at the measured tool0 scan pose.\n"
           if args.register_current_pose else "\n"),
        encoding="utf-8",
    )
    print(f"PREPARATION_2_OCT_FOLDER_PASS scan_id={scan_id}", flush=True)
    if args.register_current_pose:
        print("PREPARATION_3_OCT_REGISTRATION_PASS active_tcp=tool0", flush=True)


if __name__ == "__main__":
    main()
