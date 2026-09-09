"""Save headless robot, asset, and installed-controller identity evidence."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pybullet as p

from simulation.robot.robot_scene import HOME, JOINT_NAMES, ROOT, load_robot


def check_scene(output):
    """Verify real URDF forward kinematics and installed-package import isolation."""
    import laser_ablation

    package_path = Path(laser_ablation.__file__).resolve()
    expected_package = (ROOT / "mppi/src/laser_ablation/__init__.py").resolve()
    if not hasattr(laser_ablation, "__path__") or package_path != expected_package:
        raise RuntimeError("Robot scene requires the bundled editable MPPI package")
    output.mkdir(parents=True, exist_ok=False)
    client = p.connect(p.DIRECT)
    try:
        model = load_robot(client)
        poses = []
        for offset in (0.0, 0.1, -0.1):
            joints = HOME + np.array([offset, 0, -offset, 0, 0, 0])
            for joint, value in zip(model.joints, joints, strict=True):
                p.resetJointState(model.body, joint, value, physicsClientId=client)
            frames = {}
            for name, index in model.links.items():
                frame = p.getLinkState(model.body, index, computeForwardKinematics=True,
                                       physicsClientId=client)
                if not np.isfinite(np.array([*frame[4], *frame[5]])).all():
                    raise RuntimeError(f"Nonfinite forward kinematics for {name}")
                frames[name] = {"position_m": frame[4], "quaternion_xyzw": frame[5]}
            poses.append({"joints_rad": joints.tolist(), "frames": frames})
        result = {
            "urdf_loaded": True, "package_path": str(package_path),
            "python": sys.version, "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("numpy", "pybullet", "laser-ablation-icra2027")},
            "assets": model.asset_hashes, "joint_names": JOINT_NAMES,
            "links": model.links, "pose_checks": poses,
        }
        (output / "robot_scene.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        p.disconnect(client)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    check_scene(parser.parse_args().output_dir)
