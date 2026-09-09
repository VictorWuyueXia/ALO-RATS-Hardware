"""Validated URDF assets and explicitly simulated robot construction."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "assets/ur5e/urdf/ur5e_fixed.urdf"
HOME = np.deg2rad([92.25, -59.66, 73.01, -194.21, -65.34, 88.00])
JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
REQUIRED_LINKS = ("base", "tool0", "ee_link", "ee2_link")


def validate_assets():
    """Reject malformed transforms or missing meshes before opening a robot model."""
    tree = ET.parse(URDF).getroot()
    for element in tree.iter():
        for key, count in (("xyz", 3), ("rpy", 3), ("size", 3), ("scale", 3)):
            if key in element.attrib:
                values = np.array([float(value) for value in element.attrib[key].split()])
                if values.shape != (count,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid URDF {element.tag}/{key}")
    assets = [URDF] + [(URDF.parent / item.attrib["filename"]).resolve()
                       for item in tree.findall(".//mesh")]
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in assets}


@dataclass(frozen=True)
class RobotModel:
    client: int
    body: int
    joints: tuple[int, ...]
    links: dict[str, int]
    limits: np.ndarray
    asset_hashes: dict[str, str]


def load_robot(client):
    """Load only into the caller's connected PyBullet client; never choose hardware."""
    if not p.isConnected(client):
        raise ValueError("Robot construction requires a connected PyBullet client")
    assets = validate_assets()
    body = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=client)
    rows = [p.getJointInfo(body, i, physicsClientId=client)
            for i in range(p.getNumJoints(body, physicsClientId=client))]
    names = {row[1].decode(): row[0] for row in rows}
    links = {row[12].decode(): row[0] for row in rows}
    for name in REQUIRED_LINKS:
        if name not in links:
            raise ValueError(f"URDF is missing link {name}")
    joints = tuple(names[name] for name in JOINT_NAMES)
    movable = tuple(row[0] for row in rows if row[2] != p.JOINT_FIXED)
    if movable != joints:
        raise ValueError("URDF must expose exactly the six ordered arm joints")
    limits = np.array([[rows[joint][8], rows[joint][9]] for joint in joints])
    if np.any(HOME < limits[:, 0]) or np.any(HOME > limits[:, 1]):
        raise ValueError("Nominal home pose violates the URDF limits")
    for joint, value in zip(joints, HOME, strict=True):
        p.resetJointState(body, joint, value, physicsClientId=client)
    return RobotModel(client, body, joints, links, limits, assets)
