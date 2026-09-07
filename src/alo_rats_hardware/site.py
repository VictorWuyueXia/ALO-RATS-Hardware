"""Validated laboratory-specific robot connection and safe-pose configuration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class SiteConfiguration:
    """Store only the physical endpoint and the pose approved for this laboratory."""

    robot_ip: str
    safe_joint_pose_rad: np.ndarray
    move_speed_rad_s: float
    move_acceleration_rad_s2: float

    def __post_init__(self):
        pose = np.asarray(self.safe_joint_pose_rad, dtype=float)
        if not self.robot_ip or pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("Site configuration requires a robot IP and six finite safe-pose joints")
        if self.move_speed_rad_s <= 0 or self.move_acceleration_rad_s2 <= 0:
            raise ValueError("Site motion speed and acceleration must be positive")
        pose.setflags(write=False)
        object.__setattr__(self, "safe_joint_pose_rad", pose)


def load_site(path):
    """Load one explicit site file; omitted hardware calibration is never inferred."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Site configuration must be a YAML mapping")
    required = {"robot_ip", "safe_joint_pose_rad", "move_speed_rad_s", "move_acceleration_rad_s2"}
    if set(values) != required:
        raise ValueError(f"Site configuration keys must be exactly {sorted(required)}")
    return SiteConfiguration(**values)
