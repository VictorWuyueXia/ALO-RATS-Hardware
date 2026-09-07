"""Explicit planning/base/world transforms for the collaborator's negative-Z laser ray."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.spatial.transform import Rotation

from laser_ablation.core.actions import PhysicalAction
from laser_ablation.physics.super_gaussian import laser_axis
from surface_scan import rigid_transform


# Nominal simulation distance keeps both supplied optical-link boxes outside the specimen.
STANDOFF_M = 0.040
INTERCEPT_TOLERANCE_MM = 0.025
AXIS_TOLERANCE_RAD = 0.001


def pose_matrix(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


@dataclass(frozen=True)
class BeamCalibration:
    world_from_base_m: np.ndarray
    base_from_planning_m: np.ndarray

    def __post_init__(self):
        for name in ("world_from_base_m", "base_from_planning_m"):
            object.__setattr__(self, name, rigid_transform(getattr(self, name)))

    @property
    def world_from_planning_m(self):
        return self.world_from_base_m @ self.base_from_planning_m

    @property
    def identity(self):
        return sha256(self.world_from_base_m.tobytes() + self.base_from_planning_m.tobytes()
                      + np.array([STANDOFF_M]).tobytes()).hexdigest()

    def world_points(self, points_mm):
        transform = self.world_from_planning_m
        return np.asarray(points_mm) @ transform[:3, :3].T / 1000 + transform[:3, 3]

    def requested_pose(self, action, plane_z_mm):
        """Retain the legacy radial-forward roll convention and an explicit nominal standoff."""
        reference = np.array([action.x_mm, action.y_mm, plane_z_mm])
        base_point = self.base_from_planning_m @ np.r_[reference / 1000, 1]
        outward = self.base_from_planning_m[:3, :3] @ (-laser_axis(action.tilt_x_rad, action.tilt_y_rad))
        forward = np.array([base_point[0], base_point[1], 0.0])
        forward -= (forward @ outward) * outward
        if np.linalg.norm(forward) < 1e-8:
            raise ValueError("Radial-forward tool roll is undefined at this beam pose")
        forward /= np.linalg.norm(forward)
        rotation = np.column_stack((forward, np.cross(outward, forward), outward))
        base_pose = np.eye(4)
        base_pose[:3, :3] = rotation
        base_pose[:3, 3] = base_point[:3] + STANDOFF_M * outward
        return self.world_from_base_m @ base_pose

    def achieved_action(self, world_from_laser_m, plane_z_mm, energy_j):
        """Recover the actual beam intercept; never replace it with the requested coordinates."""
        local_pose = np.linalg.inv(self.world_from_planning_m) @ rigid_transform(world_from_laser_m)
        point = local_pose[:3, 3] * 1000
        direction = -local_pose[:3, 2]
        if direction[2] >= -1e-8:
            raise ValueError("Achieved laser ray is reversed or parallel to the planning plane")
        distance = (plane_z_mm - point[2]) / direction[2]
        if distance <= 0:
            raise ValueError("Laser origin is not above the reference-plane intersection")
        intercept = point + distance * direction
        return PhysicalAction(float(intercept[0]), float(intercept[1]),
                              float(np.arctan2(direction[1], -direction[2])),
                              float(np.arcsin(np.clip(-direction[0], -1, 1))), energy_j)


def beam_errors(requested, achieved):
    xy = float(np.linalg.norm(requested.as_array()[:2] - achieved.as_array()[:2]))
    first = laser_axis(requested.tilt_x_rad, requested.tilt_y_rad)
    second = laser_axis(achieved.tilt_x_rad, achieved.tilt_y_rad)
    angle = float(np.arctan2(np.linalg.norm(np.cross(first, second)), first @ second))
    return xy, angle
