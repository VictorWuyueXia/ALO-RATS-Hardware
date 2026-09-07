"""Explicit discrete collision checks for the supplied URDF and nominal specimen fixture."""

from itertools import combinations

import numpy as np
import pybullet as p

from beam_adapter import pose_matrix


class CollisionScene:
    """Check nonadjacent rigid groups and every declared environmental collision body."""

    def __init__(self, model, calibration, bounds_mm):
        self.model, self.client = model, model.client
        rows = [p.getJointInfo(model.body, index, physicsClientId=self.client)
                for index in range(p.getNumJoints(model.body, physicsClientId=self.client))]
        groups = {-1: -1}
        for row in rows:
            groups[row[0]] = groups[row[16]] if row[2] == p.JOINT_FIXED else row[0]
        adjacent = {tuple(sorted((groups[row[0]], groups[row[16]])))
                    for row in rows if row[2] != p.JOINT_FIXED}
        self.links = tuple(row[0] for row in rows
                           if p.getCollisionShapeData(model.body, row[0], physicsClientId=self.client))
        if len(self.links) != 9:
            raise ValueError("Collision policy requires seven arm meshes and both optical-link boxes")
        self.pairs = tuple((a, b) for a, b in combinations(self.links, 2)
                           if groups[a] != groups[b] and tuple(sorted((groups[a], groups[b]))) not in adjacent)
        # A fixed tissue-volume box is a conservative obstacle; beam passage has no collision body.
        bounds = np.asarray(bounds_mm)
        centre = bounds.mean(axis=1)
        half = np.diff(bounds, axis=1).ravel() / 2000
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        transform = calibration.world_from_planning_m
        from scipy.spatial.transform import Rotation
        body = p.createMultiBody(0, shape, basePosition=calibration.world_points(centre),
                                 baseOrientation=Rotation.from_matrix(transform[:3, :3]).as_quat(),
                                 physicsClientId=self.client)
        self.obstacles = [body]

    def check(self):
        """Distances beyond the 0.1 m search range are recorded as that lower bound."""
        body = self.model.body
        self_points = p.getClosestPoints(body, body, distance=0.1, physicsClientId=self.client)
        pairs = set(self.pairs)
        points = [row for row in self_points if (row[3], row[4]) in pairs]
        for obstacle in self.obstacles:
            points.extend(p.getClosestPoints(
                body, obstacle, distance=0.1, physicsClientId=self.client))
        minimum = min([0.1, *(row[8] for row in points)])
        if minimum <= 0:
            worst = min(points, key=lambda row: row[8])
            raise ValueError(f"COLLISION_REJECTED: bodies {worst[1:3]}, links {worst[3:5]}, "
                             f"minimum signed distance {minimum:.6g} m")
        return minimum


def link_pose(model, name):
    frame = p.getLinkState(model.body, model.links[name], computeForwardKinematics=True,
                           physicsClientId=model.client)
    return pose_matrix(frame[4], frame[5])
