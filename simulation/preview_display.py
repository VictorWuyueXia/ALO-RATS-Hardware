"""Robot overview and explicitly magnified tissue inspection in the PyBullet viewer."""

import numpy as np
import pybullet as p

from preview_tissue import ENERGY_J, TARGETS_MM


class PreviewDisplay:
    """Render the synthetic scan, target markers, and achieved beam without physics bodies."""

    def __init__(self, robot, tissue):
        self.robot = robot
        self.tissue = tissue
        self.client = robot.client
        self.display_scale = 1.0
        self.surface_body = None
        self.surface_shape = None
        self.beam_id = -1
        self.status_id = -1
        self.selected_id = -1
        self.scan_ids = []
        self.task_ids = []
        self.selected_index = 0
        self.last_beam_pulsing = False
        self.robot_colors = [(row[1], row[7]) for row in p.getVisualShapeData(
            robot.body, physicsClientId=self.client,
        )]
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client)
        self.overview()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)

    def draw_surface(self, tissue):
        """Replace the visible mesh only after a newly simulated surface observation."""
        if self.surface_body is not None:
            p.removeBody(self.surface_body, physicsClientId=self.client)
        vertices = self.world(tissue.observe())
        cells = np.arange(61 * 61).reshape(61, 61)[:-1, :-1].ravel()
        triangles = np.column_stack((cells, cells + 1, cells + 61,
                                     cells + 1, cells + 62, cells + 61)).reshape(-1, 3)
        dz_dy, dz_dx = np.gradient(tissue.surface[:, 2].reshape(61, 61), 0.1)
        normals = np.column_stack((-dz_dx.ravel(), -dz_dy.ravel(), np.ones(61 * 61)))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals = normals @ self.robot.rotation.T
        self.surface_shape = p.createVisualShape(
            p.GEOM_MESH, vertices=vertices.tolist(), indices=triangles.ravel().tolist(),
            normals=normals.tolist(),
            rgbaColor=[0.82, 0.48, 0.40, 1], flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
            physicsClientId=self.client,
        )
        self.surface_body = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=self.surface_shape,
            physicsClientId=self.client,
        )
        # Wire profiles make submillimeter crater depth visible in the close-up.
        for identifier in self.scan_ids:
            p.removeUserDebugItem(identifier, physicsClientId=self.client)
        self.scan_ids = []
        grid = vertices.reshape(61, 61, 3)
        for row in grid[::6]:
            for a, b in zip(row[:-1:2], row[2::2]):
                self.scan_ids.append(p.addUserDebugLine(
                    a, b, [0.35, 0.15, 0.10], 1, physicsClientId=self.client,
                ))

    def draw_task(self, tissue):
        """Show the approved footprint and nominal protected floor in local millimeters."""
        for identifier in self.task_ids:
            p.removeUserDebugItem(identifier, physicsClientId=self.client)
        self.task_ids = []
        for depth, color in ((0.03, [0.1, 0.9, 0.2]),
                             (tissue.floor_mm, [0.1, 0.7, 1]),
                             (tissue.protected_floor_mm, [1, 0.1, 0.1])):
            corners = self.world([[-1.1, -1.1, depth], [1.1, -1.1, depth],
                                        [1.1, 1.1, depth], [-1.1, 1.1, depth]])
            for a, b in zip(corners, np.roll(corners, -1, axis=0)):
                self.task_ids.append(p.addUserDebugLine(a, b, color, 2,
                                                        physicsClientId=self.client))
        for number, xy in enumerate(TARGETS_MM, 1):
            point = self.world([*xy, 0.2])
            self.task_ids.append(p.addUserDebugText(str(number), point, [0, 1, 1],
                                                   textSize=1.4, physicsClientId=self.client))
        # A green local X axis and blue local Y axis define the target coordinates.
        for endpoint, color in (([3, 0, 0.05], [0, 1, 0]), ([0, 3, 0.05], [0, 0.5, 1])):
            self.task_ids.append(p.addUserDebugLine(self.world([0, 0, 0.05]),
                                                    self.world(endpoint), color, 2,
                                                    physicsClientId=self.client))

    def world(self, points_mm):
        """Magnification changes only display geometry, never robot or tissue state."""
        return self.robot.origin + self.display_scale * (
            self.robot.world(points_mm) - self.robot.origin
        )

    def select(self, index):
        self.selected_index = index
        xy = TARGETS_MM[index]
        self.selected_id = p.addUserDebugLine(
            self.world([*xy, 0]), self.world([*xy, 3]), [1, 1, 0], 3,
            replaceItemUniqueId=self.selected_id, physicsClientId=self.client,
        )

    def beam(self, pulsing=False):
        self.last_beam_pulsing = pulsing
        position, _ = self.robot.pose()
        position = self.robot.origin + self.display_scale * (position - self.robot.origin)
        action = self.robot.achieved_action(ENERGY_J)
        endpoint = self.world([*action[:2], 0])
        self.beam_id = p.addUserDebugLine(
            position, endpoint, [1, 0.1, 0.1] if pulsing else [1, 0.8, 0], 3,
            replaceItemUniqueId=self.beam_id, physicsClientId=self.client,
        )

    def status(self, text):
        print(text, flush=True)
        self.status_id = p.addUserDebugText(
            text, [0, -0.25, 1.0], [0.1, 0.1, 0.1], textSize=1.2,
            replaceItemUniqueId=self.status_id, physicsClientId=self.client,
        )

    def overview(self):
        self.display_scale = 1.0
        for link, color in self.robot_colors:
            p.changeVisualShape(self.robot.body, link, rgbaColor=color,
                                physicsClientId=self.client)
        p.resetDebugVisualizerCamera(1.1, 45, -25, [0.1, -0.35, 0.4],
                                     physicsClientId=self.client)
        self.refresh_geometry()

    def closeup(self):
        """Magnify the tissue thirtyfold to avoid the viewer's near-plane clipping."""
        self.display_scale = 30.0
        for link, color in self.robot_colors:
            p.changeVisualShape(self.robot.body, link, rgbaColor=[*color[:3], 0],
                                physicsClientId=self.client)
        direction = self.robot.rotation @ np.array([0.3, -0.8, 1.4])
        direction /= np.linalg.norm(direction)
        yaw = np.rad2deg(np.arctan2(direction[0], -direction[1]))
        pitch = -np.rad2deg(np.arcsin(direction[2]))
        p.resetDebugVisualizerCamera(0.16, yaw, pitch, self.robot.origin,
                                     physicsClientId=self.client)
        self.refresh_geometry()
        print("Tissue inspection: 30x DISPLAY ONLY; all reported distances remain mm.", flush=True)

    def refresh_geometry(self):
        """Redraw view-dependent visuals as a single batch without altering the plant."""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client)
        self.draw_surface(self.tissue)
        self.draw_task(self.tissue)
        self.select(self.selected_index)
        if self.beam_id >= 0:
            self.beam(self.last_beam_pulsing)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)
