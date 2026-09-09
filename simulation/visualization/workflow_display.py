"""Live PyBullet operator view driven by reconstructed observations, not hidden tissue truth."""

from time import sleep
from pathlib import Path

import numpy as np
import pybullet as p

from simulation.robot.robot_executor import OperatorAbort
from simulation.oct.scan_adapter import export_surface_envelope


class WorkflowDisplay:
    """Approve one automatic simulated treatment; Esc aborts, C inspects tissue, V shows the robot."""

    def __init__(self, client, frame_directory):
        self.client = client
        self.gui = p.getConnectionInfo(client)["connectionMethod"] == p.GUI
        self.frame_directory = Path(frame_directory)
        if self.gui:
            self.frame_directory.mkdir()
        self.robot = self.state = None
        self.scale = 1.0
        self.surface_body = None
        self.overlay_ids = []
        self.status_id = self.requested_beam_id = self.achieved_beam_id = -1
        self.last_beam = None

    def bind(self, robot, task):
        self.robot, self.state = robot, task.state
        self.robot_colors = [(row[1], row[7]) for row in p.getVisualShapeData(
            robot.body, physicsClientId=self.client)]
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            self.overview()
            self.capture("initial")

    def poll(self):
        if not p.isConnected(self.client):
            raise OperatorAbort("Simulation window closed")
        if not self.gui:
            return {}
        keys = p.getKeyboardEvents(physicsClientId=self.client)
        if any(keys.get(key, 0) & p.KEY_WAS_TRIGGERED for key in (27, ord("q"))):
            raise OperatorAbort("Operator requested simulation abort")
        if self.robot is not None:
            if keys.get(ord("c"), 0) & p.KEY_WAS_TRIGGERED:
                self.closeup()
            if keys.get(ord("v"), 0) & p.KEY_WAS_TRIGGERED:
                self.overview()
        return keys

    def wait_for_start(self):
        self.status("SIMULATION ONLY: Enter starts automatic MPPI treatment; Esc aborts; C/V changes view")
        while self.gui:
            keys = self.poll()
            if any(keys.get(key, 0) & p.KEY_WAS_TRIGGERED for key in (p.B3G_RETURN, 10, 13)):
                return
            sleep(1 / 240)

    def wait_for_close(self):
        self.status("Treatment stopped; inspect with C/V; Esc closes the simulation")
        try:
            while True:
                self.poll()
                sleep(1 / 240)
        except OperatorAbort:
            return

    def status(self, message):
        print(message, flush=True)
        if self.gui:
            self.status_id = p.addUserDebugText(message, [0, -0.25, 0.85], [0.1, 0.1, 0.1],
                                                textSize=1.1, replaceItemUniqueId=self.status_id,
                                                physicsClientId=self.client)

    def points(self, xyz):
        origin = self.robot.calibration.world_from_planning_m[:3, 3]
        return origin + self.scale * (self.robot.calibration.world_points(xyz) - origin)

    def observation(self, observation, session):
        self.state = observation.state
        if self.gui:
            self.draw()
            self.capture(f"observed_{session.confirmed_pulses:03d}")
        from laser_ablation.metrics import evaluate_ablation
        self.status(f"Observed pulse {session.confirmed_pulses}: remaining "
                    f"{evaluate_ablation(self.state).remaining_pct:.3f}%; repairs {session.repairs}")

    def draw(self):
        """Magnification affects display-only meshes and labels, never collision or beam calculations."""
        if self.surface_body is not None:
            p.removeBody(self.surface_body, physicsClientId=self.client)
        surface = export_surface_envelope(self.state)
        vertices = self.points(surface)
        nx, ny, _ = self.state.grid_shape
        cells = np.arange(nx * ny).reshape(nx, ny)[:-1, :-1].ravel()
        # Counterclockwise faces and registered normals keep the observed top surface visible.
        triangles = np.column_stack((cells, cells + ny, cells + 1,
                                     cells + 1, cells + ny, cells + ny + 1)).ravel()
        dz_dx, dz_dy = np.gradient(surface[:, 2].reshape(nx, ny), self.state.spacing_mm)
        normals = np.column_stack((-dz_dx.ravel(), -dz_dy.ravel(), np.ones(nx * ny)))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals = normals @ self.robot.calibration.world_from_planning_m[:3, :3].T
        shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices.tolist(), indices=triangles.tolist(),
                                     normals=normals.tolist(),
                                     rgbaColor=[0.8, 0.5, 0.4, 0.8], flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                                     physicsClientId=self.client)
        self.surface_body = p.createMultiBody(0, baseVisualShapeIndex=shape, physicsClientId=self.client)
        for item in self.overlay_ids:
            p.removeUserDebugItem(item, physicsClientId=self.client)
        self.overlay_ids = []
        axes = (self.state.x_axis_mm, self.state.y_axis_mm, self.state.z_axis_mm)
        for mask, color in ((self.state.target_mask, [0.05, 0.15, 0.55]),
                            (self.state.constraint_mask, [0.55, 0.02, 0.04])):
            indices = np.argwhere(mask)
            indices = indices[::max(1, len(indices) // 1800)]
            xyz = np.column_stack([axis[indices[:, i]] for i, axis in enumerate(axes)])
            self.overlay_ids.append(p.addUserDebugPoints(self.points(xyz), [color] * len(xyz),
                                                         pointSize=4, physicsClientId=self.client))
        for endpoint, color in (([3, 0, 0], [0, 1, 0]), ([0, 3, 0], [0, 0.5, 1])):
            self.overlay_ids.append(p.addUserDebugLine(self.points([0, 0, 0]), self.points(endpoint),
                                                       color, 2, physicsClientId=self.client))
        if self.last_beam is not None:
            self.beam(*self.last_beam)

    def beam(self, requested, achieved, pulsing):
        self.last_beam = (requested, achieved, pulsing)
        if not self.gui:
            return
        from laser_ablation.physics.super_gaussian import laser_axis
        for name, action, color in (("requested_beam_id", requested, [0, 0.6, 1]),
                                     ("achieved_beam_id", achieved, [1, 0, 0] if pulsing else [1, 1, 0])):
            q = np.array([action.x_mm, action.y_mm, self.state.plane_z_mm])
            start = q - 4 * laser_axis(action.tilt_x_rad, action.tilt_y_rad)
            setattr(self, name, p.addUserDebugLine(self.points(start), self.points(q), color, 2,
                                                   replaceItemUniqueId=getattr(self, name),
                                                   physicsClientId=self.client))

    def overview(self):
        self.scale = 1.0
        for link, color in self.robot_colors:
            p.changeVisualShape(self.robot.body, link, rgbaColor=color, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(0.85, 45, -25, [0.05, -0.35, 0.4], physicsClientId=self.client)
        self.draw()

    def closeup(self):
        self.scale = 30.0
        for link, color in self.robot_colors:
            p.changeVisualShape(self.robot.body, link, rgbaColor=[*color[:3], 0], physicsClientId=self.client)
        transform = self.robot.calibration.world_from_planning_m
        direction = transform[:3, :3] @ np.array([0.3, -0.8, 1.4])
        direction /= np.linalg.norm(direction)
        p.resetDebugVisualizerCamera(0.18, np.rad2deg(np.arctan2(direction[0], -direction[1])),
                                     -np.rad2deg(np.arcsin(direction[2])), transform[:3, 3],
                                     physicsClientId=self.client)
        self.draw()
        self.status("Tissue view is 30x DISPLAY ONLY; all task dimensions remain mm")

    def capture(self, name):
        """Save the live OpenGL scene rather than replaying a previously computed robot path."""
        from PIL import Image
        # Let the native viewer publish its new camera matrix before reading it back.
        sleep(0.1)
        camera = p.getDebugVisualizerCamera(physicsClientId=self.client)
        pixels = p.getCameraImage(1000, 800, camera[2], camera[3],
                                  renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.client)[2]
        Image.fromarray(np.asarray(pixels).reshape(800, 1000, 4).astype("uint8")).save(
            self.frame_directory / f"{name}.png")
