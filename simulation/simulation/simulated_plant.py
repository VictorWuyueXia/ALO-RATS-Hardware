"""Private exact-voxel plant with an ideal processed volumetric-OCT boundary."""

from dataclasses import asdict
from time import time

from laser_ablation.control.interaction import snapshot
from laser_ablation.metrics import evaluate_ablation
from laser_ablation.physics.exact_voxel import ExactVoxelSimulator
from alo_rats_hardware.records import save_voxels
from simulation.oct.scan_adapter import export_volume


class SimulatedPlant:
    """Apply achieved beam geometry once; expose only processed scans to the controller workflow."""

    def __init__(self, initial_state, physics, bounds, calibration, frame_id, disturbed_pulse):
        self._state = snapshot(initial_state)
        self._model = ExactVoxelSimulator(physics, bounds)
        self.calibration, self.frame_id = calibration, frame_id
        self.disturbed_pulse = disturbed_pulse
        self.executions = {}

    def fire(self, command_id, achieved_action):
        """Perturb response only on the designated pulse; commanded energy and physics stay unchanged."""
        if command_id in self.executions:
            raise RuntimeError("Virtual pulse was already executed; re-firing is prohibited")
        pulse = len(self.executions) + 1
        scale = 0.8 if pulse == self.disturbed_pulse else 1.0
        after = self._model.step(self._state, achieved_action, response_scale=scale)
        self._state = after
        self.executions[command_id] = {"pulse": pulse, "response_scale": scale,
                                       "achieved_action": achieved_action, "completed_at_s": time()}
        return self.executions[command_id]

    def observe(self, scan_pose_base_m):
        """Return complete registered segmentation while retaining the actual acquisition pose."""
        return export_volume(self._state, f"scan_{len(self.executions):03d}", time(),
                             self.frame_id, scan_pose_base_m)

    def evaluation_metrics(self):
        """Evaluation consumes these truth metrics; they never influence controller decisions."""
        return asdict(evaluate_ablation(self._state))

    def save_evaluation_state(self, path):
        save_voxels(path, self._state)
