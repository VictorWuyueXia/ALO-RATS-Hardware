"""Nominal surface observation and virtual pulses using the repository's model."""

import importlib.util
from pathlib import Path

import numpy as np


# Load only the numerical model; never import either hardware execution script.
MODEL_PATH = Path(__file__).resolve().parents[1] / "simulation/cut_simulator.py"
SPEC = importlib.util.spec_from_file_location("preview_cut_model", MODEL_PATH)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
TARGETS_MM = np.array([[0, 0], [-0.65, -0.65], [0.65, -0.65],
                      [-0.65, 0.65], [0.65, 0.65]])
ENERGY_J = 3.5
STOP_REMAINING_PCT = 80.0


class PreviewTissue:
    """Keep a nominal scan surface and score a shallow, designated square target."""

    def __init__(self):
        x, y = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61))
        self.surface = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        self.target = np.all(np.abs(self.surface[:, :2]) <= 1.1, axis=1)
        self.floor_mm = -0.8
        self.protected_floor_mm = -2.0
        self.pulses = 0
        # Parameters come from planner/ablation_planner.py; the fifth action value is J.
        self.model = MODEL.Simulator(1, 1.939, 1 / 0.333885349, 0.483187452,
                                     11.071, 12.72661)

    @property
    def remaining_pct(self):
        residual = np.maximum(self.surface[self.target, 2] - self.floor_mm, 0)
        return float(100 * residual.sum() / (self.target.sum() * -self.floor_mm))

    def observe(self):
        """Return a new synthetic surface observation, not an acquired OCT scan."""
        return self.surface.copy()

    def pulse(self, action):
        """Apply one achieved-beam action, rejecting unsupported or unsafe previews."""
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Pulse requires five finite mm/rad/J values")
        if self.remaining_pct <= STOP_REMAINING_PCT:
            raise ValueError("The 80%-remaining proof-of-concept gate is already reached")
        if np.any(np.abs(action[:2]) > 1.2) or np.any(np.abs(action[2:4]) > 0.002):
            raise ValueError("Achieved beam is outside the nominal preview envelope")
        if action[4] != ENERGY_J:
            raise ValueError("Preview pulses use the fixed nominal energy only")
        candidate = self.model.simulate_SG(self.surface, action)
        if not np.isfinite(candidate).all() or candidate[:, 2].min() < self.protected_floor_mm:
            raise ValueError("Virtual pulse crosses the designated protected floor")
        if not np.any(candidate[:, 2] < self.surface[:, 2]):
            raise ValueError("Virtual pulse removes no tissue")
        self.surface = candidate
        self.pulses += 1
        return self.observe()
