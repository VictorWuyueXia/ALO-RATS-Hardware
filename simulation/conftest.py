"""Match the dedicated launcher import path when pytest starts from the robot root."""

from pathlib import Path
import sys


# The root contains an executable laser_ablation.py, not the installed MPPI package.
ROBOT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != ROBOT_ROOT]
