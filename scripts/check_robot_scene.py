"""Run the UR5e asset and forward-kinematics check without physical devices."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/robot_scene"
    subprocess.run([sys.executable, str(root / "simulation/check_robot_scene.py"),
                    "--output-dir", str(output)], check=True)
