"""Run the UR5e asset and forward-kinematics check without physical devices."""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / f"outputs/robot_scene-{timestamp}"
    subprocess.run([sys.executable, "-m", "simulation.paths.check_robot_scene",
                    "--output-dir", str(output)], check=True, cwd=root)
    print(output)
