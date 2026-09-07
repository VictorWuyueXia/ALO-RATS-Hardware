"""Explicitly deny physical device imports and Python socket connections in simulation."""

import importlib.abc
import sys


FORBIDDEN_MODULES = ("rtde", "urx", "ndyag", "oct", "UR5Controller",
                     "planned_cut_execute", "laser_control_pwm", "module_laser_utils")


def enforce_simulation_isolation():
    """Install fail-stop guards before constructing any simulation or controller component."""
    violations = []
    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(part.startswith(FORBIDDEN_MODULES) for part in fullname.split(".")):
                violations.append({"kind": "hardware_import", "module": fullname})
                raise RuntimeError(f"Physical module is forbidden in simulation: {fullname}")
    def audit(event, args):
        if event == "socket.connect":
            violations.append({"kind": "socket_connect"})
            raise RuntimeError("Socket connections are forbidden in simulation")
    if any(part.startswith(FORBIDDEN_MODULES) for name in sys.modules for part in name.split(".")):
        raise RuntimeError("A physical module was loaded before the simulation guard")
    sys.meta_path.insert(0, Guard())
    sys.addaudithook(audit)
    return violations
