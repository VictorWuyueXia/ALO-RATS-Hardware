"""Explicitly deny physical device imports and non-local Python socket connections in simulation."""

import importlib.abc
import sys


FORBIDDEN_MODULES = ("rtde", "urx", "ndyag", "oct", "UR5Controller",
                     "planned_cut_execute", "laser_control_pwm", "module_laser_utils")


def enforce_simulation_isolation():
    """Install fail-stop guards before constructing any simulation or controller component."""
    violations = []
    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + ".") or fullname.startswith(name + "_")
                   for name in FORBIDDEN_MODULES):
                violations.append({"kind": "hardware_import", "module": fullname})
                raise RuntimeError(f"Physical module is forbidden in simulation: {fullname}")
    def audit(event, args):
        if event == "socket.connect":
            # Qt uses a Windows loopback socket pair for its local GUI event loop.
            address = args[1]
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                return
            violations.append({"kind": "socket_connect"})
            raise RuntimeError("Non-local socket connections are forbidden in simulation")
    if any(name == forbidden or name.startswith(forbidden + ".") or name.startswith(forbidden + "_")
           for name in sys.modules for forbidden in FORBIDDEN_MODULES):
        raise RuntimeError("A physical module was loaded before the simulation guard")
    sys.meta_path.insert(0, Guard())
    sys.addaudithook(audit)
    return violations
