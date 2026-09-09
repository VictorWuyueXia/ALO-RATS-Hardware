"""Validated laboratory configuration for robot, OCT, registration, and laser interfaces."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from .surface_scan import rigid_transform


@dataclass(frozen=True)
class SiteConfiguration:
    """Hold one explicit laboratory configuration with physical execution disabled by default."""

    physical_execution_enabled: bool
    robot: dict
    registration: dict
    oct: dict
    laser: dict

    def __post_init__(self):
        if not isinstance(self.physical_execution_enabled, bool):
            raise ValueError("physical_execution_enabled must be Boolean")
        robot, registration, oct_values, laser = map(
            dict, (self.robot, self.registration, self.oct, self.laser))
        expected = {
            "robot": {"ip", "safe_joint_pose_rad", "scan_joint_pose_rad", "move_speed_rad_s",
                      "move_acceleration_rad_s2", "maximum_joint_delta_rad",
                      "maximum_joint_error_rad"},
            "registration": {"calibration_id", "base_from_planning_m", "tcp_from_oct_m",
                             "tcp_from_laser_m", "laser_standoff_m",
                             "maximum_intercept_error_mm", "maximum_axis_error_rad"},
            "oct": {"shared_root", "file_pattern", "expected_b_scans", "image_shape_px",
                    "pixel_spacing_lateral_scan_depth_mm", "axis_order", "axis_signs",
                    "surface_margin_px", "surface_threshold_u8", "planning_volume_bounds_mm",
                    "planning_frame_id", "stable_observations", "poll_interval_s", "scan_timeout_s"},
            "laser": {"host", "port", "timeout_s", "frequency_hz", "pulse_duration_s",
                      "startup_delay_s", "energy_to_duty_cycle", "status_key", "stopped_value",
                      "watchdog_qualified", "calibration_id"},
        }
        for name, values in (("robot", robot), ("registration", registration),
                             ("oct", oct_values), ("laser", laser)):
            if set(values) != expected[name]:
                raise ValueError(f"Site {name} keys must be exactly {sorted(expected[name])}")

        for name in ("safe_joint_pose_rad", "scan_joint_pose_rad"):
            pose = np.asarray(robot[name], dtype=float)
            if pose.shape != (6,) or not np.isfinite(pose).all():
                raise ValueError(f"Robot {name} requires six finite joints")
            pose.setflags(write=False)
            robot[name] = pose
        if (not robot["ip"] or min(robot["move_speed_rad_s"], robot["move_acceleration_rad_s2"],
                                   robot["maximum_joint_delta_rad"],
                                   robot["maximum_joint_error_rad"]) <= 0):
            raise ValueError("Robot endpoint, motion values, and joint limits are required")

        for name in ("base_from_planning_m", "tcp_from_oct_m", "tcp_from_laser_m"):
            registration[name] = rigid_transform(registration[name])
        if (not registration["calibration_id"]
                or min(registration["laser_standoff_m"], registration["maximum_intercept_error_mm"],
                       registration["maximum_axis_error_rad"]) <= 0):
            raise ValueError("Registration identity and positive beam tolerances are required")

        image_shape = np.asarray(oct_values["image_shape_px"], dtype=int)
        spacing = np.asarray(oct_values["pixel_spacing_lateral_scan_depth_mm"], dtype=float)
        order = np.asarray(oct_values["axis_order"], dtype=int)
        signs = np.asarray(oct_values["axis_signs"], dtype=int)
        bounds = np.asarray(oct_values["planning_volume_bounds_mm"], dtype=float)
        if (not oct_values["shared_root"] or not oct_values["file_pattern"]
                or image_shape.shape != (2,) or np.any(image_shape <= 0)
                or spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0)
                or sorted(order.tolist()) != [0, 1, 2] or signs.shape != (3,)
                or not np.isin(signs, (-1, 1)).all()
                or bounds.shape != (3, 2) or np.any(~np.isfinite(bounds))
                or np.any(bounds[:, 1] <= bounds[:, 0])):
            raise ValueError("OCT geometry, axes, spacing, path, and planning bounds are invalid")
        if (oct_values["expected_b_scans"] < 2 or not oct_values["planning_frame_id"]
                or not 0 <= oct_values["surface_threshold_u8"] <= 255
                or oct_values["surface_margin_px"] < 1
                or 2 * oct_values["surface_margin_px"] >= image_shape[0]
                or oct_values["stable_observations"] < 2
                or min(oct_values["poll_interval_s"], oct_values["scan_timeout_s"]) <= 0
                or oct_values["scan_timeout_s"] <= (oct_values["stable_observations"] - 1)
                * oct_values["poll_interval_s"]):
            raise ValueError("OCT acquisition counts, threshold, margin, timing, and frame are invalid")
        for name, value in (("image_shape_px", image_shape),
                            ("pixel_spacing_lateral_scan_depth_mm", spacing),
                            ("axis_order", order), ("axis_signs", signs),
                            ("planning_volume_bounds_mm", bounds)):
            value.setflags(write=False)
            oct_values[name] = value

        table = np.asarray(laser["energy_to_duty_cycle"], dtype=float)
        if table.size == 0:
            table = np.empty((0, 2))
        if (not laser["host"] or not 0 < laser["port"] < 65536
                or min(laser["timeout_s"], laser["frequency_hz"], laser["pulse_duration_s"]) <= 0
                or laser["frequency_hz"] > 290 or laser["startup_delay_s"] < 0
                or not isinstance(laser["watchdog_qualified"], bool)
                or table.ndim != 2 or table.shape[1] != 2 or np.any(~np.isfinite(table))
                or (len(table) and (np.any(np.diff(table[:, 0]) <= 0)
                                    or np.any(table[:, 1] <= 0) or np.any(table[:, 1] > 99)))):
            raise ValueError("Laser endpoint, timing, status, watchdog, and calibration values are invalid")
        placeholder_values = (
            registration["calibration_id"], oct_values["shared_root"],
            oct_values["planning_frame_id"], laser["status_key"], laser["stopped_value"],
            laser["calibration_id"],
        )
        if self.physical_execution_enabled and (
                len(table) < 2 or not laser["watchdog_qualified"]
                or not laser["status_key"] or laser["stopped_value"] is None
                or laser["stopped_value"] == "" or not laser["calibration_id"]
                or any("REPLACE" in str(value) for value in placeholder_values)):
            raise ValueError(
                "Physical execution requires measured identifiers, energy calibration, and pulse watchdog")
        table.setflags(write=False)
        laser["energy_to_duty_cycle"] = table
        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "registration", registration)
        object.__setattr__(self, "oct", oct_values)
        object.__setattr__(self, "laser", laser)

    @property
    def identity(self):
        values = {"physical_execution_enabled": self.physical_execution_enabled}
        for section in ("robot", "registration", "oct", "laser"):
            values[section] = {name: value.tolist() if isinstance(value, np.ndarray) else value
                               for name, value in getattr(self, section).items()}
        return sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def load_site(path):
    """Load one explicit site file; omitted hardware calibration is never inferred."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"physical_execution_enabled", "robot", "registration", "oct", "laser"}
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError(f"Site configuration keys must be exactly {sorted(required)}")
    return SiteConfiguration(**values)
