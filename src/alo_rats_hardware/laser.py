"""Fail-closed TCP client for the JSON PWM protocol deployed with see_plan_cut."""

import json
import socket
from time import sleep, time

import numpy as np

from laser_ablation.control.interaction import ExecutionReceipt

from .site import SiteConfiguration


class LaserPWMClient:
    """Send one JSON command per TCP connection and verify stopped status after each pulse."""

    def __init__(self, site: SiteConfiguration, records):
        self.site, self.records = site, records
        self.values = site.laser
        self.command_sequence = 0
        self.emission_possible = False

    def _send_command(self, command):
        """Use the exact unframed JSON request/response exchange present in the deployed code."""
        self.records.event("laser_command", sequence=self.command_sequence, command=command)
        with socket.create_connection((self.values["host"], self.values["port"]),
                                      timeout=self.values["timeout_s"]) as connection:
            connection.settimeout(self.values["timeout_s"])
            connection.sendall(json.dumps(command).encode("utf-8"))
            payload = connection.recv(1024)
        if not payload:
            raise RuntimeError("Raspberry Pi PWM service returned no response")
        response = json.loads(payload.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("Raspberry Pi PWM response must be a JSON mapping")
        self.records.event("laser_response", sequence=self.command_sequence, response=response)
        self.command_sequence += 1
        return response

    def stop(self):
        """Request stopped PWM and verify the configured field in a subsequent status reply."""
        stop_reply = self._send_command({"action": "stop"})
        status_reply = self._send_command({"action": "status"})
        key = self.values["status_key"]
        if key not in status_reply or status_reply[key] != self.values["stopped_value"]:
            raise RuntimeError("Raspberry Pi PWM status does not confirm stopped output")
        self.emission_possible = False
        return {"stop": stop_reply, "status": status_reply}

    def execute(self, request, achieved_action):
        """Map requested joules to calibrated duty cycle and complete one verified PWM pulse."""
        if not self.site.physical_execution_enabled or not self.values["watchdog_qualified"]:
            raise RuntimeError("Physical laser execution is disabled or its pulse watchdog is unqualified")
        table = self.values["energy_to_duty_cycle"]
        energy = request.action.energy_j
        if energy < table[0, 0] or energy > table[-1, 0]:
            raise ValueError("Requested energy lies outside the measured PWM calibration range")
        duty_cycle = float(np.interp(energy, table[:, 0], table[:, 1]))
        self._send_command({"action": "start", "duty_cycle": 0})
        sleep(self.values["startup_delay_s"])
        self.emission_possible = True
        self._send_command({"action": "set_pwm", "duty_cycle": duty_cycle,
                            "frequency": self.values["frequency_hz"]})
        sleep(self.values["pulse_duration_s"])
        replies = self.stop()
        receipt = ExecutionReceipt(
            request.command_id, request.action, achieved_action, "completed", time(),
            f"energy_to_duty_cycle:{self.values['calibration_id']}", True,
        )
        self.records.event("laser_pulse", receipt=receipt, duty_cycle_pct=duty_cycle,
                           frequency_hz=self.values["frequency_hz"],
                           pulse_duration_s=self.values["pulse_duration_s"], replies=replies)
        return receipt
