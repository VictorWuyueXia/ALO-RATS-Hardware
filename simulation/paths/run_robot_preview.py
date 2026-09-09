"""Select, move, pulse, and reobserve with no planner and no hardware connections."""

from itertools import chain, repeat
import time

import pybullet as p

from simulation.robot.preview_robot import PreviewRobot
from simulation.visualization.preview_tissue import PreviewTissue, TARGETS_MM, ENERGY_J, STOP_REMAINING_PCT


HELP = """
SIMULATION ONLY — no OCT acquisition, RTDE, PWM, MPC, or MPPI.
The tissue scan, target, registration, and 20 mm working distance are nominal.
Click the PyBullet window to give it keyboard focus:
  1–5    Select a predefined target (yellow marker)
  Enter  Confirm selection and animate robot motion
  Space  Confirm ONE virtual 3.5 J pulse after motion finishes
  H      Return to the original home pose, without a pulse
  C / V  30x tissue inspection (display only, robot hidden) / robot overview
  Esc    Quit; Ctrl+C in the terminal also quits
Green outline: target footprint; blue: target floor; red: protected floor.
Stop at <=80% remaining target volume (proof of concept, NOT full resection).
"""


class PreviewInteraction:
    """Require independent operator confirmation for robot motion and virtual pulse."""

    def __init__(self, client):
        self.robot = PreviewRobot(client)
        self.tissue = PreviewTissue()
        self.observation = self.tissue.observe()
        self.selected = 0
        self.ready = False
        self.path = None
        self.returning_home = False

    @property
    def complete(self):
        return self.tissue.remaining_pct <= STOP_REMAINING_PCT

    def select(self, index):
        if self.path is not None:
            raise ValueError("Wait for motion to finish before changing the target")
        if index not in range(len(TARGETS_MM)):
            raise ValueError("Choose target 1 through 5")
        self.selected = index
        self.ready = False

    def move(self, home=False):
        if self.path is not None:
            raise ValueError("A robot motion is already in progress")
        if self.complete and not home:
            raise ValueError("The proof-of-concept stopping criterion is reached")
        path = self.robot.path_to(None if home else TARGETS_MM[self.selected])
        self.path = iter(chain(path, repeat(path[-1], 240)))
        self.returning_home = home
        self.ready = False

    def tick(self):
        """Advance motors without blocking the GUI's camera or quit controls."""
        finished = False
        if self.path is not None:
            try:
                self.robot.hold(next(self.path))
            except StopIteration:
                self.path = None
                if not self.returning_home:
                    self.robot.require_target(TARGETS_MM[self.selected])
                    self.ready = True
                finished = True
        p.stepSimulation(physicsClientId=self.robot.client)
        return finished

    def fire(self):
        if not self.ready or self.path is not None:
            raise ValueError("Confirm and finish a target motion before pulsing")
        self.robot.require_target(TARGETS_MM[self.selected])
        action = self.robot.achieved_action(ENERGY_J)
        self.observation = self.tissue.pulse(action)
        self.ready = False
        return action


def main():
    """Run the sole GUI backend; construction failure never selects another backend."""
    from simulation.visualization.preview_display import PreviewDisplay

    print(HELP, flush=True)
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("PyBullet GUI connection failed")
    try:
        p.setTimeStep(1 / 240, physicsClientId=client)
        p.setGravity(0, 0, 0, physicsClientId=client)
        interaction = PreviewInteraction(client)
        display = PreviewDisplay(interaction.robot, interaction.tissue)
        display.select(0)
        display.status("Nominal scan ready. Select 1-5, then Enter to move.")
        while p.isConnected(client):
            keys = p.getKeyboardEvents(physicsClientId=client)
            pressed = {key for key, state in keys.items() if state & p.KEY_WAS_TRIGGERED}
            if 27 in pressed:
                break
            if ord('c') in pressed:
                display.closeup()
            if ord('v') in pressed:
                display.overview()
            # Motion cannot be queued or pulsed twice by holding a keyboard key.
            if interaction.path is None:
                for index in range(5):
                    if ord(str(index + 1)) in pressed and not interaction.complete:
                        interaction.select(index)
                        display.select(index)
                        display.status(f"Target {index + 1} selected. Enter confirms motion.")
                if ord('h') in pressed:
                    interaction.move(home=True)
                    display.status("Returning home; no pulse.")
                elif pressed & {p.B3G_RETURN, 10, 13} and not interaction.complete:
                    interaction.move()
                    display.status("Moving in simulation. Esc quits.")
                elif 32 in pressed and interaction.ready:
                    action = interaction.fire()
                    display.draw_surface(interaction.tissue)
                    display.beam(pulsing=True)
                    print(f"Achieved mm/rad/J action: {action.tolist()}", flush=True)
                    suffix = ("POC complete. H: home; Esc: quit." if interaction.complete
                              else "Select 1-5, Enter, then Space for another pulse.")
                    display.status(f"Pulse {interaction.tissue.pulses}; new synthetic scan; "
                                   f"{interaction.tissue.remaining_pct:.1f}% remaining. {suffix}")
            if interaction.tick():
                if interaction.ready:
                    display.beam()
                    display.status("Motion complete. C: close-up; Space: confirm virtual pulse.")
                else:
                    display.status("Home. Select 1-5 and Enter, or Esc to quit.")
            time.sleep(1 / 240)
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
