"""Applies a `RobotCommand` to the simulated robot's actuators. PLACEHOLDER.

Planned: integrate the base twist into the x/y/yaw position servos with velocity and
acceleration limits, pass arm targets and gripper command through, and gate the grasp
constraint on the physical grasp conditions (see ``ringcell.sim.grasp``).
"""
from __future__ import annotations

from ..interfaces import RobotCommand


class ActuatorBridge:
    def __init__(self, model, config):
        self.model, self.config = model, config

    def apply(self, data, command: RobotCommand) -> None:
        raise NotImplementedError("actuator bridge is not implemented yet")
