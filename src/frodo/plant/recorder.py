"""Measured robot motion for the evidence log (``sample`` events).

Plant side: reads the true base pose, tool position and arm joint angles from the
simulator, independent of what the robot software believes or commands. Speeds are mean
speeds over the sample interval (displacement / time), the way an external tracker would
report them.
"""
from __future__ import annotations

from math import hypot

import numpy as np

from ..robot.description import ARM_JOINTS, ARM_STOWED, BASE_JOINTS, TOOL_SITE
from .events import EventLog


class MotionRecorder:
    def __init__(self, model, events: EventLog, period_s: float, stowed_tol_rad: float = 0.05):
        self.events, self.period = events, period_s
        self._base = [int(model.joint(j).qposadr[0]) for j in BASE_JOINTS]
        self._arm = [int(model.joint(j).qposadr[0]) for j in ARM_JOINTS]
        self._site = model.site(TOOL_SITE).id
        self._tol = stowed_tol_rad
        self._last: tuple[float, np.ndarray, np.ndarray] | None = None
        self.max_base_speed = 0.0
        self.max_base_speed_unstowed = 0.0

    def pose(self, data) -> np.ndarray:
        """True base pose (x, y, yaw)."""
        return data.qpos[self._base].copy()

    def arm_stowed(self, data) -> bool:
        return bool(np.max(np.abs(data.qpos[self._arm] - ARM_STOWED)) < self._tol)

    def due(self, t: float) -> bool:
        return self._last is None or t - self._last[0] >= self.period - 1e-9

    def sample(self, data) -> dict | None:
        """Write one sample; None if one was already written at this instant."""
        t, pose, tool = float(data.time), self.pose(data), data.site_xpos[self._site].copy()
        if self._last is not None and t <= self._last[0]:
            return None
        base_speed = ee_speed = 0.0
        if self._last is not None and t > self._last[0]:
            dt = t - self._last[0]
            base_speed = hypot(*(pose[:2] - self._last[1][:2])) / dt
            ee_speed = float(np.linalg.norm(tool - self._last[2])) / dt
        self._last = (t, pose, tool)
        stowed = self.arm_stowed(data)
        self.max_base_speed = max(self.max_base_speed, base_speed)
        if not stowed:
            self.max_base_speed_unstowed = max(self.max_base_speed_unstowed, base_speed)
        return self.events.sample(t, pose[0], pose[1], pose[2], base_speed, ee_speed, stowed)
