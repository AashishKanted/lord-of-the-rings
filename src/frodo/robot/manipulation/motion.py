"""Tool-point motion profiles."""
from __future__ import annotations

import numpy as np

from ...config import get_config

_M = get_config().manipulation


class Slew:
    """Moves a point toward a goal with speed and acceleration limits."""

    def __init__(self, p, max_speed=_M.max_speed_mps, max_accel=_M.max_accel_mps2):
        self.p, self.speed = np.array(p, float), 0.0
        self.max_speed, self.max_accel = max_speed, max_accel

    def step(self, goal, dt=_M.control_dt_s):
        delta = goal - self.p
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            self.speed = 0.0
            return self.p
        self.speed = min(self.max_speed, self.speed + self.max_accel * dt,
                         np.sqrt(2 * self.max_accel * dist))
        self.p = self.p + delta / dist * min(dist, self.speed * dt)
        return self.p
