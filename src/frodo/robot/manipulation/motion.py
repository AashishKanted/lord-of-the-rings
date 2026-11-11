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


class JointMove:
    """Joint-space move through via points: straight lines in joint space, each with a
    cosine speed profile (zero velocity at both ends), timed by ``max_speed`` rad/s."""

    def __init__(self, waypoints, max_speed: float):
        self.points = [np.array(q, float) for q in waypoints]
        # peak speed of the cosine profile is pi/2 times the mean speed
        self.durations = [max(float(np.max(np.abs(b - a))) * np.pi / (2 * max_speed), 1e-3)
                          for a, b in zip(self.points[:-1], self.points[1:], strict=True)]
        self.t = 0.0

    @property
    def done(self) -> bool:
        return self.t >= sum(self.durations)

    def step(self, dt: float) -> np.ndarray:
        self.t += dt
        t = self.t
        for a, b, duration in zip(self.points[:-1], self.points[1:], self.durations, strict=True):
            if t < duration:
                return a + (b - a) * 0.5 * (1 - np.cos(np.pi * t / duration))
            t -= duration
        return self.points[-1].copy()
