"""Velocity and acceleration limiting shared by the path tracker and the docking controller.

Controllers compute a desired velocity in the WORLD frame. The limiter clamps its size,
limits how fast it may change (acceleration), and converts it to the base frame, which
is what the holonomic base is commanded in. Limiting in the world frame keeps a constant
world velocity constant while the base rotates underneath it.

One limiter is shared across the tracking -> docking switch so the command stays
continuous at the handover.
"""
from __future__ import annotations

from math import cos, hypot, sin

import numpy as np

from ...cell import CELL
from ...interfaces import Twist2D


def clamp_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = hypot(vector[0], vector[1])
    return vector if norm <= limit else vector * (limit / norm)


class VelocityLimiter:
    def __init__(self, nav_config):
        robot = CELL["robot"]
        # Configured limits can never exceed the customer's.
        self.max_linear = min(nav_config.max_linear_mps, robot["max_transit_speed_mps"])
        self.max_angular = min(nav_config.max_angular_radps, robot["max_yaw_rate_radps"])
        self.max_linear_accel = nav_config.max_linear_accel_mps2
        self.max_angular_accel = nav_config.max_angular_accel_radps2
        self.reset()

    def reset(self) -> None:
        """Forget the previous command (the base is assumed at rest)."""
        self.velocity = np.zeros(2)   # last commanded linear velocity, world frame
        self.yaw_rate = 0.0

    def limit(self, velocity_world, yaw_rate: float, yaw: float, dt: float,
              max_linear: float | None = None, max_angular: float | None = None) -> Twist2D:
        """Desired world-frame velocity -> limited base-frame command."""
        v_max = min(max_linear or self.max_linear, self.max_linear)
        w_max = min(max_angular or self.max_angular, self.max_angular)
        wanted = clamp_norm(np.asarray(velocity_world, dtype=float), v_max)
        self.velocity = self.velocity + clamp_norm(wanted - self.velocity,
                                                   self.max_linear_accel * dt)
        dw = float(np.clip(yaw_rate, -w_max, w_max)) - self.yaw_rate
        max_dw = self.max_angular_accel * dt
        self.yaw_rate += float(np.clip(dw, -max_dw, max_dw))
        c, s = cos(yaw), sin(yaw)
        vx = c * self.velocity[0] + s * self.velocity[1]      # world -> base
        vy = -s * self.velocity[0] + c * self.velocity[1]
        return Twist2D(float(vx), float(vy), float(self.yaw_rate))
