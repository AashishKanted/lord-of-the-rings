"""Fine docking: slow closed-loop regulation of the base pose onto the dock pose.

Control law (world frame, then limited and converted to the base frame):

    velocity = kp * position_error (+ ki * integral),  capped at dock_max_linear_mps and at
               sqrt(2 * dock_decel * distance) so the base can always stop on the target
    yaw_rate = kp_yaw * yaw_error (+ ki_yaw * integral), capped the same way

The velocity-commanded base is close to a pure integrator, for which P control has no
steady-state error; the integral terms default to zero and exist for a base with
friction or a deadband. Integration is frozen while the output is saturated.

Docked = position and yaw errors inside ``dock_tolerance_*`` and the commanded speed
below the customer's assembly base-speed limit, continuously for ``dock_dwell_s``.
Once docked the controller latches and commands zero. The customer limit is
25 mm / 2 deg in TRUE pose; the tolerances here apply to the ESTIMATED pose.
"""
from __future__ import annotations

from math import atan2, cos, hypot, sin, sqrt

import numpy as np

from ...cell import CELL
from ...config import get_config
from ...interfaces import Pose2D, Twist2D
from .limits import VelocityLimiter, clamp_norm


class DockingController:
    def __init__(self, config=None, limiter: VelocityLimiter | None = None):
        self.cfg = (config or get_config()).navigation
        self.limiter = limiter or VelocityLimiter(self.cfg)
        self.rest_speed = CELL["robot"]["assembly_base_speed_max_mps"]
        self.reset()

    def reset(self) -> None:
        """Call when a new docking approach starts."""
        self.docked = False
        self._settled_s = 0.0
        self._integral = np.zeros(3)
        self.position_error = float("inf")   # last errors, for logs and status
        self.yaw_error = float("inf")

    def step(self, pose: Pose2D, dock: Pose2D, dt: float) -> tuple[Twist2D, bool]:
        """Return (base-frame command, docked)."""
        cfg = self.cfg
        error = np.array([dock.x - pose.x, dock.y - pose.y])
        yaw_error = atan2(sin(dock.yaw - pose.yaw), cos(dock.yaw - pose.yaw))
        distance = hypot(error[0], error[1])
        self.position_error, self.yaw_error = distance, yaw_error
        if self.docked:
            return self.limiter.limit(np.zeros(2), 0.0, pose.yaw, dt), True

        v_cap = min(cfg.dock_max_linear_mps, sqrt(2.0 * cfg.dock_decel_mps2 * distance))
        w_cap = min(cfg.dock_max_angular_radps,
                    sqrt(2.0 * cfg.dock_decel_radps2 * abs(yaw_error)))
        wanted_v = cfg.dock_kp * error + cfg.dock_ki * self._integral[:2]
        wanted_w = cfg.dock_kp_yaw * yaw_error + cfg.dock_ki_yaw * self._integral[2]
        velocity = clamp_norm(wanted_v, v_cap)
        yaw_rate = float(np.clip(wanted_w, -w_cap, w_cap))
        if np.allclose(velocity, wanted_v):          # anti-windup: integrate only unsaturated
            self._integral[:2] += error * dt
        if yaw_rate == wanted_w:
            self._integral[2] += yaw_error * dt
        command = self.limiter.limit(velocity, yaw_rate, pose.yaw, dt,
                                     cfg.dock_max_linear_mps, cfg.dock_max_angular_radps)

        inside = distance < cfg.dock_tolerance_m and abs(yaw_error) < cfg.dock_tolerance_rad
        at_rest = hypot(command.vx, command.vy) < self.rest_speed
        self._settled_s = self._settled_s + dt if inside and at_rest else 0.0
        if self._settled_s >= cfg.dock_dwell_s:
            self.docked = True
            command = self.limiter.limit(np.zeros(2), 0.0, pose.yaw, dt)
        return command, self.docked
