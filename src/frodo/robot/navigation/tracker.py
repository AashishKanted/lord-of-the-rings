"""Holonomic path tracker for travel between stations.

A reference point moves along the path (straight segments) with a trapezoidal speed
profile: accelerate at ``profile_accel_mps2``, cruise at ``cruise_speed_mps``, and
brake in time for every corner (``corner_speed_mps``) and for the end of the path.
Because yaw is planned in proportion to distance, the cruise speed is also capped so
the planned yaw rate stays below ``cruise_yaw_rate_radps``.

Command = feedforward (the reference velocity) + feedback on the pose error, split
into along-track (P), cross-track (PD) and heading (PD). The shared `VelocityLimiter`
then enforces the hard velocity/acceleration limits and converts to the base frame.

If the base falls behind (e.g. the safety monitor is holding it), the reference waits
instead of running away.
"""
from __future__ import annotations

from math import atan2, cos, hypot, sin

import numpy as np

from ...config import get_config
from ...interfaces import Pose2D, Twist2D
from .limits import VelocityLimiter


def _wrap(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


class PathTracker:
    def __init__(self, config=None, limiter: VelocityLimiter | None = None):
        self.cfg = (config or get_config()).navigation
        self.limiter = limiter or VelocityLimiter(self.cfg)
        self._points = np.zeros((0, 2))
        self.length = 0.0

    def set_path(self, path: list[Pose2D], end_speed: float = 0.0) -> None:
        """Start tracking ``path``; the reference arrives at its end with ``end_speed``."""
        points, yaws = [np.array([path[0].x, path[0].y])], [path[0].yaw]
        for pose in path[1:]:
            point = np.array([pose.x, pose.y])
            if np.linalg.norm(point - points[-1]) > 1e-6:      # drop repeated waypoints
                points.append(point)
                yaws.append(yaws[-1] + _wrap(pose.yaw - yaws[-1]))  # unwrapped
        self._points = np.array(points)
        self._yaws = np.array(yaws)
        self._s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self._points, axis=0),
                                                                  axis=1))])
        self.length = float(self._s[-1])
        # Speed the reference must not exceed when passing each later vertex.
        self._vertex_speed = np.full(len(points), self.cfg.corner_speed_mps)
        self._vertex_speed[-1] = end_speed
        self.s = 0.0          # reference position along the path, m
        self.speed = 0.0      # reference speed along the path, m/s
        self._segment = -1
        self._previous_error: tuple[float, float] | None = None

    @property
    def reference(self) -> Pose2D:
        point, _, yaw, _, _ = self._reference(self.s)
        return Pose2D(float(point[0]), float(point[1]), _wrap(yaw))

    def _reference(self, s: float):
        """Point, unit tangent, yaw, yaw change per metre and segment index at distance s."""
        if len(self._points) < 2:
            return self._points[0], np.array([1.0, 0.0]), self._yaws[0], 0.0, 0
        i = int(np.clip(np.searchsorted(self._s, s, side="right") - 1, 0, len(self._s) - 2))
        span = self._s[i + 1] - self._s[i]
        fraction = float(np.clip((s - self._s[i]) / span, 0.0, 1.0))
        tangent = (self._points[i + 1] - self._points[i]) / span
        yaw_per_m = (self._yaws[i + 1] - self._yaws[i]) / span
        return (self._points[i] + fraction * span * tangent, tangent,
                self._yaws[i] + fraction * span * yaw_per_m, yaw_per_m, i)

    def _speed_limit(self, s: float, yaw_per_m: float) -> float:
        """Highest reference speed at s that still allows braking for what lies ahead."""
        cfg = self.cfg
        limit = cfg.cruise_speed_mps
        if abs(yaw_per_m) > 1e-9:
            limit = min(limit, cfg.cruise_yaw_rate_radps / abs(yaw_per_m))
        ahead = self._s > s + 1e-9
        ahead[-1] = True
        braking = np.sqrt(self._vertex_speed[ahead] ** 2
                          + 2.0 * cfg.profile_accel_mps2 * np.maximum(self._s[ahead] - s, 0.0))
        return min(limit, float(braking.min()))

    def step(self, pose: Pose2D, dt: float) -> tuple[Twist2D, bool]:
        """Return (base-frame command, reached_end)."""
        cfg = self.cfg
        if len(self._points) == 0:
            return self.limiter.limit(np.zeros(2), 0.0, pose.yaw, dt), True
        position = np.array([pose.x, pose.y])

        # 1. Advance the reference along the path (trapezoidal profile).
        point, tangent, _, yaw_per_m, _ = self._reference(self.s)
        if float((point - position) @ tangent) > cfg.tracker_max_lag_m:
            self.speed = 0.0                      # base is held back: wait for it
        else:
            self.speed = min(self.speed + cfg.profile_accel_mps2 * dt,
                             self._speed_limit(self.s, yaw_per_m))
        s_new = min(self.s + self.speed * dt, self.length)
        feedforward = (s_new - self.s) / dt       # zero once the reference is at the end
        self.s = s_new

        # 2. Pose error to the reference, in path coordinates.
        point, tangent, yaw_ref, yaw_per_m, segment = self._reference(self.s)
        normal = np.array([-tangent[1], tangent[0]])
        error = point - position
        along, cross = float(error @ tangent), float(error @ normal)
        heading = _wrap(yaw_ref - pose.yaw)
        cross_rate = heading_rate = 0.0
        if self._previous_error is not None and segment == self._segment:
            cross_rate = (cross - self._previous_error[0]) / dt
            heading_rate = _wrap(heading - self._previous_error[1]) / dt
        self._previous_error, self._segment = (cross, heading), segment

        # 3. Feedforward + feedback, then hard limits.
        velocity = ((feedforward + cfg.tracker_kp_along * along) * tangent
                    + (cfg.tracker_kp_cross * cross + cfg.tracker_kd_cross * cross_rate) * normal)
        yaw_rate = (feedforward * yaw_per_m + cfg.tracker_kp_yaw * heading
                    + cfg.tracker_kd_yaw * heading_rate)
        command = self.limiter.limit(velocity, yaw_rate, pose.yaw, dt)
        reached = (self.s >= self.length - 1e-9
                   and hypot(error[0], error[1]) < cfg.tracker_goal_tolerance_m)
        return command, reached
