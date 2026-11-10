"""Holonomic path tracker for travel between stations. PLACEHOLDER.

Planned: trapezoidal speed profile along the path, feedforward along-track velocity
plus PD on cross-track and heading error, output limited to the [navigation] maxima.
"""
from __future__ import annotations

from ...interfaces import Pose2D, Twist2D


class PathTracker:
    def __init__(self, config):
        self.config = config

    def set_path(self, path: list[Pose2D]) -> None:
        raise NotImplementedError

    def step(self, pose: Pose2D, dt: float) -> tuple[Twist2D, bool]:
        """Return (command, reached_end)."""
        raise NotImplementedError
