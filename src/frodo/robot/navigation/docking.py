"""Fine docking: slow closed-loop pose regulation onto the dock. PLACEHOLDER.

Planned: PI(D) on x, y, yaw error in the dock frame at low speed, done when inside
``dock_tolerance_*`` and at rest for a short dwell. Customer limit is 25 mm / 2 deg;
manipulation wants a few mm.
"""
from __future__ import annotations

from ...interfaces import Pose2D, Twist2D


class DockingController:
    def __init__(self, config):
        self.config = config

    def step(self, pose: Pose2D, dock: Pose2D, dt: float) -> tuple[Twist2D, bool]:
        """Return (command, docked)."""
        raise NotImplementedError
