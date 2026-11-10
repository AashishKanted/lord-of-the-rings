"""Global path on the known map. PLACEHOLDER.

Planned: occupancy grid from the customer spec (walls, conveyors) inflated by
``navigation.inflation_m``; A* between the current pose and a pre-dock pose; shortcut
smoothing into straight segments.
"""
from __future__ import annotations

from ...interfaces import Pose2D


class GlobalPlanner:
    def __init__(self, config):
        self.config = config

    def plan(self, start: Pose2D, goal: Pose2D) -> list[Pose2D]:
        raise NotImplementedError
