"""Geometry of a station as seen from the robot base frame.

The base frame is x forward, y left, z up, origin on the floor under the base centre.
Station-local coordinates are the customer's: x along belt travel, origin at the conveyor
centre. A `StationFrame` is built from a base pose estimate (the localizer's, taken when
the robot has docked), so it carries that estimate's error and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np

from ..cell import CELL, STATIONS
from ..interfaces import Pose2D


def world_to_base(pose: Pose2D, xy) -> np.ndarray:
    """World xy point -> base-frame xy for a base at ``pose``."""
    c, s = cos(pose.yaw), sin(pose.yaw)
    dx, dy = xy[0] - pose.x, xy[1] - pose.y
    return np.array([c * dx + s * dy, -s * dx + c * dy])


@dataclass(frozen=True)
class StationFrame:
    name: str
    center: np.ndarray     # conveyor centre on the belt surface, base frame xyz
    belt_dir: np.ndarray   # unit vector of belt travel (station-local +x), base frame xyz
    lateral: np.ndarray    # station-local +y, base frame xyz

    @classmethod
    def from_pose(cls, station: str, base_pose: Pose2D) -> StationFrame:
        st = STATIONS[station]
        center = world_to_base(base_pose, st.center[:2])
        yaw = st.yaw - base_pose.yaw
        return cls(station, np.array([*center, CELL["conveyor"]["height_m"]]),
                   np.array([cos(yaw), sin(yaw), 0.0]), np.array([-sin(yaw), cos(yaw), 0.0]))

    def local_x(self, point) -> float:
        """Station-local x (distance along belt travel from the conveyor centre)."""
        return float((np.asarray(point)[:2] - self.center[:2]) @ self.belt_dir[:2])

    def point(self, local_x: float, local_y: float, z: float) -> np.ndarray:
        """Station-local (x, y) at height z above the floor -> base-frame xyz."""
        p = self.center + local_x * self.belt_dir + local_y * self.lateral
        return np.array([p[0], p[1], z])
