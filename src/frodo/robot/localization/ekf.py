"""Pose filter: odometry + gyro prediction, scan-match correction. PLACEHOLDER.

State [x, y, yaw] in the map frame. Predict at the odometry rate; correct whenever
`ScanMatcher` returns a pose. This is a standard EKF with an identity measurement model.
"""
from __future__ import annotations

from ...interfaces import Pose2D, SensorFrame


class Localizer:
    def __init__(self, config, initial: Pose2D):
        self.config, self.pose = config, initial

    def update(self, frame: SensorFrame) -> Pose2D:
        raise NotImplementedError
