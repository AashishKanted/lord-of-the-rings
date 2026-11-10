"""Scan-to-map matching against the known cell map. PLACEHOLDER.

Planned: the map is a set of line segments (walls, conveyor faces) from the customer
spec. Point-to-line ICP: project lidar points with the predicted pose, associate each
with its nearest segment, solve the small 2D least-squares correction, iterate.
"""
from __future__ import annotations

from ...interfaces import LidarScan, Pose2D


class ScanMatcher:
    def __init__(self, config):
        self.config = config

    def match(self, scans: list[LidarScan], prior: Pose2D) -> tuple[Pose2D, object] | None:
        """Return (pose, covariance) or None when the match is unreliable."""
        raise NotImplementedError
