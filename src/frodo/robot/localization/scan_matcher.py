"""Scan-to-map matching: point-to-line ICP of the lidar points against the known map.

Given a predicted base pose, each iteration

1. puts the lidar points in the world with the current pose estimate,
2. pairs every point with its nearest map segment and drops pairs farther than a gate
   (unmapped things such as a person, and gross mismatches, fall outside the gate),
3. solves the linear least-squares problem for the small correction (dx, dy, dyaw)
   that minimises the perpendicular distances to the paired lines (Gauss-Newton).

The gate starts at ``icp_max_correspondence_m`` and tightens to ``icp_gate_sigmas`` robust standard
deviations of the current residuals (never below ``icp_min_correspondence_m``), so a
coarse prior can still converge while the final fit ignores anything not on the map.

The result is refused (``None``) when too few points fit the map or when the fitted
lines do not constrain both directions (e.g. only one straight wall in view).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np

from ...config import get_config
from ...interfaces import LidarScan, Pose2D
from ...log import get_logger
from ..map import closest_on_segments, get_map

log = get_logger(__name__)

MAD_TO_STD = 1.4826  # median absolute deviation -> standard deviation (Gaussian)


@dataclass(frozen=True)
class ScanMatch:
    pose: Pose2D
    covariance: np.ndarray    # 3x3 for [x, y, yaw]: residual variance * inv(J^T J)
    inliers: int              # points used in the final fit
    inlier_fraction: float    # inliers / valid lidar returns
    rms_m: float              # RMS point-to-line residual of the inliers
    iterations: int


def scan_points_in_base(scans: list[LidarScan], min_range: float, max_range: float,
                        stride: int = 1) -> np.ndarray:
    """Valid returns of all scans as points (N, 2) in the base frame."""
    points = []
    for scan in scans:
        r, a = np.asarray(scan.ranges)[::stride], np.asarray(scan.angles)[::stride]
        ok = np.isfinite(r) & (r > min_range) & (r < max_range)
        beam = a[ok] + scan.mount.yaw
        points.append(np.column_stack([scan.mount.x + r[ok] * np.cos(beam),
                                       scan.mount.y + r[ok] * np.sin(beam)]))
    return np.vstack(points) if points else np.zeros((0, 2))


class ScanMatcher:
    def __init__(self, config=None, segments: np.ndarray | None = None):
        cfg = config or get_config()
        self.cfg = cfg.localization
        self.lidar = cfg.sensors.lidar
        self.segments = get_map(cfg).segments if segments is None else segments
        a, b = self.segments[:, :2], self.segments[:, 2:]
        direction = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
        self._start = a
        self._normal = np.column_stack([-direction[:, 1], direction[:, 0]])

    def match(self, scans: list[LidarScan], prior: Pose2D) -> ScanMatch | None:
        """Refine ``prior`` so the scans line up with the map; None when unreliable."""
        cfg = self.cfg
        base = scan_points_in_base(scans, cfg.scan_min_range_m, self.lidar.range_max_m,
                                   cfg.icp_point_stride)
        if len(base) < cfg.icp_min_inliers:
            return None
        x, y, yaw = prior.x, prior.y, prior.yaw
        gate = cfg.icp_max_correspondence_m
        iterations = 0
        while iterations < cfg.icp_max_iterations:
            iterations += 1
            c, s = cos(yaw), sin(yaw)
            lever = base @ np.array([[c, s], [-s, c]])        # base -> point, world axes
            world = lever + [x, y]
            index, dist = closest_on_segments(world, self.segments)
            inlier = dist < gate
            if inlier.sum() < cfg.icp_min_inliers:
                log.debug("scan match refused: %d inliers", inlier.sum())
                return None
            normal = self._normal[index[inlier]]
            residual = np.einsum("ij,ij->i", normal, world[inlier] - self._start[index[inlier]])
            # d(world point)/d(yaw) is the lever arm rotated by 90 degrees.
            turn = np.column_stack([-lever[inlier, 1], lever[inlier, 0]])
            jac = np.column_stack([normal, np.einsum("ij,ij->i", normal, turn)])
            info = jac.T @ jac
            # Both translation directions must be constrained: the 2x2 block's eigenvalues
            # are the shares of points constraining the weakest and strongest direction.
            if np.linalg.eigvalsh(info[:2, :2])[0] < cfg.icp_min_direction_fraction * len(jac):
                log.debug("scan match refused: degenerate geometry")
                return None
            step = np.linalg.solve(info, -jac.T @ residual)
            x, y, yaw = x + step[0], y + step[1], yaw + step[2]
            robust_std = MAD_TO_STD * float(np.median(np.abs(residual)))
            gate = float(np.clip(cfg.icp_gate_sigmas * robust_std, cfg.icp_min_correspondence_m,
                                 cfg.icp_max_correspondence_m))
            if np.hypot(step[0], step[1]) < cfg.icp_tolerance_m and \
                    abs(step[2]) < cfg.icp_tolerance_rad and gate <= cfg.icp_min_correspondence_m:
                break

        fraction = float(inlier.mean())
        if fraction < cfg.icp_min_inlier_fraction:
            log.debug("scan match refused: only %.0f%% of points fit the map", 100 * fraction)
            return None
        rms = float(np.sqrt(np.mean(residual ** 2)))
        variance = max(rms, self.lidar.range_noise_std_m) ** 2
        return ScanMatch(Pose2D(float(x), float(y), atan2(sin(yaw), cos(yaw))),
                         variance * np.linalg.inv(info), int(inlier.sum()), fraction, rms,
                         iterations)
