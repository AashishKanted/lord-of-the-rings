"""Pose filter: odometry + gyro prediction, scan-match correction.

Extended Kalman filter on the state ``[x, y, yaw]`` in the map frame.

- predict (every control tick): integrate the odometry twist, which is measured in the
  base frame, using the gyro for the yaw rate when available. Covariance grows by the
  process noise.
- correct (when lidar scans arrive, ~10 Hz): the scan matcher refines the predicted
  pose against the map; that pose is the measurement (measurement model H = I). A
  chi-square gate on the innovation drops matches that disagree with the prediction
  by more than the combined uncertainty allows.
- sensor latency: a scan is stamped with its capture time, which is a little before the
  tick that delivers it. The filter keeps a short history of its own estimates, matches
  the scan against the estimate it had at the capture time, and applies the resulting
  correction to the current state (the pose error barely changes over a few ticks).

The room is symmetric, so there is no global relocalisation: the filter starts from
the known mission start pose and only ever tracks.
"""
from __future__ import annotations

from collections import deque
from math import atan2, cos, hypot, sin, sqrt

import numpy as np

from ...cell import CELL
from ...config import get_config
from ...interfaces import Pose2D, SensorFrame, Twist2D
from ...log import get_logger
from .scan_matcher import ScanMatch, ScanMatcher

log = get_logger(__name__)


def wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


class Localizer:
    def __init__(self, config=None, initial: Pose2D | None = None,
                 matcher: ScanMatcher | None = None):
        cfg = config or get_config()
        self.cfg = cfg.localization
        self.matcher = matcher or ScanMatcher(cfg)
        self.reset(initial or Pose2D(*CELL["start_pose_world"]))

    def reset(self, pose: Pose2D) -> None:
        """Restart the filter at a known pose with the configured initial uncertainty."""
        cfg = self.cfg
        self.state = np.array([pose.x, pose.y, pose.yaw], dtype=float)
        self.covariance = np.diag([cfg.initial_std_xy_m ** 2, cfg.initial_std_xy_m ** 2,
                                   cfg.initial_std_yaw_rad ** 2])
        self._t: float | None = None           # time of the last frame
        self._t_corrected: float | None = None  # time of the last accepted scan match
        self._twist = Twist2D()
        self._history: deque[tuple[float, np.ndarray]] = deque()  # recent (t, state)
        self.last_match: ScanMatch | None = None
        self.rejected_in_a_row = 0

    # ---- outputs ----
    @property
    def pose(self) -> Pose2D:
        return Pose2D(float(self.state[0]), float(self.state[1]), wrap_angle(self.state[2]))

    @property
    def std_xy(self) -> float:
        """Largest 1-sigma position uncertainty, m."""
        return sqrt(float(np.linalg.eigvalsh(self.covariance[:2, :2])[-1]))

    @property
    def std_yaw(self) -> float:
        return sqrt(float(self.covariance[2, 2]))

    @property
    def healthy(self) -> bool:
        """True while scan corrections are recent and the uncertainty is small."""
        if self._t is None or self._t_corrected is None:
            return False
        return (self._t - self._t_corrected <= self.cfg.health_timeout_s
                and self.std_xy <= self.cfg.health_max_std_xy_m
                and self.std_yaw <= self.cfg.health_max_std_yaw_rad)

    # ---- filter ----
    def update(self, frame: SensorFrame) -> Pose2D:
        """One control tick: predict with odometry/gyro, correct with any lidar scans."""
        dt = 0.0 if self._t is None else frame.t - self._t
        self._t = frame.t
        if frame.odometry is not None:
            self._twist = frame.odometry.twist
        yaw_rate = self._twist.wz
        if self.cfg.use_gyro and frame.imu is not None:
            yaw_rate = frame.imu.gyro_z
        if dt > 0.0:
            self.predict(self._twist, yaw_rate, dt)
        self._history.append((frame.t, self.state.copy()))
        while frame.t - self._history[0][0] > self.cfg.scan_history_s:
            self._history.popleft()
        # Scans captured at the same instant (both lidars) are matched together.
        for t_scan in sorted({scan.t for scan in frame.lidars}):
            scans = [scan for scan in frame.lidars if scan.t == t_scan]
            then = self._state_at(t_scan)
            match = self.matcher.match(scans, Pose2D(then[0], then[1], wrap_angle(then[2])))
            if match is None:
                self.rejected_in_a_row += 1
            elif self.correct(match, then):
                self._t_corrected = frame.t
        return self.pose

    def _state_at(self, t: float) -> np.ndarray:
        """The filter's own estimate at (the stored tick nearest to) time t."""
        return min(self._history, key=lambda item: abs(item[0] - t))[1].copy()

    def predict(self, twist: Twist2D, yaw_rate: float, dt: float) -> None:
        cfg = self.cfg
        yaw_mid = self.state[2] + 0.5 * yaw_rate * dt
        c, s = cos(yaw_mid), sin(yaw_mid)
        dx = (c * twist.vx - s * twist.vy) * dt       # base-frame velocity -> world step
        dy = (s * twist.vx + c * twist.vy) * dt
        self.state += [dx, dy, yaw_rate * dt]
        jac = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]])
        # Random-walk process noise plus a part that scales with the distance travelled
        # (wheel slip / scale error).
        travelled = hypot(twist.vx, twist.vy) * dt
        q_xy = cfg.process_noise_xy_m ** 2 * dt + (cfg.odometry_error_frac * travelled) ** 2
        q_yaw = cfg.process_noise_yaw_rad ** 2 * dt
        self.covariance = jac @ self.covariance @ jac.T + np.diag([q_xy, q_xy, q_yaw])

    def correct(self, match: ScanMatch, reference: np.ndarray | None = None) -> bool:
        """Fuse a scan-match pose. ``reference`` is the estimate the filter had when the scan
        was captured (default: the current state). Returns False if the gate rejected it."""
        cfg = self.cfg
        reference = self.state if reference is None else reference
        # The ICP covariance only covers range noise; the floor covers map and
        # calibration errors that do not average out over the points of one scan.
        noise = match.covariance + np.diag([cfg.scan_noise_xy_m ** 2, cfg.scan_noise_xy_m ** 2,
                                            cfg.scan_noise_yaw_rad ** 2])
        innovation = np.array([match.pose.x - reference[0], match.pose.y - reference[1],
                               wrap_angle(match.pose.yaw - reference[2])])
        s_inv = np.linalg.inv(self.covariance + noise)
        distance2 = float(innovation @ s_inv @ innovation)
        if distance2 > cfg.gate_chi2:
            self.rejected_in_a_row += 1
            log.warning("scan match gated out: d2=%.1f innovation=(%.3f m, %.3f m, %.4f rad)",
                        distance2, *innovation)
            return False
        gain = self.covariance @ s_inv
        shift = gain @ innovation
        self.state += shift
        for _, past in self._history:      # keep the stored estimates consistent
            past += shift
        keep = np.eye(3) - gain
        self.covariance = keep @ self.covariance @ keep.T + gain @ noise @ gain.T  # Joseph form
        self.last_match = match
        self.rejected_in_a_row = 0
        return True
