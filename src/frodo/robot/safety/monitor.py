"""Independent base safety supervisor (rudimentary).

Sits between the motion sources and the actuators. It looks at the raw lidar returns, not
at the localization output or the planned path, and at the base command that is about to
be sent. It can only reduce that command.

Rules, with distances measured from the edge of the base footprint:

- translation: lidar returns inside the corridor the footprint is about to sweep (the
  commanded direction of travel) scale the speed down linearly from ``slow_distance_m`` to
  zero at ``stop_distance_m``;
- rotation: a return inside the circle the footprint's corners sweep (plus
  ``rotation_margin_m``) stops the base;
- docking mask: while docking, the orchestrator passes the target conveyor as a
  `StationFrame`. Returns from inside that conveyor's outline are expected, so they are
  exempt from the slow/stop distances, but the base still stops if one comes closer than
  ``dock_min_clearance_m``. That lets the base close in on the conveyor without touching it;
- stale data: no lidar scan for ``scan_timeout_s`` stops the base.
- work zone (latched): while the robot works at a station, ANY lidar return within
  ``human_zone_radius_m`` of the footprint, other than that station's own conveyor, latches
  a stop of base and arm (`check_zone`). The latch stays until the zone is clear AND an
  operator acknowledges (`acknowledge`). The stops above release by themselves.

Every object is treated the same: a 2D lidar at 12 cm sees a slice of legs, boxes and
trolleys alike and cannot tell a person from them, so there is no person detection. That
would need a 3D lidar (or a camera) and a person detector on its point cloud. This is a
simulation behaviour, not a certified protective function.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

import numpy as np

from ...cell import CELL
from ...interfaces import RobotCommand, SensorFrame, Twist2D
from ...log import get_logger
from ..frames import StationFrame
from ..localization import scan_points_in_base

log = get_logger(__name__)
_CONVEYOR = CELL["conveyor"]


@dataclass(frozen=True)
class SafetyState:
    speed_scale: float       # 1 = free, 0 = stopped
    stopped: bool
    reason: str = ""
    nearest_m: float | None = None    # nearest relevant return, from the footprint edge
    latched: bool = False             # work-zone stop: base AND arm, until acknowledged


class SafetyMonitor:
    def __init__(self, config):
        self.cfg = config.safety
        self._lidar = config.sensors.lidar
        self._min_range = config.localization.scan_min_range_m
        self._half = self.cfg.footprint_m / 2
        self._scans: dict[str, object] = {}       # latest scan per lidar
        self.state = SafetyState(1.0, False)
        self._clear_since: float | None = None
        self._latch: str | None = None        # reason of the active work-zone stop
        self.zone_clear = True                # last work-zone check found nothing

    # ---- work-zone stop (latched) ----
    @property
    def latched(self) -> bool:
        return self._latch is not None

    def check_zone(self, frame: SensorFrame, station: StationFrame | None) -> bool:
        """Call every tick. ``station``: where the robot is working (None in transit, when
        the zone rule does not apply). Returns True while the stop is latched."""
        for scan in frame.lidars:
            self._scans[scan.name] = scan
        if station is None:
            self.zone_clear = True
            return self.latched
        reason = self._zone_intruder(frame.t, station)
        self.zone_clear = reason is None
        if reason is not None and self._latch is None:
            self._latch = reason
            log.warning("t=%.2f SAFETY STOP (latched, base and arm): %s", frame.t, reason)
        if self.latched:
            self.state = SafetyState(0.0, True, self._latch, latched=True)
        return self.latched

    def acknowledge(self, t: float) -> bool:
        """Operator recovery. Refused while anything is still inside the zone."""
        if self._latch is None:
            return True
        if not self.zone_clear:
            log.warning("t=%.2f acknowledge refused: the work zone is still occupied", t)
            return False
        log.info("t=%.2f safety stop acknowledged, zone clear: resuming", t)
        self._latch = None
        self.state = SafetyState(1.0, False)
        return True

    def _zone_intruder(self, t: float, station: StationFrame) -> str | None:
        cfg = self.cfg
        scans = list(self._scans.values())
        if not scans or any(t - scan.t > cfg.scan_timeout_s for scan in scans) \
                or len(scans) < len(self._lidar.mounts):
            return "lidar data missing or stale"
        points = scan_points_in_base(scans, self._min_range, self._lidar.range_max_m)
        if len(points) == 0:
            return None
        points = points[~self._inside_conveyor(points, station)]   # the station itself
        edge = self._edge_distance(points)
        inside = edge < cfg.human_zone_radius_m
        if inside.sum() < cfg.zone_min_points:
            return None
        return (f"object {1e3 * edge[inside].min():.0f} mm from the base inside the "
                f"{cfg.human_zone_radius_m:.1f} m work zone")

    def filter(self, frame: SensorFrame, wanted: RobotCommand,
               dock_mask: StationFrame | None = None) -> tuple[RobotCommand, SafetyState]:
        """Return the command that is allowed, and why it was changed (if it was)."""
        for scan in frame.lidars:
            self._scans[scan.name] = scan
        if self.latched:
            return replace(wanted, base=Twist2D()), self.state
        # Hysteresis: once stopped, the obstacle has to be `clear_hysteresis_m` further away
        # for `clear_time_s` before motion resumes, so range noise cannot make the base creep.
        held = self.state.stopped
        state = self._assess(frame.t, wanted.base, dock_mask,
                             self.cfg.clear_hysteresis_m if held else 0.0)
        if held and not state.stopped:
            self._clear_since = frame.t if self._clear_since is None else self._clear_since
            if frame.t - self._clear_since < self.cfg.clear_time_s and self._moving(wanted.base):
                state = self.state
        else:
            self._clear_since = None
        if state.stopped and not held:
            log.warning("t=%.2f SAFETY STOP: %s", frame.t, state.reason)
        elif held and not state.stopped:
            log.info("t=%.2f safety stop cleared", frame.t)
        self.state = state
        if state.speed_scale >= 1.0:
            return wanted, state
        k, base = state.speed_scale, wanted.base
        return replace(wanted, base=Twist2D(k * base.vx, k * base.vy, k * base.wz)), state

    # ---- assessment ----
    def _moving(self, twist: Twist2D) -> bool:
        return (hypot(twist.vx, twist.vy) >= self.cfg.motion_epsilon_mps
                or abs(twist.wz) >= self.cfg.motion_epsilon_radps)

    def _assess(self, t: float, twist: Twist2D, dock_mask: StationFrame | None,
                extra: float = 0.0) -> SafetyState:
        """``extra`` widens every stop threshold (used while a stop is being held)."""
        cfg = self.cfg
        speed = hypot(twist.vx, twist.vy)
        if not self._moving(twist):
            return SafetyState(1.0, False)            # nothing to supervise: no base motion asked
        scans = list(self._scans.values())
        if not scans or any(t - scan.t > cfg.scan_timeout_s for scan in scans) \
                or len(scans) < len(self._lidar.mounts):
            return SafetyState(0.0, True, "lidar data missing or stale")
        points = scan_points_in_base(scans, self._min_range, self._lidar.range_max_m)
        if len(points) == 0:
            return SafetyState(1.0, False)
        edge = self._edge_distance(points)
        if dock_mask is not None:
            expected = self._inside_conveyor(points, dock_mask)
            if expected.any() and edge[expected].min() < cfg.dock_min_clearance_m + extra:
                return SafetyState(0.0, True, f"conveyor {dock_mask.name} only "
                                   f"{1e3 * edge[expected].min():.0f} mm from the base",
                                   float(edge[expected].min()))
            points, edge = points[~expected], edge[~expected]
            if len(points) == 0:
                return SafetyState(1.0, False)
        if abs(twist.wz) >= cfg.motion_epsilon_radps:
            swept = self._half * np.sqrt(2) + cfg.rotation_margin_m + extra
            centre = np.linalg.norm(points, axis=1)
            if centre.min() < swept:
                return SafetyState(0.0, True, f"obstacle {1e3 * centre.min():.0f} mm from the "
                                   "base centre while turning", float(edge.min()))
        if speed < cfg.motion_epsilon_mps:
            return SafetyState(1.0, False, nearest_m=float(edge.min()))
        # Corridor swept by the footprint along the commanded direction of travel.
        u = np.array([twist.vx, twist.vy]) / speed
        along = points @ u
        across = np.abs(points @ np.array([-u[1], u[0]]))
        reach = self._half * (abs(u[0]) + abs(u[1]))          # footprint extent along u
        ahead = (along > 0) & (across <= reach + cfg.corridor_margin_m)
        if not ahead.any():
            return SafetyState(1.0, False)
        gap = float(np.maximum(along[ahead] - reach, 0.0).min())
        scale = float(np.clip((gap - cfg.stop_distance_m - extra)
                              / (cfg.slow_distance_m - cfg.stop_distance_m), 0.0, 1.0))
        if scale <= 0.0:
            return SafetyState(0.0, True, f"obstacle {1e3 * gap:.0f} mm ahead (stop distance "
                               f"{1e3 * cfg.stop_distance_m:.0f} mm)", gap)
        return SafetyState(scale, False, "slowing for an obstacle ahead" if scale < 1 else "", gap)

    def _edge_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance of each base-frame point from the square footprint (0 inside it)."""
        outside = np.maximum(np.abs(points) - self._half, 0.0)
        return np.linalg.norm(outside, axis=1)

    def _inside_conveyor(self, points: np.ndarray, station: StationFrame) -> np.ndarray:
        margin = self.cfg.dock_mask_margin_m
        rel = points - station.center[:2]
        along, across = rel @ station.belt_dir[:2], rel @ station.lateral[:2]
        return (np.abs(along) <= _CONVEYOR["length_m"] / 2 + margin) & \
            (np.abs(across) <= _CONVEYOR["width_m"] / 2 + margin)
