"""One object for the orchestrator: "go to dock A" -> base commands until docked.

    nav = Navigator()
    nav.go_to_dock("A")
    each tick:  command, status = nav.step(estimated_pose, dt)

Regimes:

    PLANNING  first step after go_to_dock: A* from the current pose to the pre-dock pose,
              which is ``dock_switch_distance_m`` straight back from the dock
    TRACKING  follow the path at travel speed
    HOLDING   optional: wait at rest at the pre-dock pose until `release_hold`
    DOCKING   from the pre-dock pose, slow closed-loop regulation onto the dock pose
    DOCKED    latched; zero command
    FAILED    no path; zero command

The pose given to `step` is the localizer's estimate. The navigator never sees true pose.
"""
from __future__ import annotations

from enum import Enum
from math import cos, sin

import numpy as np

from ...cell import STATIONS
from ...config import get_config
from ...interfaces import Pose2D, Twist2D
from ...log import get_logger
from ..map import CellMap, get_map
from .docking import DockingController
from .limits import VelocityLimiter
from .planner import GlobalPlanner, path_length
from .tracker import PathTracker

log = get_logger(__name__)


class NavStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    TRACKING = "tracking"
    HOLDING = "holding"
    DOCKING = "docking"
    DOCKED = "docked"
    FAILED = "failed"


def dock_pose(station: str) -> Pose2D:
    return Pose2D(*(float(v) for v in STATIONS[station].dock))


def pre_dock_pose(dock: Pose2D, distance: float) -> Pose2D:
    """The pose ``distance`` straight behind the dock (the dock yaw faces the conveyor)."""
    return Pose2D(dock.x - distance * cos(dock.yaw), dock.y - distance * sin(dock.yaw), dock.yaw)


class Navigator:
    def __init__(self, config=None, cell_map: CellMap | None = None):
        cfg = config or get_config()
        self.cfg = cfg.navigation
        self.limiter = VelocityLimiter(self.cfg)
        self.planner = GlobalPlanner(cfg, cell_map or get_map(cfg))
        self.tracker = PathTracker(cfg, self.limiter)
        self.docking = DockingController(cfg, self.limiter)
        self.status = NavStatus.IDLE
        self.station: str | None = None
        self.dock: Pose2D | None = None
        self.path: list[Pose2D] = []
        self._hold = False

    def go_to_dock(self, station: str, hold_at_pre_dock: bool = False) -> None:
        """Request a move to a station's dock; planning happens on the next `step`. With
        ``hold_at_pre_dock`` the base stops at the pre-dock pose until `release_hold`."""
        self.station, self.dock = station, dock_pose(station)
        self._hold = hold_at_pre_dock
        self.path = []
        self.docking.reset()
        self.status = NavStatus.PLANNING

    def release_hold(self) -> None:
        """Allow (or resume) the final docking after a hold at the pre-dock pose."""
        self._hold = False
        if self.status is NavStatus.HOLDING:
            log.info("hold released, docking at %s", self.station)
            self.docking.reset()
            self.status = NavStatus.DOCKING

    def cancel(self) -> None:
        """Drop the current goal; subsequent steps brake to rest."""
        self.status = NavStatus.IDLE

    def step(self, pose: Pose2D, dt: float) -> tuple[Twist2D, NavStatus]:
        """Advance one control tick with the estimated pose; returns (command, status)."""
        if self.status is NavStatus.PLANNING:
            self._plan(pose)
        if self.status is NavStatus.TRACKING:
            command, reached = self.tracker.step(pose, dt)
            if reached and self._hold:
                log.info("pre-dock reached for %s, holding", self.station)
                self.status = NavStatus.HOLDING
            elif reached:
                log.info("pre-dock reached for %s, switching to docking", self.station)
                self.status = NavStatus.DOCKING
            return command, self.status
        if self.status is NavStatus.DOCKING:
            command, docked = self.docking.step(pose, self.dock, dt)
            if docked:
                log.info("docked at %s: estimated error %.1f mm, %.2f deg", self.station,
                         1e3 * self.docking.position_error,
                         np.degrees(self.docking.yaw_error))
                self.status = NavStatus.DOCKED
            return command, self.status
        # IDLE, HOLDING, DOCKED, FAILED: brake to rest within the acceleration limits.
        return self.limiter.limit(np.zeros(2), 0.0, pose.yaw, dt), self.status

    def _plan(self, pose: Pose2D) -> None:
        goal = pre_dock_pose(self.dock, self.cfg.dock_switch_distance_m)
        self.path = self.planner.plan(pose, goal)
        if not self.path:
            log.error("no path to dock %s from %s", self.station, pose)
            self.status = NavStatus.FAILED
            return
        log.info("path to dock %s: %d waypoints, %.2f m", self.station, len(self.path),
                 path_length(self.path))
        # Arrive at the pre-dock pose already moving at docking speed.
        self.tracker.set_path(self.path, end_speed=self.cfg.dock_max_linear_mps)
        self.status = NavStatus.TRACKING
