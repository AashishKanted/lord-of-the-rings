"""A person walks into the robot's work zone during a ring transfer, waits, and leaves.

The person is a body in the scene (a 1.7 m tall cylinder, legs-to-shoulders) that walks on
the floor. The robot is not told anything: it only sees the person as lidar returns and
stops because something is inside its work zone, as it would for any object.

Timing (config [scenarios.human_stop]): the walk starts so that the person crosses into
the zone ``enter_after_arrival_s`` after kit ``kit_id`` entered the work window, i.e. while
the robot is moving that kit's ring; the person stays inside for ``duration_s`` (2-4 s,
from crossing in to crossing out). An operator presses "acknowledge" ``ack_delay_s`` after
the person has left.

Evidence written here (customer side, from the true positions): ``human_enter`` and
``human_exit`` when the person's body crosses the zone boundary (1 m from the robot's
footprint). ``stop_command`` and ``resume`` come from the robot.
"""
from __future__ import annotations

from math import cos, sin

import mujoco
import numpy as np

from ..cell import STATIONS
from ..log import get_logger
from .base import Scenario

log = get_logger(__name__)
RADIUS, HEIGHT = 0.18, 1.70


def footprint_distance(base_pose, point_xy, half: float) -> float:
    """Distance from a point to the robot's square footprint (0 inside)."""
    x, y, yaw = base_pose
    dx, dy = point_xy[0] - x, point_xy[1] - y
    local = np.array([cos(yaw) * dx + sin(yaw) * dy, -sin(yaw) * dx + cos(yaw) * dy])
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


class HumanStop(Scenario):
    name = "human_stop"

    def __init__(self, config):
        self.cfg = c = config.scenarios.human_stop
        self.time_limit_s = c.time_limit_s
        self.half = config.safety.footprint_m / 2
        self.zone = config.safety.human_zone_radius_m
        # Customer rule: 2-4 s. Kept 0.05 s inside so that the crossing times, which are
        # sampled on the 10 ms control tick, cannot land just outside it.
        if not 2.05 <= c.duration_s <= 3.95:
            raise ValueError("scenarios.human_stop.duration_s must be 2.05-3.95 s "
                             "(customer rule: 2-4 s)")
        st = STATIONS[c.station]
        x, y, yaw = st.dock
        side = np.array([-sin(yaw), cos(yaw)])          # the robot's left, seen from the dock
        back = -np.array([cos(yaw), sin(yaw)])
        self.start_xy = np.array([x, y]) + c.offset_back_m * back + c.start_side_m * side
        self.stand_xy = np.array([x, y]) + c.offset_back_m * back + c.stand_side_m * side
        walk = np.linalg.norm(self.stand_xy - self.start_xy)
        # where along the walk the body crosses the zone boundary (from the true dock pose)
        crossing = c.start_side_m - (self.half + self.zone + RADIUS)
        self._to_boundary_s = crossing / c.walk_speed_mps
        inside_walk_s = (walk - crossing) / c.walk_speed_mps
        self.stand_s = c.duration_s - 2 * inside_walk_s
        if self.stand_s < 0:
            raise ValueError("duration_s is too short for the walk in and out")
        self.phase, self.t0 = "waiting", None
        self.times: dict[str, float] = {}

    def add_to_scene(self, spec) -> None:
        body = spec.worldbody.add_body(name="person", mocap=True, pos=[*self.start_xy, 0.0])
        body.add_geom(name="person_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[RADIUS, HEIGHT / 2, 0], pos=[0, 0, HEIGHT / 2],
                      rgba=[0.2, 0.35, 0.8, 1])
        body.add_geom(name="person_head", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.11, 0, 0],
                      pos=[0, 0, HEIGHT + 0.08], rgba=[0.9, 0.75, 0.6, 1])

    def tick(self, model, data, robot, line) -> None:
        c, t = self.cfg, data.time
        mocap = model.body("person").mocapid[0]
        if self.phase == "waiting":
            kit = next((k for k in line.kits if k["kit_id"] == c.kit_id), None)
            if kit and kit["arrived_t"] is not None and \
                    t >= kit["arrived_t"] + c.enter_after_arrival_s - self._to_boundary_s:
                self.phase, self.t0 = "walking_in", t
                self.times["walk_start"] = t
                log.info("t=%.2f person starts walking toward the robot", t)
        elif self.phase == "walking_in":
            if self._walk(data, mocap, self.start_xy, self.stand_xy, t - self.t0):
                self.phase, self.t0 = "standing", t
        elif self.phase == "standing" and t - self.t0 >= self.stand_s:
            self.phase, self.t0 = "walking_out", t
        elif self.phase == "walking_out":
            if self._walk(data, mocap, self.stand_xy, self.start_xy, t - self.t0):
                self.phase = "gone"
        # customer-side evidence from the true positions
        inside = footprint_distance(data.qpos[:3], data.mocap_pos[mocap][:2], self.half) \
            - RADIUS < self.zone
        if inside and "enter" not in self.times:
            self.times["enter"] = t
            line.events.human_enter(t)
            log.info("t=%.2f person entered the work zone", t)
        elif not inside and "enter" in self.times and "exit" not in self.times:
            self.times["exit"] = t
            line.events.human_exit(t)
            log.info("t=%.2f person left the work zone", t)
        if "exit" in self.times and "ack" not in self.times and \
                t >= self.times["exit"] + c.ack_delay_s:
            self.times["ack"] = t
            log.info("t=%.2f operator presses acknowledge", t)
            robot.operator_acknowledge()

    def _walk(self, data, mocap, a, b, elapsed) -> bool:
        length = np.linalg.norm(b - a)
        s = min(elapsed * self.cfg.walk_speed_mps, length)
        data.mocap_pos[mocap][:2] = a + (b - a) * s / length
        return s >= length

    def report(self) -> dict:
        return {"person": {"kit_id": self.cfg.kit_id, "station": self.cfg.station,
                           "enter_after_arrival_s": self.cfg.enter_after_arrival_s,
                           "duration_s": self.cfg.duration_s,
                           "walk_speed_mps": self.cfg.walk_speed_mps,
                           "stand_xy": [round(float(v), 3) for v in self.stand_xy],
                           "times": {k: round(v, 3) for k, v in self.times.items()}}}
