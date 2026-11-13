"""Arm controller: stow / unstow and the pick-and-place cycle on a moving kit.

Everything is in the robot base frame. Inputs each tick are the measured joints and the
tracked `KitObservation` of the kit being worked on; the output is joint targets and a
gripper command. The controller never sees simulator state.

Cartesian IK tracker: a phase machine picks a tool goal relative to a moving reference
(the ring's grasp point, then the bottle). A speed/acceleration-limited setpoint (`Slew`)
moves toward that goal in the moving frame, the belt velocity is added as a short lead,
and differential IK (`ArmIK`) turns the result into joint targets.

    STOWED -UNSTOWING-> WAITING -begin()-> PREGRASP -> DESCEND -> CLOSE -> LIFT
        -> TRANSFER -> LOWER (closed-loop centring) -> RELEASE -> RETREAT -> RETURN -> WAITING
    WAITING -STOWING-> STOWED

Stow and unstow are joint-space moves through the home posture.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...cell import CELL, KIT_TOP_Z
from ...config import get_config
from ...interfaces import JointState
from ...log import get_logger
from ..description import ARM_HOME, ARM_JOINTS, ARM_STOWED
from ..frames import StationFrame
from ..perception.types import KitObservation
from .ik import ArmFK, ArmIK, tool_rotation_from_axes
from .motion import JointMove, Slew

log = get_logger(__name__)
NECK_TOP = CELL["bottle"]["height_m"]
_B, _R = CELL["bottle"], CELL["ring"]
# Height of the ring centre above the bottle base when the ring rests level on the shoulder
# cone (where the cone's radius equals the ring's inner radius).
SEAT_Z = (_B["body_height_m"] + _B["shoulder_height_m"]
          * (_B["body_diameter_m"] - _R["inner_diameter_m"])
          / (_B["body_diameter_m"] - _B["neck_diameter_m"]) + _R["thickness_m"] / 2)
_KIT = CELL["kit"]
RING_AHEAD_OF_BOTTLE = _KIT["ring_xy_in_kit_m"][0] - _KIT["bottle_xy_in_kit_m"][0]
REACHED_M = 1e-4


class ArmPhase(Enum):
    STOWED = "stowed"
    UNSTOWING = "unstowing"
    WAITING = "waiting"        # hovering over the ring lane, ready for `begin`
    PREGRASP = "pregrasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    TRANSFER = "transfer"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"
    RETURN = "return"
    STOWING = "stowing"
    PUTBACK = "putback"        # giving a kit up: carry the held ring back over its nest ...
    PUTDOWN = "putdown"        # ... and lower it there


CARRYING = (ArmPhase.LIFT, ArmPhase.TRANSFER, ArmPhase.LOWER)
CYCLE_PHASES = (ArmPhase.PREGRASP, ArmPhase.DESCEND, ArmPhase.CLOSE, *CARRYING,
                ArmPhase.PUTBACK, ArmPhase.PUTDOWN, ArmPhase.RELEASE, ArmPhase.RETREAT)


@dataclass
class ArmCommand:
    arm_q: np.ndarray
    gripper: float                # 0 open .. 1 closed
    request_grasp: bool = False
    release_grasp: bool = False


@dataclass
class CycleResult:
    track_id: int
    started_t: float
    grasped_t: float | None = None
    released_t: float | None = None
    ended_t: float | None = None
    release_radial_m: float | None = None   # ring-over-neck error at release, as estimated
    held_ring_looks: int = 0                # camera measurements of the ring in the hand
    failure: str | None = None


class ArmController:
    def __init__(self, kin_model, config=None):
        """``kin_model`` is the robot's own kinematic model (MuJoCo model of the robot)."""
        cfg = config or get_config()
        self.m = cfg.manipulation
        self.model = kin_model
        self.phase = ArmPhase.STOWED
        self.station: StationFrame | None = None
        self.ik: ArmIK | None = None
        self.fk: ArmFK | None = None
        self.q_ready: np.ndarray | None = None
        self.hover: np.ndarray | None = None
        self.cycle: CycleResult | None = None     # the cycle in progress
        self.last_cycle: CycleResult | None = None
        self.released_event = False               # True for the one tick the ring is let go
        self._q_cmd = ARM_STOWED.copy()
        self._move: JointMove | None = None
        self._slew: Slew | None = None
        self._origin = np.zeros(3)
        self._reference = "base"        # what `_origin` rides on: "base", "ring" or "bottle"
        self._phase_t0 = 0.0
        self._last_gripper = 0.0
        self._gripper_cmd = self._open_cmd = self.m.grasps[self.m.grasp].preshape / 255.0
        self._putting_back = False
        self._stop = None                       # (slew, target) of a safety stop in progress
        self._blocked_s, self._limit_warned = 0.0, False
        self._tcp_history: deque[tuple[float, np.ndarray]] = deque()
        self._looks: list[np.ndarray] = []
        if self.m.grasp not in self.m.grasps:
            raise ValueError(f"manipulation.grasp '{self.m.grasp}' is not one of "
                             f"{sorted(self.m.grasps)}")
        self.g = self.m.grasps[self.m.grasp]      # geometry of the selected grasp style
        self._open = self.g.preshape / 255.0
        self._closed = self.m.grip_closed / 255.0

    # ---- station setup ----
    def set_station(self, station: StationFrame) -> None:
        """Prepare IK and the waiting pose for a station (call when docked, arm stowed)."""
        self.station = station
        # horizontal unit vector from the robot toward the belt, perpendicular to belt travel
        toward = station.lateral if station.center[:2] @ station.lateral[:2] > 0 else -station.lateral
        self._toward_belt = toward
        rot = tool_rotation_from_axes(station.belt_dir, toward)
        q = self.model.qpos0.copy()
        q[[self.model.joint(j).qposadr[0] for j in ARM_JOINTS]] = ARM_HOME
        self.ik, self.fk = ArmIK(self.model, q, rot), ArmFK(self.model, rot)
        # tool height that keeps the fingers clear of passing bottle tops (safety stop)
        self._safe_z = KIT_TOP_Z + NECK_TOP + self.m.carry_clearance_m + self.m.stop_clearance_m
        # Tool point relative to the ring centre at the grasp. The tool point is where the
        # pads meet when closed; on an open 2F-85 the pads sit higher and further out, which
        # the per-grasp offsets account for.
        self._grasp_from_ring = (self.g.offset_along_belt_m * station.belt_dir
                                 + self.g.offset_toward_belt_m * toward
                                 + [0, 0, self.g.offset_up_m])
        ring_x_at_entry = CELL["conveyor"]["work_entry_local_x_m"] + RING_AHEAD_OF_BOTTLE
        self.hover = station.point(ring_x_at_entry + self.m.hover_ahead_m + self.g.offset_along_belt_m,
                                   _KIT["ring_xy_in_kit_m"][1], KIT_TOP_Z + self.m.hover_height_m)
        # Each round also re-centres the IK's posture preference on the result, so repeat
        # until the waiting pose is actually reached.
        for _ in range(6):
            self.q_ready = self.ik.solve_static(self.hover).copy()
            if np.linalg.norm(self.ik.tcp() - self.hover) < 1e-4:
                break
        else:
            log.warning("waiting pose at station %s only reached to %.1f mm", station.name,
                        1e3 * np.linalg.norm(self.ik.tcp() - self.hover))

    # ---- requests ----
    @property
    def waiting(self) -> bool:
        return self.phase is ArmPhase.WAITING

    @property
    def stowed(self) -> bool:
        return self.phase is ArmPhase.STOWED

    def unstow(self) -> None:
        if self.phase is not ArmPhase.STOWED or self.q_ready is None:
            raise RuntimeError(f"cannot unstow from {self.phase} (station set: {self.station})")
        self._move = JointMove([self._q_cmd, ARM_HOME, self.q_ready], self.m.joint_speed_radps)
        self.phase = ArmPhase.UNSTOWING

    def stow(self) -> None:
        if self.phase is ArmPhase.STOWED:
            return
        if self.phase is not ArmPhase.WAITING:
            raise RuntimeError(f"cannot stow from {self.phase}")
        self._move = JointMove([self._q_cmd, ARM_HOME, ARM_STOWED], self.m.joint_speed_radps)
        self.phase = ArmPhase.STOWING

    def begin(self, track_id: int, t: float) -> None:
        """Start a pick-and-place cycle on a tracked kit."""
        if self.phase is not ArmPhase.WAITING:
            raise RuntimeError(f"cannot begin a cycle from {self.phase}")
        self.cycle = CycleResult(track_id, t)
        self._putting_back = False
        self._grasp_offset, self._trim, self._centred_s = np.zeros(3), np.zeros(3), 0.0
        self._gripped_ticks, self._blocked_s, self._limit_warned = 0, 0.0, False
        self._looks: list[np.ndarray] = []      # measured tool-to-ring offsets, ring in hand
        self._enter(ArmPhase.PREGRASP, t)

    # ---- safety stop ----
    def hold(self, dt: float) -> ArmCommand:
        """Safety stop. The belt keeps running, so a tool that is low over the kits would be
        hit by the next bottle: it first rises straight up (fast, finished well inside the
        1 s settle limit) to clear the bottle tops, then holds still. A tool that is already
        high just holds. The controller's own state does not advance meanwhile."""
        if self._stop is None:
            tcp = self.ik.tcp()
            low = self.phase in CYCLE_PHASES and tcp[2] < self._safe_z
            self._stop = (Slew(tcp, self.m.stop_retract_speed_mps, self.m.stop_retract_accel_mps2),
                          np.array([tcp[0], tcp[1], self._safe_z]) if low else None)
        slew, target = self._stop
        if target is not None and np.linalg.norm(slew.p - target) > REACHED_M:
            self._q_cmd = self.ik.step(self._limit(slew.step(target, dt)), dt)
        return ArmCommand(self._q_cmd.copy(), self._gripper_cmd)

    @property
    def needs_grasp(self) -> bool:
        """In a cycle, but the ring is not held yet."""
        return self.phase in (ArmPhase.PREGRASP, ArmPhase.DESCEND, ArmPhase.CLOSE)

    def resume(self, t: float) -> None:
        """Continue after a stop. The kit kept moving (and the tool may have risen), so the
        setpoint is re-anchored where the arm is now and moves on at the normal limits. A
        ring that was being lowered goes back to centring above the bottle first."""
        if self.cycle is not None and self._slew is not None:
            self._slew.p = self.ik.tcp() - self._origin
            self._reference = "stale"         # next `_follow` keeps the commanded point
            self._slew.speed = 0.0
            if self.phase is ArmPhase.LOWER:
                self._trim, self._centred_s = np.zeros(3), 0.0
                self._enter(ArmPhase.TRANSFER, t)
        self._stop = None
        self._phase_t0, self._blocked_s = t, 0.0

    def abandon(self, t: float, reason: str) -> None:
        """Give up the current kit (for example: too far downstream after a stop). A held
        ring is put back on its nest; otherwise the tool lifts clear of the kit. No
        ASSEMBLY_DONE follows; the plant rejects the kit at inspection."""
        if self.cycle is None or self.phase not in CYCLE_PHASES or self._putting_back \
                or self.cycle.released_t is not None:     # nothing left to give up
            return
        log.warning("t=%.2f giving up kit (track %d) in %s: %s", t, self.cycle.track_id,
                    self.phase.value, reason)
        self.cycle.failure = reason
        self._putting_back = True
        if self.phase in CARRYING:
            self._enter(ArmPhase.PUTBACK, t)
        else:                                  # not holding it yet: straight up, then home
            self._grasp_offset = self._grasp_from_ring.copy()
            self._enter(ArmPhase.RETREAT, t)

    # ---- per-tick ----
    def step(self, joints: JointState, obs: KitObservation | None, t: float,
             dt: float) -> ArmCommand:
        command = self._step(joints, obs, t, dt)
        self._gripper_cmd = command.gripper
        return command

    def _step(self, joints: JointState, obs: KitObservation | None, t: float,
              dt: float) -> ArmCommand:
        self.released_event = False
        gripper_rate = (joints.gripper - self._last_gripper) / dt
        self._last_gripper = joints.gripper
        if self.phase is ArmPhase.STOWED:
            return ArmCommand(self._q_cmd, self._open)
        if self.phase in (ArmPhase.UNSTOWING, ArmPhase.STOWING):
            return self._joint_move(t, dt)
        if self.phase is ArmPhase.WAITING:
            return self._track(self._slew.step(self.hover, dt), dt, self._open)
        if self.phase is ArmPhase.RETURN:
            rel = self._slew.step(self.hover, dt)
            if np.linalg.norm(rel - self.hover) < REACHED_M:
                self._finish(t)
            return self._track(rel, dt, self._open)
        if obs is None:
            return self._abort(t, dt, "kit track lost")
        self._tcp_history.append((t, self.fk.tcp(joints.arm_q)))
        while t - self._tcp_history[0][0] > 0.5:
            self._tcp_history.popleft()
        # Blocked-tool watchdog: the arm is position controlled, so a tool that stays away
        # from where it was commanded is pushing on something.
        error = float(np.linalg.norm(self.fk.tcp(joints.arm_q) - self.ik.tcp()))
        self._blocked_s = self._blocked_s + dt if error > self.m.max_tool_error_m else 0.0
        if self._blocked_s > self.m.tool_error_time_s:
            return self._abort(t, dt, f"tool {1e3 * error:.0f} mm from its commanded position "
                                      "(blocked?)")
        return self._cycle(joints, obs, t, dt, gripper_rate)

    # ---- ring in hand ----
    @property
    def held_ring_estimate(self) -> np.ndarray | None:
        """Where the robot believes the held ring's centre is (base frame), while carrying."""
        if self.phase not in CARRYING or not self._tcp_history:
            return None
        return self._tcp_history[-1][1] - self._grasp_offset

    def measure_held_ring(self, t_capture: float, ring) -> None:
        """A look at the ring in the hand: update the tool-to-ring offset (mean of the looks).
        The grasp-time value assumes the ring did not move as the fingers closed."""
        if self.phase not in CARRYING or not self._tcp_history:
            return
        tcp = min(self._tcp_history, key=lambda item: abs(item[0] - t_capture))[1]
        self._looks.append(tcp - np.asarray(ring, float))
        self._grasp_offset = np.mean(self._looks, axis=0)

    # ---- joint-space moves ----
    def _joint_move(self, t: float, dt: float) -> ArmCommand:
        self._q_cmd = self._move.step(dt)
        if self._move.done:
            if self.phase is ArmPhase.UNSTOWING:
                self.ik.set_arm(self._q_cmd)
                self._slew, self._origin, self._reference = Slew(self.ik.tcp()), np.zeros(3), "base"
                self._enter(ArmPhase.WAITING, t)
            else:
                self._enter(ArmPhase.STOWED, t)
            self._move = None
        return ArmCommand(self._q_cmd, self._open)

    # ---- Cartesian cycle ----
    def _cycle(self, joints, obs, t, dt, gripper_rate) -> ArmCommand:
        m, belt = self.m, self.station.belt_dir
        ring, bottle = obs.ring_in_base, obs.bottle_in_base
        grasp_point = ring + self._grasp_from_ring
        tcp = self.fk.tcp(joints.arm_q)                          # measured tool point
        in_phase = t - self._phase_t0
        command = ArmCommand(self._q_cmd, self._open)
        phase = self.phase
        # The setpoint lives in a frame that rides on the belt: the ring's grasp point until
        # the ring is held, the bottle afterwards.
        self._follow("ring" if phase in (ArmPhase.PREGRASP, ArmPhase.DESCEND, ArmPhase.CLOSE)
                     else "bottle", grasp_point if phase in
                     (ArmPhase.PREGRASP, ArmPhase.DESCEND, ArmPhase.CLOSE) else bottle)

        if phase is ArmPhase.PREGRASP:
            goal = np.array([0, 0, m.pregrasp_height_m])
            if self._at(goal):
                self._enter(ArmPhase.DESCEND, t)
        elif phase is ArmPhase.DESCEND:
            goal = np.zeros(3)
            if self._at(goal) and np.linalg.norm(tcp - grasp_point) < m.grasp_position_tol_m:
                self._enter(ArmPhase.CLOSE, t)
        elif phase is ArmPhase.CLOSE:
            goal = np.zeros(3)
            command.gripper = self._closed
            # Grasp check from the robot's own sensors: the fingers have stopped on something
            # of about the ring wall's width (an empty gripper closes further), with the tool
            # still at the planned grasp point. Only then is the grasp constraint requested.
            gripped = (in_phase > m.grasp_settle_s and abs(gripper_rate) < m.grasp_still_rate_per_s
                       and self.g.detect_min <= joints.gripper <= self.g.detect_max
                       and np.linalg.norm(tcp - grasp_point) < 2 * m.grasp_position_tol_m)
            command.request_grasp = gripped
            self._gripped_ticks = self._gripped_ticks + 1 if gripped else 0
            if self._gripped_ticks >= 2:      # the request has been sent for one full tick
                self._grasp_offset = tcp - ring
                self.cycle.grasped_t = t
                self._follow("bottle", bottle)
                goal = self._slew.p
                self._enter(ArmPhase.LIFT, t)
            elif in_phase > m.grasp_timeout_s:
                return self._abort(t, dt, f"no grasp detected (gripper at {joints.gripper:.2f})")
        elif phase in CARRYING:
            command.gripper = self._closed
            lift_z, release_z = NECK_TOP + m.carry_clearance_m, SEAT_Z + self.g.release_above_seat_m
            ring_goal = {ArmPhase.LIFT: RING_AHEAD_OF_BOTTLE * belt + [0, 0, lift_z],
                         ArmPhase.TRANSFER: np.array([0, 0, lift_z]),
                         ArmPhase.LOWER: np.array([0, 0, release_z])}[phase]
            goal = ring_goal + self._grasp_offset + self._trim
            if self._at(goal):
                if phase is ArmPhase.LIFT:
                    self._enter(ArmPhase.TRANSFER, t)
                elif phase is ArmPhase.TRANSFER:
                    # Above the bottle: go down only once the ring has been seen in the hand
                    # (it slides in the fingers as they close), or after a short wait.
                    if len(self._looks) >= m.held_ring_min_looks or in_phase > m.held_ring_wait_s:
                        if len(self._looks) < m.held_ring_min_looks:
                            log.warning("t=%.2f ring in hand not seen; placing on the grasp-time "
                                        "estimate", t)
                        self.cycle.held_ring_looks = len(self._looks)
                        self._enter(ArmPhase.LOWER, t)
                else:
                    # The seat only self-centres within ~2 mm, so servo the observed
                    # ring-over-neck error out (integral action) before letting go.
                    error = (bottle - (tcp - self._grasp_offset))[:2]
                    radial = float(np.linalg.norm(error))
                    self._trim[:2] += m.place_trim_gain_per_s * dt * error
                    self._centred_s = self._centred_s + dt if radial < m.release_radial_m else 0.0
                    timed_out = in_phase > m.place_timeout_s
                    if self._centred_s >= m.release_dwell_s or timed_out:
                        if timed_out:
                            log.warning("t=%.2f releasing uncentred after %.1f s: %.1f mm off",
                                        t, in_phase, 1e3 * radial)
                        self.cycle.released_t, self.cycle.release_radial_m = t, radial
                        self.released_event = True
                        self._enter(ArmPhase.RELEASE, t)
        elif phase is ArmPhase.PUTBACK:       # over the ring's nest at carrying height
            command.gripper = self._closed
            goal = RING_AHEAD_OF_BOTTLE * belt + [0, 0, NECK_TOP + m.carry_clearance_m] \
                + self._grasp_offset
            if self._at(goal):
                self._enter(ArmPhase.PUTDOWN, t)
        elif phase is ArmPhase.PUTDOWN:       # lower onto the tray, then let go (no DONE)
            command.gripper = self._closed
            goal = RING_AHEAD_OF_BOTTLE * belt + [0, 0, _R["thickness_m"] / 2
                                                  + m.putback_clearance_m] + self._grasp_offset
            if self._at(goal):
                self._enter(ArmPhase.RELEASE, t)
        elif phase is ArmPhase.RELEASE:
            goal = self._slew.p
            command.release_grasp = True
            if in_phase > m.release_settle_s:
                self._enter(ArmPhase.RETREAT, t)
        elif self._putting_back:   # RETREAT after giving up: straight up from the ring's nest
            goal = RING_AHEAD_OF_BOTTLE * belt + [0, 0, NECK_TOP + m.retreat_up_m] \
                + self._grasp_offset
            if self._at(goal):
                self._follow("base", np.zeros(3))
                self._enter(ArmPhase.RETURN, t)
        else:   # RETREAT: straight up and back from the bottle, still moving with the belt
            goal = np.array([0, 0, NECK_TOP + m.retreat_up_m]) + self._grasp_offset \
                - m.retreat_back_m * self._toward_belt
            if self._at(goal):
                self._follow("base", np.zeros(3))
                self._enter(ArmPhase.RETURN, t)
        if self.phase is ArmPhase.RELEASE:
            command.release_grasp = True
        rel = self._slew.step(goal, dt) if self.phase is not ArmPhase.RETURN else self._slew.p
        lead = obs.belt_velocity * m.belt_lead_s
        command.arm_q = self._q_cmd = self.ik.step(self._limit(self._origin + rel + lead, t), dt)
        return command

    def _abort(self, t: float, dt: float, reason: str) -> ArmCommand:
        """Give up on the current kit: open the gripper and go back to the waiting pose."""
        log.warning("t=%.2f cycle on track %d aborted in %s: %s", t, self.cycle.track_id,
                    self.phase.value, reason)
        self.cycle.failure = reason
        self._follow("base", np.zeros(3))
        self._enter(ArmPhase.RETURN, t)
        command = self._track(self._slew.step(self.hover, dt), dt, self._open)
        command.release_grasp = True
        return command

    def _finish(self, t: float) -> None:
        self.cycle.ended_t = t
        self.last_cycle, self.cycle = self.cycle, None
        self._enter(ArmPhase.WAITING, t)

    # ---- helpers ----
    def _track(self, target, dt: float, gripper: float) -> ArmCommand:
        self._q_cmd = self.ik.step(self._limit(self._origin + target), dt)
        return ArmCommand(self._q_cmd, gripper)

    def _limit(self, target, t: float | None = None) -> np.ndarray:
        """Keep the tool target inside the station workspace box (virtual walls): never below
        the kits, never beyond the belt or past the reachable stretch of it."""
        st, m = self.station, self.m
        rel = np.asarray(target, float) - st.center
        along = float(np.clip(rel[:2] @ st.belt_dir[:2], *m.workspace_local_x_m))
        across = float(np.clip(rel[:2] @ st.lateral[:2], *m.workspace_local_y_m))
        z = float(np.clip(target[2], KIT_TOP_Z - m.workspace_below_kit_top_m, m.workspace_top_z_m))
        limited = st.center + along * st.belt_dir + across * st.lateral
        limited[2] = z
        if np.linalg.norm(limited - target) > 1e-6 and not self._limit_warned:
            log.warning("%sarm target %s is outside the workspace box; limited to %s",
                        "" if t is None else f"t={t:.2f} ", np.round(target, 3),
                        np.round(limited, 3))
            self._limit_warned = True
        return limited

    def _at(self, goal) -> bool:
        return bool(np.linalg.norm(self._slew.p - goal) < REACHED_M)

    def _follow(self, reference: str, origin) -> None:
        """Set the moving reference frame's origin. Switching to another reference keeps
        the commanded point where it is; the same reference just carries it along."""
        origin = np.asarray(origin, float)
        if reference != self._reference:
            self._slew.p = self._slew.p + self._origin - origin
            self._reference = reference
        self._origin = origin

    def _enter(self, phase: ArmPhase, t: float) -> None:
        log.debug("t=%.2f arm %s -> %s", t, self.phase.value, phase.value)
        self.phase, self._phase_t0 = phase, t
