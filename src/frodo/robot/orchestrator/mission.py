"""Mission orchestrator: owns the `Mode` and runs the robot modules each control tick.

    command = orchestrator.step(frame)        # SensorFrame in, RobotCommand out

Tick order:
    localization -> line-controller poll -> mode logic (navigation or perception +
    arm controller) -> RobotCommand

Mode flow of the nominal mission:
    IDLE -> NAVIGATE -> DOCK -> MANIPULATE (cycles until the batch is done) -> STOW
         -> NAVIGATE -> DOCK -> MANIPULATE -> STOW -> DONE

What the robot believes about kits comes only from the line controller (arrival ids,
inspection results, accepted counts) and from perception. ASSEMBLY_DONE is sent when the
robot has let go of a ring inside the work window; it is a claim, not acceptance.

The safety monitor (`frodo.robot.safety`) filters the base command last, from raw lidar
data. While docking it is told which conveyor the robot is closing in on.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ...cell import CELL, STATIONS
from ...config import get_config
from ...interfaces import Mode, RobotCommand, SensorFrame, Twist2D
from ...log import get_logger
from ..frames import StationFrame
from ..localization import Localizer
from ..manipulation.arm_controller import ArmCommand, ArmController
from ..navigation.navigator import Navigator, NavStatus
from ..perception import Perception
from ..plc import RESULT_ACCEPTED, LineClient, LineError, LineStatus
from ..safety.monitor import SafetyMonitor

log = get_logger(__name__)
WORK_EXIT_X = CELL["conveyor"]["work_exit_local_x_m"]
LEAVE_POLICIES = ("after_accepted", "after_done")


@dataclass(frozen=True)
class RobotEvent:
    """Something the robot did that the mission runner records as evidence."""
    kind: str               # "dock" | "assembly_done"
    station: str
    kit_id: int | None = None


class Orchestrator:
    def __init__(self, kin_model, line: LineClient, config=None):
        cfg = self.config = config or get_config()
        self.mission = cfg.mission
        if self.mission.leave_policy not in LEAVE_POLICIES:
            raise ValueError(f"mission.leave_policy must be one of {LEAVE_POLICIES}")
        self.dt = cfg.sim.control_dt_s
        self.order = list(self.mission.station_order)
        self.batch = self.mission.assemblies_per_station
        self.line = line
        self.localizer = Localizer(cfg)
        self.navigator = Navigator(cfg)
        self.perception = Perception(cfg, kin_model=kin_model)
        self.arm = ArmController(kin_model, cfg)
        self.safety = SafetyMonitor(cfg) if cfg.safety.enabled else None
        self.mode = Mode.IDLE
        self.station: str | None = None           # station being driven to or worked at
        self.station_frame: StationFrame | None = None
        self.events: list[RobotEvent] = []        # drained by the mission runner
        self.attempts: list[dict] = []            # robot-side record of every cycle
        self.line_status: LineStatus | None = None
        self._next_poll = 0.0
        self._seen_arrival = {s: 0 for s in self.order}
        self._seen_inspection = {s: 0 for s in self.order}
        self._open: dict[str, set[int]] = {s: set() for s in self.order}  # done, not inspected
        self._pending: deque[int] = deque()       # arrived kit ids waiting for the arm
        self._current: dict | None = None         # attempt in progress
        self._handled: set[int] = set()           # perception tracks already worked on

    # ---- line bookkeeping ----
    def _accepted(self, station: str) -> int:
        if self.line_status is None:
            return 0
        return self.line_status.stations[STATIONS[station].id].accepted

    def _settled(self, station: str) -> bool:
        """The line controller shows the full batch accepted."""
        return self._accepted(station) >= self.batch

    def _covered(self, station: str) -> bool:
        """Accepted plus reported-done kits still on their way to inspection fill the batch."""
        return self._accepted(station) + len(self._open[station]) >= self.batch

    def _goal(self) -> str | None:
        """The station that needs the robot next, or None when none does right now."""
        early = self.mission.leave_policy == "after_done"
        for name in self.order:
            if not (self._covered(name) if early else self._settled(name)):
                return name
        return None

    def _poll(self, t: float) -> None:
        if t < self._next_poll:
            return
        self._next_poll = t + 1.0 / self.mission.plc_poll_hz
        try:
            self.line_status = status = self.line.poll()
        except (OSError, EOFError, RuntimeError) as exc:
            log.warning("t=%.2f line controller poll failed: %s", t, exc)
            return
        for name in self.order:
            st = status.stations[STATIONS[name].id]
            if st.latest_kit != self._seen_arrival[name]:
                self._seen_arrival[name] = st.latest_kit
                if name == self.station and self.mode is Mode.MANIPULATE:
                    self._pending.append(st.latest_kit)
                    log.info("t=%.2f kit %d arrived at %s", t, st.latest_kit, name)
                else:
                    log.warning("t=%.2f kit %d arrived at %s while the robot is not working "
                                "there", t, st.latest_kit, name)
            if st.inspected_kit != self._seen_inspection[name]:
                self._seen_inspection[name] = st.inspected_kit
                self._open[name].discard(st.inspected_kit)
                log.info("t=%.2f line: kit %d at %s %s (accepted %d, rejected %d)", t,
                         st.inspected_kit, name,
                         "ACCEPTED" if st.result == RESULT_ACCEPTED else "REJECTED",
                         st.accepted, st.rejected)

    def _command_line(self, t: float, what: str, send) -> bool:
        try:
            send()
            return True
        except LineError as exc:
            log.error("t=%.2f %s failed: %s", t, what, exc)
            return False

    # ---- tick ----
    def step(self, frame: SensorFrame) -> RobotCommand:
        t = frame.t
        pose = self.localizer.update(frame)
        self._poll(t)
        base, arm = Twist2D(), None

        if self.mode is Mode.IDLE:
            self._head_for(self._goal(), t)
        if self.mode in (Mode.NAVIGATE, Mode.DOCK):
            base = self._navigate(pose, t)
        elif self.mode is Mode.MANIPULATE:
            base, _ = self.navigator.step(pose, self.dt)      # docked: holds at rest
            arm = self._manipulate(frame, t)
        elif self.mode is Mode.STOW:
            base, _ = self.navigator.step(pose, self.dt)
            arm = self.arm.step(frame.joints, None, t, self.dt)
            if self.arm.stowed:
                self._head_for(self._goal(), t)
        if arm is None:
            arm = self.arm.step(frame.joints, None, t, self.dt)
        command = RobotCommand(base=base, arm_q=arm.arm_q, gripper=arm.gripper,
                               request_grasp=arm.request_grasp, release_grasp=arm.release_grasp)
        if self.safety is not None:
            # The conveyor being docked at is expected in the lidar data; nothing else is.
            mask = StationFrame.from_pose(self.station, pose) if self.mode is Mode.DOCK else None
            command, _ = self.safety.filter(frame, command, mask)
        return command

    def _set_mode(self, mode: Mode, t: float) -> None:
        if mode is not self.mode:
            log.info("t=%.2f mode %s -> %s%s", t, self.mode.value, mode.value,
                     f" ({self.station})" if self.station else "")
            self.mode = mode

    def _head_for(self, station: str | None, t: float) -> None:
        """Arm is stowed: drive to the next station, or finish."""
        if station is None:
            if all(self._settled(name) for name in self.order):
                self._set_mode(Mode.DONE, t)
            elif self.mode is not Mode.IDLE:
                self._set_mode(Mode.IDLE, t)      # nothing to do until the line reports
            return
        self.station = station
        earlier = self.order[:self.order.index(station)]
        hold = not all(self._settled(name) for name in earlier)
        self.navigator.go_to_dock(station, hold_at_pre_dock=hold)
        self._set_mode(Mode.NAVIGATE, t)

    def _navigate(self, pose, t: float) -> Twist2D:
        goal = self._goal()
        if goal is not None and goal != self.station:
            log.warning("t=%.2f station %s needs the robot again; turning back", t, goal)
            self._head_for(goal, t)
        earlier = self.order[:self.order.index(self.station)]
        if all(self._settled(name) for name in earlier):
            self.navigator.release_hold()
        base, status = self.navigator.step(pose, self.dt)
        if status is NavStatus.DOCKING:
            self._set_mode(Mode.DOCK, t)
        elif status is NavStatus.DOCKED:
            self._on_docked(pose, t)
        elif status is NavStatus.FAILED:
            self._set_mode(Mode.FAULT, t)
        return base

    def _on_docked(self, pose, t: float) -> None:
        name, sid = self.station, STATIONS[self.station].id
        self.events.append(RobotEvent("dock", name))
        self.station_frame = StationFrame.from_pose(name, pose)
        self.arm.set_station(self.station_frame)
        self.arm.unstow()
        self.perception.start(self.station_frame)
        self._pending.clear()
        self._command_line(t, f"READY at {name}", lambda: self.line.ready(sid))
        self._set_mode(Mode.MANIPULATE, t)

    def _manipulate(self, frame: SensorFrame, t: float) -> ArmCommand:
        self.perception.update(frame)
        if self.arm.waiting:
            if self._current is not None:
                self._close_attempt()
            if self._pending:
                self._start_cycle(t)
            else:
                goal = self._goal()
                done_everywhere = all(self._settled(name) for name in self.order)
                if done_everywhere or (goal is not None and goal != self.station):
                    self._leave(t)
        believed = self.arm.held_ring_estimate
        if believed is not None:
            seen = self.perception.held_ring(frame, believed)
            if seen is not None:
                self.arm.measure_held_ring(*seen)
        track = self._current["track_id"] if self._current else None
        obs = self.perception.get(track)
        command = self.arm.step(frame.joints, obs, t, self.dt)
        if self.arm.released_event and obs is not None:
            self._report_done(obs, t)
        return command

    def _start_cycle(self, t: float) -> None:
        kit_id = self._pending.popleft()
        lo, hi = self.mission.pick_window_local_x_m
        candidates = [obs for obs in self.perception.observations()
                      if obs.track_id not in self._handled
                      and lo <= self.station_frame.local_x(obs.bottle_in_base) <= hi]
        attempt = {"kit_id": kit_id, "station": self.station, "started_t": t, "track_id": None,
                   "reported_done": False, "failure": None}
        self.attempts.append(attempt)
        if not candidates:
            attempt["failure"] = "no tracked kit inside the pick window"
            log.warning("t=%.2f kit %d: %s; skipped", t, kit_id, attempt["failure"])
            return
        # the kit that arrived first is the one furthest downstream
        obs = max(candidates, key=lambda o: self.station_frame.local_x(o.bottle_in_base))
        attempt["track_id"] = obs.track_id
        self._handled.add(obs.track_id)
        self._current = attempt
        self.arm.begin(obs.track_id, t)
        log.info("t=%.2f cycle start: kit %d (track %d), bottle at local x %.3f m", t, kit_id,
                 obs.track_id, self.station_frame.local_x(obs.bottle_in_base))

    def _report_done(self, obs, t: float) -> None:
        attempt, name = self._current, self.station
        bottle_x = self.station_frame.local_x(obs.bottle_in_base)
        if bottle_x > WORK_EXIT_X - self.mission.done_margin_m:
            attempt["failure"] = "ring released after the work window"
            log.warning("t=%.2f kit %d: %s; not reported", t, attempt["kit_id"], attempt["failure"])
            return
        if self._command_line(t, f"ASSEMBLY_DONE kit {attempt['kit_id']}", lambda: self.line.
                              assembly_done(STATIONS[name].id, attempt["kit_id"])):
            attempt["reported_done"] = True
            self._open[name].add(attempt["kit_id"])
            self.events.append(RobotEvent("assembly_done", name, attempt["kit_id"]))
            log.info("t=%.2f ASSEMBLY_DONE kit %d at %s (%.2f s after cycle start)", t,
                     attempt["kit_id"], name, t - attempt["started_t"])

    def _close_attempt(self) -> None:
        cycle = self.arm.last_cycle
        if cycle is not None and cycle.track_id == self._current["track_id"]:
            self._current.update(grasped_t=cycle.grasped_t, released_t=cycle.released_t,
                                 ended_t=cycle.ended_t,
                                 release_radial_m=cycle.release_radial_m,
                                 held_ring_looks=cycle.held_ring_looks)
            self._current["failure"] = self._current["failure"] or cycle.failure
        self._current = None

    def _leave(self, t: float) -> None:
        name = self.station
        self.perception.stop()
        self._command_line(t, f"NOT_READY at {name}",
                           lambda: self.line.ready(STATIONS[name].id, False))
        self.arm.stow()
        self._set_mode(Mode.STOW, t)
