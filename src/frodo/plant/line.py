"""Customer line: belts, kit feeder, PLC test double, seat monitoring, inspection, evidence.

Plant side. It reads simulator ground truth and never reads a robot decision: the only
robot inputs are the commands the robot writes over Modbus/TCP (READY gates the feeder;
ASSEMBLY_DONE is recorded by the controller and is not acceptance).

    with Line(model, events_path="results/x/events.jsonl") as line:
        while running:
            mujoco.mj_step(model, data)
            line.step(data)            # once per physics step

Behaviour (customer packet, "Conveyor and feeder"):

- Both belts run for the whole run at their own constant speed, whether or not they carry
  kits. ``belt_travel_m`` / ``belt_speed`` are the conveyor encoder for the sensor layer.
- A station is fed only while the robot has reported READY there over Modbus and every
  earlier station in ``mission.station_order`` has its five accepted kits. That is how the
  active station is selected; there is no separate plant-side switch.
- The first kit is released ``release_phase_s`` after the feeder is enabled; after that a
  release slot comes by every 450 mm of belt travel. A slot is used only while
  ``accepted + in_flight < 5`` (in flight = released, not yet inspected), so a rejection
  frees a slot and its replacement leaves at the next pitch.
- ``kit_arrived`` / ``PLC arrive`` happen when the bottle centre really crosses the work
  entry line; the seat monitor of that kit runs every step from then until the bottle
  crosses the inspection line, where the last measurement goes to ``PLC inspect``.
- Seat hold is counted from work entry. ``seat_verified`` is written when the hold first
  completes, also when that is after the work window (the ledger marks it late).

Event timestamps are ``data.time``: the mission must start at simulation time zero.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..cell import STATIONS
from ..config import get_config
from ..log import get_logger
from . import feeder
from .events import EventLog
from .feeder import Carrier, RunParameters
from .inspection import SeatMonitor
from .plc import LinePLC

log = get_logger(__name__)


@dataclass
class _Belt:
    speed: float
    phase_s: float
    pool: list[Carrier]
    travel: float = 0.0
    enabled: bool = False
    next_slot: float | None = None      # belt travel at which the next release slot passes
    last_release: float | None = None   # belt travel at the last release
    starved: bool = False


class Line:
    def __init__(self, model, config=None, scenario: str = "nominal", *,
                 params: RunParameters | None = None,
                 belt_speed_mps: float | dict | None = None,
                 release_phase_s: float | dict | None = None,
                 events_path: str | Path | None = None, port: int | None = None):
        """``params`` (e.g. ``feeder.draw_run_parameters(seed)``) or explicit speed/phase;
        default is the nominal customer speed with zero release phase. ``port`` overrides
        ``plant.plc_port`` (0 = any free port)."""
        self.model, self.scenario = model, scenario
        self.config = config or get_config()
        plant = self.config.plant
        self.params = params or feeder.run_parameters(belt_speed_mps, release_phase_s)
        self.order = list(self.config.mission.station_order)
        self._gain = plant.carrier_position_gain_per_s
        self._belts: dict[str, _Belt] = {}
        for name, st in STATIONS.items():
            pool = feeder.find_pool(model, st, self.config)
            if len(pool) < feeder.min_pool_size():
                log.warning("station %s has %d pooled kits; %d are needed to keep the pitch",
                            name, len(pool), feeder.min_pool_size())
            self._belts[name] = _Belt(self.params.belt_speed_mps[name],
                                      self.params.release_phase_s[name], pool)
        self.plc = LinePLC(plant.plc_host, plant.plc_port if port is None else port)
        self.events = EventLog(events_path)
        self.kits: list[dict] = []   # ledger: one record per released kit, in release order
        self._next_kit_id = plant.first_kit_id
        self._raw = None
        if events_path and plant.seat_raw_log:
            self.raw_path = Path(events_path).with_name("seat_samples.jsonl")
            self._raw = self.raw_path.open("w", encoding="utf-8")
        self._initialised = False

    # ---- lifecycle ----
    def start(self) -> Line:
        self.plc.start()
        return self

    def close(self) -> None:
        self.plc.close()
        self.events.close()
        if self._raw:
            self._raw.close()
            self._raw = None

    def __enter__(self) -> Line:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def port(self) -> int:
        return self.plc.port

    # ---- conveyor encoder (read-only, for the simulator's sensor layer) ----
    def belt_speed(self, station: str, data=None) -> float | None:
        """Belt surface speed in m/s; None for an unknown station. Shape matches
        ``SensorSuite.set_belt_speed_source(fn(station_name, data))``."""
        belt = self._belts.get(station)
        return None if belt is None else belt.speed

    def belt_travel_m(self, station: str) -> float:
        """Belt travel since the start of the run, as of the last ``step``."""
        return self._belts[station].travel

    def conveyors(self) -> dict[str, tuple[float, float]]:
        """station name -> (travel_m, speed_mps)."""
        return {name: (b.travel, b.speed) for name, b in self._belts.items()}

    # ---- status ----
    def counts(self) -> dict[str, dict[str, int]]:
        """Accepted/rejected per station from the controller, plus kits in flight."""
        state = self.plc.state()
        return {name: {"accepted": state.accepted[st.id], "rejected": state.rejected[st.id],
                       "in_flight": self._in_flight(self._belts[name])}
                for name, st in STATIONS.items()}

    @staticmethod
    def _in_flight(belt: _Belt) -> int:
        return sum(c.on_belt and c.record["inspected_t"] is None for c in belt.pool)

    # ---- simulation ----
    def step(self, data) -> None:
        """Advance plant logic by one simulation step (call after each ``mj_step``)."""
        if not self._initialised:
            self._park_all(data)
        state = self.plc.state()
        t = data.time
        for name, belt in self._belts.items():
            st = STATIONS[name]
            belt.travel = belt.speed * t
            earlier = self.order[:self.order.index(name)] if name in self.order else []
            enabled = (name in self.order and state.ready[st.id]
                       and all(state.accepted[STATIONS[p].id] >= feeder.BATCH for p in earlier))
            if enabled and not belt.enabled:
                belt.next_slot = belt.travel + belt.phase_s * belt.speed
                if belt.last_release is not None:
                    belt.next_slot = max(belt.next_slot, belt.last_release + feeder.PITCH_M)
                log.info("t=%.3f feeder %s enabled", t, name)
            belt.enabled = enabled
            if enabled and belt.travel >= belt.next_slot:
                if state.accepted[st.id] + self._in_flight(belt) < feeder.BATCH:
                    self._release(data, belt, name)
                belt.next_slot += feeder.PITCH_M
            for carrier in belt.pool:
                self._step_carrier(data, belt, carrier)

    def _park_all(self, data) -> None:
        """Initial fixture state: every pooled kit in its magazine slot, ring on its nest."""
        for belt in self._belts.values():
            for c in belt.pool:
                if abs(c.x(data) - c.slot_x) > 1e-6:
                    c.place_kit(data, c.slot_x, 0.0)
                    c.place_ring_on_nest(data, c.slot_x, 0.0)
        self._initialised = True

    def _release(self, data, belt: _Belt, name: str) -> None:
        carrier = next((c for c in belt.pool if not c.on_belt and c.ring_on_nest), None)
        if carrier is None:
            if not belt.starved:
                log.warning("t=%.3f feeder %s has no free kit; release slot passes empty",
                            data.time, name)
            belt.starved = True
            return
        belt.starved = False
        if self._next_kit_id > 65535:
            raise RuntimeError("kit ids exhausted (Modbus register is 16 bit)")
        kit_id, self._next_kit_id = self._next_kit_id, self._next_kit_id + 1
        carrier.place_kit(data, feeder.INFEED_X, belt.speed)
        carrier.place_ring_on_nest(data, feeder.INFEED_X, belt.speed)
        carrier.on_belt, carrier.release_travel = True, belt.travel
        carrier.record = {"kit_id": kit_id, "station": name, "carrier": carrier.name,
                          "belt_speed_mps": belt.speed, "released_t": data.time,
                          "arrived_t": None, "seat_verified_t": None, "cycle_s": None,
                          "seat_in_window": None, "inspected_t": None, "robot_done": None,
                          "accepted": None}
        self.kits.append(carrier.record)
        belt.last_release = belt.travel
        log.info("t=%.3f release kit %d at %s (%s)", data.time, kit_id, name, carrier.name)

    def _step_carrier(self, data, belt: _Belt, c: Carrier) -> None:
        x = c.x(data)
        if not c.on_belt:
            if not c.ring_on_nest and not c.ring_held(data):
                c.place_ring_on_nest(data, c.slot_x, 0.0)
            data.ctrl[c.drive] = self._gain * (c.slot_x - x)
            return
        # Carrier follows the belt: speed feed-forward plus a position loop on belt travel,
        # so a push from the robot cannot make the kit lag the belt permanently.
        target_x = feeder.INFEED_X + belt.travel - c.release_travel
        data.ctrl[c.drive] = belt.speed + self._gain * (target_x - x)
        rec, st, t = c.record, c.station, data.time
        bottle_x = x + feeder.BOTTLE_X_IN_KIT
        if rec["arrived_t"] is None and bottle_x >= feeder.WORK_ENTRY_X:
            self.plc.arrive(st.id, rec["kit_id"])
            self.events.kit_arrived(t, st.id, rec["kit_id"], belt.speed)
            rec["arrived_t"] = t
            c.monitor = SeatMonitor(self.model, c.name)
            log.info("t=%.3f kit %d entered the work zone at %s", t, rec["kit_id"], st.name)
        if c.monitor is not None:
            measurement = c.monitor.update(data)
            self._log_raw(t, rec, c.monitor.last_raw)
            if rec["seat_verified_t"] is None and c.monitor.seat_verified_t is not None:
                self.events.seat_verified(t, st.id, rec["kit_id"], measurement)
                rec["seat_verified_t"] = t
                rec["cycle_s"] = t - rec["arrived_t"]
                rec["seat_in_window"] = bottle_x <= feeder.WORK_EXIT_X
                log.info("t=%.3f kit %d seat verified, cycle %.2f s%s", t, rec["kit_id"],
                         rec["cycle_s"], "" if rec["seat_in_window"] else " (after work window)")
            if bottle_x >= feeder.INSPECTION_X:
                accepted, done = self.plc.inspect(st.id, rec["kit_id"], measurement)
                self.events.inspection(t, st.id, rec["kit_id"], measurement, belt.speed)
                rec.update(inspected_t=t, accepted=accepted, robot_done=done)
                c.monitor = None
                log.info("t=%.3f kit %d inspection at %s: %s (robot done=%s)", t, rec["kit_id"],
                         st.name, "ACCEPTED" if accepted else "REJECTED", done)
        if x >= feeder.OUTFEED_X:
            # Off the downstream end: back to the magazine. The ring follows once the robot
            # is not holding it (a held ring is never moved by the plant).
            c.on_belt, c.ring_on_nest = False, False
            c.place_kit(data, c.slot_x, 0.0)

    def _log_raw(self, t: float, rec: dict, raw: dict) -> None:
        """Per-step seat samples for recomputing the hold; skips the untouched-on-nest state."""
        if self._raw is None or (raw["released"] and raw["kit_contact"]
                                 and not raw["bottle_contact"]):
            return
        self._raw.write(json.dumps({"t": t, "kit_id": rec["kit_id"], **raw}) + "\n")
