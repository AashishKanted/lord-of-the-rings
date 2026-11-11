"""Evidence event log in the docs/INTERFACES.md JSONL format.

One writer per run. The plant writes ``kit_arrived``, ``seat_verified`` and
``inspection`` from its own measurements; the mission runner calls the remaining
methods (``sample``, ``dock``, ``assembly_done``, human and ``plc_retry`` events) with
what the robot side actually did. Nothing here decides acceptance.

Timestamps are simulated seconds from mission start and must not go backwards: the
starter checker rejects unordered logs, so an out-of-order write raises here instead of
producing a file that fails later.
"""
from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

from ..cell import STATIONS
from ..log import get_logger

log = get_logger(__name__)

SEAT_FIELDS = ("dx_m", "dy_m", "z_m", "tilt_rad", "hold_s", "released", "bottle_contact",
               "kit_contact")


def station_id(station: int | str) -> int:
    """Accept a station name ("A") or its integer id (1); return the id used in evidence."""
    if isinstance(station, str):
        return STATIONS[station].id
    if station not in {s.id for s in STATIONS.values()}:
        raise ValueError(f"unknown station id {station}")
    return int(station)


class EventLog:
    """Append-only evidence log, kept in memory and (optionally) streamed to a file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        self._last_t = 0.0
        self._file = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")

    # ---- generic ----
    def emit(self, kind: str, t: float, **fields) -> dict:
        t = float(t)
        if not isfinite(t) or t < 0 or t < self._last_t:
            raise ValueError(f"event '{kind}' at t={t} is not monotonic (last {self._last_t})")
        self._last_t = t
        event = {"t": t, "type": kind, **fields}
        self.events.append(event)
        if self._file:
            self._file.write(json.dumps(event) + "\n")
        return event

    def of_type(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            log.info("wrote %d events to %s", len(self.events), self.path)

    # ---- plant-measured events ----
    def kit_arrived(self, t, station, kit_id: int, belt_speed_mps: float) -> dict:
        return self.emit("kit_arrived", t, station=station_id(station), kit_id=int(kit_id),
                         belt_speed_mps=float(belt_speed_mps))

    def seat_verified(self, t, station, kit_id: int, measurement: dict) -> dict:
        return self.emit("seat_verified", t, station=station_id(station), kit_id=int(kit_id),
                         **{k: measurement[k] for k in SEAT_FIELDS})

    def inspection(self, t, station, kit_id: int, measurement: dict,
                   belt_speed_mps: float) -> dict:
        return self.emit("inspection", t, station=station_id(station), kit_id=int(kit_id),
                         **{k: measurement[k] for k in SEAT_FIELDS},
                         belt_speed_mps=float(belt_speed_mps))

    # ---- robot / scenario events, written by the mission runner ----
    def sample(self, t, x: float, y: float, yaw: float, base_speed: float, ee_speed: float,
               arm_stowed: bool) -> dict:
        return self.emit("sample", t, x=float(x), y=float(y), yaw=float(yaw),
                         base_speed=float(base_speed), ee_speed=float(ee_speed),
                         arm_stowed=bool(arm_stowed))

    def dock(self, t, station, x: float, y: float, yaw: float) -> dict:
        return self.emit("dock", t, station=station_id(station), x=float(x), y=float(y),
                         yaw=float(yaw))

    def assembly_done(self, t, station, kit_id: int) -> dict:
        """The robot's completion command was applied. Not acceptance."""
        return self.emit("assembly_done", t, station=station_id(station), kit_id=int(kit_id))

    def human_enter(self, t) -> dict:
        return self.emit("human_enter", t)

    def stop_command(self, t) -> dict:
        return self.emit("stop_command", t)

    def human_exit(self, t) -> dict:
        return self.emit("human_exit", t)

    def resume(self, t) -> dict:
        return self.emit("resume", t)

    def plc_retry(self, t, first_request_hex: str, retry_request_hex: str,
                  controller_applied_count: int, accepted_before: int,
                  accepted_after: int) -> dict:
        return self.emit("plc_retry", t, first_request_hex=first_request_hex,
                         retry_request_hex=retry_request_hex,
                         controller_applied_count=int(controller_applied_count),
                         accepted_before=int(accepted_before), accepted_after=int(accepted_after))
