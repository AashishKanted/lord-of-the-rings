"""Acceptance geometry and a small line-controller model, independent of robot code."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite


def inspection_passes(value: dict) -> bool:
    """Check measured rigid-ring seating, not a robot-supplied success flag."""
    try:
        dx, dy, z, tilt, hold = [float(value[k]) for k in
            ("dx_m", "dy_m", "z_m", "tilt_rad", "hold_s")]
        return (
            all(isfinite(x) for x in [dx, dy, z, tilt, hold])
            and hypot(dx, dy) <= .004
            and .130 <= z <= .136
            and abs(tilt) <= .17453292519943295
            and hold >= .5
            and value["released"] is True
            and value["bottle_contact"] is True
            and value["kit_contact"] is False
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


@dataclass
class Kit:
    station: int
    done: bool = False
    inspection: bool | None = None


@dataclass
class LineController:
    """Robot completion and customer inspection are distinct, idempotent events."""

    kits: dict[int, Kit] = field(default_factory=dict)
    ready: dict[int, bool] = field(default_factory=lambda: {1: False, 2: False})
    accepted: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    rejected: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    latest: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    inspected: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    inspection_result: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    last_sequence: int = 0
    last_payload: tuple | None = None
    last_status: int = 0
    commands: list[dict] = field(default_factory=list)

    def arrive(self, station: int, kit_id: int):
        if station not in (1, 2) or not 1 <= kit_id <= 65535 or kit_id in self.kits:
            raise ValueError("Invalid or duplicate kit arrival")
        self.kits[kit_id] = Kit(station)
        self.latest[station] = kit_id

    def command(self, sequence: int, station: int, kit_id: int, command: int):
        payload = (station, kit_id, command)
        if not 1 <= sequence <= 65535 or station not in (1, 2):
            raise ValueError("Invalid command sequence or station")
        if sequence == self.last_sequence:
            if payload != self.last_payload:
                raise ValueError("Sequence reused for a different command")
            self.last_status = 2  # acknowledged duplicate, never re-applied
            return
        if sequence != self.last_sequence + 1:
            raise ValueError("Sequence must increase by one; restart server before wrap")
        if command in (1, 2):
            self.ready[station] = command == 1  # READY / NOT_READY
        elif command == 3:  # ASSEMBLY_DONE, not ACCEPTED
            kit = self.kits.get(kit_id)
            if kit is None or kit.station != station:
                raise ValueError("Unknown kit or wrong station")
            if kit.done:
                raise ValueError("Kit already completed; retry with the original sequence")
            kit.done = True
        elif command == 4:  # ROBOT_FAULT
            self.ready[station] = False
        else:
            raise ValueError("Unsupported robot command")
        self.last_sequence, self.last_payload, self.last_status = sequence, payload, 1
        self.commands.append({"sequence": sequence, "station": station,
                              "kit_id": kit_id, "command": command})

    def inspect(self, station: int, kit_id: int, measurement: dict):
        kit = self.kits.get(kit_id)
        if kit is None or kit.station != station:
            raise ValueError("Inspection for unknown kit or wrong station")
        result = kit.done and inspection_passes(measurement)
        if kit.inspection is not None:
            if kit.inspection != result:
                raise ValueError("Conflicting duplicate inspection")
            return
        kit.inspection = result
        self.inspected[station] = kit_id
        self.inspection_result[station] = 1 if result else 2
        (self.accepted if result else self.rejected)[station] += 1

    def registers(self):
        values = [0] * 64
        values[0:4] = [1, self.last_sequence, self.last_status, len(self.kits)]
        for station, offset in ((1, 10), (2, 20)):
            values[offset:offset + 7] = [int(self.ready[station]), self.latest[station],
                self.inspected[station], self.inspection_result[station],
                self.accepted[station], self.rejected[station], 30]
        return values
