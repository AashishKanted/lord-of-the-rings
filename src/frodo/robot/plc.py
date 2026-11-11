"""Robot-side adapter for the customer line controller (Modbus/TCP, docs/INTERFACES.md).

The robot learns about arrivals, inspections and accepted counts only by reading
registers here, and reports READY / NOT_READY / ASSEMBLY_DONE / ROBOT_FAULT by writing
the command block. ASSEMBLY_DONE is the robot's claim, never acceptance.

Command sequence rule: each new command uses ``last sequence + 1``. If a write fails at
the socket level its outcome is unknown, so the SAME sequence and payload are sent again;
the controller applies a sequence once and acknowledges a repeat as a duplicate. The
sequence only advances after an acknowledged write.
"""
from __future__ import annotations

from dataclasses import dataclass

from advaitic_systems.modbus import PLCClient

from ..log import get_logger

log = get_logger(__name__)

READY, NOT_READY, ASSEMBLY_DONE, ROBOT_FAULT = 1, 2, 3, 4
RESULT_NONE, RESULT_ACCEPTED, RESULT_REJECTED = 0, 1, 2
_STATION_REGISTERS = {1: 10, 2: 20}


class LineError(RuntimeError):
    """The controller refused a command, or it could not be delivered."""


@dataclass(frozen=True)
class StationStatus:
    ready: bool
    latest_kit: int        # id of the newest kit that crossed the work-entry line (0 = none)
    inspected_kit: int     # id of the newest inspected kit (0 = none)
    result: int            # RESULT_* of that inspection
    accepted: int
    rejected: int


@dataclass(frozen=True)
class LineStatus:
    last_sequence: int
    last_status: int       # 1 applied, 2 duplicate acknowledged
    total_arrivals: int
    stations: dict[int, StationStatus]


class LineClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 1502, attempts: int = 3):
        self._client = PLCClient(host, port)
        self.attempts = attempts
        self.sequence: int | None = None    # last acknowledged sequence; None until first use
        self.ledger: list[dict] = []        # every acknowledged command, with attempt count

    # ---- reads ----
    def poll(self) -> LineStatus:
        reg = self._client.read(0, 27)
        stations = {sid: StationStatus(bool(reg[o]), reg[o + 1], reg[o + 2], reg[o + 3],
                                       reg[o + 4], reg[o + 5])
                    for sid, o in _STATION_REGISTERS.items()}
        return LineStatus(reg[1], reg[2], reg[3], stations)

    # ---- commands ----
    def ready(self, station_id: int, ready: bool = True) -> None:
        self._send(station_id, 0, READY if ready else NOT_READY)

    def assembly_done(self, station_id: int, kit_id: int) -> None:
        self._send(station_id, kit_id, ASSEMBLY_DONE)

    def fault(self, station_id: int) -> None:
        self._send(station_id, 0, ROBOT_FAULT)

    def _send(self, station_id: int, kit_id: int, command: int) -> None:
        if self.sequence is None:
            self.sequence = self.poll().last_sequence
        sequence = self.sequence + 1
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                self._client.command(sequence, station_id, kit_id, command)
            except (OSError, EOFError) as exc:      # outcome unknown: repeat the same write
                last_error = exc
                log.warning("line command seq=%d attempt %d failed (%s); retrying with the "
                            "same sequence", sequence, attempt, exc)
                continue
            except RuntimeError as exc:             # Modbus exception: the controller said no
                raise LineError(f"line controller refused seq={sequence} station={station_id} "
                                f"kit={kit_id} command={command}: {exc}") from exc
            self.sequence = sequence
            self.ledger.append({"sequence": sequence, "station": station_id, "kit_id": kit_id,
                                "command": command, "attempts": attempt})
            return
        raise LineError(f"line command seq={sequence} not delivered after {self.attempts} "
                        f"attempts: {last_error}")
