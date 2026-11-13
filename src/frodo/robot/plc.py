"""Robot-side adapter for the customer line controller (Modbus/TCP, docs/INTERFACES.md).

The robot learns about arrivals, inspections and accepted counts only by reading
registers here, and reports READY / NOT_READY / ASSEMBLY_DONE / ROBOT_FAULT by writing
the command block. ASSEMBLY_DONE is the robot's claim, never acceptance.

Command sequence rule: each new command uses ``last sequence + 1``. If a write fails at
the socket level its outcome is unknown (the controller may or may not have applied it), so
the SAME sequence and payload are sent again; the controller applies a sequence once and
acknowledges a repeat as a duplicate. The sequence only advances after an acknowledged
write. A command that is still unacknowledged after all attempts stays pending and is sent
again, unchanged, before any newer command: reusing its sequence for a different command
would be refused if the first one had in fact been applied.

Every request frame sent is kept in `requests` (time, hex) so a retry can be shown with
the actual bytes.
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

from advaitic_systems.modbus import PLCClient, read_exact

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


class _RecordingClient(PLCClient):
    """The starter's reference client, with every request frame kept as sent."""

    def __init__(self, host, port, timeout_s: float):
        super().__init__(host, port)
        self.timeout_s = timeout_s
        self.sent: list[bytes] = []

    def request(self, pdu):
        self.transaction = (self.transaction + 1) % 65536
        frame = struct.pack(">HHHB", self.transaction, 0, len(pdu) + 1, 1) + pdu
        self.sent.append(frame)
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as stream:
            stream.sendall(frame)
            tx, protocol, length, unit = struct.unpack(">HHHB", read_exact(stream, 7))
            if (tx, protocol, unit) != (self.transaction, 0, 1) or not 2 <= length <= 254:
                raise RuntimeError("Invalid Modbus response header")
            response = read_exact(stream, length - 1)
            if response[0] & 0x80:
                raise RuntimeError(f"Modbus exception {response[1]}")
            return response


class LineClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 1502, attempts: int = 3,
                 timeout_s: float = 2.0):
        self._client = _RecordingClient(host, port, timeout_s)
        self.attempts = attempts
        self.sequence: int | None = None    # last acknowledged sequence; None until first use
        self.ledger: list[dict] = []        # every acknowledged command, with its frames
        self._pending: tuple[int, int, int, int] | None = None   # (seq, station, kit, command)

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
        if self._pending is not None:       # settle the unacknowledged command first
            self._deliver(*self._pending)
        self._deliver(self.sequence + 1, station_id, kit_id, command)

    def _deliver(self, sequence: int, station_id: int, kit_id: int, command: int) -> None:
        """Send one command until it is acknowledged (applied, or a duplicate of an applied
        one). Raises LineError if refused, or if still unacknowledged (it stays pending)."""
        first = len(self._client.sent)
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                self._client.command(sequence, station_id, kit_id, command)
            except (OSError, EOFError) as exc:      # outcome unknown: repeat the same write
                last_error = exc
                log.warning("line command seq=%d attempt %d: no acknowledgement (%s); "
                            "reconnecting and sending it again unchanged", sequence, attempt,
                            str(exc) or type(exc).__name__)
                continue
            except RuntimeError as exc:             # Modbus exception: the controller said no
                self._pending = None
                raise LineError(f"line controller refused seq={sequence} station={station_id} "
                                f"kit={kit_id} command={command}: {exc}") from exc
            self.sequence, self._pending = sequence, None
            self.ledger.append({"sequence": sequence, "station": station_id, "kit_id": kit_id,
                                "command": command, "attempts": attempt,
                                "frames_hex": [f.hex() for f in self._client.sent[first:]]})
            return
        self._pending = (sequence, station_id, kit_id, command)
        raise LineError(f"line command seq={sequence} not acknowledged after {self.attempts} "
                        f"attempts ({last_error}); it will be sent again before the next one")
