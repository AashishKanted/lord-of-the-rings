"""Hosts the starter's line-controller test double (``PLCServer``) on a daemon thread.

Follows the "Attach the simulator" pattern of docs/INTERFACES.md: the simulation thread
touches ``server.model`` only while holding ``server.lock``; the robot reaches the same
model only through Modbus/TCP.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from advaitic_systems.modbus import PLCServer

from ..log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LineState:
    """Snapshot of the controller, keyed by integer station id."""
    ready: dict[int, bool]
    accepted: dict[int, int]
    rejected: dict[int, int]


class LinePLC:
    def __init__(self, host: str, port: int):
        self._address = (host, port)
        self.server: PLCServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Actual TCP port (differs from the requested one when that was 0)."""
        if self.server is None:
            raise RuntimeError("PLC server is not running")
        return self.server.server_address[1]

    def start(self) -> None:
        if self.server is not None:
            return
        self.server = PLCServer(self._address)
        self._thread = threading.Thread(target=self.server.serve_forever, name="plc-server",
                                        daemon=True)
        self._thread.start()
        log.info("PLC test double listening on %s:%d", self._address[0], self.port)

    def close(self) -> None:
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()
        self.server = self._thread = None

    def state(self) -> LineState:
        with self.server.lock:
            m = self.server.model
            return LineState(dict(m.ready), dict(m.accepted), dict(m.rejected))

    def arrive(self, station: int, kit_id: int) -> None:
        with self.server.lock:
            self.server.model.arrive(station, kit_id)

    def inspect(self, station: int, kit_id: int, measurement: dict) -> tuple[bool, bool]:
        """Report a downstream measurement. Returns (accepted, robot_reported_done)."""
        with self.server.lock:
            self.server.model.inspect(station, kit_id, measurement)
            kit = self.server.model.kits[kit_id]
            return bool(kit.inspection), bool(kit.done)

    def command_ledger(self) -> list[dict]:
        """Copy of the commands the controller applied (evidence for the retry scenario)."""
        with self.server.lock:
            return [dict(c) for c in self.server.model.commands]
