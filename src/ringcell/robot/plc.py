"""Robot-side line-controller adapter over Modbus/TCP. PLACEHOLDER.

Planned: wrap ``advaitic_systems.modbus.PLCClient`` with the command sequence ledger
(retry an ambiguous write with the same sequence and payload, advance only when
acknowledged), arrival-ID tracking and reconnect handling.
"""
from __future__ import annotations


class LineClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 1502):
        self.host, self.port = host, port

    def ready(self, station_id: int, ready: bool = True) -> None:
        raise NotImplementedError

    def assembly_done(self, station_id: int, kit_id: int) -> None:
        raise NotImplementedError

    def poll(self) -> dict:
        """Latest arrival/inspection registers for both stations."""
        raise NotImplementedError
