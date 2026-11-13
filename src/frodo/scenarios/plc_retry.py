"""The acknowledgement of one ASSEMBLY_DONE is lost after the line controller applied it.

A small TCP relay sits between the robot's line client and the line controller. It
forwards every request and response unchanged, except once: for the ``drop_nth_done``-th
ASSEMBLY_DONE it forwards the request, receives the controller's acknowledgement, and then
closes the robot's connection without passing the acknowledgement on. The robot sees a
broken connection, cannot know whether the command was applied, reconnects and sends the
same frame again (same sequence and payload). The controller recognises the duplicate.

Evidence: every frame in both directions (``plc_socket_trace.jsonl``), the controller's
application ledger (``plc_ledger.json``) and a ``plc_retry`` event with the first and the
retried request frames, how often the controller applied that sequence, and the accepted
count before and after.
"""
from __future__ import annotations

import json
import socket
import struct
import threading

from advaitic_systems.modbus import read_exact

from ..log import get_logger
from .base import Scenario

log = get_logger(__name__)
ASSEMBLY_DONE = 3


def _done_payload(frame: bytes):
    """(sequence, station, kit, command) if the frame is an ASSEMBLY_DONE write."""
    pdu = frame[7:]
    if len(pdu) == 14 and pdu[0] == 16:
        payload = struct.unpack(">4H", pdu[6:])
        if payload[3] == ASSEMBLY_DONE:
            return payload
    return None


class AckDropRelay:
    def __init__(self, target_port: int, drop_nth_done: int, clock, accepted):
        """``clock()``: simulation time for the trace; ``accepted(station_id)``: accepted
        count from the controller (read just before the dropped request is forwarded)."""
        self.target = ("127.0.0.1", target_port)
        self.drop_nth, self.clock, self.accepted = drop_nth_done, clock, accepted
        self.trace: list[dict] = []
        self.dropped: dict | None = None
        self.retry: dict | None = None
        self._dones = 0
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(0.2)
        self.port = self._server.getsockname()[1]
        self._running = True
        self._thread = threading.Thread(target=self._serve, name="plc-fault-relay", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join()
        self._server.close()

    def _serve(self) -> None:
        while self._running:
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            with conn:
                self._relay(conn)

    def _relay(self, conn) -> None:
        while True:
            try:
                header = read_exact(conn, 7)
            except (EOFError, ConnectionError, TimeoutError):
                return
            frame = header + read_exact(conn, struct.unpack(">HHHB", header)[2] - 1)
            payload = _done_payload(frame)
            drop = False
            if payload is not None:
                self._dones += 1
                if self._dones == self.drop_nth and self.dropped is None:
                    drop = True
                    before = self.accepted(payload[1])
            self._log("robot->line", frame)
            with socket.create_connection(self.target, timeout=2) as upstream:
                upstream.sendall(frame)
                head = read_exact(upstream, 7)
                response = head + read_exact(upstream, struct.unpack(">HHHB", head)[2] - 1)
            self._log("line->relay", response)
            if drop:
                self.dropped = {"t": self.clock(), "request_hex": frame.hex(),
                                "payload": payload, "accepted_before": before}
                self.trace.append({"t": self.clock(), "event": "acknowledgement dropped, "
                                   "robot connection closed"})
                log.warning("t=%.2f FAULT: ASSEMBLY_DONE seq=%d kit=%d applied by the line, "
                            "acknowledgement dropped", self.clock(), payload[0], payload[2])
                return                                   # close without answering
            if self.dropped is not None and self.retry is None and payload is not None \
                    and payload == self.dropped["payload"]:
                self.retry = {"t": self.clock(), "request_hex": frame.hex(),
                              "response_hex": response.hex(),
                              "accepted_after": self.accepted(payload[1])}
            conn.sendall(response)
            self._log("line->robot", response)

    def _log(self, direction: str, frame: bytes) -> None:
        self.trace.append({"t": self.clock(), "dir": direction, "hex": frame.hex()})


class PlcRetry(Scenario):
    name = "plc_retry"

    def __init__(self, config):
        self.cfg = config.scenarios.plc_retry
        self.time_limit_s = self.cfg.time_limit_s
        self.relay: AckDropRelay | None = None
        self._line = None
        self._data = None
        self._emitted = False
        self.ledger: list[dict] = []

    def robot_port(self, line_port: int) -> int:
        return self.relay.port

    def start(self, line, data) -> None:
        self._line, self._data = line, data
        self.relay = AckDropRelay(line.port, self.cfg.drop_nth_done, lambda: float(data.time),
                                  lambda sid: line.plc.state().accepted[sid])

    def tick(self, model, data, robot, line) -> None:
        r = self.relay
        if self._emitted or r.dropped is None or r.retry is None:
            return
        seq = r.dropped["payload"][0]
        applied = sum(c["sequence"] == seq for c in line.plc.command_ledger())
        line.events.plc_retry(data.time, r.dropped["request_hex"], r.retry["request_hex"],
                              applied, r.dropped["accepted_before"], r.retry["accepted_after"])
        self._emitted = True
        log.info("t=%.2f retry evidence: seq %d applied %d time(s) by the line", data.time,
                 seq, applied)

    def report(self) -> dict:
        r = self.relay
        return {"plc_fault": {"drop_nth_done": self.cfg.drop_nth_done,
                              "dropped": r.dropped and {**r.dropped,
                                                        "payload": list(r.dropped["payload"])},
                              "retry": r.retry, "evidence_written": self._emitted}}

    def write_files(self, out_dir) -> None:
        with open(out_dir / "plc_socket_trace.jsonl", "w", encoding="utf-8") as f:
            for row in self.relay.trace:
                f.write(json.dumps(row) + "\n")
        (out_dir / "plc_ledger.json").write_text(json.dumps(self._line.plc.command_ledger(),
                                                            indent=1))

    def close(self) -> None:
        if self.relay is not None:
            self.relay.close()
