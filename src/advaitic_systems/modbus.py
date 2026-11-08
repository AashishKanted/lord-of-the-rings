"""Local Modbus/TCP test double: FC03 and FC16, not a production PLC runtime."""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import struct
import threading
from pathlib import Path

from .contract import LineController


def read_exact(stream, count):
    data = b""
    while len(data) < count:
        part = stream.recv(count - len(data))
        if not part:
            raise EOFError
        data += part
    return data


class ModbusHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(10)
        while True:
            try:
                header = read_exact(self.request, 7)
                transaction, protocol, length, unit = struct.unpack(">HHHB", header)
                if protocol != 0 or not 2 <= length <= 254:
                    return
                pdu = read_exact(self.request, length - 1)
                with self.server.lock:
                    response = self.server.process(unit, pdu)
                self.request.sendall(struct.pack(">HHHB", transaction, 0, len(response) + 1, unit)
                                     + response)
            except (EOFError, ConnectionError, socket.timeout):
                return


class PLCServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address):
        self.model = LineController()
        self.lock = threading.Lock()
        super().__init__(address, ModbusHandler)

    def process(self, unit: int, pdu: bytes):
        function = pdu[0]
        if unit != 1:
            return bytes([function | 0x80, 2])
        try:
            if function == 3 and len(pdu) == 5:
                start, count = struct.unpack(">HH", pdu[1:])
                if not 1 <= count <= 64 or start + count > 64:
                    return bytes([0x83, 2])
                values = self.model.registers()[start:start + count]
                return bytes([3, 2 * count]) + struct.pack(">" + "H" * count, *values)
            if function == 16 and len(pdu) >= 6:
                start, count, byte_count = struct.unpack(">HHB", pdu[1:6])
                if start != 100 or count != 4:
                    return bytes([0x90, 2])  # inspection/counters are read-only to robot
                if byte_count != 8 or len(pdu) != 14:
                    return bytes([0x90, 3])
                self.model.command(*struct.unpack(">4H", pdu[6:]))
                return struct.pack(">BHH", 16, start, count)
            return bytes([function | 0x80, 1])
        except (ValueError, struct.error):
            return bytes([function | 0x80, 3])


class PLCClient:
    """Minimal reference robot adapter; candidates may use a standard library instead."""

    def __init__(self, host="127.0.0.1", port=1502):
        self.host, self.port, self.transaction = host, port, 0

    def request(self, pdu):
        self.transaction = (self.transaction + 1) % 65536
        with socket.create_connection((self.host, self.port), timeout=2) as stream:
            stream.sendall(struct.pack(">HHHB", self.transaction, 0, len(pdu) + 1, 1) + pdu)
            tx, protocol, length, unit = struct.unpack(">HHHB", read_exact(stream, 7))
            if (tx, protocol, unit) != (self.transaction, 0, 1) or not 2 <= length <= 254:
                raise RuntimeError("Invalid Modbus response header")
            response = read_exact(stream, length - 1)
            if response[0] & 0x80:
                raise RuntimeError(f"Modbus exception {response[1]}")
            return response

    def read(self, start=0, count=27):
        result = self.request(struct.pack(">BHH", 3, start, count))
        if result[:2] != bytes([3, count * 2]) or len(result) != 2 + count * 2:
            raise RuntimeError("Invalid register response")
        return list(struct.unpack(">" + "H" * count, result[2:]))

    def command(self, sequence, station, kit_id, command):
        pdu = struct.pack(">BHHB4H", 16, 100, 4, 8, sequence, station, kit_id, command)
        result = self.request(pdu)
        if result != struct.pack(">BHH", 16, 100, 4):
            raise RuntimeError("Invalid write acknowledgement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=1502)
    parser.add_argument("--events", type=Path,
                        help="Optional customer fixture JSON; seeds arrivals/inspection for adapter tests")
    args = parser.parse_args()
    with PLCServer(("127.0.0.1", args.port)) as server:
        if args.events:
            for event in json.loads(args.events.read_text()):
                if event["type"] == "arrive":
                    server.model.arrive(event["station"], event["kit_id"])
                elif event["type"] == "inspect":
                    server.model.inspect(event["station"], event["kit_id"], event["measurement"])
        print(f"Local test PLC listening on 127.0.0.1:{args.port}; not a complete plant simulator",
              flush=True)
        server.serve_forever()
