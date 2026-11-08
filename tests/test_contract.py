import socket
import struct
import threading
import time

import pytest

from advaitic_systems.contract import LineController, inspection_passes
from advaitic_systems.modbus import PLCClient, PLCServer

GOOD = dict(dx_m=0., dy_m=0., z_m=.133, tilt_rad=0., hold_s=.5,
            released=True, bottle_contact=True, kit_contact=False)


@pytest.mark.parametrize("key,value", [("dx_m", .005), ("z_m", .14), ("tilt_rad", .2),
    ("hold_s", .49), ("released", False), ("bottle_contact", False), ("kit_contact", True),
    ("z_m", float("nan")), ("dx_m", float("inf")), ("dx_m", 10**1000)])
def test_bad_geometry_rejected(key, value):
    assert not inspection_passes({**GOOD, key: value})


def test_inspection_independent_and_idempotent():
    line = LineController()
    line.arrive(1, 42)
    line.command(1, 1, 42, 3)
    assert line.accepted[1] == 0
    line.command(1, 1, 42, 3)
    assert len(line.commands) == 1
    assert line.last_status == 2
    with pytest.raises(ValueError, match="original sequence"):
        line.command(2, 1, 42, 3)
    assert line.last_sequence == 1
    line.inspect(1, 42, GOOD)
    line.inspect(1, 42, GOOD)
    assert line.accepted[1] == 1
    with pytest.raises(ValueError):
        line.command(1, 1, 42, 4)
    with pytest.raises(ValueError):
        line.inspect(1, 42, {**GOOD, "released": False})


def test_inspection_without_completion_rejects():
    line = LineController()
    line.arrive(2, 4)
    line.inspect(2, 4, GOOD)
    assert line.accepted[2] == 0 and line.rejected[2] == 1


def test_real_socket_reconnect_retry_and_readonly_registers():
    with PLCServer(("127.0.0.1", 0)) as server:
        server.model.arrive(1, 42)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = PLCClient(port=server.server_address[1])
            # Send an actual write but close without reading its acknowledgement.
            pdu = struct.pack(">BHHB4H", 16, 100, 4, 8, 1, 1, 42, 3)
            with socket.create_connection(server.server_address, timeout=2) as stream:
                stream.sendall(struct.pack(">HHHB", 77, 0, len(pdu) + 1, 1) + pdu)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with server.lock:
                    applied = server.model.last_sequence == 1
                if applied:
                    break
                time.sleep(.01)
            assert applied
            # New TCP connection, same command: ambiguous acknowledgement recovery.
            PLCClient(port=server.server_address[1]).command(1, 1, 42, 3)
            assert client.read(0, 3) == [1, 1, 2]
            assert len(server.model.commands) == 1
            assert client.read(14, 1) == [0]
            with server.lock:
                server.model.inspect(1, 42, GOOD)
            assert client.read(14, 1) == [1]
            with pytest.raises(RuntimeError):
                client.request(struct.pack(">BHHBH", 16, 14, 1, 2, 9))
            with pytest.raises(RuntimeError):
                client.command(2, 2, 42, 3)
            assert client.read(14, 1) == [1]
        finally:
            server.shutdown()
            thread.join()
