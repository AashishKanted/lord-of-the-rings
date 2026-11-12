"""Robot-side line client against the starter's Modbus/TCP test double, over real TCP."""
import threading

import pytest

from advaitic_systems.modbus import PLCServer
from frodo.robot.plc import ASSEMBLY_DONE, READY, RESULT_ACCEPTED, LineClient, LineError

GOOD_SEAT = dict(dx_m=0.0, dy_m=0.0, z_m=0.133, tilt_rad=0.0, hold_s=0.5, released=True,
                 bottle_contact=True, kit_contact=False)


@pytest.fixture
def server():
    with PLCServer(("127.0.0.1", 0)) as srv:
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        yield srv
        srv.shutdown()
        thread.join()


def client_for(server):
    return LineClient("127.0.0.1", server.server_address[1])


def test_ready_arrival_done_and_inspection_are_seen_through_the_registers(server):
    line = client_for(server)
    line.ready(1)
    assert line.poll().stations[1].ready and not line.poll().stations[2].ready
    with server.lock:
        server.model.arrive(1, 7)
    assert line.poll().stations[1].latest_kit == 7
    line.assembly_done(1, 7)
    status = line.poll()
    assert status.last_sequence == 2 and status.stations[1].accepted == 0   # done != accepted
    with server.lock:
        server.model.inspect(1, 7, GOOD_SEAT)
    st = line.poll().stations[1]
    assert (st.inspected_kit, st.result, st.accepted, st.rejected) == (7, RESULT_ACCEPTED, 1, 0)
    line.ready(1, False)
    assert not line.poll().stations[1].ready
    assert [c["command"] for c in line.ledger] == [READY, ASSEMBLY_DONE, 2]
    assert [c["sequence"] for c in line.ledger] == [1, 2, 3]


def test_lost_acknowledgement_is_retried_with_the_same_sequence_and_applied_once(server):
    line = client_for(server)
    line.ready(1)
    with server.lock:
        server.model.arrive(1, 3)
    real, calls = line._client.command, []

    def acknowledgement_lost_once(sequence, station, kit_id, command):
        calls.append((sequence, station, kit_id, command))
        real(sequence, station, kit_id, command)          # the controller applies it ...
        if len(calls) == 1:
            raise ConnectionResetError("ack dropped")     # ... but the robot never hears back

    line._client.command = acknowledgement_lost_once
    line.assembly_done(1, 3)
    assert calls == [(2, 1, 3, ASSEMBLY_DONE)] * 2        # same sequence, same payload
    assert line.ledger[-1]["attempts"] == 2 and line.sequence == 2
    with server.lock:
        applied = [c for c in server.model.commands if c["command"] == ASSEMBLY_DONE]
        assert len(applied) == 1 and server.model.last_status == 2   # duplicate acknowledged


def test_refused_command_raises_and_does_not_advance_the_sequence(server):
    line = client_for(server)
    line.ready(1)
    with pytest.raises(LineError):
        line.assembly_done(1, 99)          # unknown kit
    assert line.sequence == 1
    line.ready(2)                          # the next command still uses sequence 2
    assert line.poll().last_sequence == 2


def test_client_picks_up_the_controllers_sequence_on_first_use(server):
    client_for(server).ready(1)
    late = client_for(server)              # e.g. robot software restarted
    late.ready(2)
    assert late.sequence == 2 and late.poll().stations[2].ready


def test_command_unacknowledged_after_all_attempts_is_settled_before_the_next(server):
    line = client_for(server)
    line.ready(1)
    with server.lock:
        server.model.arrive(1, 5)
    real = line._client.command

    def applied_but_never_acknowledged(sequence, station, kit_id, command):
        real(sequence, station, kit_id, command)
        raise ConnectionResetError("ack dropped")

    line._client.command = applied_but_never_acknowledged
    with pytest.raises(LineError):
        line.assembly_done(1, 5)                 # applied once, the robot cannot know
    assert line.sequence == 1 and line._pending == (2, 1, 5, ASSEMBLY_DONE)
    line._client.command = real
    line.ready(1, False)                         # resends seq 2 unchanged, then seq 3
    assert [c["sequence"] for c in line.ledger] == [1, 2, 3]
    with server.lock:
        assert [c["sequence"] for c in server.model.commands] == [1, 2, 3]   # each applied once
    frames = line.ledger[1]["frames_hex"]
    assert len(frames) == 1 and len(frames[0]) == 42     # one complete FC16 request frame
