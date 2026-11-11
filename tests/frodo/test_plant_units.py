"""Plant pieces that need no physics: parameters, pool geometry, event log, PLC host,
and the feeder gating logic.

The feeder-logic test moves the carriers kinematically and STUBS the seat measurement, so
it says nothing about physics or about any robot. It only checks the feeder rules.
"""
import json
import socket

import mujoco
import pytest

from advaitic_systems.evidence import check
from advaitic_systems.modbus import PLCClient
from frodo.cell import CELL
from frodo.plant import feeder
from frodo.plant import line as line_module
from frodo.plant.events import EventLog
from frodo.plant.line import Line
from frodo.plant.plc import LinePLC
from frodo.sim.scene import build_spec

READY, ASSEMBLY_DONE = 1, 3


def test_pool_fits_the_belt_at_customer_pitch():
    assert feeder.min_pool_size() == 4
    layout = feeder.pool_layout()
    assert set(layout) == {"A", "B"}
    upstream_end = -CELL["conveyor"]["length_m"] / 2
    for slots in layout.values():
        assert len(slots) >= feeder.min_pool_size()
        # parked kits are clear of the belt and of each other
        assert max(slots) + CELL["kit"]["length_m"] / 2 < upstream_end
        gaps = [a - b for a, b in zip(slots, slots[1:], strict=False)]
        assert all(g > CELL["kit"]["length_m"] for g in gaps)


def test_run_parameters_are_seeded_and_inside_customer_range():
    lo, hi = CELL["conveyor"]["test_speed_range_mps"]
    assert feeder.draw_run_parameters(1) == feeder.draw_run_parameters(1)
    assert feeder.draw_run_parameters(1) != feeder.draw_run_parameters(2)
    for seed in range(20):
        p = feeder.draw_run_parameters(seed)
        assert all(lo <= v <= hi for v in p.belt_speed_mps.values())
        assert all(0 <= v <= 1 for v in p.release_phase_s.values())
    assert feeder.run_parameters().belt_speed_mps == {"A": 0.03, "B": 0.03}
    with pytest.raises(ValueError):
        feeder.run_parameters(belt_speed_mps=0.04)


def test_event_log_writes_jsonl_and_refuses_time_going_backwards(tmp_path):
    log = EventLog(tmp_path / "run" / "events.jsonl")
    seat = dict(dx_m=0.0, dy_m=0.0, z_m=0.133, tilt_rad=0.0, hold_s=0.5, released=True,
                bottle_contact=True, kit_contact=False)
    log.sample(0.0, 1.0, 3.0, 0.0, 0.0, 0.0, True)
    log.dock(1.0, "A", 3.0, 2.0, -1.5707963)
    log.kit_arrived(2.0, 1, 7, 0.03)
    log.assembly_done(3.0, "A", 7)
    log.seat_verified(4.0, 1, 7, {**seat, "extra": "dropped"})
    log.human_enter(5.0)
    log.stop_command(5.1)
    log.human_exit(7.5)
    log.resume(8.0)
    log.inspection(30.0, 1, 7, seat, 0.03)
    with pytest.raises(ValueError):
        log.sample(29.0, 1.0, 3.0, 0.0, 0.0, 0.0, True)
    with pytest.raises(KeyError):
        log.dock(31.0, "Z", 0, 0, 0)
    log.close()
    lines = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert lines == log.events
    assert [e["type"] for e in lines] == [
        "sample", "dock", "kit_arrived", "assembly_done", "seat_verified", "human_enter",
        "stop_command", "human_exit", "resume", "inspection"]
    assert "extra" not in lines[4] and lines[1]["station"] == 1
    # The starter checker must be able to parse every event type we write (no "Event N:").
    assert not [v for v in check(lines)["violations"] if v.startswith("Event")]


def test_plc_host_starts_and_releases_its_port():
    plc = LinePLC("127.0.0.1", 0)
    plc.start()
    port = plc.port
    assert PLCClient(port=port).read(0, 1) == [1]
    plc.arrive(1, 42)
    assert PLCClient(port=port).read(11, 1) == [42]
    plc.close()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.5)


class _StubSeat:
    """NOT a measurement: always reports a seated ring, to exercise the accept path of the
    feeder logic without physics."""

    def __init__(self, model, kit_name):
        self.hold_s, self.seat_verified_t, self.last_raw = 0.0, None, None

    def update(self, data):
        self.hold_s += 0.02
        if self.seat_verified_t is None and self.hold_s >= 0.5:
            self.seat_verified_t = data.time
        self.last_raw = dict(dx_m=0.0, dy_m=0.0, z_m=0.133, tilt_rad=0.0, hold_s=self.hold_s,
                             released=True, bottle_contact=True, kit_contact=False)
        return dict(self.last_raw)


def test_feeder_rules_with_stubbed_seat_and_kinematic_carriers(monkeypatch):
    """Five accepted at A, then a fresh five at B; never more than five accepted+in flight.

    Carriers are integrated kinematically (no mj_step) and the seat measurement is a stub;
    ASSEMBLY_DONE is sent by this test over Modbus. This is a logic test, not a robot run.
    """
    monkeypatch.setattr(line_module, "SeatMonitor", _StubSeat)
    model = build_spec(feeder.pool_layout()).compile()
    data = mujoco.MjData(model)
    dt, speeds = 0.02, {"A": 0.035, "B": 0.025}
    with Line(model, belt_speed_mps=speeds, release_phase_s={"A": 0.5, "B": 0.25},
              port=0) as line:
        plc, seq = PLCClient(port=line.port), 0
        for station in (1, 2):   # ready at both from the start: B must still wait for A
            seq += 1
            plc.command(seq, station, 0, READY)
        done, b_first_release, max_load = set(), None, 0
        drives = [(c.drive, c.slide_qpos) for b in line._belts.values() for c in b.pool]
        while data.time < 260 and sum(k["inspected_t"] is not None for k in line.kits) < 10:
            data.time += dt
            line.step(data)
            for drive, qpos in drives:     # kinematic carrier: follow the commanded speed
                data.qpos[qpos] += data.ctrl[drive] * dt
            for kit in line.kits:
                if kit["arrived_t"] is not None and kit["kit_id"] not in done:
                    seq += 1
                    station = 1 if kit["station"] == "A" else 2
                    plc.command(seq, station, kit["kit_id"], ASSEMBLY_DONE)
                    line.events.assembly_done(data.time, station, kit["kit_id"])
                    done.add(kit["kit_id"])
                if kit["station"] == "B" and b_first_release is None:
                    b_first_release = kit["released_t"]
            counts = line.counts()
            max_load = max(max_load, *(c["accepted"] + c["in_flight"] for c in counts.values()))
        registers = plc.read(0, 27)
        kits, events = line.kits, line.events.events

    assert [k["station"] for k in kits] == ["A"] * 5 + ["B"] * 5
    assert [k["kit_id"] for k in kits] == list(range(1, 11))
    assert all(k["accepted"] for k in kits)
    assert (registers[14], registers[24], registers[15], registers[25]) == (5, 5, 0, 0)
    assert max_load == 5
    fifth_a_inspection = kits[4]["inspected_t"]
    assert b_first_release >= fifth_a_inspection
    assert b_first_release - fifth_a_inspection == pytest.approx(0.25, abs=2 * dt)
    for name, v in speeds.items():
        arrivals = [k["arrived_t"] for k in kits if k["station"] == name]
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:], strict=False)]
        assert gaps == pytest.approx([feeder.PITCH_M / v] * 4, abs=2 * dt)
        for k in (k for k in kits if k["station"] == name):
            assert k["inspected_t"] - k["arrived_t"] == pytest.approx(0.85 / v, abs=2 * dt)
    # The starter checker finds nothing wrong with the plant's own events. (It still
    # reports the missing robot side: no motion samples, no dock.)
    plant_words = ("Event", "pitch", "Feeder", "feeder", "Belt speed", "Inspection",
                   "Seat", "seat", "Expected five")
    violations = check(events)["violations"]
    assert not [v for v in violations if any(w in v for w in plant_words)], violations
