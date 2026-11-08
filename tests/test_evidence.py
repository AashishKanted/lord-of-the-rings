import json
import struct
import subprocess
import sys
from math import pi

import pytest
from test_contract import GOOD

from advaitic_systems.evidence import check
from advaitic_systems.timing import budget, move_time


def fixture():
    """Synthetic trace for checker tests. NOT output of a robot simulation."""
    events = []
    for i in range(2001):
        t = i / 10
        if t < 10:
            x, y, yaw, speed = 1 + .2*t, 3 - .1*t, -pi/2*t/10, .224
        elif 99 <= t <= 109:
            x, y, yaw, speed = 3, 2 + .2*(t-99), -pi/2 + pi*(t-99)/10, .2
        elif t < 99:
            x, y, yaw, speed = 3, 2, -pi/2, 0
        else:
            x, y, yaw, speed = 3, 4, pi/2, 0
        events.append(dict(type="sample", t=t, x=x, y=y, yaw=yaw,
                           base_speed=speed, ee_speed=0, arm_stowed=True))
    for station, start, y, yaw in ((1, 10.1, 2, -pi/2), (2, 110, 4, pi/2)):
        events.append(dict(type="dock", t=start, station=station, x=3, y=y, yaw=yaw))
        for i in range(5):
            kit_id, t = station*100+i, start+i*15
            events.extend([
                dict(type="kit_arrived", t=t, station=station, kit_id=kit_id,
                     belt_speed_mps=.03),
                dict(type="assembly_done", t=t+2, station=station, kit_id=kit_id),
                dict(type="seat_verified", t=t+2, station=station, kit_id=kit_id, **GOOD),
                dict(type="inspection", t=t+.85/.03, station=station, kit_id=kit_id,
                     belt_speed_mps=.03, **GOOD)])
    return sorted(events, key=lambda e: e["t"])


def completion_frame(transaction, sequence=1, station=2, kit_id=201):
    pdu = struct.pack(">BHHB4H", 16, 100, 4, 8, sequence, station, kit_id, 3)
    return (struct.pack(">HHHB", transaction, 0, len(pdu) + 1, 1) + pdu).hex()


def test_nominal_synthetic_plumbing():
    assert check(fixture())["passed"]


@pytest.mark.parametrize("mutation", ["geometry", "teleport", "early_inspect", "late_cycle", "bad_dock", "missing_sample"])
def test_checker_rejects_invalid_trace(mutation):
    events = fixture()
    if mutation == "geometry":
        next(e for e in events if e["type"] == "inspection")["z_m"] = .2
    elif mutation == "teleport":
        next(e for e in events if e["type"] == "sample" and e["t"] == 1)["x"] = 5
    elif mutation == "early_inspect":
        next(e for e in events if e["type"] == "inspection")["t"] = 15
        events.sort(key=lambda e: e["t"])
    elif mutation == "late_cycle":
        next(e for e in events if e["type"] == "seat_verified")["t"] = 24
        events.sort(key=lambda e: e["t"])
    elif mutation == "bad_dock":
        next(e for e in events if e["type"] == "dock")["x"] = 3.1
    else:
        events = [e for e in events if not (e["type"] == "sample" and 20 <= e["t"] <= 21)]
    assert not check(events)["passed"]


def test_human_stop_and_resume():
    events = fixture()
    next(e for e in events if e["type"] == "assembly_done" and e["kit_id"] == 200)["t"] = 115
    next(e for e in events if e["type"] == "seat_verified" and e["kit_id"] == 200)["t"] = 115
    events += [dict(type="human_enter", t=111), dict(type="stop_command", t=111.1),
               dict(type="human_exit", t=114), dict(type="resume", t=114.1)]
    events.sort(key=lambda e: e["t"])
    assert check(events, "human_stop")["passed"]
    next(e for e in events if e["type"] == "stop_command")["t"] = 111.4
    events.sort(key=lambda e: e["t"])
    assert not check(events, "human_stop")["passed"]


def test_checker_rejects_feeder_and_inspection_false_positives():
    events = fixture()
    for event in events:
        if event.get("kit_id") == 101:
            event["t"] -= 15
    events.sort(key=lambda e: e["t"])
    assert "Kit arrivals violated 450 mm pitch" in check(events)["violations"]

    events = fixture()
    next(e for e in events if e["type"] == "inspection" and e["kit_id"] == 100)["t"] = 90
    events.sort(key=lambda e: e["t"])
    assert "Inspection did not occur at downstream crossing" in check(events)["violations"]

    events = fixture()
    next(e for e in events if e["type"] == "inspection")["belt_speed_mps"] = .035
    assert "Inspection belt speed differs from arrival" in check(events)["violations"]

    events = fixture()
    next(e for e in events if e["type"] == "kit_arrived" and e["kit_id"] == 101)["belt_speed_mps"] = .035
    assert "Belt speed changed within a station run" in check(events)["violations"]

    events = fixture() + [dict(type="kit_arrived", t=85.1, station=1, kit_id=999,
                               belt_speed_mps=.03)]
    events.sort(key=lambda e: e["t"])
    assert "Feeder exceeded five accepted plus in-flight kits" in check(events)["violations"]


def test_cycle_uses_measured_seating_not_robot_completion():
    events = fixture()
    next(e for e in events if e["type"] == "seat_verified" and e["kit_id"] == 100)["t"] = 24
    events.sort(key=lambda e: e["t"])
    assert "Assembly cycle missed" in check(events)["violations"]

    events = [e for e in fixture() if not (e["type"] == "seat_verified" and e["kit_id"] == 100)]
    assert "Accepted inspection lacks measured seat verification" in check(events)["violations"]


def test_human_interruption_duration_is_enforced():
    events = fixture()
    for kind, t in (("assembly_done", 115), ("seat_verified", 115)):
        next(e for e in events if e["type"] == kind and e["kit_id"] == 200)["t"] = t
    events += [dict(type="human_enter", t=111), dict(type="stop_command", t=111.1),
               dict(type="human_exit", t=111.2), dict(type="resume", t=111.3)]
    events.sort(key=lambda e: e["t"])
    assert "Human interruption must last 2–4 s" in check(events, "human_stop")["violations"]


def test_human_stop_must_interrupt_active_kit():
    events = fixture() + [dict(type="human_enter", t=120), dict(type="stop_command", t=120.1),
                          dict(type="human_exit", t=123), dict(type="resume", t=123.1)]
    events.sort(key=lambda e: e["t"])
    assert "Human interruption outside an active kit" in check(events, "human_stop")["violations"]


def test_station_b_feeder_waits_for_five_accepted_at_a():
    events = fixture()
    next(e for e in events if e["type"] == "inspection" and e["kit_id"] == 104)["t"] = 109
    next(e for e in events if e["type"] == "kit_arrived" and e["kit_id"] == 200)["t"] = 105
    events.sort(key=lambda e: e["t"])
    assert "Station B feeder started before five accepted at A" in check(events)["violations"]


def test_plc_retry():
    events = fixture() + [dict(type="plc_retry", t=130,
                              first_request_hex=completion_frame(77),
                              retry_request_hex=completion_frame(78),
                              controller_applied_count=1,
                              accepted_before=5, accepted_after=5)]
    events.sort(key=lambda e: e["t"])
    assert check(events, "plc_retry")["passed"]
    next(e for e in events if e["type"] == "plc_retry")["retry_request_hex"] = completion_frame(78, 2)
    assert "PLC retry did not preserve sequence and payload" in check(events, "plc_retry")["violations"]
    next(e for e in events if e["type"] == "plc_retry")["retry_request_hex"] = completion_frame(78)
    next(e for e in events if e["type"] == "plc_retry")["first_request_hex"] = "not a frame"
    assert not check(events, "plc_retry")["passed"]

    events = fixture() + [dict(type="plc_retry", t=130, same_sequence=True, same_payload=True,
                              accepted_before=5, accepted_after=5)]
    events.sort(key=lambda e: e["t"])
    assert not check(events, "plc_retry")["passed"]


def test_budget_is_an_estimate_with_margin():
    assert 0 < budget()["cycle_with_25_percent_reserve_s"] <= 12
    assert budget()["slowest_test_two_station_dwell_s"] + budget()["travel_dock_budget_s"] < budget()["proposed_mission_limit_s"]
    assert move_time(.2) > 0
    with pytest.raises(ValueError):
        move_time(-1)


def test_cli_exit_codes(tmp_path):
    path = tmp_path / "synthetic-checker-fixture.jsonl"
    events = fixture()
    path.write_text("\n".join(json.dumps(e) for e in events))
    args = [sys.executable, "-m", "advaitic_systems.evidence", str(path)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and json.loads(result.stdout)["passed"]
    next(e for e in events if e["type"] == "inspection")["z_m"] = .9
    path.write_text("\n".join(json.dumps(e) for e in events))
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and not json.loads(result.stdout)["passed"]
