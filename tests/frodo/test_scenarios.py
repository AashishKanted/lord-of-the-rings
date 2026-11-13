"""Scenario equipment and the robot's reactions, without a full mission."""
import threading

import numpy as np
import pytest

from advaitic_systems.modbus import PLCServer
from frodo.cell import STATIONS
from frodo.config import get_config, with_overrides
from frodo.interfaces import JointState, LidarScan, Pose2D, SensorFrame
from frodo.robot.frames import StationFrame
from frodo.robot.plc import ASSEMBLY_DONE, READY, LineClient
from frodo.robot.safety.monitor import SafetyMonitor
from frodo.scenarios import make_scenario
from frodo.scenarios.human_stop import footprint_distance
from frodo.scenarios.plc_retry import AckDropRelay

CFG = get_config()
ANGLES = np.linspace(-np.pi, np.pi, 720, endpoint=False)
DOCK_A = Pose2D(*(float(v) for v in STATIONS["A"].dock))


def scan_frame(points, t=1.0):
    ranges = np.full(len(ANGLES), np.inf)
    for x, y in points:
        ranges[np.argmin(np.abs(np.angle(np.exp(1j * (ANGLES - np.arctan2(y, x))))))] = np.hypot(x, y)
    scans = [LidarScan(t, m.name, Pose2D(0.0, 0.0, 0.0), ANGLES, ranges)
             for m in CFG.sensors.lidar.mounts]
    return SensorFrame(t=t, joints=JointState(t, np.zeros(7), 0.0), lidars=scans)


def conveyor_edge():
    gap = 0.8 - 0.25 - CFG.safety.footprint_m / 2          # docked: conveyor edge to footprint
    return [(CFG.safety.footprint_m / 2 + gap, y) for y in np.linspace(-0.6, 0.6, 25)]


def person(x, y, r=0.18):
    return [(x + r * np.cos(a), y + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)]


# ---------------- work-zone latch ----------------
def test_docked_conveyor_alone_does_not_stop_the_robot():
    monitor = SafetyMonitor(CFG)
    assert not monitor.check_zone(scan_frame(conveyor_edge()), StationFrame.from_pose("A", DOCK_A))


def test_anything_inside_the_work_zone_latches_until_clear_and_acknowledged():
    monitor, station = SafetyMonitor(CFG), StationFrame.from_pose("A", DOCK_A)
    inside = conveyor_edge() + person(0.0, 0.285 + 0.6)          # 0.6 m to the robot's left
    assert monitor.check_zone(scan_frame(inside, t=1.0), station)
    assert monitor.state.latched and monitor.state.speed_scale == 0.0
    assert not monitor.acknowledge(1.1)                          # still occupied: refused
    assert monitor.check_zone(scan_frame(conveyor_edge(), t=1.2), station)   # gone, still latched
    assert monitor.acknowledge(1.3) and not monitor.latched


def test_objects_outside_the_zone_or_in_transit_do_not_latch():
    monitor, station = SafetyMonitor(CFG), StationFrame.from_pose("A", DOCK_A)
    far = conveyor_edge() + person(0.0, 0.285 + 1.4)
    assert not monitor.check_zone(scan_frame(far), station)
    assert not monitor.check_zone(scan_frame(person(0.0, 0.6)), None)    # not docked: no zone


def test_footprint_distance_used_for_the_human_evidence():
    pose = (3.0, 2.0, -np.pi / 2)
    assert footprint_distance(pose, (3.0, 2.0), 0.285) == 0.0
    assert footprint_distance(pose, (4.285 + 0.5, 2.0), 0.285) == pytest.approx(1.5)


def test_human_scenario_duration_must_stay_inside_the_customer_range():
    for bad in (1.9, 4.0):
        with pytest.raises(ValueError):
            make_scenario("human_stop", with_overrides(
                CFG, {"scenarios": {"human_stop": {"duration_s": bad}}}))


# ---------------- PLC fault relay ----------------
@pytest.fixture
def line_server():
    with PLCServer(("127.0.0.1", 0)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield server
        server.shutdown()
        thread.join()


def test_relay_drops_one_acknowledgement_and_the_retry_is_applied_once(line_server):
    accepted = lambda sid: line_server.model.accepted[sid]   # noqa: E731
    relay = AckDropRelay(line_server.server_address[1], 1, lambda: 0.0, accepted)
    try:
        robot = LineClient("127.0.0.1", relay.port)
        robot.ready(1)
        with line_server.lock:
            line_server.model.arrive(1, 4)
        robot.assembly_done(1, 4)                  # first try loses its ack, retry succeeds
        assert relay.dropped is not None and relay.retry is not None
        first, retry = relay.dropped["request_hex"], relay.retry["request_hex"]
        assert len(first) == len(retry) == 42
        assert first[4:] == retry[4:] and first[:4] != retry[:4]   # new transaction id only
        assert robot.ledger[-1]["command"] == ASSEMBLY_DONE and robot.ledger[-1]["attempts"] == 2
        with line_server.lock:
            applied = [c for c in line_server.model.commands if c["command"] == ASSEMBLY_DONE]
            assert len(applied) == 1 and line_server.model.last_status == 2
        assert relay.dropped["accepted_before"] == relay.retry["accepted_after"] == 0
        robot.ready(1, False)                      # later traffic passes through untouched
        assert [c["command"] for c in robot.ledger] == [READY, ASSEMBLY_DONE, 2]
        dropped = [row for row in relay.trace if row.get("event")]
        assert len(dropped) == 1
    finally:
        relay.close()
