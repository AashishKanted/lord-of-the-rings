"""Base safety monitor: synthetic lidar scans, then a physical obstacle in the MuJoCo cell."""
from math import hypot

import mujoco
import numpy as np
import pytest

from frodo.cell import STATIONS
from frodo.config import get_config
from frodo.interfaces import (
    JointState,
    LidarScan,
    Mode,
    Pose2D,
    RobotCommand,
    SensorFrame,
    Twist2D,
)
from frodo.robot.frames import StationFrame
from frodo.robot.plc import LineStatus, StationStatus
from frodo.robot.safety.monitor import SafetyMonitor
from frodo.sim.actuation import ActuatorBridge
from frodo.sim.scene import build_spec, reset
from frodo.sim.sensors import SensorSuite

CFG = get_config()
S = CFG.safety
HALF = S.footprint_m / 2
ANGLES = np.linspace(-np.pi, np.pi, 720, endpoint=False)


def frame(points, t=1.0):
    """A frame whose lidars (both at the base centre here) return exactly `points` (base xy)."""
    ranges = np.full(len(ANGLES), np.inf)
    for x, y in points:
        ranges[np.argmin(np.abs(np.angle(np.exp(1j * (ANGLES - np.arctan2(y, x))))))] = hypot(x, y)
    scans = [LidarScan(t, m.name, Pose2D(0.0, 0.0, 0.0), ANGLES, ranges)
             for m in CFG.sensors.lidar.mounts]
    return SensorFrame(t=t, joints=JointState(t, np.zeros(7), 0.0), lidars=scans)


def drive(vx=0.0, vy=0.0, wz=0.0):
    return RobotCommand(base=Twist2D(vx, vy, wz))


def test_free_space_passes_the_command_through():
    command, state = SafetyMonitor(CFG).filter(frame([(3.0, 0.0)]), drive(0.3))
    assert command.base.vx == 0.3 and state.speed_scale == 1.0 and not state.stopped


def test_obstacle_ahead_slows_then_stops_the_base():
    monitor = SafetyMonitor(CFG)
    mid = (S.slow_distance_m + S.stop_distance_m) / 2
    command, state = monitor.filter(frame([(HALF + mid, 0.0)]), drive(0.3))
    assert state.speed_scale == pytest.approx(0.5, abs=0.02) and not state.stopped
    assert command.base.vx == pytest.approx(0.15, abs=0.01)
    command, state = monitor.filter(frame([(HALF + S.stop_distance_m - 0.05, 0.1)]), drive(0.3))
    assert state.stopped and command.base == Twist2D(0.0, 0.0, 0.0)
    assert "ahead" in state.reason


def check(points, command, mask=None):
    """Verdict of a fresh monitor (no stop being held) on one frame."""
    return SafetyMonitor(CFG).filter(frame(points), command, mask)[1]


def test_only_the_direction_of_travel_counts():
    close_behind = [(-(HALF + 0.20), 0.0)]
    assert check(close_behind, drive(0.3)).speed_scale == 1.0        # driving away
    assert check(close_behind, drive(-0.3)).stopped                  # backing into it
    beside = [(0.0, HALF + S.corridor_margin_m + 0.15)]
    assert check(beside, drive(0.3)).speed_scale == 1.0              # outside the corridor
    assert check(beside, drive(0.0, 0.3)).stopped                    # sliding into it


def test_turning_stops_for_anything_the_corners_would_hit():
    corner_circle = HALF * np.sqrt(2)
    assert check([(corner_circle + 0.02, 0.0)], drive(wz=0.3)).stopped
    assert not check([(corner_circle + 0.20, 0.0)], drive(wz=0.3)).stopped


def test_a_stop_is_held_until_the_obstacle_has_clearly_gone():
    monitor = SafetyMonitor(CFG)
    edge = HALF + S.stop_distance_m
    assert monitor.filter(frame([(edge - 0.01, 0.0)], t=1.0), drive(0.3))[1].stopped
    # range noise puts it just outside the stop distance: still held
    assert monitor.filter(frame([(edge + 0.02, 0.0)], t=1.1), drive(0.3))[1].stopped
    # clearly further away, but not yet for long enough
    far = edge + S.clear_hysteresis_m + 0.10
    assert monitor.filter(frame([(far, 0.0)], t=1.2), drive(0.3))[1].stopped
    command, state = monitor.filter(frame([(far, 0.0)], t=1.2 + S.clear_time_s + 0.01), drive(0.3))
    assert not state.stopped and 0 < command.base.vx < 0.3      # moving again, in the slow zone


def test_arm_and_gripper_commands_are_not_touched_and_idle_base_is_not_flagged():
    monitor = SafetyMonitor(CFG)
    wanted = RobotCommand(base=Twist2D(), arm_q=np.ones(7), gripper=0.5, request_grasp=True)
    command, state = monitor.filter(frame([(HALF + 0.05, 0.0)]), wanted)
    assert command is wanted and not state.stopped


def test_missing_or_stale_lidar_data_stops_the_base():
    monitor = SafetyMonitor(CFG)
    no_scan = SensorFrame(t=0.0, joints=JointState(0.0, np.zeros(7), 0.0))
    assert monitor.filter(no_scan, drive(0.3))[1].reason == "lidar data missing or stale"
    monitor.filter(frame([(3.0, 0.0)], t=1.0), drive(0.3))
    late = SensorFrame(t=1.0 + S.scan_timeout_s + 0.1, joints=JointState(0.0, np.zeros(7), 0.0))
    command, state = monitor.filter(late, drive(0.3))
    assert state.stopped and command.base.vx == 0.0


def conveyor_edge_points(gap):
    """Lidar returns along the near edge of conveyor A, `gap` ahead of the footprint."""
    return [(HALF + gap, y) for y in np.linspace(-0.6, 0.6, 25)]


def test_dock_mask_lets_the_base_close_in_on_the_conveyor_but_not_touch_it():
    monitor = SafetyMonitor(CFG)
    dock = Pose2D(*(float(v) for v in STATIONS["A"].dock))
    gap_at_dock = 0.8 - 0.25 - HALF                      # conveyor edge to footprint, docked
    assert S.dock_min_clearance_m < gap_at_dock < S.stop_distance_m
    at_dock = frame(conveyor_edge_points(gap_at_dock))
    mask = StationFrame.from_pose("A", dock)
    assert monitor.filter(at_dock, drive(0.05))[1].stopped                  # unmasked: too close
    command, state = SafetyMonitor(CFG).filter(at_dock, drive(0.05), mask)
    assert not state.stopped and command.base.vx == 0.05                    # masked: expected
    # 12 cm too far forward (pose estimate says so too): the conveyor is now inside the limit
    ahead = Pose2D(dock.x, dock.y - 0.12, dock.yaw)
    too_close = frame(conveyor_edge_points(gap_at_dock - 0.12))
    state = monitor.filter(too_close, drive(0.05), StationFrame.from_pose("A", ahead))[1]
    assert state.stopped and "conveyor A" in state.reason


def test_dock_mask_does_not_hide_something_standing_in_front_of_the_conveyor():
    monitor = SafetyMonitor(CFG)
    dock = Pose2D(*(float(v) for v in STATIONS["A"].dock))
    pre_dock = Pose2D(dock.x, dock.y + 0.4, dock.yaw)
    points = conveyor_edge_points(0.8 - 0.25 - HALF + 0.4) + [(HALF + 0.30, 0.05)]
    state = monitor.filter(frame(points), drive(0.08), StationFrame.from_pose("A", pre_dock))[1]
    assert state.stopped and "ahead" in state.reason


class _IdleLine:
    """Line controller stub: nothing arrives, nothing is accepted."""
    def poll(self):
        none = StationStatus(False, 0, 0, 0, 0, 0)
        return LineStatus(0, 0, 0, {1: none, 2: none})

    def ready(self, *args, **kwargs):
        pass


def test_robot_stops_short_of_a_box_left_on_its_path():
    """MuJoCo: an unmapped box on the straight line from the start pose to dock A."""
    from frodo.robot.orchestrator.mission import Orchestrator
    spec = build_spec()
    box_xy, box_half = np.array([2.0, 2.7]), 0.10
    body = spec.worldbody.add_body(name="stray_box", pos=[*box_xy, 0.15])
    body.add_geom(name="stray_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[box_half, box_half, 0.15])
    model = spec.compile()
    data = mujoco.MjData(model)
    reset(model, data)
    sensors, bridge = SensorSuite(model, CFG, seed=0, enable_cameras=False), ActuatorBridge(model, CFG)
    robot = Orchestrator(build_spec().compile(), _IdleLine(), CFG)
    base = [model.joint(j).qposadr[0] for j in ("joint_x", "joint_y", "joint_th")]
    box_geom, robot_root = model.geom("stray_box").id, model.body("base_link").id
    touched, stopped_at = False, None
    while data.time < 14.0:
        command = robot.step(sensors.read(data))
        if robot.safety.state.stopped and data.time > 0.5 and stopped_at is None:
            stopped_at = data.time
        bridge.apply(data, command)
        for _ in range(5):
            mujoco.mj_step(model, data)
        for g1, g2 in data.contact.geom[:data.ncon]:
            other = g2 if g1 == box_geom else g1 if g2 == box_geom else None
            touched |= other is not None and model.body_rootid[model.geom_bodyid[other]] == robot_root
    x, y, _ = data.qpos[base]
    centre_gap = hypot(box_xy[0] - x, box_xy[1] - y)
    assert not touched
    assert stopped_at is not None and robot.safety.state.stopped
    assert robot.mode is Mode.NAVIGATE                         # waiting, not docked
    assert np.linalg.norm(data.qvel[base[0]:base[0] + 2]) < 1e-3
    # stopped roughly a stop distance short of the box (footprint and box half-sizes added)
    assert S.stop_distance_m - 0.05 < centre_gap - HALF - box_half < S.stop_distance_m + 0.25
