"""Planner, tracker, docking and the closed loop on the kinematic test harness (no MuJoCo).

The closed-loop tests feed the navigator the EKF estimate only; the harness's true pose
is used for scoring. They are harness results, not MuJoCo robot runs.
"""
from math import degrees, hypot, pi

import numpy as np
import pytest

from frodo.apps.nav_demo import HarnessOptions, box_segments, run_mission, summarize
from frodo.cell import CELL
from frodo.config import get_config
from frodo.interfaces import Pose2D
from frodo.robot.map import closest_on_segments, get_map
from frodo.robot.navigation import (
    DockingController,
    GlobalPlanner,
    Navigator,
    NavStatus,
    PathTracker,
    dock_pose,
    path_length,
    pre_dock_pose,
)

CFG = get_config()
NAV = CFG.navigation
START = Pose2D(*CELL["start_pose_world"])
DT = CFG.sim.control_dt_s


def _assert_path_free(path, grid):
    for p, q in zip(path, path[1:], strict=False):
        assert grid.segment_free((p.x, p.y), (q.x, q.y)), (p, q)


@pytest.mark.parametrize("start,station", [(START, "A"), (START, "B"), (dock_pose("A"), "B"),
                                           (dock_pose("B"), "A")],
                         ids=["start-A", "start-B", "A-B", "B-A"])
def test_planner_reaches_docks(start, station):
    grid = get_map(CFG)
    goal = pre_dock_pose(dock_pose(station), NAV.dock_switch_distance_m)
    path = GlobalPlanner(CFG).plan(start, goal)
    assert path and path[0] == start and path[-1] == goal
    assert len(path) <= 4
    _assert_path_free(path, grid)
    assert path_length(path) < 1.2 * hypot(goal.x - start.x, goal.y - start.y) + 1e-9
    # The final docking move is also clear of the inflated obstacles.
    dock = dock_pose(station)
    assert grid.segment_free((goal.x, goal.y), (dock.x, dock.y))


def test_planner_goes_around_a_conveyor():
    grid = get_map(CFG)
    start, goal = Pose2D(1.0, 0.6, 0.0), Pose2D(5.0, 0.6, pi)
    assert not grid.segment_free((start.x, start.y), (goal.x, goal.y))
    path = GlobalPlanner(CFG).plan(start, goal)
    assert 3 <= len(path) <= 10          # rounded inflation corners need a few points
    _assert_path_free(path, grid)
    for p in path:
        assert grid.clearance(p.x, p.y) >= NAV.inflation_m - 1e-9
    assert GlobalPlanner(CFG).plan(start, Pose2D(3.0, 1.2, 0.0)) == []   # inside conveyor A


def _track(path, end_speed=0.0, seconds=40.0):
    """Ideal base (integrates the command exactly). Returns commands, final pose, worst
    distance to the reference point (lag) and worst distance off the path (cross-track)."""
    lines = np.array([[p.x, p.y, q.x, q.y] for p, q in zip(path, path[1:], strict=False)])
    tracker = PathTracker(CFG)
    tracker.set_path(path, end_speed)
    x, y, yaw = path[0].x, path[0].y, path[0].yaw
    commands, worst_lag, worst_cross = [], 0.0, 0.0
    for _ in range(int(seconds / DT)):
        cmd, reached = tracker.step(Pose2D(x, y, yaw), DT)
        commands.append(cmd)
        c, s = np.cos(yaw), np.sin(yaw)
        x += (c * cmd.vx - s * cmd.vy) * DT
        y += (s * cmd.vx + c * cmd.vy) * DT
        yaw += cmd.wz * DT
        ref = tracker.reference
        worst_lag = max(worst_lag, hypot(ref.x - x, ref.y - y))
        worst_cross = max(worst_cross, closest_on_segments(np.array([[x, y]]), lines)[1][0])
        if reached:
            break
    return commands, Pose2D(x, y, yaw), (worst_lag, worst_cross), reached


def _check_limits(commands):
    v = np.array([[c.vx, c.vy, c.wz] for c in commands])
    speed = np.hypot(v[:, 0], v[:, 1])
    assert speed.max() <= min(NAV.max_linear_mps, CELL["robot"]["max_transit_speed_mps"]) + 1e-9
    assert np.abs(v[:, 2]).max() <= min(NAV.max_angular_radps,
                                        CELL["robot"]["max_yaw_rate_radps"]) + 1e-9
    # World-frame acceleration: undo the base rotation before differencing.
    yaw = np.concatenate([[0.0], np.cumsum(v[:-1, 2]) * DT])
    world = np.column_stack([np.cos(yaw) * v[:, 0] - np.sin(yaw) * v[:, 1],
                             np.sin(yaw) * v[:, 0] + np.cos(yaw) * v[:, 1]])
    accel = np.linalg.norm(np.diff(world, axis=0), axis=1) / DT
    assert accel.max() <= NAV.max_linear_accel_mps2 * 1.02 + 1e-6
    assert np.abs(np.diff(v[:, 2])).max() / DT <= NAV.max_angular_accel_radps2 + 1e-6
    return speed.max(), np.abs(v[:, 2]).max(), accel.max()


def test_tracker_respects_limits_and_follows_path():
    path = [Pose2D(1.0, 3.0, 0.0), Pose2D(2.5, 3.0, 1.0), Pose2D(2.5, 4.0, 2.0),
            Pose2D(4.5, 3.0, 3.0)]
    commands, final, (lag, cross), reached = _track(path)
    speed, yaw_rate, accel = _check_limits(commands)
    print(f"\ntracker: max speed {speed:.3f} m/s, yaw rate {yaw_rate:.3f} rad/s, "
          f"accel {accel:.3f} m/s^2, worst lag {1e3 * lag:.1f} mm, cross-track {1e3 * cross:.1f} mm, "
          f"{len(commands) * DT:.1f} s")
    assert reached
    assert speed > 0.9 * NAV.cruise_speed_mps        # it does reach cruise
    assert hypot(final.x - 4.5, final.y - 3.0) < 0.01 and abs(final.yaw - 3.0) < 0.01
    assert cross < 0.02 and lag < 0.06


def test_tracker_yaw_rate_cap_slows_a_turn_on_a_short_path():
    commands, final, _, reached = _track([Pose2D(3.0, 2.0, -pi / 2), Pose2D(3.0, 2.5, pi / 2)])
    speed, yaw_rate, _ = _check_limits(commands)
    assert reached and abs(abs(final.yaw) - pi / 2) < 0.05
    assert speed < 0.5 * NAV.cruise_speed_mps        # translation waits for the rotation
    assert yaw_rate > 0.9 * NAV.cruise_yaw_rate_radps


def test_tracker_reference_waits_for_a_held_base():
    tracker = PathTracker(CFG)
    tracker.set_path([Pose2D(1.0, 3.0, 0.0), Pose2D(4.0, 3.0, 0.0)])
    for _ in range(500):                              # base held at the start for 5 s
        tracker.step(Pose2D(1.0, 3.0, 0.0), DT)
    assert tracker.s <= NAV.tracker_max_lag_m + 0.01


def test_docking_converges_and_latches():
    dock = dock_pose("A")
    controller = DockingController(CFG)
    x, y, yaw = dock.x + 0.03, dock.y + 0.40, dock.yaw + 0.1
    commands, docked, t = [], False, 0.0
    while not docked and t < 20.0:
        cmd, docked = controller.step(Pose2D(x, y, yaw), dock, DT)
        commands.append(cmd)
        c, s = np.cos(yaw), np.sin(yaw)
        x, y, yaw = x + (c * cmd.vx - s * cmd.vy) * DT, y + (s * cmd.vx + c * cmd.vy) * DT, \
            yaw + cmd.wz * DT
        t += DT
    assert docked
    speed, _, _ = _check_limits(commands)
    assert speed <= NAV.dock_max_linear_mps + 1e-9
    assert hypot(x - dock.x, y - dock.y) < NAV.dock_tolerance_m
    assert abs(yaw - dock.yaw) < NAV.dock_tolerance_rad
    assert y >= dock.y - 0.001                        # no overshoot towards the conveyor
    cmd, still = controller.step(Pose2D(x, y, yaw), dock, DT)
    assert still and hypot(cmd.vx, cmd.vy) < CELL["robot"]["assembly_base_speed_max_mps"]


def test_navigator_fails_cleanly_without_a_path():
    nav = Navigator(CFG)
    nav.go_to_dock("A")
    cmd, status = nav.step(Pose2D(3.0, 1.2, 0.0), DT)    # "inside" conveyor A: no free cell near
    assert status is NavStatus.FAILED and (cmd.vx, cmd.vy, cmd.wz) == (0.0, 0.0, 0.0)


def _check_mission(legs, trace, label):
    stats = summarize(trace, legs)
    print(f"\n[kinematic harness] {label}")
    for leg in legs:
        print(f"  dock {leg.station}: {leg.status.value} in {leg.travel_s:.2f} s, TRUE error "
              f"{leg.true_error_mm:.2f} mm / {leg.true_yaw_error_deg:.3f} deg, estimate error "
              f"{leg.estimate_error_mm:.2f} mm / {leg.estimate_yaw_error_deg:.3f} deg")
    print(f"  EKF max {stats['ekf_pos_max_mm']:.1f} mm, odometry-only max "
          f"{stats['odo_pos_max_mm']:.1f} mm, max speed {stats['max_true_speed_mps']:.3f} m/s, "
          f"max yaw rate {stats['max_true_yaw_rate_radps']:.3f} rad/s, min clearance "
          f"{stats['min_footprint_clearance_m']:.3f} m")
    robot = CELL["robot"]
    for leg in legs:
        assert leg.status is NavStatus.DOCKED
        assert leg.true_error_mm < 1e3 * robot["dock_position_tolerance_m"] / 2.5   # < 10 mm
        assert leg.true_yaw_error_deg < degrees(robot["dock_yaw_tolerance_rad"]) / 4  # < 0.5 deg
        assert leg.travel_s < 20.0
    assert stats["max_true_speed_mps"] <= robot["max_transit_speed_mps"]
    assert stats["max_true_yaw_rate_radps"] <= robot["max_yaw_rate_radps"]
    assert stats["min_footprint_clearance_m"] > 0.05
    assert stats["unhealthy_ticks"] == 0
    return stats


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_loop_start_to_a_to_b(seed):
    legs, trace = run_mission(seed)
    _check_mission(legs, trace, f"seed {seed}, nominal sensors")


def test_closed_loop_with_odometry_scale_error_and_back_to_a():
    options = HarnessOptions(odometry_scale=(1.03, 0.97, 1.0))
    legs, trace = run_mission(0, stations=("B", "A", "B"), options=options)
    stats = _check_mission(legs, trace, "seed 0, 3 % odometry scale error, start-B-A-B")
    assert stats["ekf_pos_max_mm"] < 0.5 * stats["odo_pos_max_mm"]


def test_closed_loop_with_an_unmapped_obstacle_in_view():
    """A static unmapped box beside the route is seen by the lidars; docking stays accurate.
    (Stopping for obstacles is the safety monitor's job, not tested here.)"""
    options = HarnessOptions(extra_segments=box_segments(4.6, 3.0, 0.5))
    legs, trace = run_mission(0, options=options)
    _check_mission(legs, trace, "seed 0, unmapped 0.5 m box at (4.6, 3.0)")
