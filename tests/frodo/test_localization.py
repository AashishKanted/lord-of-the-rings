"""Scan matcher and EKF against the kinematic test harness (synthetic scans, no MuJoCo)."""
from dataclasses import replace
from math import cos, hypot, radians, sin

import numpy as np
import pytest

from frodo.apps.nav_demo import (
    HarnessOptions,
    KinematicCell,
    box_segments,
    lidar_scans,
    pose_error,
)
from frodo.config import get_config
from frodo.interfaces import LidarScan, Pose2D, Twist2D
from frodo.robot.localization import Localizer, ScanMatcher
from frodo.robot.map import build_map, cell_segments

CFG = get_config()
TRUE_POSES = [Pose2D(1.0, 3.0, 0.0), Pose2D(3.0, 2.0, -np.pi / 2), Pose2D(3.0, 4.0, np.pi / 2),
              Pose2D(2.2, 2.7, 0.7), Pose2D(4.6, 3.3, -2.5)]
# Prior error (dx m, dy m, dyaw deg): mm/deg-level up to 10 cm / 5 deg.
OFFSETS = [(0.003, -0.002, 0.2), (-0.01, 0.01, -1.0), (0.05, -0.04, 2.0), (0.10, 0.0, 5.0),
           (-0.07, 0.07, -5.0)]


def test_map_has_walls_and_conveyors():
    segments = cell_segments()
    assert segments.shape == (12, 4)
    grid = build_map(0.05, 0.45)
    assert grid.is_free(1.0, 3.0) and grid.is_free(3.0, 2.0) and grid.is_free(3.0, 4.0)
    assert not grid.is_free(3.0, 1.2)       # inside conveyor A
    assert not grid.is_free(0.2, 3.0)       # too close to the west wall
    assert not grid.is_free(3.0, 1.8)       # closer than the inflation to conveyor A
    assert not grid.segment_free((1.0, 0.7), (5.0, 0.7))


@pytest.mark.parametrize("true", TRUE_POSES, ids=lambda p: f"{p.x},{p.y}")
def test_icp_recovers_pose_offsets(true):
    matcher = ScanMatcher(CFG)
    rng = np.random.default_rng(0)
    for dx, dy, dyaw in OFFSETS:
        scans = lidar_scans(true, 0.0, CFG, rng)
        match = matcher.match(scans, Pose2D(true.x + dx, true.y + dy, true.yaw + radians(dyaw)))
        assert match is not None, (dx, dy, dyaw)
        distance, angle = pose_error(match.pose, true)
        assert distance < 0.002 and angle < radians(0.1), (dx, dy, dyaw, distance, angle)
        assert match.inlier_fraction > 0.9


def test_icp_ignores_unmapped_obstacles():
    """A person-sized box mid-room and a box against a wall must not bias the pose."""
    true = Pose2D(2.2, 2.7, 0.7)
    extra = np.vstack([box_segments(3.2, 3.2, 0.4), box_segments(0.25, 2.0, 0.4)])
    rng = np.random.default_rng(1)
    matcher = ScanMatcher(CFG)
    clean = lidar_scans(true, 0.0, CFG, None)
    scans = lidar_scans(true, 0.0, CFG, rng, extra)
    blocked = sum(int(np.sum(np.abs(s.ranges - c.ranges) > 0.05))
                  for s, c in zip(scans, clean, strict=True))
    assert blocked > 50                      # the obstacles really are in the scan
    match = matcher.match(scans, Pose2D(true.x + 0.02, true.y - 0.02, true.yaw + radians(1.0)))
    assert match is not None
    distance, angle = pose_error(match.pose, true)
    assert distance < 0.003 and angle < radians(0.15)
    assert match.inlier_fraction < 0.97      # obstacle points were left out of the fit


def test_icp_refuses_unreliable_scans():
    matcher = ScanMatcher(CFG)
    true = Pose2D(2.2, 2.7, 0.7)
    scans = lidar_scans(true, 0.0, CFG, None)
    # Surrounded by unmapped clutter: almost nothing fits the map.
    boxed_in = lidar_scans(true, 0.0, CFG, None, box_segments(true.x, true.y, 1.6))
    assert matcher.match(boxed_in, true) is None
    # No returns at all.
    empty = [replace(s, ranges=np.full_like(s.ranges, np.inf)) for s in scans]
    assert matcher.match(empty, true) is None
    # Degenerate geometry: only beams that hit one straight wall constrain one direction.
    one_wall = []
    for s in scans:
        beam = true.yaw + s.mount.yaw + s.angles
        ox = true.x + cos(true.yaw) * s.mount.x - sin(true.yaw) * s.mount.y
        hits_east = np.isclose(ox + s.ranges * np.cos(beam), 6.0, atol=1e-6)
        one_wall.append(LidarScan(s.t, s.name, s.mount, s.angles,
                                  np.where(hits_east, s.ranges, np.inf)))
    assert sum(int(np.isfinite(s.ranges).sum()) for s in one_wall) > 100
    assert matcher.match(one_wall, true) is None


def _scripted_run(seed, options, seconds=20.0):
    """Open-loop figure through the room; returns per-tick true, EKF and odometry poses."""
    sim = KinematicCell(CFG, seed=seed, options=options)
    ekf, odo = Localizer(CFG), Localizer(CFG)
    rows = []
    while sim.t < seconds:
        frame = sim.sense()
        estimate = ekf.update(frame)
        reckoned = odo.update(replace(frame, lidars=[]))
        rows.append((sim.true_pose, estimate, reckoned, ekf.healthy))
        t = sim.t
        sim.apply(Twist2D(0.25 * cos(0.4 * t), 0.2 * sin(0.3 * t), 0.4 * sin(0.5 * t)))
    return rows


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ekf_tracks_and_beats_odometry(seed):
    rows = _scripted_run(seed, HarnessOptions(odometry_scale=(1.02, 0.98, 1.0)))
    ekf_err = np.array([pose_error(e, t) for t, e, _, _ in rows])
    odo_err = np.array([pose_error(o, t) for t, _, o, _ in rows])
    print(f"\nseed {seed}: EKF max {1e3 * ekf_err[:, 0].max():.1f} mm / "
          f"{np.degrees(ekf_err[:, 1].max()):.2f} deg, rms {1e3 * np.sqrt(np.mean(ekf_err[:, 0] ** 2)):.1f} mm"  # noqa: E501
          f" | odometry-only max {1e3 * odo_err[:, 0].max():.1f} mm / "
          f"{np.degrees(odo_err[:, 1].max()):.2f} deg")
    assert ekf_err[:, 0].max() < 0.015 and ekf_err[:, 1].max() < radians(0.5)
    assert ekf_err[:, 0].max() < 0.5 * odo_err[:, 0].max()
    assert ekf_err[-1, 0] < 0.5 * odo_err[-1, 0]
    assert all(healthy for *_, healthy in rows[20:])


def test_ekf_gates_out_a_wrong_match_and_reports_health():
    ekf = Localizer(CFG, Pose2D(2.0, 3.0, 0.0))
    scans = lidar_scans(Pose2D(2.0, 3.0, 0.0), 0.0, CFG, None)
    good = ekf.matcher.match(scans, ekf.pose)
    assert ekf.correct(good)
    before = ekf.pose
    wrong = replace(good, pose=Pose2D(good.pose.x + 0.3, good.pose.y, good.pose.yaw))
    assert not ekf.correct(wrong)
    assert ekf.pose == before and ekf.rejected_in_a_row == 1

    # Without scans the filter coasts on odometry and must flag itself unhealthy.
    sim = KinematicCell(CFG, seed=0)
    ekf = Localizer(CFG)
    for _ in range(30):
        ekf.update(sim.sense())
        sim.apply(Twist2D(0.2, 0.0, 0.0))
    assert ekf.healthy
    for _ in range(int(1.5 * CFG.localization.health_timeout_s / sim.dt)):
        ekf.update(replace(sim.sense(), lidars=[]))
        sim.apply(Twist2D(0.2, 0.0, 0.0))
    assert not ekf.healthy
    assert hypot(ekf.pose.x - sim.true_pose.x, ekf.pose.y - sim.true_pose.y) < 0.02
