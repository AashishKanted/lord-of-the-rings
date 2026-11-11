"""Simulated lidar, odometry, IMU, encoder and joint sensors (no rendering)."""
from math import cos, pi, sin

import numpy as np
import pytest

from frodo.cell import CELL
from frodo.config import get_config
from frodo.interfaces import RobotCommand, Twist2D
from frodo.sim.actuation import ActuatorBridge
from frodo.sim.lidar import coverage
from frodo.sim.sensors import SensorSuite

CFG = get_config()
LIDAR = CFG.sensors.lidar
DT = CFG.sim.control_dt_s


def suite_for(model, **kw):
    return SensorSuite(model, enable_cameras=False, **kw)


def read_scans(suite, data):
    """Scan now, bypassing rate and latency."""
    return [lidar.scan(data, np.random.default_rng(1)) for lidar in suite.lidars]


def expected_range(origin, heading, obstacles):
    """Distance along a ray to the nearest axis-aligned box [(xmin, ymin, xmax, ymax)] seen
    from outside, or to the room walls seen from inside."""
    lx, ly = CELL["cell_size_m"]
    dx, dy = cos(heading), sin(heading)
    best = np.inf
    for wall, o, d in ((0.0, origin[0], dx), (lx, origin[0], dx), (0.0, origin[1], dy),
                       (ly, origin[1], dy)):
        if abs(d) > 1e-12 and (wall - o) / d > 0:
            best = min(best, (wall - o) / d)
    for xmin, ymin, xmax, ymax in obstacles:
        for lo, hi, o, d, o2, d2, lo2, hi2 in ((xmin, xmax, origin[0], dx, origin[1], dy, ymin, ymax),
                                               (ymin, ymax, origin[1], dy, origin[0], dx, xmin, xmax)):
            for face in (lo, hi):
                if abs(d) > 1e-12 and (face - o) / d > 0:
                    r = (face - o) / d
                    if lo2 <= o2 + r * d2 <= hi2:
                        best = min(best, r)
    return best


def conveyor_boxes():
    c = CELL["conveyor"]
    return [(s["conveyor_center_world"][0] - c["length_m"] / 2,
             s["conveyor_center_world"][1] - c["width_m"] / 2,
             s["conveyor_center_world"][0] + c["length_m"] / 2,
             s["conveyor_center_world"][1] + c["width_m"] / 2) for s in CELL["stations"].values()]


@pytest.mark.parametrize("pose", [(3.0, 3.0, 0.0), (1.0, 3.0, 0.0), (2.2, 2.6, 0.7),
                                  (3.0, 2.0, -pi / 2)])
def test_lidar_ranges_match_room_geometry(empty_model, fresh, pose):
    data = fresh(empty_model, pose)
    suite = suite_for(empty_model)
    boxes = conveyor_boxes()
    for scan in read_scans(suite, data):
        yaw = pose[2]
        origin = (pose[0] + cos(yaw) * scan.mount.x - sin(yaw) * scan.mount.y,
                  pose[1] + sin(yaw) * scan.mount.x + cos(yaw) * scan.mount.y)
        expect = np.array([expected_range(origin, yaw + scan.mount.yaw + a, boxes)
                           for a in scan.angles])
        assert np.isfinite(scan.ranges).all()
        err = scan.ranges - expect
        # Rays grazing a corner can land on either side of an edge; ignore a handful.
        close = np.abs(err) < 5 * LIDAR.range_noise_std_m
        assert close.mean() > 0.98, f"{scan.name}: {(~close).sum()} rays off"
        assert abs(err[close].mean()) < 0.001
        assert 0.5 * LIDAR.range_noise_std_m < err[close].std() < 1.5 * LIDAR.range_noise_std_m


def test_lidar_message_layout(empty_model, fresh):
    data = fresh(empty_model)
    suite = suite_for(empty_model)
    scans = read_scans(suite, data)
    assert [s.name for s in scans] == [m.name for m in LIDAR.mounts]
    for scan, mount in zip(scans, LIDAR.mounts, strict=True):
        assert scan.angles.shape == scan.ranges.shape == (LIDAR.rays,)
        assert scan.angles[0] == pytest.approx(-np.deg2rad(LIDAR.fov_deg) / 2)
        assert scan.angles[-1] == pytest.approx(np.deg2rad(LIDAR.fov_deg) / 2)
        assert (scan.mount.x, scan.mount.y) == pytest.approx((mount.x, mount.y))
        assert scan.mount.yaw == pytest.approx(np.deg2rad(mount.yaw_deg))
        assert scan.t == data.time


def test_lidars_cover_360_degrees_without_self_occlusion(empty_model):
    suite = suite_for(empty_model)
    assert not any(lidar.blind.any() for lidar in suite.lidars)
    for radius in (0.40, 1.0, 5.0):
        _, covered = coverage(suite.lidars, radius)
        assert covered.all(), f"gap at {radius} m"
    # The only unseen area is inside the square spanned by the two mounts (base corners).
    _, near = coverage(suite.lidars, 0.30)
    assert not near.all()


def test_lidar_no_return_is_inf(empty_model, fresh):
    data = fresh(empty_model, (3.0, 3.0, 0.0))
    suite = suite_for(empty_model)
    lidar = suite.lidars[0]
    lidar.range_max = 1.0            # conveyors are 1.2+ m away, walls further
    try:
        scan = lidar.scan(data, np.random.default_rng(0))
    finally:
        lidar.range_max = float(LIDAR.range_max_m)
    assert np.isinf(scan.ranges).all()


def test_lidar_sees_a_kit_free_scene_not_the_robot(kit_model, fresh):
    """Kits ride above the scan plane and the robot never appears in its own scan."""
    data = fresh(kit_model, (3.0, 2.0, -pi / 2))
    suite = suite_for(kit_model)
    for scan in read_scans(suite, data):
        assert scan.ranges.min() > 0.25


def test_multi_rate_and_latency(empty_model, fresh, step):
    data = fresh(empty_model)
    suite, bridge = suite_for(empty_model), ActuatorBridge(empty_model)
    seconds = 2.0
    frames = step(empty_model, data, bridge, RobotCommand(), seconds, suite)
    assert all(f.joints is not None and f.joints.t == f.t for f in frames)
    assert sum(f.odometry is not None for f in frames) == len(frames)
    assert sum(f.imu is not None for f in frames) == len(frames)
    with_scans = [f for f in frames if f.lidars]
    assert len(with_scans) == pytest.approx(seconds * LIDAR.rate_hz, abs=1)
    assert all(len(f.lidars) == len(LIDAR.mounts) for f in with_scans)
    # Capture times are spaced by the period and delivered `latency_s` later.
    stamps = np.array([f.lidars[0].t for f in with_scans])
    assert np.allclose(np.diff(stamps), 1 / LIDAR.rate_hz, atol=1e-6)
    delays = np.array([f.t - f.lidars[0].t for f in with_scans])
    assert np.allclose(delays, np.ceil(LIDAR.latency_s / DT - 1e-6) * DT, atol=1e-6)
    assert all(f.camera is None for f in frames)       # cameras disabled here
    assert all(f.conveyors == [] for f in frames)      # no kits, no plant hook


def test_determinism_with_fixed_seed(empty_model, fresh, step):
    def record(seed):
        data = fresh(empty_model)
        suite, bridge = suite_for(empty_model, seed=seed), ActuatorBridge(empty_model)
        frames = step(empty_model, data, bridge, RobotCommand(Twist2D(0.2, 0.1, 0.3)), 0.5, suite)
        scans = np.concatenate([s.ranges for f in frames for s in f.lidars])
        odom = np.array([[f.odometry.twist.vx, f.odometry.twist.vy, f.odometry.twist.wz,
                          f.imu.gyro_z, *f.imu.accel_xy] for f in frames])
        return scans, odom, np.array([f.joints.arm_q for f in frames])

    a, b, c = record(7), record(7), record(8)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[0], c[0])
    assert not np.array_equal(a[1], c[1])


def test_reset_replays_the_same_noise(empty_model, fresh):
    suite = suite_for(empty_model)
    data = fresh(empty_model)
    first = suite.read(data).joints.arm_q
    suite.reset()
    assert np.array_equal(first, suite.read(data).joints.arm_q)


def test_odometry_twist_is_in_the_base_frame(empty_model, fresh, step):
    """Robot facing +y world, driving forward: odometry says +vx, the world says +y."""
    data = fresh(empty_model, (3.0, 3.0, pi / 2))
    suite, bridge = suite_for(empty_model), ActuatorBridge(empty_model)
    frames = step(empty_model, data, bridge, RobotCommand(Twist2D(vx=0.2, vy=-0.1)), 1.5, suite)
    twist = frames[-1].odometry.twist
    assert twist.vx == pytest.approx(0.2, rel=0.05)
    assert twist.vy == pytest.approx(-0.1, rel=0.05)
    assert abs(twist.wz) < 1e-3
    assert data.qvel[1] > 0.15 and data.qvel[0] > 0.05


def test_odometry_integration_drift(empty_model, fresh, step):
    """Dead reckoning over an L-shaped path with a turn stays within the modelled error."""
    odo = CFG.sensors.odometry
    for seed in range(3):
        data = fresh(empty_model)
        suite, bridge = suite_for(empty_model, seed=seed), ActuatorBridge(empty_model)
        pose, last_t, distance = np.array(CELL["start_pose_world"], float), None, 0.0
        for twist, seconds in ((Twist2D(vx=0.3), 4.0), (Twist2D(wz=0.5), 3.0),
                               (Twist2D(vx=0.2, vy=0.1), 4.0), (Twist2D(), 1.0)):
            for f in step(empty_model, data, bridge, RobotCommand(twist), seconds, suite):
                tw = f.odometry.twist
                if last_t is not None:
                    dt = f.odometry.t - last_t
                    yaw = pose[2] + 0.5 * tw.wz * dt
                    pose += dt * np.array([cos(yaw) * tw.vx - sin(yaw) * tw.vy,
                                           sin(yaw) * tw.vx + cos(yaw) * tw.vy, tw.wz])
                    distance += dt * np.hypot(tw.vx, tw.vy)
                last_t = f.odometry.t
        true = data.qpos[:3]
        assert distance > 2.0
        drift = np.hypot(*(pose[:2] - true[:2]))
        # 3-sigma of the scale error on distance, plus the heading error swinging the path.
        assert drift < 3 * 2 * odo.scale_error_std_frac * distance + 0.005, f"seed {seed}"
        assert abs(pose[2] - true[2]) < 3 * odo.scale_error_std_frac * 1.5 + 0.002
        assert drift > 1e-5      # it is a noisy sensor, not ground truth


def test_imu_gyro_tracks_yaw_rate_with_bias(empty_model, fresh, step):
    imu = CFG.sensors.imu
    data = fresh(empty_model)
    suite, bridge = suite_for(empty_model), ActuatorBridge(empty_model)
    step(empty_model, data, bridge, RobotCommand(Twist2D(wz=0.4)), 1.0, suite)
    frames = step(empty_model, data, bridge, RobotCommand(Twist2D(wz=0.4)), 3.0, suite)
    gyro = np.array([f.imu.gyro_z for f in frames])
    assert gyro.mean() == pytest.approx(0.4 + imu.gyro_bias_radps, abs=4 * imu.gyro_noise_std_radps
                                        / np.sqrt(len(gyro)))
    assert gyro.std() == pytest.approx(imu.gyro_noise_std_radps, rel=0.3)
    accel = np.array([f.imu.accel_xy for f in frames])
    assert np.abs(accel.mean(axis=0)).max() < 0.01


def test_imu_measures_planar_acceleration(empty_model, fresh, step):
    data = fresh(empty_model, (3.0, 3.0, pi / 2))
    suite, bridge = suite_for(empty_model), ActuatorBridge(empty_model)
    frames = step(empty_model, data, bridge, RobotCommand(Twist2D(vx=0.4)), 0.3, suite)
    ax = np.array([f.imu.accel_xy[0] for f in frames[5:]])
    assert ax.mean() == pytest.approx(CFG.actuation.max_linear_accel_mps2, rel=0.1)


def test_joint_state(empty_model, fresh):
    from frodo.robot.description import ARM_HOME
    data = fresh(empty_model, None, ARM_HOME)
    joints = suite_for(empty_model).read(data).joints
    assert joints.arm_q.shape == (7,)
    assert np.allclose(joints.arm_q, ARM_HOME, atol=5 * CFG.sensors.joints.position_noise_std_rad
                       + 1e-9)
    assert joints.gripper == pytest.approx(0.0, abs=0.02)


def test_gripper_measurement_follows_command(empty_model, fresh, step):
    data = fresh(empty_model)
    suite, bridge = suite_for(empty_model), ActuatorBridge(empty_model)
    frames = step(empty_model, data, bridge, RobotCommand(gripper=1.0), 1.5, suite)
    assert frames[-1].joints.gripper > 0.9
    assert data.ctrl[empty_model.actuator("fingers_actuator").id] == 255


def test_conveyor_encoder(kit_model, fresh, step):
    conv = CFG.sensors.conveyor
    speed, seconds = 0.03, 3.0
    data = fresh(kit_model)
    data.ctrl[kit_model.actuator("kit_A0_drive").id] = speed     # stand-in for the plant
    suite, bridge = suite_for(kit_model), ActuatorBridge(kit_model)
    slide = kit_model.joint("kit_A0_slide").qposadr[0]
    travel = []
    frames = step(kit_model, data, bridge, RobotCommand(), seconds, suite,
                  on_tick=lambda: travel.append(float(data.qpos[slide])))
    signals = [s for f in frames for s in f.conveyors]
    at = [x for f, x in zip(frames, travel, strict=True) if f.conveyors]
    assert len(signals) == pytest.approx(seconds * conv.rate_hz, abs=1)
    assert {s.station for s in signals} == {"A"}                 # belt B has no kit, no source
    last = signals[-1]
    # The encoder counts from its first sample, so compare increments.
    assert last.encoder_m - signals[0].encoder_m == pytest.approx(
        at[-1] - at[0], abs=2 * conv.resolution_m + 3e-4)
    assert last.encoder_m > 0.08
    assert last.speed_mps == pytest.approx(speed, abs=5 * conv.speed_noise_std_mps)
    assert all(b.encoder_m >= a.encoder_m for a, b in zip(signals, signals[1:], strict=False))


def test_plant_can_supply_belt_speed(empty_model, fresh, step):
    data = fresh(empty_model)
    suite, bridge = suite_for(empty_model), ActuatorBridge(empty_model)
    suite.set_belt_speed_source(lambda station, data: {"A": 0.03, "B": 0.025}[station])
    frames = step(empty_model, data, bridge, RobotCommand(), 2.0, suite)
    last = {s.station: s for f in frames for s in f.conveyors}
    assert last["A"].encoder_m == pytest.approx(0.06, abs=0.001)
    assert last["B"].encoder_m == pytest.approx(0.05, abs=0.001)


def test_sensor_frame_carries_no_ground_truth(empty_model, fresh):
    """Guard on the message set: a frame holds exactly the customer's sensor list."""
    frame = suite_for(empty_model).read(fresh(empty_model))
    assert set(vars(frame)) == {"t", "joints", "odometry", "imu", "lidars", "camera", "conveyors",
                                "ground_truth"}
    # the oracle slot exists for perception.method = "oracle" but the sensor suite never fills it
    assert frame.ground_truth is None
