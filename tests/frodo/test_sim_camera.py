"""RGB-D cameras: frame layout, intrinsics, projection consistency, rate, selection.

Needs an off-screen GL context (MUJOCO_GL=egl, set in conftest); skipped where unavailable.
"""
import mujoco
import numpy as np
import pytest

from frodo.cell import CELL, STATIONS
from frodo.config import get_config
from frodo.interfaces import RobotCommand
from frodo.robot.description import ARM_HOME
from frodo.sim.actuation import ActuatorBridge
from frodo.sim.sensors import SensorSuite

CFG = get_config()
CAMS = CFG.sensors.cameras


@pytest.fixture(scope="module")
def gl(kit_model):
    try:
        renderer = mujoco.Renderer(kit_model, 8, 8)
    except Exception as exc:  # noqa: BLE001 - any GL initialisation failure means "no GPU/EGL"
        pytest.skip(f"off-screen rendering unavailable: {exc}")
    renderer.close()


@pytest.fixture
def docked(kit_model, fresh, gl):
    data = fresh(kit_model, STATIONS["A"].dock, ARM_HOME)
    suite = SensorSuite(kit_model)
    yield suite, data
    suite.close()


def base_frame(model, data, world_xyz):
    base = model.body("base_link").id
    return data.xmat[base].reshape(3, 3).T @ (np.asarray(world_xyz) - data.xpos[base])


@pytest.mark.parametrize("name", CAMS.enabled)
def test_frame_shapes_and_intrinsics(docked, kit_model, name):
    suite, data = docked
    frame = suite.render_camera(name, data)
    assert frame.name == name and frame.t == data.time
    assert frame.rgb.shape == (CAMS.height, CAMS.width, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (CAMS.height, CAMS.width) and frame.depth.dtype == np.float32
    fx, fy, cx, cy = frame.intrinsics
    fovy = np.deg2rad(kit_model.cam_fovy[kit_model.camera(name).id])
    assert fy == pytest.approx(CAMS.height / 2 / np.tan(fovy / 2))
    assert fx == fy and (cx, cy) == (CAMS.width / 2, CAMS.height / 2)
    pose = frame.cam_in_base
    assert pose.shape == (4, 4) and np.allclose(pose[3], [0, 0, 0, 1])
    assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    valid = frame.depth > 0
    assert valid.mean() > 0.1      # the rest is beyond depth_range_m (open space)
    near, far = CAMS.depth_range_m
    # The range gate applies to the true depth; noise (sigma = k z^2) is added afterwards.
    slack = 6 * CAMS.depth_noise_std_m
    assert frame.depth[valid].min() >= near - slack * near ** 2
    assert frame.depth[valid].max() <= far + slack * far ** 2


def test_mast_pose_matches_config(docked):
    suite, data = docked
    frame = suite.render_camera("mast", data)
    assert frame.cam_in_base[:3, 3] == pytest.approx(CAMS.mast.pos_m)
    yaw, pitch = np.deg2rad(CAMS.mast.yaw_deg), np.deg2rad(CAMS.mast.pitch_deg)
    forward = [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)]
    assert frame.cam_in_base[:3, 2] == pytest.approx(forward)       # optical z = view direction
    assert frame.cam_in_base[2, 1] < 0                              # optical y points down


def test_project_deproject_consistency_with_cam_in_base(docked, kit_model):
    """Known points -> pixel via cam_in_base -> depth image -> back to the base frame."""
    suite, data = docked
    frame = suite.render_camera("mast", data)
    ring = base_frame(kit_model, data, data.body("kit_A0_ring").xpos
                      + [0.032, 0, CELL["ring"]["thickness_m"] / 2])   # on the ring's top face
    bottle_top = base_frame(kit_model, data, data.body("kit_A0_bottle").xpos
                            + [0, 0, CELL["bottle"]["height_m"]])
    for point in (ring, bottle_top):
        u, v, z = frame.project(point)
        assert 0 <= u < CAMS.width and 0 <= v < CAMS.height and z > 0
        measured = frame.depth[int(round(float(v))), int(round(float(u)))]
        assert measured == pytest.approx(z, abs=0.006)
        back = frame.deproject(u, v)
        assert np.linalg.norm(back - point) < 0.008
        assert np.linalg.norm(frame.deproject(u, v, z) - point) < 1e-9   # exact inverse


def test_wrist_cam_in_base_follows_the_arm(docked, kit_model):
    suite, data = docked
    frame = suite.render_camera("wrist", data)
    cam = kit_model.camera("wrist").id
    assert frame.cam_in_base[:3, 3] == pytest.approx(
        base_frame(kit_model, data, data.cam_xpos[cam]), abs=1e-9)
    # A point 0.5 m straight ahead of the lens projects to the image centre.
    ahead = frame.cam_in_base[:3, 3] + 0.5 * frame.cam_in_base[:3, 2]
    u, v, z = frame.project(ahead)
    assert (float(u), float(v), float(z)) == pytest.approx((CAMS.width / 2, CAMS.height / 2, 0.5))


def test_ring_is_blue_where_it_projects(docked, kit_model):
    suite, data = docked
    frame = suite.render_camera("mast", data)
    point = base_frame(kit_model, data, data.body("kit_A0_ring").xpos + [0.032, 0, 0.004])
    u, v, _ = frame.project(point)
    r, g, b = frame.rgb[int(round(float(v))), int(round(float(u)))].astype(int)
    assert b > r + 40 and b > g


def test_only_perception_source_in_frame_at_camera_rate(docked, kit_model, step):
    suite, data = docked
    bridge = ActuatorBridge(kit_model)
    seconds = 1.0
    frames = step(kit_model, data, bridge, RobotCommand(arm_q=ARM_HOME), seconds, suite)
    images = [f.camera for f in frames if f.camera is not None]
    assert len(images) == pytest.approx(seconds * CAMS.rate_hz, abs=1)
    assert {i.name for i in images} == {CAMS.perception_source}
    assert set(suite._cameras) == {CAMS.perception_source}      # the other one never rendered
    delays = [f.t - f.camera.t for f in frames if f.camera is not None]
    dt = CFG.sim.control_dt_s
    assert np.allclose(delays, np.ceil(CAMS.latency_s / dt - 1e-6) * dt, atol=1e-6)


def test_depth_noise_is_seeded(kit_model, fresh, gl):
    data = fresh(kit_model, STATIONS["A"].dock, ARM_HOME)
    shots = []
    for seed in (3, 3, 4):
        suite = SensorSuite(kit_model, seed=seed)
        shots.append(suite.render_camera("mast", data).depth)
        suite.close()
    assert np.array_equal(shots[0], shots[1])
    assert not np.array_equal(shots[0], shots[2])
    diff = (shots[0] - shots[2])[(shots[0] > 0) & (shots[2] > 0)]
    assert 0 < np.abs(diff).mean() < 10 * CAMS.depth_noise_std_m


def test_disabled_camera_cannot_be_rendered(kit_model, fresh):
    suite = SensorSuite(kit_model, enable_cameras=False)
    data = fresh(kit_model)
    assert suite.read(data).camera is None
    with pytest.raises(ValueError):
        suite.render_camera("mast", data)
    with pytest.raises(ValueError):
        SensorSuite(kit_model).render_camera("overview", data)
