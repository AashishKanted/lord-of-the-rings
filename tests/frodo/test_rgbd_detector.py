"""RGB-D detector on rendered wrist-camera frames (depth-band segmenter, no model).

Still frames with the robot docked and the arm at its waiting pose. Simulator truth is
used only to score the result.
"""
import mujoco
import numpy as np
import pytest

from frodo.cell import CELL, STATIONS
from frodo.config import get_config, with_overrides
from frodo.interfaces import JointState, Pose2D, SensorFrame
from frodo.plant.feeder import find_pool
from frodo.robot.frames import StationFrame
from frodo.robot.manipulation.arm_controller import ArmController
from frodo.robot.perception.geometry import fit_ring, rim
from frodo.robot.perception.rgbd_detector import RgbdDetector
from frodo.sim.oracle import OracleKitSource
from frodo.sim.scene import build_spec, reset
from frodo.sim.sensors import SensorSuite

CFG = with_overrides(get_config(), {"perception": {"method": "rgbd", "segmenter": "depth_bands"}})
BOTTLE_H = CELL["bottle"]["height_m"]


@pytest.fixture(scope="module")
def cell(empty_model):
    st = STATIONS["A"]
    station = StationFrame.from_pose("A", Pose2D(*(float(v) for v in st.dock)))
    arm = ArmController(empty_model, CFG)
    arm.set_station(station)
    model = build_spec({"A": [0.0]}).compile()
    data = mujoco.MjData(model)
    suite = SensorSuite(model, CFG, seed=0)

    def look(kit_x, ring_on_kit=True):
        """Camera frame with the kit centre at station-local x; returns (frame, truth)."""
        reset(model, data, st.dock, arm.q_ready)
        kit = find_pool(model, st, CFG)[0]
        kit.place_kit(data, kit_x, 0.0)
        kit.place_ring_on_nest(data, kit_x if ring_on_kit else kit_x + 0.6, 0.0)
        mujoco.mj_forward(model, data)
        return suite.render_camera("wrist", data), OracleKitSource(model).read(data)[0]
    yield station, look
    suite.close()


def test_wrist_camera_looks_straight_down_from_the_waiting_pose(cell):
    _, look = cell
    image, _ = look(-0.05)
    assert image.cam_in_base[:3, 2] == pytest.approx([0, 0, -1], abs=0.01)   # optical axis


@pytest.mark.parametrize("kit_x", [-0.10, -0.05, 0.0])
def test_kit_is_found_within_a_millimetre(cell, kit_x):
    station, look = cell
    image, truth = look(kit_x)
    detector = RgbdDetector(CFG)
    frame = SensorFrame(t=image.t, joints=JointState(image.t, np.zeros(7), 0.0), camera=image)
    kits = detector.detect(frame, station)
    assert len(kits) == 1
    assert np.linalg.norm(kits[0].ring_in_base - truth.ring_in_base) < 0.001
    assert np.linalg.norm(kits[0].bottle_in_base - truth.bottle_in_base) < 0.001
    assert kits[0].t == image.t


def test_no_camera_frame_means_no_new_detection(cell):
    station, _ = cell
    frame = SensorFrame(t=0.0, joints=JointState(0.0, np.zeros(7), 0.0))
    assert RgbdDetector(CFG).detect(frame, station) is None


def test_bottle_without_its_ring_is_reported_ringless_and_starts_no_track(cell):
    from frodo.interfaces import ConveyorSignal
    from frodo.robot.perception.tracker import KitTracker
    station, look = cell
    image, _ = look(-0.05, ring_on_kit=False)
    report = RgbdDetector(CFG).analyse(image, station)
    assert len(report.bottles) == 1 and len(report.kits) == 1
    assert report.kits[0].ring_in_base is None
    tracker = KitTracker(CFG)
    tracker.reset(station)
    assert tracker.update(image.t, [ConveyorSignal(image.t, "A", 0.0, 0.03)], report.kits, "t") == []


def test_half_hidden_ring_is_rejected_not_misplaced(cell):
    station, look = cell
    image, truth = look(-0.05)
    detector = RgbdDetector(CFG)
    ring_mask = next(m for m in detector.segmenter.masks(image)
                     if fit_ring(image, m, CFG.perception).valid)
    u, _, _ = image.project(truth.ring_in_base)
    hidden = ring_mask.copy()
    hidden[:, :int(u) + 5] = False                 # something covers a bit more than half
    fit = fit_ring(image, hidden, CFG.perception)
    assert rim(hidden).sum() > CFG.perception.min_rim_points and not fit.valid


def test_robot_computes_the_camera_pose_from_its_own_joint_measurements(cell, empty_model):
    """The robot does not need the simulator's camera pose: forward kinematics of the
    measured joints gives the same pose."""
    from frodo.robot.perception.camera_pose import CameraPose
    station, look = cell
    image, _ = look(-0.05)
    pose = CameraPose(empty_model, "wrist")
    arm = ArmController(empty_model, CFG)
    arm.set_station(station)
    pose.record(JointState(image.t, arm.q_ready, 0.0))
    fk = pose.at(image.t)
    assert fk[:3, 3] == pytest.approx(image.cam_in_base[:3, 3], abs=1e-4)
    assert fk[:3, :3] == pytest.approx(image.cam_in_base[:3, :3], abs=1e-4)
