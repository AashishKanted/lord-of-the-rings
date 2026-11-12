"""Perception scaffold: station frame geometry, belt-encoder tracker, oracle backend."""
import numpy as np
import pytest

from frodo.cell import CELL, STATIONS
from frodo.config import get_config, with_overrides
from frodo.interfaces import ConveyorSignal, GroundTruthKit, JointState, Pose2D, SensorFrame
from frodo.robot.frames import StationFrame
from frodo.robot.perception import KitDetection, Perception
from frodo.robot.perception.detector import make_detector
from frodo.robot.perception.tracker import KitTracker

CFG = with_overrides(get_config(), {"perception": {"method": "oracle"}})
BELT_Z = CELL["conveyor"]["height_m"]


def dock(name):
    return Pose2D(*(float(v) for v in STATIONS[name].dock))


@pytest.mark.parametrize("name", ["A", "B"])
def test_both_stations_look_the_same_from_their_dock(name):
    frame = StationFrame.from_pose(name, dock(name))
    assert frame.center == pytest.approx([0.8, 0.0, BELT_Z], abs=1e-9)    # belt 0.8 m ahead
    assert frame.belt_dir == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)     # travelling to the left
    assert frame.lateral == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)     # robot on the +y side
    point = frame.point(-0.3, 0.045, 0.9)
    assert point == pytest.approx([0.755, -0.3, 0.9], abs=1e-9)
    assert frame.local_x(point) == pytest.approx(-0.3)


def test_station_frame_follows_the_pose_estimate():
    pose = dock("A")
    shifted = StationFrame.from_pose("A", Pose2D(pose.x + 0.01, pose.y, pose.yaw))
    # base 10 mm further along world +x = 10 mm further along the belt = centre 10 mm to its right
    assert shifted.center[:2] == pytest.approx([0.8, -0.01], abs=1e-9)


def signal(t, travel, speed=0.03):
    return [ConveyorSignal(t, "A", travel, speed)]


def detection(t, bottle_y, x=0.755):
    bottle = np.array([x, bottle_y, 0.775])
    return KitDetection(t, bottle + [0.0, 0.14, 0.004], bottle)


def test_tracker_carries_a_kit_forward_with_the_encoder():
    tracker = KitTracker(CFG)
    station = StationFrame.from_pose("A", dock("A"))
    tracker.reset(station)
    first = tracker.update(0.0, signal(0.0, 0.0), [detection(0.0, -0.30)], "test")
    assert len(first) == 1 and first[0].age_s == 0.0
    # no new detection for 0.5 s: the belt moved 15 mm, and so must the kit
    later = tracker.update(0.5, signal(0.5, 0.015), None, "test")
    assert later[0].track_id == first[0].track_id
    assert later[0].bottle_in_base == pytest.approx([0.755, -0.285, 0.775], abs=1e-9)
    assert later[0].ring_in_base[1] == pytest.approx(-0.145, abs=1e-9)
    assert later[0].age_s == pytest.approx(0.5)
    assert later[0].belt_velocity == pytest.approx([0.0, 0.03, 0.0])
    # between encoder samples the travel is extrapolated with the measured speed
    between = tracker.update(0.6, [], None, "test")
    assert between[0].bottle_in_base[1] == pytest.approx(-0.282, abs=1e-9)


def test_tracker_matches_detections_and_starts_new_tracks():
    tracker = KitTracker(CFG)
    tracker.reset(StationFrame.from_pose("A", dock("A")))
    a = tracker.update(0.0, signal(0.0, 0.0), [detection(0.0, 0.10)], "test")[0]
    # 1 s later: the first kit 30 mm on (seen 2 mm off its prediction), a second one a pitch behind
    obs = tracker.update(1.0, signal(1.0, 0.03), [detection(1.0, 0.132), detection(1.0, -0.32)],
                         "test")
    by_id = {o.track_id: o for o in obs}
    assert set(by_id) == {a.track_id, a.track_id + 1}
    assert by_id[a.track_id].bottle_in_base[1] == pytest.approx(0.132)     # detection wins
    assert by_id[a.track_id + 1].bottle_in_base[1] == pytest.approx(-0.32)


def test_tracker_drops_stale_tracks_and_kits_that_left_the_view():
    tracker = KitTracker(CFG)
    tracker.reset(StationFrame.from_pose("A", dock("A")))
    tracker.update(0.0, signal(0.0, 0.0), [detection(0.0, 0.0)], "test")
    assert tracker.update(1.0, signal(1.0, 0.03), None, "test")
    assert tracker.update(0.0 + CFG.perception.track_timeout_s + 0.1, [], None, "test") == []
    tracker.update(10.0, signal(10.0, 0.30), [detection(10.0, 0.74)], "test")
    assert tracker.update(10.5, signal(10.5, 0.45), None, "test") == []    # carried past the view


def frame_with(t, kits, conveyors=()):
    return SensorFrame(t=t, joints=JointState(t, np.zeros(7), 0.0), conveyors=list(conveyors),
                       ground_truth=kits)


def test_oracle_backend_reports_only_kits_on_the_docked_belt_at_its_rate():
    perception = Perception(CFG)
    assert perception.detector.name == "oracle"
    perception.start(StationFrame.from_pose("A", dock("A")))
    here = GroundTruthKit("kit_A0", np.array([0.755, -0.16, 0.779]), np.array([0.755, -0.30, 0.775]))
    upstream = GroundTruthKit("kit_A1", np.array([0.755, -0.76, 0.779]),
                              np.array([0.755, -0.90, 0.775]))        # outside the view
    other_belt = GroundTruthKit("kit_B0", np.array([-2.8, 0.1, 0.779]), np.array([-2.8, 0.0, 0.775]))
    obs = perception.update(frame_with(0.0, [here, upstream, other_belt], signal(0.0, 0.0)))
    assert len(obs) == 1 and obs[0].source == "oracle"
    assert obs[0].bottle_in_base == pytest.approx(here.bottle_in_base)
    assert perception.get(obs[0].track_id) is obs[0] and perception.get(None) is None
    # 10 ms later the oracle has no new sample (30 Hz): the track is carried forward instead
    moved = GroundTruthKit("kit_A0", here.ring_in_base + [0, 0.5, 0], here.bottle_in_base + [0, 0.5, 0])
    obs = perception.update(frame_with(0.01, [moved], signal(0.01, 0.0003)))
    assert obs[0].bottle_in_base[1] == pytest.approx(-0.2997) and obs[0].age_s == pytest.approx(0.01)
    perception.stop()
    assert not perception.active and perception.update(frame_with(1.0, [here])) == []


def test_oracle_backend_without_ground_truth_reports_nothing():
    perception = Perception(CFG)
    perception.start(StationFrame.from_pose("A", dock("A")))
    assert perception.update(frame_with(0.0, None, signal(0.0, 0.0))) == []


def test_detector_backends_are_selected_by_config():
    assert make_detector(get_config()).name == "rgbd/depth_bands"       # the default
    with pytest.raises(ValueError):
        make_detector(with_overrides(CFG, {"perception": {"method": "magic"}}))
