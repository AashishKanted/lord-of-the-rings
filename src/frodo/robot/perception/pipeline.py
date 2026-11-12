"""Perception facade used by the orchestrator.

    perception.start(station_frame)       # orchestrator: begin watching this station
    observations = perception.update(frame)   # every tick while active
    perception.get(track_id)              # latest observation of one kit, or None
    perception.stop()

Detector (pluggable, ``perception.method``) -> KitTracker (encoder carry-forward) ->
`KitObservation`s in the robot base frame.
"""
from __future__ import annotations

from dataclasses import replace

from ...interfaces import SensorFrame
from ...log import get_logger
from ..frames import StationFrame
from .camera_pose import CameraPose
from .detector import Detector, make_detector
from .tracker import KitTracker
from .types import KitObservation

log = get_logger(__name__)


class Perception:
    def __init__(self, config, detector: Detector | None = None, kin_model=None):
        """With ``kin_model`` (the robot's kinematic model) the camera's pose is computed from
        the measured joints; without it the pose delivered with the image is used (tests)."""
        self.detector = detector or make_detector(config)
        self.camera_pose = CameraPose(kin_model, config.sensors.cameras.perception_source) \
            if kin_model is not None else None
        self.tracker = KitTracker(config)
        self.station: StationFrame | None = None
        self._latest: dict[int, KitObservation] = {}
        if self.detector.name == "oracle":
            log.warning("perception backend is ORACLE: kit poses come from simulator truth")

    @property
    def active(self) -> bool:
        return self.station is not None

    def start(self, station: StationFrame) -> None:
        self.station = station
        self.detector.reset()
        self.tracker.reset(station)
        self._latest = {}
        log.info("perception started at station %s (%s)", station.name, self.detector.name)

    def stop(self) -> None:
        self.station, self._latest = None, {}

    def update(self, frame: SensorFrame) -> list[KitObservation]:
        frame = self._own_camera_pose(frame)
        if self.station is None:
            return []
        detections = self.detector.detect(frame, self.station)
        observations = self.tracker.update(frame.t, frame.conveyors, detections,
                                           self.detector.name)
        self._latest = {obs.track_id: obs for obs in observations}
        return observations

    def held_ring(self, frame: SensorFrame, expected):
        """Where the ring in the gripper really is: (capture time, centre xyz) or None."""
        return self.detector.find_held_ring(self._own_camera_pose(frame, record=False), expected)

    def _own_camera_pose(self, frame: SensorFrame, record: bool = True) -> SensorFrame:
        """Replace the camera pose that came with the image by forward kinematics of the
        robot's own joint measurements at the image's capture time."""
        if self.camera_pose is None:
            return frame
        if record:
            self.camera_pose.record(frame.joints)
        if frame.camera is None:
            return frame
        image = replace(frame.camera, cam_in_base=self.camera_pose.at(frame.camera.t))
        return replace(frame, camera=image)

    def get(self, track_id: int | None) -> KitObservation | None:
        return self._latest.get(track_id)

    def observations(self) -> list[KitObservation]:
        """All kits tracked as of the last `update`."""
        return list(self._latest.values())
