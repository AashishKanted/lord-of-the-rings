"""Detector interface and factory. A detector turns one `SensorFrame` into kit detections.

Backends are selected by ``perception.method``:

- ``oracle``     true poses supplied by the simulator (`frodo.robot.perception.oracle`).
                 ORACLE: for developing the control stack, not a perception result.
- ``rgbd``       segment the camera image (`segmenters`), fit ring and bottle from the depth
                 inside each mask (`geometry`), verify and pair them (`rgbd_detector`).

To add a backend, implement `Detector` and register it in `make_detector`. The tracker,
the orchestrator and the arm controller do not change.
"""
from __future__ import annotations

from typing import Protocol

from ...interfaces import SensorFrame
from ..frames import StationFrame
from .types import KitDetection


class Detector(Protocol):
    name: str

    def reset(self) -> None:
        """Forget state between stations."""

    def detect(self, frame: SensorFrame, station: StationFrame) -> list[KitDetection] | None:
        """Kits seen in this tick's data, or None when there is no new sample this tick
        (an empty list means a sample was processed and showed no kit)."""

    def find_held_ring(self, frame: SensorFrame, expected):
        """(capture time, ring centre xyz) of the ring in the gripper, near `expected`, or
        None when this tick has no measurement of it."""


def make_detector(config) -> Detector:
    method = config.perception.method
    if method == "oracle":
        from .oracle import OracleDetector
        return OracleDetector(config)
    if method == "rgbd":
        from .rgbd_detector import RgbdDetector
        return RgbdDetector(config)
    raise ValueError(f"unknown perception.method '{method}' (have: oracle, rgbd)")
