"""Detector interface and factory. A detector turns one `SensorFrame` into kit detections.

Backends are selected by ``perception.method``:

- ``oracle``     true poses supplied by the simulator (`frodo.robot.perception.oracle`).
                 ORACLE: for developing the control stack, not a perception result.
- ``classical``  PLACEHOLDER. Planned: colour threshold for the ring ([perception] HSV
                 range), depth to 3D points, circle fit for the ring centre; bottle neck
                 from the depth blob at the known neck height.

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


class ClassicalDetector:
    name = "classical"

    def __init__(self, config):
        self.config = config

    def reset(self) -> None:
        pass

    def detect(self, frame: SensorFrame, station: StationFrame) -> list[KitDetection] | None:
        raise NotImplementedError("classical RGB-D ring/bottle detection is not implemented yet")


def make_detector(config) -> Detector:
    method = config.perception.method
    if method == "oracle":
        from .oracle import OracleDetector
        return OracleDetector(config)
    if method == "classical":
        return ClassicalDetector(config)
    raise ValueError(f"unknown perception.method '{method}' (have: oracle, classical)")
