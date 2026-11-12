"""ORACLE detector: passes on the true kit poses the simulator put in the sensor frame.

It stands in for a real detector while the control stack is developed. To make it behave
like one it samples at ``perception.oracle_rate_hz``, only reports kits inside
``perception.view_half_length_m`` of the conveyor centre, and can add Gaussian position
noise (``perception.oracle_noise_std_m``, default 0).
"""
from __future__ import annotations

import numpy as np

from ...interfaces import SensorFrame
from ...log import get_logger
from ..frames import StationFrame
from .types import KitDetection

log = get_logger(__name__)


class OracleDetector:
    name = "oracle"

    def __init__(self, config):
        self.cfg = config.perception
        self._rng = np.random.default_rng(config.sim.seed)
        self._warned = False
        self.reset()

    def reset(self) -> None:
        self._next_t = -np.inf

    def detect(self, frame: SensorFrame, station: StationFrame) -> list[KitDetection] | None:
        if frame.t < self._next_t - 1e-9:
            return None
        self._next_t = frame.t + 1.0 / self.cfg.oracle_rate_hz
        if frame.ground_truth is None:
            if not self._warned:
                log.error("perception.method is 'oracle' but the sensor frame has no ground truth")
                self._warned = True
            return None
        std = self.cfg.oracle_noise_std_m
        out = []
        for kit in frame.ground_truth:
            if abs(station.local_x(kit.bottle_in_base)) > self.cfg.view_half_length_m:
                continue
            if abs((kit.bottle_in_base - station.center)[:2] @ station.lateral[:2]) > 0.5:
                continue        # a kit on the other station's belt
            ring, bottle = kit.ring_in_base, kit.bottle_in_base
            if std > 0:
                ring = ring + self._rng.normal(0.0, std, 3)
                bottle = bottle + self._rng.normal(0.0, std, 3)
            out.append(KitDetection(frame.t, np.array(ring), np.array(bottle)))
        return out

    def find_held_ring(self, frame: SensorFrame, expected: np.ndarray):
        """ORACLE: the true ring nearest to where the robot believes it is holding one."""
        if not frame.ground_truth:
            return None
        ring = min((kit.ring_in_base for kit in frame.ground_truth),
                   key=lambda r: np.linalg.norm(r - expected))
        return (frame.t, np.array(ring)) if np.linalg.norm(ring - expected) < 0.05 else None
