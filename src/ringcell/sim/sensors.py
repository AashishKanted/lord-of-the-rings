"""Simulated robot sensors: turns simulator state into a `SensorFrame`. PLACEHOLDER.

Planned (branch feat/sim-sensors):
- two 2D lidars on opposite base corners (ray casts via ``mujoco.mj_multiRay``), noise, rate
- wheel odometry (base joint velocities + noise) and IMU (yaw rate + bias)
- RGB-D frames from the cameras enabled in [sensors.cameras]
- conveyor encoder per station
"""
from __future__ import annotations

from ..interfaces import SensorFrame


class SensorSuite:
    def __init__(self, model, config):
        self.model, self.config = model, config

    def read(self, data) -> SensorFrame:
        raise NotImplementedError("simulated sensors are not implemented yet")
