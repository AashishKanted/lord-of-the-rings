"""Independent safety supervisor. PLACEHOLDER.

Sits between every motion source and the actuators. Inputs: raw lidar scans (not the
localization output) and the command the planners want to send. Output: the command
that is allowed.

Planned behaviour:
- obstacle in the direction of travel closer than ``slow_distance_m``: scale speed
  linearly down, reaching zero at ``stop_distance_m``
- anything unmapped within ``human_zone_radius_m`` of the base footprint, or inside
  the active workstation zone: latch a stop (base and arm) within
  ``max_stop_latency_s``; motion stays blocked until explicit recovery
- stale or missing sensor data counts as an obstacle
"""
from __future__ import annotations

from dataclasses import dataclass

from ...interfaces import RobotCommand, SensorFrame


@dataclass(frozen=True)
class SafetyState:
    speed_scale: float       # 1 = free, 0 = stopped
    stop_latched: bool
    reason: str = ""


class SafetyMonitor:
    def __init__(self, config):
        self.config = config

    def filter(self, frame: SensorFrame, wanted: RobotCommand) -> tuple[RobotCommand, SafetyState]:
        raise NotImplementedError

    def acknowledge(self) -> bool:
        """Explicit recovery; only succeeds when the zone is clear."""
        raise NotImplementedError
