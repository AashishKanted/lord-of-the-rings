"""The customer cell specification, shared by the plant and the robot.

Read-only view of ``spec/customer_cell.json`` (published customer data, so both
sides may use it). Nothing here is ground-truth simulator state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import cos, sin

import numpy as np

from .config import ROOT

CELL = json.loads((ROOT / "spec" / "customer_cell.json").read_text())


@dataclass(frozen=True)
class Station:
    name: str
    id: int
    center: np.ndarray  # conveyor centre on the belt surface, world
    yaw: float
    dock: np.ndarray  # x, y, yaw

    def to_world(self, local_xy, z=0.0):
        """Station-local (x along belt travel, y) -> world xyz."""
        c, s = cos(self.yaw), sin(self.yaw)
        x, y = local_xy
        return np.array([self.center[0] + c * x - s * y, self.center[1] + s * x + c * y, z])


STATIONS = {name: Station(name, v["id"], np.array(v["conveyor_center_world"]), v["yaw_rad"],
                          np.array(v["dock_world"]))
            for name, v in CELL["stations"].items()}
KIT_TOP_Z = CELL["conveyor"]["height_m"] + CELL["kit"]["height_m"]
