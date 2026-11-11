"""ORACLE kit poses for the oracle perception backend.

Reads the true ring and bottle poses of every kit that is on a belt and expresses them in
the robot base frame, the frame a robot-mounted camera would measure in. The mission
runner attaches the result to ``SensorFrame.ground_truth`` only when
``perception.method = "oracle"``; such a run is not a sensor-driven perception result.
"""
from __future__ import annotations

import numpy as np

from ..cell import CELL, STATIONS
from ..interfaces import GroundTruthKit


class OracleKitSource:
    def __init__(self, model):
        self._base = model.body("base_link").id
        self._half_belt = CELL["conveyor"]["length_m"] / 2
        self._kits = []      # (name, station, kit body, ring body, bottle body)
        for st in STATIONS.values():
            index = 0
            while True:
                name = f"kit_{st.name}{index}"
                try:
                    kit = model.body(name).id
                except KeyError:
                    break
                self._kits.append((name, st, kit, model.body(f"{name}_ring").id,
                                   model.body(f"{name}_bottle").id))
                index += 1

    def read(self, data) -> list[GroundTruthKit]:
        """Kits whose centre is over a belt (parked pool kits are left out)."""
        origin, rot = data.xpos[self._base], data.xmat[self._base].reshape(3, 3)
        out = []
        for name, st, kit, ring, bottle in self._kits:
            along = (data.xpos[kit][:2] - st.center[:2]) @ [np.cos(st.yaw), np.sin(st.yaw)]
            if abs(along) <= self._half_belt:
                out.append(GroundTruthKit(name, (data.xpos[ring] - origin) @ rot,
                                          (data.xpos[bottle] - origin) @ rot))
        return out
