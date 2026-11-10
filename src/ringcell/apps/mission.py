"""Full mission runner: sim + plant + robot in one deterministic loop. PLACEHOLDER.

    loop every control tick:
        frame   = sensors.read(data)          # ringcell.sim.sensors
        command = orchestrator.step(frame)    # ringcell.robot (includes safety filter)
        bridge.apply(data, command)           # ringcell.sim.actuation
        for each physics step: mj_step; line.step(data)   # ringcell.plant
"""
from __future__ import annotations

from ..log import get_logger

log = get_logger(__name__)


def run(scenario: str = "nominal", out_dir: str = "results/nominal") -> None:
    raise NotImplementedError("mission runner is not implemented yet")
