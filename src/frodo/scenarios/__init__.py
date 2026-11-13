"""Acceptance scenarios: what happens to the cell besides the nominal line.

A scenario is test equipment, not robot software: it may read simulator state (where the
person really is, which bytes went over the wire) and it writes the customer-side evidence
for its fault. The robot only meets it through its sensors and its Modbus connection.

    nominal      nothing extra
    human_stop   a person walks into the robot's work zone during a ring transfer
    plc_retry    the acknowledgement of one ASSEMBLY_DONE is lost after the line applied it
"""
from __future__ import annotations

from .base import Scenario
from .human_stop import HumanStop
from .plc_retry import PlcRetry

SCENARIOS = ("nominal", "human_stop", "plc_retry")


def make_scenario(name: str, config) -> Scenario:
    if name == "nominal":
        return Scenario()
    if name == "human_stop":
        return HumanStop(config)
    if name == "plc_retry":
        return PlcRetry(config)
    raise ValueError(f"unknown scenario '{name}' (have: {', '.join(SCENARIOS)})")
