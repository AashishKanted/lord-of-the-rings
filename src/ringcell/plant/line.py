"""Customer line: kit feeder, belt, PLC server, downstream inspection, event log. PLACEHOLDER.

Planned (branch feat/plant):
- spawn/recycle kits at the spec pitch and drive belts at the scenario speed
- run ``advaitic_systems.modbus.PLCServer`` on a thread; call ``arrive`` on real
  work-entry crossings and ``inspect`` with `SeatMonitor` measurements at the
  inspection line
- write the evidence JSONL in the INTERFACES.md format
- scenario hooks: human enters/leaves, PLC connection drop
"""
from __future__ import annotations


class Line:
    def __init__(self, model, config, scenario: str = "nominal"):
        self.model, self.config, self.scenario = model, config, scenario

    def step(self, data) -> None:
        """Advance plant logic by one simulation step."""
        raise NotImplementedError("plant line is not implemented yet")
