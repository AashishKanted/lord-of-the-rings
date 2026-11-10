"""Mission orchestrator: owns the `Mode` and wires the modules each tick. PLACEHOLDER.

Tick order:
    frame -> localization -> perception -> (mode logic picks the active controller)
          -> wanted command -> safety filter -> RobotCommand

Mode flow for the nominal mission:
    IDLE -> NAVIGATE(A) -> DOCK(A) -> MANIPULATE x5 accepted -> STOW
         -> NAVIGATE(B) -> DOCK(B) -> MANIPULATE x5 accepted -> DONE
SAFETY_STOP can pre-empt any mode and returns to the interrupted mode after recovery.
Counts of accepted kits come from the line controller, never from the robot itself.
"""
from __future__ import annotations

from ...interfaces import Mode, RobotCommand, SensorFrame


class Orchestrator:
    def __init__(self, config):
        self.config, self.mode = config, Mode.IDLE

    def step(self, frame: SensorFrame) -> RobotCommand:
        raise NotImplementedError
