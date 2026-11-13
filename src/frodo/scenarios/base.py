"""The hooks a scenario can use. The nominal scenario uses none of them."""
from __future__ import annotations


class Scenario:
    name = "nominal"
    time_limit_s: float | None = None     # None: the mission's own limit

    def add_to_scene(self, spec) -> None:
        """Add bodies before the model is compiled."""

    def start(self, line, data) -> None:
        """The line is running; ``data`` is the live simulation state."""

    def robot_port(self, line_port: int) -> int:
        """The TCP port the robot's line client connects to."""
        return line_port

    def tick(self, model, data, robot, line) -> None:
        """After every control tick (physics has been stepped)."""

    def report(self) -> dict:
        """Scenario results for summary.json."""
        return {}

    def write_files(self, out_dir) -> None:
        """Extra raw evidence next to events.jsonl."""

    def close(self) -> None:
        pass
