"""Interactive mission: the full mission in the MuJoCo viewer, with a box you can move.

    scripts/run.sh play                      # camera-driven perception (slower than real time)
    scripts/run.sh play --perception oracle  # true kit poses: runs at about real time

Moving the box: double-click it to select it, then hold Ctrl and drag with the RIGHT mouse
button to slide it across the floor (Ctrl + left drag rotates it). Put it on the robot's
path, for example between the start pose and dock A, or between the docks.

The box is an obstacle the robot does not know about: it is not on the map. The robot only
learns about it from its lidars, through the safety monitor. The marker above the robot
shows the safety state: green = free, orange = slowing, red = stopped. The robot waits
while the box blocks its path and continues when you move the box away (it does not plan
around it). The safety monitor only governs base motion: a box pushed against the docked
robot does not stop the arm.

Same robot software, plant and evidence log as `scripts/run.sh mission`; the run's files
go to results/interactive/. The box exists only in this app, not in the mission runner.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from ..config import ROOT
from ..log import get_logger, setup_logging
from .mission import run

log = get_logger(__name__)
BOX_START = (2.0, 4.6)               # world xy: beside the route, out of the way at the start
BOX_HALF = (0.15, 0.15, 0.25)        # 30 x 30 x 50 cm: tall enough for the 12 cm lidars
STATE_COLOURS = {"free": (0.1, 0.8, 0.2, 0.9), "slow": (1.0, 0.6, 0.0, 0.9),
                 "stop": (0.9, 0.1, 0.1, 0.9)}


def add_box(spec) -> None:
    """A mocap body: the viewer moves it directly, physics does not push it around."""
    body = spec.worldbody.add_body(name="test_box", mocap=True, pos=[*BOX_START, BOX_HALF[2]])
    body.add_geom(name="test_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOX_HALF),
                  rgba=[0.85, 0.55, 0.2, 1.0])


class ViewerTick:
    """Keeps the viewer in sync with the simulation and paces it to real time."""

    def __init__(self, realtime: bool = True):
        self.viewer, self.realtime = None, realtime
        self._wall0 = self._sim0 = None
        self._last_state = None

    def __call__(self, model, data, robot) -> bool:
        if self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(model, data)
            self._base = model.body("base_link").id
            self._wall0, self._sim0 = time.perf_counter(), data.time
            log.info("viewer open: double-click the orange box, then Ctrl + right-drag to move it")
        if not self.viewer.is_running():
            return False
        self._marker(data, robot)
        self.viewer.sync()
        if self.realtime:          # wait if the simulation is ahead of the wall clock
            ahead = (data.time - self._sim0) - (time.perf_counter() - self._wall0)
            if ahead > 0:
                time.sleep(ahead)
        return True

    def _marker(self, data, robot) -> None:
        state = robot.safety.state if robot.safety is not None else None
        kind = "free" if state is None or state.speed_scale >= 1 else \
            "stop" if state.stopped else "slow"
        if kind != self._last_state:
            log.info("t=%.2f safety marker: %s%s", data.time, kind,
                     f" ({state.reason})" if state is not None and state.reason else "")
            self._last_state = kind
        scene = self.viewer.user_scn
        scene.ngeom = 1
        mujoco.mjv_initGeom(scene.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.06, 0.0, 0.0]), data.xpos[self._base] + [0, 0, 1.75],
                            np.eye(3).flatten(), np.array(STATE_COLOURS[kind], dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--perception", choices=["oracle", "rgbd"],
                        help="override [perception] method (oracle runs at about real time)")
    parser.add_argument("--fast", action="store_true", help="do not slow down to real time")
    parser.add_argument("--port", type=int, default=0, help="Modbus/TCP port (default: any free)")
    args = parser.parse_args()
    setup_logging()
    tick = ViewerTick(realtime=not args.fast)
    summary = run("nominal", ROOT / "results" / "interactive", args.seed, port=args.port,
                  perception=args.perception, add_to_scene=add_box, on_tick=tick)
    if tick.viewer is not None:
        tick.viewer.close()
    print(f"accepted {summary['counts']['A']['accepted']} at A, "
          f"{summary['counts']['B']['accepted']} at B; simulated {summary['simulated_s']} s")


if __name__ == "__main__":
    main()
