"""Full mission runner: simulator + customer line + robot software in one deterministic loop.

    scripts/run.sh mission --seed 0
    uv run python -m frodo mission --seed 0 --out results/nominal_seed0

    every control tick:
        frame   = sensors.read(data)          # frodo.sim.sensors (+ ORACLE kit poses if configured)
        command = orchestrator.step(frame)    # frodo.robot
        bridge.apply(data, command)           # frodo.sim.actuation
        every physics step: mj_step; line.step(data)      # frodo.plant

The robot software only gets the `SensorFrame` and the line controller's Modbus registers.
The customer line (``frodo.plant``) measures seating and decides acceptance on its own.
With ``perception.method = "oracle"`` the kit poses in the sensor frame are simulator
truth: such a run exercises localization, navigation, the arm controller, the line
protocol and the plant, but it is NOT a sensor-driven perception result.

Outputs in the run directory: ``events.jsonl`` (evidence format), ``seat_samples.jsonl``
(raw seat measurements), ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from math import atan2, cos, degrees, hypot, sin
from pathlib import Path

import mujoco

from advaitic_systems.evidence import check

from ..cell import STATIONS
from ..config import ROOT, get_config, with_overrides
from ..interfaces import Mode
from ..log import get_logger, setup_logging
from ..plant import feeder
from ..plant.line import Line
from ..plant.recorder import MotionRecorder
from ..robot.orchestrator.mission import Orchestrator
from ..robot.plc import LineClient
from ..sim.actuation import ActuatorBridge
from ..sim.oracle import OracleKitSource
from ..sim.scene import build_spec, reset
from ..sim.sensors import SensorSuite

log = get_logger(__name__)
SCENARIOS = ("nominal",)


def run(scenario: str = "nominal", out_dir: str | Path | None = None, seed: int | None = None,
        config=None, belt_speed_mps: float | None = None, port: int | None = None,
        time_limit_s: float | None = None, video: str | Path | None = None,
        perception: str | None = None) -> dict:
    """Run one mission and return its summary (also written to ``summary.json``)."""
    if scenario not in SCENARIOS:
        raise NotImplementedError(f"scenario '{scenario}' is not implemented (have: {SCENARIOS})")
    cfg = config or get_config()
    if perception is not None:
        cfg = with_overrides(cfg, {"perception": {"method": perception}})
    seed = cfg.sim.seed if seed is None else seed
    out = Path(out_dir or ROOT / "results" / f"{scenario}_seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    limit = cfg.mission.time_limit_s if time_limit_s is None else time_limit_s
    oracle_perception = cfg.perception.method == "oracle"
    params = feeder.run_parameters(belt_speed_mps) if belt_speed_mps is not None \
        else feeder.draw_run_parameters(seed, cfg)

    model = build_spec(feeder.pool_layout(cfg)).compile()
    robot_model = build_spec().compile()      # the robot's own kinematic model: no kits
    data = mujoco.MjData(model)
    reset(model, data)                        # mission start pose, arm stowed
    sensors = SensorSuite(model, cfg, seed=seed, enable_cameras=not oracle_perception)
    bridge = ActuatorBridge(model, cfg)
    oracle = OracleKitSource(model) if oracle_perception else None
    steps_per_tick = round(cfg.sim.control_dt_s / model.opt.timestep)
    docks, wall = [], time.perf_counter()
    camera_geom, camera_touched = model.geom("wrist_camera_body").id, set()
    recorder_video = _VideoWriter(model, Path(video), cfg) if video else None

    with Line(model, cfg, scenario, params=params, events_path=out / "events.jsonl",
              port=cfg.plant.plc_port if port is None else port) as line:
        sensors.set_belt_speed_source(line.belt_speed)
        recorder = MotionRecorder(model, line.events, cfg.mission.sample_period_s)
        robot = Orchestrator(robot_model, LineClient(cfg.plant.plc_host, line.port), cfg)
        log.info("mission start: scenario=%s seed=%d belt=%s phase=%s perception=%s", scenario,
                 seed, params.belt_speed_mps, params.release_phase_s, cfg.perception.method)
        finished = False
        while data.time < limit:
            frame = sensors.read(data)
            if oracle is not None:
                frame.ground_truth = oracle.read(data)
            command = robot.step(frame)
            for event in robot.events:        # what the robot declared, with measured poses
                if event.kind == "dock":
                    x, y, yaw = recorder.pose(data)
                    recorder.sample(data)     # the checker wants a motion sample at the dock
                    line.events.dock(data.time, event.station, x, y, yaw)
                    docks.append(_dock_report(event.station, data.time, x, y, yaw))
                elif event.kind == "assembly_done":
                    line.events.assembly_done(data.time, event.station, event.kit_id)
            robot.events.clear()
            if recorder.due(data.time):
                recorder.sample(data)
            counts = line.counts()
            if robot.mode in (Mode.DONE, Mode.FAULT):
                finished = robot.mode is Mode.DONE
                break
            bridge.apply(data, command)
            for _ in range(steps_per_tick):
                mujoco.mj_step(model, data)
                line.step(data)
            for g1, g2 in data.contact.geom[:data.ncon]:      # did the camera hit anything?
                if camera_geom in (g1, g2):
                    other = g2 if g1 == camera_geom else g1
                    camera_touched.add(model.geom(other).name
                                       or model.body(model.geom_bodyid[other]).name)
            if recorder_video is not None:
                recorder_video.frame(data, robot.mode.value, robot.station, frame.camera)
        mujoco.mj_kinematics(model, data)
        recorder.sample(data)                 # close the log with a final motion sample
        counts, kits = line.counts(), [dict(k) for k in line.kits]
        events = list(line.events.events)
        commands = line.plc.command_ledger()
    sensors.close()
    if recorder_video is not None:
        recorder_video.close()
    wall = time.perf_counter() - wall

    verdict = check(events, scenario)
    accepted_kits = [k for k in kits if k["accepted"]]
    cycles = [k["cycle_s"] for k in accepted_kits if k["cycle_s"] is not None]
    summary = {
        "scenario": scenario,
        "seed": seed,
        "perception": ("ORACLE: kit poses are simulator truth, not a sensor-driven perception "
                       "result" if oracle_perception else
                       f"sensor-driven: wrist RGB-D camera, {robot.perception.detector.name}"),
        "wrist_camera_contacts": sorted(camera_touched),
        "localization_navigation": "simulated lidar + odometry + IMU (sensor-driven)",
        "safety_monitor": ("rudimentary lidar slow/stop on the base; no human-zone latch, arm "
                           "not supervised" if cfg.safety.enabled else "disabled"),
        "leave_policy": cfg.mission.leave_policy,
        "belt_speed_mps": params.belt_speed_mps,
        "release_phase_s": params.release_phase_s,
        "robot_finished": finished,
        "robot_mode_at_end": robot.mode.value,
        "simulated_s": round(float(data.time), 3),
        "wall_s": round(wall, 1),
        "mujoco_version": mujoco.__version__,
        "counts": counts,
        "accepted_cycle_s": {"min": min(cycles, default=None), "max": max(cycles, default=None)},
        "docks": docks,
        "max_base_speed_mps": round(recorder.max_base_speed, 4),
        "max_base_speed_arm_unstowed_mps": round(recorder.max_base_speed_unstowed, 5),
        "kits": kits,
        "robot_attempts": robot.attempts,
        "line_commands_applied": len(commands),
        "starter_checker": {**verdict, "note": "systems-check verifies log consistency only"},
        "events_file": str(out / "events.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("mission end: t=%.1f s, accepted %s, checker %s (%d violations), wall %.0f s",
             data.time, {k: v["accepted"] for k, v in counts.items()},
             "PASSED" if verdict["passed"] else "FAILED", len(verdict["violations"]), wall)
    return summary


class _VideoWriter:
    """Renders the overview camera with an inset close-up of the station being worked.
    Uses simulator state for drawing only; the robot software never sees it."""

    def __init__(self, model, path: Path, cfg, fps: int = 30):
        import imageio.v2 as imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.fps, self._n = path, fps, 0
        self._writer = imageio.get_writer(path, fps=fps, macro_block_size=None)
        self._main = mujoco.Renderer(model, 720, 960)
        self._inset = mujoco.Renderer(model, 240, 320)
        self._wrist = None

    def frame(self, data, mode: str, station: str | None, camera=None) -> None:
        """`camera`: the robot's own RGB-D frame of this tick, if one arrived; the last one
        is shown bottom right (what perception works from)."""
        if camera is not None:
            self._wrist = camera.rgb[::2, ::2]
        if data.time * self.fps < self._n:
            return
        self._n += 1
        from PIL import Image, ImageDraw
        self._main.update_scene(data, camera="overview")
        image = self._main.render().copy()
        self._inset.update_scene(data, camera=f"close_{station or 'A'}")
        image[8:248, 632:952] = self._inset.render()
        if self._wrist is not None:
            image[472:712, 632:952] = self._wrist
        pil = Image.fromarray(image)
        ImageDraw.Draw(pil).text((12, 10), f"t={data.time:6.1f} s  mode={mode}"
                                 f"{'  station ' + station if station else ''}", fill="white")
        self._writer.append_data(__import__("numpy").asarray(pil))

    def close(self) -> None:
        self._writer.close()
        self._main.close()
        self._inset.close()
        log.info("wrote video %s (%d frames)", self.path, self._n)


def _dock_report(station: str, t: float, x: float, y: float, yaw: float) -> dict:
    dock = STATIONS[station].dock
    return {"station": station, "t": round(t, 3),
            "true_error_mm": round(1e3 * hypot(x - dock[0], y - dock[1]), 2),
            "true_yaw_error_deg": round(degrees(abs(atan2(sin(yaw - dock[2]),
                                                          cos(yaw - dock[2])))), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    parser.add_argument("--seed", type=int, default=None, help="belt speed, release phase and "
                        "sensor noise are drawn from this (default: [sim] seed)")
    parser.add_argument("--belt-speed", type=float, help="fixed belt speed in m/s for both "
                        "stations instead of the seeded draw")
    parser.add_argument("--out", type=Path, help="run directory (default results/<scenario>_seed<N>)")
    parser.add_argument("--port", type=int, help="Modbus/TCP port (default [plant] plc_port; 0 = any)")
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--video", type=Path, help="also write an MP4 of the run (slower)")
    parser.add_argument("--perception", choices=["oracle", "rgbd"],
                        help="override [perception] method")
    args = parser.parse_args()
    setup_logging()
    summary = run(args.scenario, args.out, args.seed, belt_speed_mps=args.belt_speed,
                  port=args.port, time_limit_s=args.time_limit, video=args.video,
                  perception=args.perception)
    brief = {k: summary[k] for k in ("scenario", "seed", "perception", "belt_speed_mps",
                                     "robot_finished", "simulated_s", "wall_s", "counts",
                                     "accepted_cycle_s", "docks", "starter_checker")}
    print(json.dumps(brief, indent=2, default=float))


if __name__ == "__main__":
    main()
