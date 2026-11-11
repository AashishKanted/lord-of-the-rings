"""Sensor and actuator-bridge demo: drive a scripted twist and dump what the sensors saw.

The base is driven open loop (start pose -> in front of station A) through the
`ActuatorBridge`; nothing localizes or plans here. Outputs in ``results/sensors/``:

- ``lidar_topdown.png``  both lidar scans in the world frame, true and odometry paths
- ``<camera>_rgb.png``, ``<camera>_depth.png``  one frame from every enabled camera
- ``summary.json``  odometry-integrated pose against the true pose, sensor counts

The plot places scans with the TRUE base pose (diagnostic view of the simulator, not a
localization result).

    uv run python -m frodo sensors [--out results/sensors] [--belt-speed 0.03]
"""
from __future__ import annotations

import argparse
import json
import time
from math import cos, pi, sin
from pathlib import Path

import mujoco
import numpy as np

from ..cell import CELL, STATIONS
from ..config import get_config
from ..interfaces import RobotCommand, Twist2D
from ..log import get_logger, setup_logging
from ..robot.description import ARM_HOME, ARM_STOWED, BASE_JOINTS
from ..sim.actuation import ActuatorBridge
from ..sim.camera import depth_to_image
from ..sim.scene import build_spec, reset
from ..sim.sensors import SensorSuite

log = get_logger(__name__)
PX_PER_M, MARGIN_PX = 110, 40
LIDAR_COLOURS = ((230, 97, 1), (26, 133, 255))
KIT_START_X = -0.45


def script(max_v: float, max_w: float) -> list[tuple[str, Twist2D, float]]:
    """Open-loop twist segments (name, twist, seconds) from the start pose to dock A."""
    start, dock = CELL["start_pose_world"], STATIONS["A"].dock
    return [("forward", Twist2D(vx=max_v), (dock[0] - start[0]) / max_v),
            ("turn", Twist2D(wz=-max_w), abs(dock[2] - start[2]) / max_w),
            ("approach", Twist2D(vx=max_v), (start[1] - dock[1]) / max_v),
            ("settle", Twist2D(), 1.5)]


def integrate(pose: np.ndarray, twist: Twist2D, dt: float) -> np.ndarray:
    """Dead reckoning: advance (x, y, yaw) by a base-frame twist held for dt."""
    yaw = pose[2] + 0.5 * twist.wz * dt
    return pose + dt * np.array([cos(yaw) * twist.vx - sin(yaw) * twist.vy,
                                 sin(yaw) * twist.vx + cos(yaw) * twist.vy, twist.wz])


def scan_points_world(scan, base_pose) -> np.ndarray:
    """Lidar returns -> world xy, through the mount pose and the given base pose."""
    ok = np.isfinite(scan.ranges)
    a = scan.angles[ok] + scan.mount.yaw
    bx, by = scan.mount.x + scan.ranges[ok] * np.cos(a), scan.mount.y + scan.ranges[ok] * np.sin(a)
    c, s = cos(base_pose[2]), sin(base_pose[2])
    return np.stack([base_pose[0] + c * bx - s * by, base_pose[1] + s * bx + c * by], axis=1)


def plot_topdown(path: Path, scans, scan_pose, true_path, odom_path) -> None:
    from PIL import Image, ImageDraw
    lx, ly = CELL["cell_size_m"]
    size = (int(lx * PX_PER_M) + 2 * MARGIN_PX, int(ly * PX_PER_M) + 2 * MARGIN_PX + 60)

    def px(xy):
        return MARGIN_PX + xy[0] * PX_PER_M, MARGIN_PX + (ly - xy[1]) * PX_PER_M

    img = Image.new("RGB", size, (252, 252, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([px((0, ly)), px((lx, 0))], outline=(60, 60, 60), width=2)
    c = CELL["conveyor"]
    for st in STATIONS.values():
        corners = [st.to_world([sx * c["length_m"] / 2, sy * c["width_m"] / 2])[:2]
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        draw.polygon([px(p) for p in corners], fill=(225, 225, 222), outline=(120, 120, 120))
        draw.text(px(st.center[:2]), f"conveyor {st.name}", fill=(90, 90, 90), anchor="mm")
    draw.line([px(p) for p in true_path], fill=(40, 40, 40), width=2)
    draw.line([px(p) for p in odom_path], fill=(160, 60, 200), width=1)
    for scan, colour in zip(scans, LIDAR_COLOURS, strict=False):
        for p in scan_points_world(scan, scan_pose):
            u, v = px(p)
            draw.ellipse([u - 1.5, v - 1.5, u + 1.5, v + 1.5], fill=colour)
        cs, sn = cos(scan_pose[2]), sin(scan_pose[2])
        mount = (scan_pose[0] + cs * scan.mount.x - sn * scan.mount.y,
                 scan_pose[1] + sn * scan.mount.x + cs * scan.mount.y)
        u, v = px(mount)
        draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=colour, outline=(0, 0, 0))
    half = max(abs(m.x) for m in get_config().sensors.lidar.mounts)
    cs, sn = cos(scan_pose[2]), sin(scan_pose[2])
    body = [(scan_pose[0] + cs * x - sn * y, scan_pose[1] + sn * x + cs * y)
            for x, y in ((half, half), (-half, half), (-half, -half), (half, -half))]
    draw.polygon([px(p) for p in body], outline=(0, 0, 0))
    nose = (scan_pose[0] + cs * half, scan_pose[1] + sn * half)
    draw.line([px(scan_pose[:2]), px(nose)], fill=(0, 0, 0), width=2)
    y0 = size[1] - 62
    legend = [((40, 40, 40), "true base path"), ((160, 60, 200), "odometry-integrated path"),
              *((col, f"lidar {s.name} returns (t={s.t:.2f} s)")
                for s, col in zip(scans, LIDAR_COLOURS, strict=False))]
    for i, (colour, label) in enumerate(legend):
        u, v = MARGIN_PX + 330 * (i % 2), y0 + 22 * (i // 2)
        draw.rectangle([u, v + 2, u + 14, v + 10], fill=colour)
        draw.text((u + 20, v), label, fill=(30, 30, 30))
    draw.text((MARGIN_PX, 12), "Lidar scans placed with the TRUE base pose (simulator "
              "diagnostic); room 6 m x 6 m", fill=(30, 30, 30))
    img.save(path)


def run(out: Path, belt_speed: float = CELL["conveyor"]["nominal_speed_mps"]) -> dict:
    from PIL import Image
    cfg = get_config()
    robot = CELL["robot"]
    model = build_spec({"A": [KIT_START_X]}).compile()
    data = mujoco.MjData(model)
    reset(model, data)
    suite, bridge = SensorSuite(model, cfg), ActuatorBridge(model, cfg)
    # Stand-in for the plant: run the kit carrier at belt speed so the encoder has a signal.
    data.ctrl[model.actuator("kit_A0_drive").id] = belt_speed
    base_qpos = [int(model.joint(j).qposadr[0]) for j in BASE_JOINTS]
    steps = round(cfg.sim.control_dt_s / model.opt.timestep)

    odom_pose = np.array(CELL["start_pose_world"], dtype=float)   # known start pose
    last_odom_t, true_path, odom_path = None, [], []
    counts = {"ticks": 0, "odometry": 0, "imu": 0, "lidar": 0, "camera": 0, "conveyor": 0}
    scans, scan_pose, conveyor, read_ms = {}, None, None, []
    pose_at = {}                       # true base pose by time, to place delayed scans
    for name, twist, seconds in script(robot["max_transit_speed_mps"],
                                       robot["max_yaw_rate_radps"]):
        log.info("t=%5.2fs segment %-8s twist=(%.2f, %.2f, %.2f) for %.2f s", data.time, name,
                 twist.vx, twist.vy, twist.wz, seconds)
        arm = ARM_HOME if name == "settle" else ARM_STOWED
        for _ in range(round(seconds / cfg.sim.control_dt_s)):
            bridge.apply(data, RobotCommand(base=twist, arm_q=arm, gripper=0.0))
            for _ in range(steps):
                mujoco.mj_step(model, data)
            t0 = time.perf_counter()
            frame = suite.read(data)
            read_ms.append(1000 * (time.perf_counter() - t0))
            true = data.qpos[base_qpos].copy()
            pose_at[round(frame.t, 6)] = true
            counts["ticks"] += 1
            if frame.odometry is not None:
                if last_odom_t is not None:
                    odom_pose = integrate(odom_pose, frame.odometry.twist,
                                          frame.odometry.t - last_odom_t)
                last_odom_t = frame.odometry.t
                counts["odometry"] += 1
            counts["imu"] += frame.imu is not None
            counts["camera"] += frame.camera is not None
            counts["lidar"] += len(frame.lidars)
            counts["conveyor"] += len(frame.conveyors)
            for scan in frame.lidars:
                scans[scan.name] = scan
            if frame.conveyors:
                conveyor = frame.conveyors[0]
            true_path.append(true[:2])
            odom_path.append(odom_pose[:2].copy())

    out.mkdir(parents=True, exist_ok=True)
    latest = [scans[k] for k in sorted(scans)]
    scan_pose = pose_at[round(latest[0].t, 6)]
    plot_topdown(out / "lidar_topdown.png", latest, scan_pose, true_path, odom_path)
    for cam in suite.camera_names:
        shot = suite.render_camera(cam, data)
        Image.fromarray(shot.rgb).save(out / f"{cam}_rgb.png")
        Image.fromarray(depth_to_image(shot.depth)).save(out / f"{cam}_depth.png")
    suite.close()

    true = data.qpos[base_qpos]
    err = odom_pose - true
    dock = STATIONS["A"].dock
    travelled = float(np.sum(np.linalg.norm(np.diff(np.array(true_path), axis=0), axis=1)))
    summary = {
        "mode": "open-loop scripted twist through ActuatorBridge; no localization",
        "seed": suite.seed,
        "sim_time_s": round(float(data.time), 3),
        "true_pose": [round(float(v), 4) for v in true],
        "odometry_pose": [round(float(v), 4) for v in odom_pose],
        "odometry_drift_m": round(float(np.hypot(err[0], err[1])), 4),
        "odometry_drift_yaw_deg": round(float(np.degrees(err[2])), 3),
        "distance_travelled_m": round(travelled, 3),
        "yaw_turned_deg": round(float(np.degrees(abs(true[2] - CELL["start_pose_world"][2]))), 2),
        "open_loop_error_to_dock_A_m": round(float(np.hypot(*(true[:2] - dock[:2]))), 5),
        "open_loop_error_to_dock_A_yaw_deg": round(float(np.degrees(
            (true[2] - dock[2] + pi) % (2 * pi) - pi)), 4),
        "messages": counts,
        "conveyor_A": None if conveyor is None else {
            "encoder_m": round(conveyor.encoder_m, 4), "speed_mps": round(conveyor.speed_mps, 5)},
        "sensor_read_ms": {"mean": round(float(np.mean(read_ms)), 3),
                           "max": round(float(np.max(read_ms)), 3)},
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/sensors"))
    parser.add_argument("--belt-speed", type=float, default=CELL["conveyor"]["nominal_speed_mps"])
    args = parser.parse_args()
    setup_logging()
    summary = run(args.out, args.belt_speed)
    print(json.dumps(summary, indent=2))
    print(f"odometry drift after {summary['distance_travelled_m']} m and "
          f"{summary['yaw_turned_deg']} deg: {1000 * summary['odometry_drift_m']:.1f} mm, "
          f"{summary['odometry_drift_yaw_deg']:.2f} deg")


if __name__ == "__main__":
    main()
