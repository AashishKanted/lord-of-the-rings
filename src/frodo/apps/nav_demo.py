"""KINEMATIC TEST HARNESS for localization + navigation. NOT the MuJoCo robot run.

    uv run python -m frodo.apps.nav_demo [--seed 0] [--out results/nav]

Drives start -> dock A -> dock B with the real robot modules (`Localizer`, `Navigator`)
closed around a pure-numpy stand-in for the simulator:

- `KinematicCell` integrates the commanded base twist (first-order actuator lag) and
  holds the TRUE pose. It produces a `SensorFrame` with noisy odometry, a biased gyro
  and two 2D lidar scans ray-cast against the map segments, using the [sensors] config.
- The robot modules only ever receive that `SensorFrame`; the true pose is used here
  for scoring and plotting only.

Nothing here involves physics, the arm, the plant or the PLC, and it produces no
acceptance evidence. It exists so the navigation stack can be developed and tested
headless, in milliseconds, before it is wired to the MuJoCo sensors.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from math import atan2, cos, degrees, hypot, sin
from pathlib import Path

import numpy as np

from ..cell import CELL, STATIONS
from ..config import ROOT, get_config
from ..interfaces import (
    ImuSample,
    JointState,
    LidarScan,
    OdometrySample,
    Pose2D,
    SensorFrame,
    Twist2D,
)
from ..log import get_logger
from ..robot.localization import Localizer
from ..robot.map import cell_segments, closest_on_segments, get_map
from ..robot.navigation import Navigator, NavStatus, dock_pose

log = get_logger(__name__)

FOOTPRINT_M = 0.57   # square bumper of the base, used here only for the collision check


# --------------------------------------------------------------------------------------
# Harness: ray-caster and kinematic base (ground truth lives here, not in frodo.robot)
# --------------------------------------------------------------------------------------
def raycast(origin, angles: np.ndarray, segments: np.ndarray, max_range: float) -> np.ndarray:
    """Range along each world-frame beam angle to the nearest segment; inf = no return."""
    direction = np.column_stack([np.cos(angles), np.sin(angles)])          # (R, 2)
    a = segments[:, :2] - np.asarray(origin, dtype=float)                  # (S, 2)
    e = segments[:, 2:] - segments[:, :2]
    denom = direction[:, None, 0] * e[None, :, 1] - direction[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / denom
        u = (a[None, :, 0] * direction[:, None, 1] - a[None, :, 1] * direction[:, None, 0]) / denom
    hit = (np.abs(denom) > 1e-12) & (r > 0.0) & (u >= 0.0) & (u <= 1.0)
    ranges = np.where(hit, r, np.inf).min(axis=1)
    return np.where(ranges <= max_range, ranges, np.inf)


def box_segments(cx: float, cy: float, size: float) -> np.ndarray:
    """A square unmapped obstacle (e.g. a person or a pallet) as four segments."""
    h = size / 2
    c = np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])
    return np.hstack([c, np.roll(c, -1, axis=0)])


def lidar_scans(pose: Pose2D, t: float, config, rng: np.random.Generator | None,
                extra_segments: np.ndarray | None = None) -> list[LidarScan]:
    """Both lidars' scans from a true base pose, per [sensors.lidar]. rng=None: no noise."""
    cfg = config.sensors.lidar
    segments = cell_segments(cfg.height_m)
    if extra_segments is not None and len(extra_segments):
        segments = np.vstack([segments, extra_segments])
    half_fov = np.radians(cfg.fov_deg) / 2
    angles = np.linspace(-half_fov, half_fov, cfg.rays)
    c, s = cos(pose.yaw), sin(pose.yaw)
    scans = []
    for mount in cfg.mounts:
        mount_pose = Pose2D(mount.x, mount.y, np.radians(mount.yaw_deg))
        origin = (pose.x + c * mount.x - s * mount.y, pose.y + s * mount.x + c * mount.y)
        ranges = raycast(origin, angles + pose.yaw + mount_pose.yaw, segments, cfg.range_max_m)
        if rng is not None:
            ranges = ranges + rng.normal(0.0, cfg.range_noise_std_m, len(ranges))
        scans.append(LidarScan(t, mount.name, mount_pose, angles, ranges))
    return scans


@dataclass
class HarnessOptions:
    """Properties of the stand-in plant (not robot tunables)."""
    actuator_tau_s: float = 0.05                 # first-order lag from command to base velocity
    odometry_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)  # systematic encoder error
    extra_segments: np.ndarray | None = None     # unmapped obstacles the lidars also see


class KinematicCell:
    """True base state + simulated sensors. Ground truth: never pass it to robot modules."""

    def __init__(self, config=None, start: Pose2D | None = None, seed: int = 0,
                 options: HarnessOptions | None = None):
        self.cfg = config or get_config()
        self.options = options or HarnessOptions()
        self.rng = np.random.default_rng(seed)
        self.dt = self.cfg.sim.control_dt_s
        self.true_pose = start or Pose2D(*CELL["start_pose_world"])
        self.true_twist = Twist2D()
        self.tick = 0
        self._lidar_every = max(1, round(1.0 / (self.cfg.sensors.lidar.rate_hz * self.dt)))

    @property
    def t(self) -> float:
        return self.tick * self.dt

    def sense(self) -> SensorFrame:
        s, tw, k = self.cfg.sensors, self.true_twist, self.options.odometry_scale
        noise = s.odometry.noise_std_frac * hypot(tw.vx, tw.vy)
        odometry = OdometrySample(self.t, Twist2D(
            k[0] * tw.vx + self.rng.normal(0.0, noise) if noise else k[0] * tw.vx,
            k[1] * tw.vy + self.rng.normal(0.0, noise) if noise else k[1] * tw.vy,
            k[2] * tw.wz + self.rng.normal(0.0, s.odometry.noise_std_frac * abs(tw.wz))))
        imu = ImuSample(self.t, tw.wz + s.imu.gyro_bias_radps
                        + self.rng.normal(0.0, s.imu.gyro_noise_std_radps), (0.0, 0.0))
        lidars = []
        if self.tick % self._lidar_every == 0:
            lidars = lidar_scans(self.true_pose, self.t, self.cfg, self.rng,
                                 self.options.extra_segments)
        return SensorFrame(self.t, JointState(self.t, np.zeros(7), 0.0), odometry, imu, lidars)

    def apply(self, command: Twist2D) -> None:
        """Advance the true state by one control tick under a base-frame twist command."""
        alpha = self.dt / (self.options.actuator_tau_s + self.dt)
        tw = self.true_twist
        tw = Twist2D(tw.vx + alpha * (command.vx - tw.vx), tw.vy + alpha * (command.vy - tw.vy),
                     tw.wz + alpha * (command.wz - tw.wz))
        p = self.true_pose
        yaw_mid = p.yaw + 0.5 * tw.wz * self.dt
        c, s = cos(yaw_mid), sin(yaw_mid)
        self.true_pose = Pose2D(p.x + (c * tw.vx - s * tw.vy) * self.dt,
                                p.y + (s * tw.vx + c * tw.vy) * self.dt,
                                atan2(sin(p.yaw + tw.wz * self.dt), cos(p.yaw + tw.wz * self.dt)))
        self.true_twist = tw
        self.tick += 1


def footprint_clearance(pose: Pose2D, segments: np.ndarray) -> float:
    """Smallest distance from the square base outline to any map segment (>0 = no contact)."""
    edge = np.linspace(-FOOTPRINT_M / 2, FOOTPRINT_M / 2, 9)
    half = np.full_like(edge, FOOTPRINT_M / 2)
    outline = np.vstack([np.column_stack([edge, half]), np.column_stack([edge, -half]),
                         np.column_stack([half, edge]), np.column_stack([-half, edge])])
    c, s = cos(pose.yaw), sin(pose.yaw)
    world = outline @ np.array([[c, s], [-s, c]]) + [pose.x, pose.y]
    return float(closest_on_segments(world, segments)[1].min())


# --------------------------------------------------------------------------------------
# Closed loop
# --------------------------------------------------------------------------------------
@dataclass
class Trace:
    """Per-tick record of a run; `true` is for scoring only."""
    t: list[float] = field(default_factory=list)
    true: list[Pose2D] = field(default_factory=list)
    estimate: list[Pose2D] = field(default_factory=list)
    odometry_only: list[Pose2D] = field(default_factory=list)
    command: list[Twist2D] = field(default_factory=list)
    true_twist: list[Twist2D] = field(default_factory=list)
    healthy: list[bool] = field(default_factory=list)
    tick_s: list[float] = field(default_factory=list)       # localizer wall time per tick
    scan_tick: list[bool] = field(default_factory=list)


@dataclass
class LegResult:
    station: str
    status: NavStatus
    travel_s: float
    true_error_mm: float          # true base position vs dock
    true_yaw_error_deg: float
    estimate_error_mm: float      # estimated vs true position at the end
    estimate_yaw_error_deg: float
    path: list[Pose2D]


def pose_error(a: Pose2D, b: Pose2D) -> tuple[float, float]:
    """(distance, |yaw difference|) between two poses."""
    return hypot(a.x - b.x, a.y - b.y), abs(atan2(sin(a.yaw - b.yaw), cos(a.yaw - b.yaw)))


def run_leg(sim: KinematicCell, localizer: Localizer, dead_reckoner: Localizer,
            navigator: Navigator, station: str, trace: Trace,
            timeout_s: float = 60.0) -> LegResult:
    """Drive to one dock. The navigator is fed the localizer's estimate only."""
    navigator.go_to_dock(station)
    t0, status = sim.t, navigator.status
    while sim.t - t0 < timeout_s:
        frame = sim.sense()
        tic = time.perf_counter()
        estimate = localizer.update(frame)
        toc = time.perf_counter()
        reckoned = dead_reckoner.update(replace(frame, lidars=[]))   # odometry + gyro only
        command, status = navigator.step(estimate, sim.dt)
        trace.t.append(sim.t)
        trace.true.append(sim.true_pose)
        trace.estimate.append(estimate)
        trace.odometry_only.append(reckoned)
        trace.command.append(command)
        trace.true_twist.append(sim.true_twist)
        trace.healthy.append(localizer.healthy)
        trace.tick_s.append(toc - tic)
        trace.scan_tick.append(bool(frame.lidars))
        if status in (NavStatus.DOCKED, NavStatus.FAILED):
            break
        sim.apply(command)
    dock = dock_pose(station)
    true_d, true_yaw = pose_error(sim.true_pose, dock)
    est_d, est_yaw = pose_error(localizer.pose, sim.true_pose)
    return LegResult(station, status, sim.t - t0, 1e3 * true_d, degrees(true_yaw), 1e3 * est_d,
                     degrees(est_yaw), list(navigator.path))


def run_mission(seed: int = 0, stations=("A", "B"), options: HarnessOptions | None = None,
                config=None):
    """start -> each dock in turn. Returns (leg results, trace)."""
    cfg = config or get_config()
    sim = KinematicCell(cfg, seed=seed, options=options)
    localizer, dead_reckoner, navigator = Localizer(cfg), Localizer(cfg), Navigator(cfg)
    trace = Trace()
    return [run_leg(sim, localizer, dead_reckoner, navigator, s, trace) for s in stations], trace


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def draw(trace: Trace, legs: list[LegResult], path: Path, config=None) -> None:
    """Top-down PNG: map, inflated zone, planned paths, true and estimated trajectories."""
    from PIL import Image, ImageDraw

    cell_map = get_map(config)
    lx, ly = CELL["cell_size_m"]
    scale, margin = 130, 40

    def px(x, y):
        return margin + x * scale, margin + (ly - y) * scale

    image = Image.new("RGB", (int(lx * scale) + 2 * margin, int(ly * scale) + 2 * margin + 60),
                      "white")
    d = ImageDraw.Draw(image)
    res = cell_map.resolution
    for iy, ix in zip(*np.nonzero(cell_map.occupied), strict=True):
        d.rectangle([px(ix * res, (iy + 1) * res), px((ix + 1) * res, iy * res)], fill="#e6e6e6")
    for x1, y1, x2, y2 in cell_map.segments:
        d.line([px(x1, y1), px(x2, y2)], fill="black", width=3)
    for leg in legs:
        d.line([px(p.x, p.y) for p in leg.path], fill="#1f77b4", width=2)
        for p in leg.path:
            d.ellipse([px(p.x - .03, p.y + .03), px(p.x + .03, p.y - .03)], outline="#1f77b4")
    d.line([px(p.x, p.y) for p in trace.odometry_only], fill="#ff9900", width=1)
    d.line([px(p.x, p.y) for p in trace.true], fill="#2ca02c", width=4)
    d.line([px(p.x, p.y) for p in trace.estimate], fill="#d62728", width=1)
    for name, station in STATIONS.items():
        x, y, yaw = station.dock
        d.ellipse([px(x - .05, y + .05), px(x + .05, y - .05)], outline="black", width=2)
        d.line([px(x, y), px(x + .25 * cos(yaw), y + .25 * sin(yaw))], fill="black", width=2)
        d.text(px(x + .12, y + .12), f"dock {name}", fill="black")
    y0 = int(ly * scale) + 2 * margin
    for i, (colour, label) in enumerate((
            ("#2ca02c", "true base trajectory (harness ground truth)"),
            ("#d62728", "EKF estimate (odometry + gyro + lidar scan matching)"),
            ("#ff9900", "odometry + gyro only"), ("#1f77b4", "planned path (A* + shortcut)"))):
        d.line([(margin + 400 * (i % 2), y0 + 12 + 22 * (i // 2)),
                (margin + 400 * (i % 2) + 30, y0 + 12 + 22 * (i // 2))], fill=colour, width=3)
        d.text((margin + 400 * (i % 2) + 38, y0 + 6 + 22 * (i // 2)), label, fill="black")
    d.text((margin, 12), "KINEMATIC TEST HARNESS (not the MuJoCo run) - grey: inflated no-go zone",
           fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def summarize(trace: Trace, legs: list[LegResult]) -> dict:
    true = np.array([[p.x, p.y, p.yaw] for p in trace.true])
    est = np.array([[p.x, p.y, p.yaw] for p in trace.estimate])
    odo = np.array([[p.x, p.y, p.yaw] for p in trace.odometry_only])
    tick, scan = np.array(trace.tick_s), np.array(trace.scan_tick)
    segments = cell_segments()

    def yaw_err(a):
        return np.abs(np.arctan2(np.sin(a[:, 2] - true[:, 2]), np.cos(a[:, 2] - true[:, 2])))

    return {
        "ekf_pos_rms_mm": 1e3 * float(np.sqrt(np.mean(np.sum((est - true)[:, :2] ** 2, axis=1)))),
        "ekf_pos_max_mm": 1e3 * float(np.linalg.norm((est - true)[:, :2], axis=1).max()),
        "ekf_yaw_max_deg": degrees(float(yaw_err(est).max())),
        "odo_pos_final_mm": 1e3 * float(np.linalg.norm((odo - true)[-1, :2])),
        "odo_pos_max_mm": 1e3 * float(np.linalg.norm((odo - true)[:, :2], axis=1).max()),
        "odo_yaw_max_deg": degrees(float(yaw_err(odo).max())),
        "max_true_speed_mps": max(hypot(tw.vx, tw.vy) for tw in trace.true_twist),
        "max_true_yaw_rate_radps": max(abs(tw.wz) for tw in trace.true_twist),
        "min_footprint_clearance_m": min(footprint_clearance(p, segments)
                                         for p in trace.true[::5]),
        "unhealthy_ticks": int(np.sum(~np.array(trace.healthy)[20:])),
        "tick_no_scan_ms_mean": 1e3 * float(tick[~scan].mean()),
        "tick_with_scan_ms_mean": 1e3 * float(tick[scan].mean()),
        "tick_with_scan_ms_max": 1e3 * float(tick[scan].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=get_config().sim.seed)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "nav")
    parser.add_argument("--odometry-scale", type=float, default=1.0,
                        help="systematic wheel-odometry scale error to inject, e.g. 1.03")
    args = parser.parse_args()

    k = args.odometry_scale
    legs, trace = run_mission(args.seed, options=HarnessOptions(odometry_scale=(k, k, 1.0)))
    print("KINEMATIC TEST HARNESS - not a MuJoCo robot run, no acceptance evidence")
    print(f"seed {args.seed}, odometry scale error {k}")
    tolerance = CELL["robot"]
    for leg in legs:
        ok = (leg.status is NavStatus.DOCKED
              and leg.true_error_mm <= 1e3 * tolerance["dock_position_tolerance_m"]
              and leg.true_yaw_error_deg <= degrees(tolerance["dock_yaw_tolerance_rad"]))
        print(f"dock {leg.station}: {leg.status.value:7s} in {leg.travel_s:5.2f} s | "
              f"TRUE error {leg.true_error_mm:5.2f} mm, {leg.true_yaw_error_deg:5.3f} deg | "
              f"estimate error {leg.estimate_error_mm:5.2f} mm, "
              f"{leg.estimate_yaw_error_deg:5.3f} deg | "
              f"{'within' if ok else 'OUTSIDE'} customer 25 mm / 2 deg")
    for key, value in summarize(trace, legs).items():
        print(f"  {key:28s} {value:.3f}" if isinstance(value, float) else f"  {key:28s} {value}")
    out = args.out / f"nav_demo_seed{args.seed}.png"
    draw(trace, legs, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
