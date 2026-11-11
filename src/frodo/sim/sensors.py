"""Simulated robot sensors: turns simulator state into a `SensorFrame`.

This is the only path by which robot software learns about the world. Each sensor reads
simulator state, adds noise from its own seeded random stream ([sim] seed), samples at its
own rate and is delivered after its own latency. Messages carry their CAPTURE time `t`;
`SensorFrame.t` is the time of the tick that delivers them.

    suite = SensorSuite(model)            # config/frodo.toml
    frame = suite.read(data)              # once per control tick, after mj_step
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from math import cos, sin

import mujoco
import numpy as np

from ..cell import STATIONS
from ..config import get_config
from ..interfaces import (
    ConveyorSignal,
    ImuSample,
    JointState,
    OdometrySample,
    RgbdFrame,
    SensorFrame,
    Twist2D,
)
from ..log import get_logger
from ..robot.description import ARM_JOINTS, BASE_JOINTS
from .camera import RgbdCamera
from .lidar import Lidar2D

log = get_logger(__name__)
_EPS = 1e-9
GRIPPER_JOINT = "right_driver_joint"

# Belt speed source: (station name, data) -> true belt speed in m/s, or None if unknown.
BeltSpeedSource = Callable[[str, mujoco.MjData], float | None]


class _Channel:
    """Sampling schedule and delivery delay of one sensor."""

    def __init__(self, rate_hz: float, latency_s: float = 0.0):
        self.period, self.latency = 1.0 / rate_hz, latency_s
        self.reset()

    def reset(self):
        self._next, self._queue = -np.inf, deque()

    def due(self, t: float) -> bool:
        if t < self._next - _EPS:
            return False
        self._next = self._next + self.period if self._next + self.period > t + _EPS \
            else t + self.period
        return True

    def push(self, t: float, message) -> None:
        self._queue.append((t, message))

    def pop(self, t: float):
        """Newest message whose latency has elapsed (older ready ones are dropped), or None."""
        message = None
        while self._queue and self._queue[0][0] + self.latency <= t + _EPS:
            message = self._queue.popleft()[1]
        return message


class SensorSuite:
    """All robot sensors. `config` is the full config (default: ``get_config()``)."""

    def __init__(self, model, config=None, *, seed: int | None = None,
                 enable_cameras: bool = True):
        self.model = model
        self.config = config = config or get_config()
        s = self.cfg = config.sensors
        self.seed = config.sim.seed if seed is None else seed
        self._base_body = model.body("base_link").id
        self._base_qpos = [int(model.joint(j).qposadr[0]) for j in BASE_JOINTS]
        self._base_dof = [int(model.joint(j).dofadr[0]) for j in BASE_JOINTS]
        self._arm_qpos = [int(model.joint(j).qposadr[0]) for j in ARM_JOINTS]
        grip = model.joint(GRIPPER_JOINT)
        self._grip_qpos, self._grip_range = int(grip.qposadr[0]), float(grip.range[1])

        self.lidars = [Lidar2D(model, m.name, s.lidar) for m in s.lidar.mounts]
        self._lidar_ch = [_Channel(s.lidar.rate_hz, s.lidar.latency_s) for _ in self.lidars]
        self._odom_ch = _Channel(s.odometry.rate_hz, s.odometry.latency_s)
        self._imu_ch = _Channel(s.imu.rate_hz, s.imu.latency_s)
        self._conv_ch = _Channel(s.conveyor.rate_hz, s.conveyor.latency_s)
        self._cam_ch = _Channel(s.cameras.rate_hz, s.cameras.latency_s)

        # Kits currently on each belt, found by name (kit_<station><n>_slide).
        self._kit_dofs = {
            st: [int(model.jnt_dofadr[j]) for j in range(model.njnt)
                 if model.joint(j).name.startswith(f"kit_{st}")
                 and model.joint(j).name.endswith("_slide")]
            for st in STATIONS}
        self._belt_source: BeltSpeedSource = self._kit_slide_speed

        cams = s.cameras
        if cams.perception_source not in cams.enabled:
            raise ValueError(f"perception_source '{cams.perception_source}' is not in "
                             f"sensors.cameras.enabled {cams.enabled}")
        self.cameras_enabled = bool(enable_cameras)
        self.camera_names = list(cams.enabled) if enable_cameras else []
        self.perception_source = cams.perception_source
        self._renderer = None          # created on the first render, shared by all cameras
        self._cameras: dict[str, RgbdCamera] = {}
        self.reset()

    # ---------- lifecycle ----------
    def reset(self) -> None:
        """Restart schedules, biases and the noise streams (same seed -> same readings)."""
        streams = np.random.SeedSequence(self.seed).spawn(7)
        (self._rng_lidar, self._rng_odom, self._rng_imu, self._rng_cam, self._rng_conv,
         self._rng_joint, rng_const) = (np.random.default_rng(x) for x in streams)
        self._odom_scale = 1.0 + rng_const.normal(0.0, self.cfg.odometry.scale_error_std_frac, 3)
        for ch in (*self._lidar_ch, self._odom_ch, self._imu_ch, self._conv_ch, self._cam_ch):
            ch.reset()
        self._encoder = {st: 0.0 for st in STATIONS}
        self._odom_prev = self._imu_prev = None
        self._last_t: float | None = None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._cameras.clear()

    def set_belt_speed_source(self, source: BeltSpeedSource) -> None:
        """Hook for the plant: supply the true belt speed per station (the default reads the
        kit carriers' slide-joint velocities and knows nothing about an empty belt)."""
        self._belt_source = source

    # ---------- per-tick read ----------
    def read(self, data) -> SensorFrame:
        t = float(data.time)
        if self._last_t is not None and t < self._last_t - _EPS:
            log.info("simulation time went backwards (%.3f -> %.3f): sensors reset",
                     self._last_t, t)
            self.reset()
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        # mj_step leaves body/site poses one physics step behind qpos; refresh them so the
        # sensors see the state at `t`. This does not change the simulation.
        mujoco.mj_kinematics(self.model, data)

        frame = SensorFrame(t=t, joints=self._joints(data, t))
        if self._odom_ch.due(t):
            self._odom_ch.push(t, self._odometry(data, t))
        frame.odometry = self._odom_ch.pop(t)
        if self._imu_ch.due(t):
            self._imu_ch.push(t, self._imu(data, t))
        frame.imu = self._imu_ch.pop(t)
        for lidar, ch in zip(self.lidars, self._lidar_ch, strict=True):
            if ch.due(t):
                ch.push(t, lidar.scan(data, self._rng_lidar))
            scan = ch.pop(t)
            if scan is not None:
                frame.lidars.append(scan)
        speeds = self._integrate_belts(data, dt)
        if self._conv_ch.due(t):
            self._conv_ch.push(t, self._conveyors(speeds, t))
        frame.conveyors = self._conv_ch.pop(t) or []
        if self.cameras_enabled:
            if self._cam_ch.due(t):
                self._cam_ch.push(t, self.render_camera(self.perception_source, data,
                                                        refresh_kinematics=False))
            frame.camera = self._cam_ch.pop(t)
        return frame

    # ---------- individual sensors ----------
    def _joints(self, data, t) -> JointState:
        noise = self._rng_joint.normal(0.0, self.cfg.joints.position_noise_std_rad, 7)
        gripper = float(np.clip(data.qpos[self._grip_qpos] / self._grip_range, 0.0, 1.0))
        return JointState(t, data.qpos[self._arm_qpos] + noise, gripper)

    @staticmethod
    def _rotate(yaw, world_xy):
        """World-frame planar vector -> frame rotated by `yaw`."""
        c, s = cos(yaw), sin(yaw)
        return c * world_xy[0] + s * world_xy[1], -s * world_xy[0] + c * world_xy[1]

    def _odometry(self, data, t) -> OdometrySample:
        """Wheel odometry: mean base-frame twist since the previous sample, as encoder
        counts give it (the servo makes the instantaneous joint velocity ripple within a
        control tick, so point samples of it would be biased)."""
        pose, prev = data.qpos[self._base_qpos].copy(), self._odom_prev
        self._odom_prev = (t, pose)
        if prev is None or t - prev[0] < _EPS:
            vx, vy = self._rotate(pose[2], data.qvel[self._base_dof[:2]])
            true = np.array([vx, vy, data.qvel[self._base_dof[2]]])
        else:
            delta = (pose - prev[1]) / (t - prev[0])
            vx, vy = self._rotate(0.5 * (pose[2] + prev[1][2]), delta[:2])
            true = np.array([vx, vy, delta[2]])
        white = self._rng_odom.normal(0.0, 1.0, 3) * self.cfg.odometry.noise_std_frac * abs(true)
        return OdometrySample(t, Twist2D(*(true * self._odom_scale + white).tolist()))

    def _imu(self, data, t) -> ImuSample:
        """Gyro: mean yaw rate since the previous sample + bias + noise. Accelerometer:
        change of base velocity between samples, in the base frame."""
        cfg = self.cfg.imu
        yaw = float(data.qpos[self._base_qpos[2]])
        vel, prev = data.qvel[self._base_dof].copy(), self._imu_prev
        self._imu_prev = (t, yaw, vel)
        if prev is None or t - prev[0] < _EPS:
            rate, (ax, ay) = vel[2], self._rotate(yaw, data.qacc[self._base_dof[:2]])
        else:
            dt = t - prev[0]
            rate, (ax, ay) = (yaw - prev[1]) / dt, self._rotate(yaw, (vel - prev[2])[:2] / dt)
        noise = self._rng_imu.normal(0.0, 1.0, 3)
        gyro = rate + cfg.gyro_bias_radps + cfg.gyro_noise_std_radps * noise[0]
        return ImuSample(t, float(gyro), (float(ax + cfg.accel_noise_std_mps2 * noise[1]),
                                          float(ay + cfg.accel_noise_std_mps2 * noise[2])))

    def _kit_slide_speed(self, station: str, data) -> float | None:
        dofs = self._kit_dofs[station]
        return float(np.mean(data.qvel[dofs])) if dofs else None

    def _integrate_belts(self, data, dt: float) -> dict[str, float]:
        speeds = {}
        for st in STATIONS:
            speed = self._belt_source(st, data)
            if speed is not None:
                speeds[st] = speed
                self._encoder[st] += speed * dt
        return speeds

    def _conveyors(self, speeds: dict[str, float], t) -> list[ConveyorSignal]:
        cfg = self.cfg.conveyor
        out = []
        for st, speed in speeds.items():
            counts = np.floor(self._encoder[st] / cfg.resolution_m)
            out.append(ConveyorSignal(t, st, float(counts * cfg.resolution_m), float(
                speed + self._rng_conv.normal(0.0, cfg.speed_noise_std_mps))))
        return out

    def render_camera(self, name: str, data, *, refresh_kinematics: bool = True) -> RgbdFrame:
        """Render one enabled camera now (expensive). `read` does this for the perception
        source at the camera rate; other enabled cameras are rendered only through here."""
        if name not in self.camera_names:
            raise ValueError(f"camera '{name}' is not enabled (enabled: {self.camera_names})")
        cams = self.cfg.cameras
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, cams.height, cams.width)
        if name not in self._cameras:
            self._cameras[name] = RgbdCamera(self.model, name, cams.width, cams.height,
                                             renderer=self._renderer)
        cam = self._cameras[name]
        if refresh_kinematics:
            mujoco.mj_kinematics(self.model, data)
        mujoco.mj_camlight(self.model, data)
        rgb, depth = cam.read(data)
        near, far = cams.depth_range_m
        valid = (depth >= near) & (depth <= far)
        noise = self._rng_cam.standard_normal(depth.shape, dtype=np.float32)
        depth = depth + np.float32(cams.depth_noise_std_m) * depth * depth * noise
        depth[~valid] = 0.0
        return RgbdFrame(float(data.time), name, rgb, depth.astype(np.float32), cam.intrinsics,
                         cam.cam_in_base(data, self._base_body))
