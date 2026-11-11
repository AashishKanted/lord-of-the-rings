"""The only data that crosses between the simulator/plant and the robot software.

Robot modules (``frodo.robot``) receive a `SensorFrame` and return a `RobotCommand`.
They never import ``frodo.sim`` or ``frodo.plant`` (enforced by a test), so no
ground truth can leak into robot decisions. Line state reaches the robot only through
Modbus/TCP (``frodo.robot.plc``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Twist2D:
    """Base velocity in the base frame: vx forward, vy left, wz counter-clockwise."""
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass(frozen=True)
class LidarScan:
    t: float
    name: str
    mount: Pose2D            # sensor pose in the base frame
    angles: np.ndarray       # rad, in the sensor frame
    ranges: np.ndarray       # m; inf = no return


@dataclass(frozen=True)
class OdometrySample:
    t: float
    twist: Twist2D           # measured base velocity (wheel encoders), base frame


@dataclass(frozen=True)
class ImuSample:
    t: float
    gyro_z: float            # rad/s
    accel_xy: tuple[float, float]


@dataclass(frozen=True)
class RgbdFrame:
    t: float
    name: str
    rgb: np.ndarray          # HxWx3 uint8
    depth: np.ndarray        # HxW float32, metres along the optical axis
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy
    cam_in_base: np.ndarray  # 4x4 camera pose in the base frame (from robot kinematics)
    # Conventions: pixel u right / v down; depth 0 = no valid return; `cam_in_base` is the
    # pose of the optical frame (x right, y down, z forward) at capture time `t`.

    def deproject(self, u, v, depth=None) -> np.ndarray:
        """Pixel(s) (u, v) -> xyz in the base frame. Depth defaults to the image value at the
        rounded pixel; results for invalid (zero) depth are meaningless."""
        fx, fy, cx, cy = self.intrinsics
        u, v = np.asarray(u, float), np.asarray(v, float)
        if depth is None:
            depth = self.depth[np.rint(v).astype(int), np.rint(u).astype(int)]
        z = np.asarray(depth, float)
        cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)
        return cam @ self.cam_in_base[:3, :3].T + self.cam_in_base[:3, 3]

    def project(self, xyz_base) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base-frame xyz -> (u, v, depth along the optical axis)."""
        fx, fy, cx, cy = self.intrinsics
        cam = (np.asarray(xyz_base, float) - self.cam_in_base[:3, 3]) @ self.cam_in_base[:3, :3]
        z = cam[..., 2]
        return cx + fx * cam[..., 0] / z, cy + fy * cam[..., 1] / z, z


@dataclass(frozen=True)
class ConveyorSignal:
    t: float
    station: str
    encoder_m: float         # belt travel since start
    speed_mps: float


@dataclass(frozen=True)
class JointState:
    t: float
    arm_q: np.ndarray        # 7 arm joint angles
    gripper: float           # 0 open .. 1 closed (measured)


@dataclass
class SensorFrame:
    """Everything the robot may know at one control tick. Slow sensors are None between
    their samples."""
    t: float
    joints: JointState
    odometry: OdometrySample | None = None
    imu: ImuSample | None = None
    lidars: list[LidarScan] = field(default_factory=list)
    camera: RgbdFrame | None = None
    conveyors: list[ConveyorSignal] = field(default_factory=list)


@dataclass
class RobotCommand:
    base: Twist2D = Twist2D()
    arm_q: np.ndarray | None = None   # joint position targets; None = hold
    gripper: float | None = None      # 0 open .. 1 closed; None = hold
    request_grasp: bool = False       # ask the sim to engage the documented grasp constraint
    release_grasp: bool = False


class Mode(Enum):
    """Top-level modes owned by the orchestrator."""
    IDLE = "idle"
    NAVIGATE = "navigate"          # path tracking between stations, arm stowed
    DOCK = "dock"                  # fine approach to the dock pose
    MANIPULATE = "manipulate"      # docked, picking and placing
    STOW = "stow"                  # fold the arm before leaving a station
    SAFETY_STOP = "safety_stop"    # held by the safety monitor until explicit recovery
    FAULT = "fault"
    DONE = "done"
