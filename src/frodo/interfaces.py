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
