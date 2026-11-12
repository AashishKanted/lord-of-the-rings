"""Where the wrist camera is, from the robot's own joint measurements.

The camera rides on the arm, so its pose in the base frame is forward kinematics of the
measured arm joints at the moment the image was taken. The robot's kinematic model (no
kits, base at the origin) gives it; nothing comes from the simulator's state.
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from ...interfaces import JointState
from ..description import ARM_JOINTS

# MuJoCo camera axes (x right, y up, z back) -> optical axes (x right, y down, z forward).
_MJ_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


class CameraPose:
    def __init__(self, kin_model, camera: str, history_s: float = 0.5):
        self.model, self.data = kin_model, mujoco.MjData(kin_model)
        self.cam = kin_model.camera(camera).id
        self.arm_q = [kin_model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.history_s = history_s
        self._joints: deque[JointState] = deque()

    def record(self, joints: JointState) -> None:
        """Call every tick with the measured joints."""
        self._joints.append(joints)
        while joints.t - self._joints[0].t > self.history_s:
            self._joints.popleft()

    def at(self, t: float) -> np.ndarray:
        """4x4 pose of the camera's optical frame in the base frame at time t (the stored
        joint measurement nearest to t)."""
        joints = min(self._joints, key=lambda j: abs(j.t - t))
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.arm_q] = joints.arm_q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)     # camera poses are computed here
        pose = np.eye(4)
        pose[:3, :3] = self.data.cam_xmat[self.cam].reshape(3, 3) @ _MJ_TO_OPTICAL
        pose[:3, 3] = self.data.cam_xpos[self.cam]
        return pose
