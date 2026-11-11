"""Arm inverse kinematics and grasp geometry for the tilted ring-wall pinch."""
from __future__ import annotations

import mink
import mujoco
import numpy as np

from ...cell import CELL
from ...config import get_config
from ..description import ARM_JOINTS, TOOL_SITE

_M = get_config().manipulation
TILT = np.deg2rad(_M.tilt_deg)
RING = CELL["ring"]
WALL_MID = (RING["outer_diameter_m"] + RING["inner_diameter_m"]) / 4  # 32 mm from ring centre
PAD_TIP_Z, PAD_HALF_WIDTH = 0.043, 0.011  # pad corner in the pinch_site frame (measured)


def tool_rotation_from_axes(finger_axis, toward_belt):
    """Gripper orientation: fingers close along ``finger_axis`` (the belt direction), approach
    tilted toward the belt. Both are horizontal unit vectors in the frame the IK works in.

    Pitching about the finger axis keeps the ring level while the wrist stays within reach.
    """
    x = np.asarray(finger_axis, float)
    z = np.asarray(toward_belt, float) * np.sin(TILT) - np.array([0, 0, 1]) * np.cos(TILT)
    return np.column_stack([x, np.cross(z, x), z])


def tool_rotation(station):
    """`tool_rotation_from_axes` in the world frame for a robot at the station's dock."""
    return tool_rotation_from_axes(
        [np.cos(station.yaw), np.sin(station.yaw), 0.0],
        [np.cos(station.dock[2]), np.sin(station.dock[2]), 0.0])


def tcp_in_site(rot):
    """Grasp point on the pads: 3.5 mm above the lower pad corner, in pinch_site axes."""
    up = rot.T @ np.array([0, 0, 1.0])
    corners = [np.array([0, s * PAD_HALF_WIDTH, PAD_TIP_Z]) for s in (-1, 1)]
    lowest = min(corners, key=lambda c: (rot @ c)[2])
    return lowest + 0.0035 * up


class ArmIK:
    """Differential IK (mink): each tick, a small QP turns a tool-pose error into joint motion.

    `model` is used as the robot's own kinematic model; `qpos` seeds the joint state.
    """

    def __init__(self, model, qpos, rot):
        self.model, self.rot = model, rot
        self.cfg = mink.Configuration(model)
        self.cfg.update(qpos)
        self.arm_q = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.arm_ctrl = [model.actuator(j).id for j in ARM_JOINTS]
        arm_dofs = {model.joint(j).dofadr[0] for j in ARM_JOINTS}
        self.freeze = mink.DofFreezingTask(model, [i for i in range(model.nv) if i not in arm_dofs])
        self.task = mink.FrameTask(TOOL_SITE, "site", position_cost=1.0, orientation_cost=1.0,
                                   lm_damping=1e-3)
        self.posture = mink.PostureTask(model, cost=1e-2)
        self.posture.set_target(self.cfg.q.copy())
        self.limits = [mink.ConfigurationLimit(model)]
        self.offset = rot @ tcp_in_site(rot)  # site -> TCP, world axes (orientation is fixed)

    def step(self, tcp_target, dt=_M.control_dt_s, gain=1.0):
        self.task.gain = gain
        self.task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.rot), tcp_target - self.offset))
        v = mink.solve_ik(self.cfg, [self.task, self.posture], dt, "daqp", damping=1e-4,
                          limits=self.limits, constraints=[self.freeze])
        self.cfg.integrate_inplace(v, dt)
        return self.cfg.q[self.arm_q]

    def tcp(self) -> np.ndarray:
        """Commanded TCP position (forward kinematics of the IK's own joint state)."""
        site = self.cfg.get_transform_frame_to_world(TOOL_SITE, "site")
        return site.translation() + self.offset

    def set_arm(self, arm_q) -> None:
        """Put the IK state at given arm joint angles (after a joint-space move)."""
        q = self.cfg.q.copy()
        q[self.arm_q] = arm_q
        self.cfg.update(q)
        self.posture.set_target(q)

    def solve_static(self, tcp_target, iters=300):
        for _ in range(iters):
            q = self.step(tcp_target, gain=0.3)
        self.posture.set_target(self.cfg.q.copy())
        return q


class ArmFK:
    """Forward kinematics of measured joint angles in the base frame (base joints at zero)."""

    def __init__(self, model, rot):
        self.model, self.data = model, mujoco.MjData(model)
        self.arm_q = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.site = model.site(TOOL_SITE).id
        self.tcp_in_site = tcp_in_site(rot)

    def tcp(self, arm_q) -> np.ndarray:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.arm_q] = arm_q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site] + \
            self.data.site_xmat[self.site].reshape(3, 3) @ self.tcp_in_site
