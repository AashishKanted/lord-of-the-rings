"""Applies a `RobotCommand` to the simulated robot's actuators.

The bridge only writes ``data.ctrl`` (plus the documented grasp weld, see
``frodo.sim.grasp``); it never writes a pose. The base is three stiff position servos
(world x, y, yaw), so the base-frame twist is rate- and acceleration-limited ([actuation])
and integrated into the servo targets. With `velocity_feedforward` the targets lead by the
servo lag (kv/kp times the velocity), so the base follows the integrated trajectory itself.

    bridge = ActuatorBridge(model)
    bridge.apply(data, command)           # once per control tick, before mj_step
"""
from __future__ import annotations

from math import cos, hypot, sin

import mujoco
import numpy as np

from ..config import get_config
from ..interfaces import RobotCommand, Twist2D
from ..log import get_logger
from ..robot.description import ARM_JOINTS, BASE_JOINTS, GRIPPER_ACTUATOR
from .grasp import activate_grasp_weld, release_grasp_weld

log = get_logger(__name__)
PAD_GEOMS = {"left_pad1": "left", "left_pad2": "left", "right_pad1": "right",
             "right_pad2": "right"}
GRIPPER_BODY = "base"     # Robotiq 2F-85 palm; the grasp welds attach rings to it


class ActuatorBridge:
    """`config` is the full config (default: ``get_config()``)."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config = config or get_config()
        self.cfg = config.actuation
        self._control_dt = config.sim.control_dt_s
        self._base_ctrl = [model.actuator(j).id for j in BASE_JOINTS]
        self._base_qpos = [int(model.joint(j).qposadr[0]) for j in BASE_JOINTS]
        self._arm_ctrl = [model.actuator(j).id for j in ARM_JOINTS]
        self._arm_range = [(model.actuator_ctrlrange[a] if model.actuator_ctrllimited[a] else None)
                           for a in self._arm_ctrl]
        self._grip_ctrl = model.actuator(GRIPPER_ACTUATOR).id
        self._grip_max = float(model.actuator_ctrlrange[self._grip_ctrl, 1])
        # Servo lag kv/kp of each base position actuator (force = kp*(ctrl - q) - kv*qdot).
        self._lag = np.array([-model.actuator_biasprm[a, 2] / model.actuator_gainprm[a, 0]
                              for a in self._base_ctrl])
        self._pads = {model.geom(name).id: side for name, side in PAD_GEOMS.items()}
        gripper = model.body(GRIPPER_BODY).id
        # weld id -> ring body id, for every grasp weld in the scene
        self._welds = {e: int(model.eq_obj2id[e]) for e in range(model.neq)
                       if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD
                       and model.eq_obj1id[e] == gripper}
        self._target: np.ndarray | None = None
        self.reset()

    def reset(self, data=None) -> None:
        """Forget motion state. With `data`, re-sync the servo targets to the current ctrl
        (call after the scene was reset to a new pose)."""
        self.twist = Twist2D()          # rate/accel-limited twist currently being executed
        self._target = None if data is None else data.ctrl[self._base_ctrl].copy()
        self._last_t: float | None = None
        self._held_weld: int | None = None
        self._refusal: str | None = None
        self._clamped = False
        if data is not None:
            for weld in self._welds:
                release_grasp_weld(data, weld)

    # ---------- state for apps / tests (sim side; robot code cannot see this) ----------
    @property
    def target_pose(self) -> np.ndarray | None:
        """Integrated base trajectory (world x, y, yaw) the servos are asked to follow."""
        return None if self._target is None else self._target.copy()

    @property
    def holding(self) -> str | None:
        """Name of the ring body held by the grasp weld, or None."""
        return None if self._held_weld is None else self.model.body(
            self._welds[self._held_weld]).name

    @property
    def last_grasp_refusal(self) -> str | None:
        return self._refusal

    # ---------- per-tick apply ----------
    def apply(self, data, command: RobotCommand) -> None:
        t = float(data.time)
        if self._target is None or (self._last_t is not None and t < self._last_t):
            self.reset(data)
        dt = self._control_dt if self._last_t is None else min(t - self._last_t,
                                                               5 * self._control_dt)
        self._last_t = t
        self._drive_base(data, command.base, dt)
        if command.arm_q is not None:
            for a, q, rng in zip(self._arm_ctrl, command.arm_q, self._arm_range, strict=True):
                data.ctrl[a] = q if rng is None else min(max(q, rng[0]), rng[1])
        if command.gripper is not None:
            data.ctrl[self._grip_ctrl] = min(max(command.gripper, 0.0), 1.0) * self._grip_max
        self._grasp(data, command)

    def limit(self, wanted: Twist2D, dt: float) -> Twist2D:
        """Next executable twist: speed limits, then acceleration limits from `self.twist`."""
        c = self.cfg
        vx, vy, wz = wanted.vx, wanted.vy, wanted.wz
        speed = hypot(vx, vy)
        if speed > c.max_linear_mps:
            vx, vy = vx * c.max_linear_mps / speed, vy * c.max_linear_mps / speed
        wz = min(max(wz, -c.max_angular_radps), c.max_angular_radps)
        dvx, dvy, dwz = vx - self.twist.vx, vy - self.twist.vy, wz - self.twist.wz
        dv, dv_max = hypot(dvx, dvy), c.max_linear_accel_mps2 * dt
        if dv > dv_max:
            dvx, dvy = dvx * dv_max / dv, dvy * dv_max / dv
        dw_max = c.max_angular_accel_radps2 * dt
        dwz = min(max(dwz, -dw_max), dw_max)
        return Twist2D(self.twist.vx + dvx, self.twist.vy + dvy, self.twist.wz + dwz)

    def _drive_base(self, data, wanted: Twist2D, dt: float) -> None:
        c = self.cfg
        self.twist = tw = self.limit(wanted, dt)
        yaw_mid = self._target[2] + 0.5 * tw.wz * dt
        world = np.array([cos(yaw_mid) * tw.vx - sin(yaw_mid) * tw.vy,
                          sin(yaw_mid) * tw.vx + cos(yaw_mid) * tw.vy, tw.wz])
        self._target += world * dt
        # Anti-windup: a blocked base must not let the target run away from it.
        err = self._target - data.qpos[self._base_qpos]
        dist = hypot(err[0], err[1])
        clamped = dist > c.max_following_error_m or abs(err[2]) > c.max_following_error_rad
        if dist > c.max_following_error_m:
            self._target[:2] -= err[:2] * (1 - c.max_following_error_m / dist)
        self._target[2] -= err[2] - min(max(err[2], -c.max_following_error_rad),
                                        c.max_following_error_rad)
        if clamped and not self._clamped:
            log.warning("t=%.2f base following error %.0f mm / %.1f deg exceeds the limit: "
                        "servo target held back (base blocked?)", data.time, 1000 * dist,
                        np.degrees(err[2]))
        self._clamped = clamped
        lead = self._lag * world if c.velocity_feedforward else 0.0
        data.ctrl[self._base_ctrl] = self._target + lead

    # ---------- grasp constraint ----------
    def pads_on_ring(self, data) -> dict[int, set[str]]:
        """ring body id -> finger sides ('left'/'right') whose pads touch it right now."""
        touching: dict[int, set[str]] = {}
        rings = set(self._welds.values())
        for g1, g2 in data.contact.geom[:data.ncon]:
            for pad, other in ((g1, g2), (g2, g1)):
                body = int(self.model.geom_bodyid[other])
                if pad in self._pads and body in rings:
                    touching.setdefault(body, set()).add(self._pads[pad])
        return touching

    def _refuse(self, data, reason: str) -> None:
        if reason != self._refusal:      # log a run of identical refusals once
            log.warning("t=%.2f grasp request refused: %s", data.time, reason)
        self._refusal = reason

    def _grasp(self, data, command: RobotCommand) -> None:
        grip_cmd = data.ctrl[self._grip_ctrl] / self._grip_max
        closed = grip_cmd >= self.cfg.grasp_min_gripper_cmd
        if self._held_weld is not None and (command.release_grasp or not closed):
            log.info("t=%.2f released %s (%s)", data.time, self.holding,
                     "release requested" if command.release_grasp else "gripper opened")
            release_grasp_weld(data, self._held_weld)
            self._held_weld = None
        if not command.request_grasp:
            self._refusal = None
            return
        if command.release_grasp:
            return self._refuse(data, "release requested in the same command")
        if self._held_weld is not None:
            return
        if not closed:
            return self._refuse(data, f"gripper commanded {grip_cmd:.2f}, closed needs >= "
                                      f"{self.cfg.grasp_min_gripper_cmd}")
        both = [ring for ring, sides in self.pads_on_ring(data).items() if len(sides) == 2]
        if not both:
            return self._refuse(data, "finger pads are not both in contact with one ring")
        for weld, ring in self._welds.items():
            if ring == both[0]:
                activate_grasp_weld(self.model, data, weld)
                self._held_weld, self._refusal = weld, None
                log.info("t=%.2f grasp weld engaged on %s", data.time, self.holding)
                return
