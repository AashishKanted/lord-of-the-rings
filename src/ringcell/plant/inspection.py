"""Plant-side (customer) seat measurement from simulator ground truth.

This module must never be imported by robot decision logic: it reads true geometry,
contacts and velocities. It produces the measurement dict of the evidence format and
the starter's ``inspection_passes`` decides acceptance.
"""
from __future__ import annotations

import mujoco
import numpy as np

from advaitic_systems.contract import inspection_passes

SETTLED_LIN_MPS, SETTLED_ANG_RADPS = 0.01, 0.1


class SeatMonitor:
    """Tracks one kit's ring relative to its (moving) bottle and accumulates hold time."""

    def __init__(self, model, kit_name: str):
        self.model, self.kit = model, kit_name
        self.ring = model.body(f"{kit_name}_ring").id
        self.bottle = model.body(f"{kit_name}_bottle").id
        self.tray = model.body(kit_name).id
        self.weld = model.equality(f"{kit_name}_grasp").id
        robot_root = model.body("base_link").id
        self.robot_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == robot_root}
        self.hold_s = 0.0
        self.seat_verified_t = None

    def measure(self, data) -> dict:
        m = self.model
        bottle_pos, bottle_rot = data.xpos[self.bottle], data.xmat[self.bottle].reshape(3, 3)
        rel = bottle_rot.T @ (data.xpos[self.ring] - bottle_pos)
        ring_rot = bottle_rot.T @ data.xmat[self.ring].reshape(3, 3)
        touching = set()
        for c in data.contact[:data.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if self.ring in (b1, b2):
                touching.add(b2 if b1 == self.ring else b1)
        vel_ring, vel_kit = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, self.ring, vel_ring, 0)
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, self.tray, vel_kit, 0)
        rel_vel = vel_ring - vel_kit  # [angular, linear], world axes
        return {
            "dx_m": float(rel[0]), "dy_m": float(rel[1]), "z_m": float(rel[2]),
            "tilt_rad": float(np.arccos(np.clip(abs(ring_rot[2, 2]), -1, 1))),
            "released": bool(not data.eq_active[self.weld]
                             and not (touching & self.robot_bodies)),
            "bottle_contact": self.bottle in touching,
            "kit_contact": self.tray in touching,
            "settled": bool(np.linalg.norm(rel_vel[3:]) <= SETTLED_LIN_MPS
                            and np.linalg.norm(rel_vel[:3]) <= SETTLED_ANG_RADPS),
        }

    def update(self, data) -> dict:
        """Call once per simulation step. Returns the measurement including hold_s."""
        raw = self.measure(data)
        valid = raw.pop("settled") and inspection_passes({**raw, "hold_s": 0.5})
        self.hold_s = self.hold_s + self.model.opt.timestep if valid else 0.0
        if self.seat_verified_t is None and self.hold_s >= 0.5:
            self.seat_verified_t = data.time
        return {**raw, "hold_s": self.hold_s}
