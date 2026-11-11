"""Simulated planar lidar: a fan of ray casts from a site on the mobile base.

The mount is a site named ``lidar_<name>`` created by ``frodo.sim.scene`` from
[sensors.lidar]. Rays are cast in the site's xy plane against everything except decoration
geoms and the base itself. Rays that the robot's own structure would block are found once
at construction (`blind`) and always report "no return", like a driver-side self filter.
"""
from __future__ import annotations

from math import atan2

import mujoco
import numpy as np

from ..interfaces import LidarScan, Pose2D
from ..log import get_logger
from .scene import LIDAR_SITE_PREFIX, VISUAL_GROUP

log = get_logger(__name__)

# Geom groups a ray can hit: everything except decoration (visual meshes, sensor housings).
_RAY_GROUPS = np.array([g != VISUAL_GROUP for g in range(mujoco.mjNGROUP)], dtype=np.uint8)


class Lidar2D:
    def __init__(self, model, name: str, cfg):
        self.model, self.name = model, name
        self.range_max = float(cfg.range_max_m)
        self.noise_std = float(cfg.range_noise_std_m)
        site = model.site(f"{LIDAR_SITE_PREFIX}{name}")
        self.site_id, self.body_id = site.id, int(site.bodyid[0])
        self._robot_root = int(model.body_rootid[self.body_id])
        w, _, _, z = model.site_quat[self.site_id]
        self.mount = Pose2D(float(model.site_pos[self.site_id, 0]),
                            float(model.site_pos[self.site_id, 1]), 2 * atan2(z, w))
        half = np.deg2rad(cfg.fov_deg) / 2
        self.angles = np.linspace(-half, half, int(cfg.rays))
        self._local = np.stack([np.cos(self.angles), np.sin(self.angles),
                                np.zeros_like(self.angles)], axis=1)
        self._geomid = np.zeros(len(self.angles), dtype=np.int32)
        self._dist = np.zeros(len(self.angles))
        self.blind = self._self_occlusion()
        if self.blind.any():
            log.info("lidar %s: %d of %d rays blocked by the robot's own structure", name,
                     int(self.blind.sum()), len(self.angles))

    def _cast(self, data, bodyexclude: int):
        origin = data.site_xpos[self.site_id]
        vec = self._local @ data.site_xmat[self.site_id].reshape(3, 3).T
        mujoco.mj_multiRay(self.model, data, origin, vec.flatten(), _RAY_GROUPS, 1, bodyexclude,
                           self._geomid, self._dist, None, len(self.angles), self.range_max)
        return self._dist, self._geomid

    def _self_occlusion(self) -> np.ndarray:
        """Rays whose nearest hit, with nothing excluded, is a part of the robot."""
        data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, data)
        _, geomid = self._cast(data, -1)
        hit = geomid >= 0
        blind = np.zeros(len(self.angles), dtype=bool)
        blind[hit] = self.model.body_rootid[self.model.geom_bodyid[geomid[hit]]] == self._robot_root
        return blind

    def scan(self, data, rng: np.random.Generator) -> LidarScan:
        """One sweep at the current simulator state (kinematics must be up to date)."""
        dist, _ = self._cast(data, self.body_id)
        ranges = np.where((dist < 0) | self.blind, np.inf, dist)
        ok = np.isfinite(ranges)
        ranges[ok] += rng.normal(0.0, self.noise_std, int(ok.sum()))
        ranges[ok & (ranges > self.range_max)] = np.inf
        np.maximum(ranges, 0.0, out=ranges)
        return LidarScan(float(data.time), self.name, self.mount, self.angles, ranges)


def coverage(lidars: list[Lidar2D], radius_m: float, samples: int = 3600) -> np.ndarray:
    """Which points of a circle of `radius_m` around the base origin lie inside the field of
    view of at least one lidar and are not self-occluded. Returns (bearing, covered) arrays
    stacked as rows; purely geometric, ignores obstacles."""
    bearing = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    px, py = radius_m * np.cos(bearing), radius_m * np.sin(bearing)
    covered = np.zeros(samples, dtype=bool)
    for lidar in lidars:
        rel = np.arctan2(py - lidar.mount.y, px - lidar.mount.x) - lidar.mount.yaw
        rel = (rel + np.pi) % (2 * np.pi) - np.pi
        half = -lidar.angles[0]
        step = lidar.angles[1] - lidar.angles[0]
        idx = np.clip(np.rint((rel + half) / step).astype(int), 0, len(lidar.angles) - 1)
        covered |= (np.abs(rel) <= half) & ~lidar.blind[idx]
    return np.stack([bearing, covered.astype(float)])
