"""Known static map of the cell, built from the customer spec (no simulator state).

Two views of the same geometry:

- line segments (walls, conveyor footprints) as seen by the low 2D lidars; the scan
  matcher fits lidar points to these
- an occupancy grid of where the base CENTRE may not be: obstacles grown by
  ``navigation.inflation_m`` (robot circumscribed radius + margin); the planner
  searches this grid

Conveyors are solid from the floor to the belt, so they show up at lidar height.
Kits and bottles ride above the lidar plane and are not in the map.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import cos, sin

import numpy as np

from ..cell import CELL, STATIONS
from ..config import get_config


def _rectangle_corners(cx, cy, yaw, length, width) -> np.ndarray:
    """Corners (4, 2) of a rectangle, counter-clockwise."""
    c, s = cos(yaw), sin(yaw)
    half = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length / 2, width / 2]
    return np.array([cx, cy]) + half @ np.array([[c, s], [-s, c]])


def _polygon_segments(corners: np.ndarray) -> np.ndarray:
    return np.hstack([corners, np.roll(corners, -1, axis=0)])


def conveyor_footprints() -> list[np.ndarray]:
    """Corner arrays (4, 2) of each conveyor footprint, world frame."""
    c = CELL["conveyor"]
    return [_rectangle_corners(st.center[0], st.center[1], st.yaw, c["length_m"], c["width_m"])
            for st in STATIONS.values()]


def cell_segments(lidar_height_m: float | None = None) -> np.ndarray:
    """Map segments (N, 4) as rows ``x1, y1, x2, y2``: room walls, then conveyors.

    Conveyors are included only if the lidar plane is below the belt surface.
    """
    lx, ly = CELL["cell_size_m"]
    parts = [_polygon_segments(np.array([[0, 0], [lx, 0], [lx, ly], [0, ly]], dtype=float))]
    if lidar_height_m is None or lidar_height_m < CELL["conveyor"]["height_m"]:
        parts += [_polygon_segments(corners) for corners in conveyor_footprints()]
    return np.vstack(parts)


def closest_on_segments(points: np.ndarray, segments: np.ndarray):
    """For points (P, 2) and segments (S, 4): index of the nearest segment and the
    distance to it, both (P,)."""
    a, b = segments[:, :2], segments[:, 2:]
    d = b - a
    length2 = np.einsum("ij,ij->i", d, d)
    rel = points[:, None, :] - a[None, :, :]                       # (P, S, 2)
    t = np.clip(np.einsum("psj,sj->ps", rel, d) / length2, 0.0, 1.0)
    gap = rel - t[:, :, None] * d[None, :, :]
    dist = np.hypot(gap[:, :, 0], gap[:, :, 1])
    index = np.argmin(dist, axis=1)
    return index, dist[np.arange(len(points)), index]


def _inside_convex(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """True for points inside a counter-clockwise convex polygon."""
    inside = np.ones(len(points), dtype=bool)
    for p, q in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        edge, rel = q - p, points - p
        inside &= edge[0] * rel[:, 1] - edge[1] * rel[:, 0] >= 0
    return inside


@dataclass(frozen=True)
class CellMap:
    segments: np.ndarray     # (N, 4) x1, y1, x2, y2 in the world frame
    resolution: float        # grid cell size, m
    inflation: float         # clearance kept between the base centre and any obstacle, m
    occupied: np.ndarray     # (ny, nx) bool; True = base centre must not be here

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World point -> grid index (ix, iy); may lie outside the grid."""
        return int(np.floor(x / self.resolution)), int(np.floor(y / self.resolution))

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        """Centre of a grid cell."""
        return (ix + 0.5) * self.resolution, (iy + 0.5) * self.resolution

    def cell_free(self, ix: int, iy: int) -> bool:
        ny, nx = self.occupied.shape
        return 0 <= ix < nx and 0 <= iy < ny and not self.occupied[iy, ix]

    def is_free(self, x: float, y: float) -> bool:
        return self.cell_free(*self.to_cell(x, y))

    def segment_free(self, p, q) -> bool:
        """True if the straight line p -> q stays in free cells (sampled at half a cell)."""
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        steps = max(1, int(np.ceil(np.linalg.norm(q - p) / (0.5 * self.resolution))))
        return all(self.is_free(*(p + (q - p) * i / steps)) for i in range(steps + 1))

    def clearance(self, x: float, y: float) -> float:
        """Distance from a point to the nearest map segment."""
        return float(closest_on_segments(np.array([[x, y]]), self.segments)[1][0])


def build_map(resolution: float, inflation: float,
              lidar_height_m: float | None = None) -> CellMap:
    segments = cell_segments(lidar_height_m)
    lx, ly = CELL["cell_size_m"]
    nx, ny = int(round(lx / resolution)), int(round(ly / resolution))
    xs, ys = (np.arange(nx) + 0.5) * resolution, (np.arange(ny) + 0.5) * resolution
    centres = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    # Obstacles for planning are always walls + conveyors, whatever the lidar sees.
    _, dist = closest_on_segments(centres, cell_segments())
    # A cell is blocked if any point of it could be closer than `inflation` to an obstacle.
    occupied = dist < inflation + resolution * np.sqrt(0.5)
    for corners in conveyor_footprints():
        occupied |= _inside_convex(centres, corners)
    return CellMap(segments, resolution, inflation, occupied.reshape(ny, nx))


@lru_cache(maxsize=None)
def _default_map(resolution: float, inflation: float, lidar_height_m: float) -> CellMap:
    return build_map(resolution, inflation, lidar_height_m)


def get_map(config=None) -> CellMap:
    """The cell map for the configured grid resolution, inflation and lidar height."""
    cfg = config or get_config()
    return _default_map(cfg.navigation.grid_resolution_m, cfg.navigation.inflation_m,
                        cfg.sensors.lidar.height_m)
