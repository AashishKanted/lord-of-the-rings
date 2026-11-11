"""Global path on the known map.

1. A* over the inflated occupancy grid (8-connected, no cutting of obstacle corners)
   from the start cell to the goal cell.
2. Shortcut smoothing: from each waypoint jump to the farthest later waypoint that is
   in straight line of sight, which leaves a few straight segments.
3. Heading plan: the base is holonomic, so yaw is simply interpolated from the start
   yaw to the goal yaw in proportion to distance travelled (shortest rotation).

The returned waypoints start exactly at ``start`` and end exactly at ``goal``.
"""
from __future__ import annotations

import heapq
from math import atan2, cos, hypot, sin, sqrt

import numpy as np

from ...config import get_config
from ...interfaces import Pose2D
from ...log import get_logger
from ..map import CellMap, get_map

log = get_logger(__name__)

NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class GlobalPlanner:
    def __init__(self, config=None, cell_map: CellMap | None = None):
        cfg = config or get_config()
        self.cfg = cfg.navigation
        self.map = cell_map or get_map(cfg)

    def plan(self, start: Pose2D, goal: Pose2D) -> list[Pose2D]:
        """Waypoints from start to goal, or an empty list if there is no path."""
        start_cell = self._nearest_free_cell(start)
        goal_cell = self._nearest_free_cell(goal)
        if start_cell is None or goal_cell is None:
            log.error("no free cell near start %s or goal %s", start, goal)
            return []
        cells = self._astar(start_cell, goal_cell)
        if not cells:
            log.error("no path from %s to %s", start, goal)
            return []
        # Interior points are cell centres; the ends are the exact requested positions.
        points = [(start.x, start.y), *(self.map.to_world(*c) for c in cells[1:-1]),
                  (goal.x, goal.y)]
        points = self._shortcut(points)
        return self._with_heading(points, start.yaw, goal.yaw)

    # ---- steps ----
    def _nearest_free_cell(self, pose: Pose2D) -> tuple[int, int] | None:
        """The pose's own cell, or the closest free one within ``planner_snap_m`` (the
        estimate can sit a few cm inside the inflated zone, e.g. when leaving a dock)."""
        cell = self.map.to_cell(pose.x, pose.y)
        if self.map.cell_free(*cell):
            return cell
        free_y, free_x = np.nonzero(~self.map.occupied)
        if len(free_x) == 0:
            return None
        res = self.map.resolution
        dist = np.hypot((free_x + 0.5) * res - pose.x, (free_y + 0.5) * res - pose.y)
        best = int(np.argmin(dist))
        if dist[best] > self.cfg.planner_snap_m:
            return None
        return int(free_x[best]), int(free_y[best])

    def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        def heuristic(cell):  # octile distance: exact cost on an empty 8-connected grid
            dx, dy = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)

        cost = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        queue = [(heuristic(start), start)]
        closed = set()
        while queue:
            _, cell = heapq.heappop(queue)
            if cell == goal:
                path = [cell]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return path[::-1]
            if cell in closed:
                continue
            closed.add(cell)
            for dx, dy in NEIGHBOURS:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not self.map.cell_free(*nxt):
                    continue
                if dx and dy and not (self.map.cell_free(cell[0] + dx, cell[1])
                                      and self.map.cell_free(cell[0], cell[1] + dy)):
                    continue  # a diagonal step must not squeeze past an occupied corner
                new_cost = cost[cell] + (sqrt(2) if dx and dy else 1.0)
                if new_cost < cost.get(nxt, float("inf")):
                    cost[nxt], parent[nxt] = new_cost, cell
                    heapq.heappush(queue, (new_cost + heuristic(nxt), nxt))
        return []

    def _shortcut(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out, i = [points[0]], 0
        while i < len(points) - 1:
            j = len(points) - 1
            # Fall back to the next A* point, which is collision-free by construction
            # (or is the short hop out of an inflated start/goal cell).
            while j > i + 1 and not self.map.segment_free(points[i], points[j]):
                j -= 1
            out.append(points[j])
            i = j
        return out

    @staticmethod
    def _with_heading(points, start_yaw: float, goal_yaw: float) -> list[Pose2D]:
        lengths = [hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(points, points[1:],
                                                                   strict=False)]
        total = sum(lengths)
        turn = atan2(sin(goal_yaw - start_yaw), cos(goal_yaw - start_yaw))
        path, travelled = [], 0.0
        for point, length in zip(points, [0.0, *lengths], strict=True):
            travelled += length
            fraction = travelled / total if total > 0 else 1.0
            path.append(Pose2D(float(point[0]), float(point[1]), start_yaw + fraction * turn))
        path[-1] = Pose2D(path[-1].x, path[-1].y, goal_yaw)
        return path


def path_length(path: list[Pose2D]) -> float:
    return sum(hypot(q.x - p.x, q.y - p.y) for p, q in zip(path, path[1:], strict=False))
