import heapq
import math
import time
from typing import List, Tuple, Dict, Optional

import numpy as np


def _within_bounds(point: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> bool:
    mins, maxs = bounds
    return np.all(point >= mins) and np.all(point <= maxs)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class SafePlanner:
    """
    Classical planners (RRT* / A*) on Maze2D: walls from the maze grid (MuJoCo frame) and the configured
    obstacles (CBF / evaluation frame) inflated by `safety_margin`.
    """

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        obstacles: List[Dict],
        maze_arr: np.ndarray,
        step_size: float = 0.5,
        goal_sample_rate: float = 0.1,
        max_nodes: int = 5000,
        rewire_radius: float = 1.0,
        goal_radius: float = 0.3,
        grid_resolution: float = 0.2,
        safety_margin: float = 0.0,
        center_offset: float = -0.5,
        wall_offset: Tuple[float, float] = (0.0, 0.0),
    ):
        self.bounds = (np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float))
        self.obstacles = obstacles or []
        self.maze_arr = maze_arr
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_nodes = max_nodes
        self.rewire_radius = rewire_radius
        self.goal_radius = goal_radius
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        # Frames (all in qpos = observation[:2] = (s0, s1)):
        #   obstacle centre = (cy + center_offset, cx + center_offset), the CBF / evaluation frame
        #     (diffuser/models/cbf.py: -0.5 on large, -0.7 on umaze/medium);
        #   wall cell (r, c) = [r - 0.5, r + 0.5] x [c - 0.5, c + 0.5] shifted by wall_offset, the MuJoCo frame
        #     (wall geom at world (r+1, c+1), particle body at world (1.2, 1.2) -> wall_offset = (-0.2, -0.2)).
        self.center_offset = float(center_offset)
        self.wall_offset = np.array(wall_offset, dtype=float)

    def plan(self, start: np.ndarray, goal: np.ndarray, algo: str = 'rrt_star') -> Tuple[Optional[List[np.ndarray]], float]:
        algo = algo.lower()
        tic = time.time()
        if algo in ('rrt_star', 'rrt*'):
            path = self._rrt_star(start, goal)
        elif algo in ('astar', 'a*'):
            path = self._astar(start, goal)
        else:
            raise ValueError(f'Unknown planner: {algo}')
        return path, time.time() - tic

    # ------------------------- RRT* helpers ------------------------- #
    def _sample_free(self, goal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.goal_sample_rate:
            return goal
        mins, maxs = self.bounds
        for _ in range(100):
            sample = mins + np.random.rand(2) * (maxs - mins)
            if (
                not self._point_blocked_by_maze(sample)
                and not self._point_in_obstacle(sample)
            ):
                return sample
        return goal

    def _steer(self, from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
        direction = to_pt - from_pt
        dist = np.linalg.norm(direction)
        if dist == 0:
            return from_pt.copy()
        step = min(self.step_size, dist)
        return from_pt + (direction / dist) * step

    def _obs_geometry(self, obs: Dict):
        """Obstacle in qpos: centre (s0, s1), radii (r0, r1) incl. safety_margin, order n (same barrier as the CBF)."""
        cx, cy = obs['center']
        rx = float(obs.get('radius_x', obs.get('radius', 1.0))) + self.safety_margin
        ry = float(obs.get('radius_y', obs.get('radius', 1.0))) + self.safety_margin
        c = np.array([cy + self.center_offset, cx + self.center_offset], dtype=float)
        return c, np.array([ry, rx], dtype=float), obs.get('order', 2)

    def _cbf_inside(self, point: np.ndarray, obs: Dict) -> bool:
        """Inside the CBF obstacle inflated by safety_margin: sum_i ((s_i - c_i) / (r_i + margin))^n < 1."""
        c, r, n = self._obs_geometry(obs)
        d = (np.asarray(point, dtype=float) - c) / r
        return float(np.sum(np.abs(d) ** n)) < 1.0

    def _point_in_obstacle(self, point: np.ndarray) -> bool:
        if not self.obstacles:
            return False
        for obs in self.obstacles:
            if self._cbf_inside(point, obs):
                return True
        return False

    def _segment_hits_obstacle(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        if not self.obstacles:
            return False
        p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
        for obs in self.obstacles:
            c, r, n = self._obs_geometry(obs)
            if n == 2 and r[0] == r[1]:
                # exact: distance from the centre to the closest point of the segment
                v = p2 - p1; vv = float(np.dot(v, v))
                t = 0.0 if vv == 0 else float(np.clip(np.dot(c - p1, v) / vv, 0.0, 1.0))
                if np.linalg.norm(p1 + t * v - c) < r[0]:
                    return True
            else:
                n_s = int(math.ceil(np.linalg.norm(p2 - p1) / 0.01))
                for a in np.linspace(0.0, 1.0, max(n_s, 1) + 1):
                    if self._cbf_inside(p1 * (1 - a) + p2 * a, obs):
                        return True
        return False

    def _cell_of(self, x: float, axis: int) -> int:
        return int(math.floor(float(x) - self.wall_offset[axis] + 0.5))

    def _is_wall(self, row: int, col: int) -> bool:
        H_a, W_a = self.maze_arr.shape
        if row < 0 or row >= H_a or col < 0 or col >= W_a:
            return True                       # out of the maze array: blocked
        return int(self.maze_arr[row, col]) == 10

    def _point_blocked_by_maze(self, point: np.ndarray) -> bool:
        # obs[0] = s0 indexes maze_arr rows, obs[1] = s1 columns; wall cell (r, c) spans
        # [r - 0.5, r + 0.5] x [c - 0.5, c + 0.5] + wall_offset (MuJoCo wall geometry in qpos).
        return self._is_wall(self._cell_of(point[0], 0), self._cell_of(point[1], 1))

    def _segment_hits_maze(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Exact: does the closed segment p1-p2 touch any wall cell box (slab test over the cells in its bbox)?"""
        p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        d = p2 - p1
        for r in range(self._cell_of(lo[0], 0), self._cell_of(hi[0], 0) + 1):
            for c in range(self._cell_of(lo[1], 1), self._cell_of(hi[1], 1) + 1):
                if not self._is_wall(r, c):
                    continue
                bmin = np.array([r - 0.5, c - 0.5]) + self.wall_offset
                bmax = bmin + 1.0
                t0, t1 = 0.0, 1.0
                hit = True
                for k in range(2):
                    if abs(d[k]) < 1e-12:
                        if p1[k] < bmin[k] or p1[k] > bmax[k]:
                            hit = False; break
                    else:
                        ta, tb = (bmin[k] - p1[k]) / d[k], (bmax[k] - p1[k]) / d[k]
                        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
                        if t0 > t1:
                            hit = False; break
                if hit:
                    return True
        return False

    def _collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        if not _within_bounds(p2, self.bounds):
            return False
        if self._point_blocked_by_maze(p2) or self._segment_hits_maze(p1, p2):
            return False
        if self._point_in_obstacle(p2) or self._segment_hits_obstacle(p1, p2):
            return False
        return True

    def _rrt_star(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        nodes = [start]
        parents = [-1]
        costs = [0.0]

        for _ in range(self.max_nodes):
            sample = self._sample_free(goal)
            dists = [np.linalg.norm(n - sample) for n in nodes]
            nearest_idx = int(np.argmin(dists))
            new_node = self._steer(nodes[nearest_idx], sample)
            if not self._collision_free(nodes[nearest_idx], new_node):
                continue

            neighbors = [i for i, n in enumerate(nodes) if np.linalg.norm(n - new_node) <= self.rewire_radius]
            best_parent = nearest_idx
            best_cost = costs[nearest_idx] + np.linalg.norm(new_node - nodes[nearest_idx])
            for ni in neighbors:
                cand_cost = costs[ni] + np.linalg.norm(new_node - nodes[ni])
                if cand_cost < best_cost and self._collision_free(nodes[ni], new_node):
                    best_cost = cand_cost
                    best_parent = ni

            nodes.append(new_node)
            parents.append(best_parent)
            costs.append(best_cost)

            for ni in neighbors:
                new_cost = best_cost + np.linalg.norm(nodes[ni] - new_node)
                if new_cost < costs[ni] and self._collision_free(new_node, nodes[ni]):
                    parents[ni] = len(nodes) - 1
                    costs[ni] = new_cost

            if np.linalg.norm(new_node - goal) <= self.goal_radius and self._collision_free(new_node, goal):
                nodes.append(goal)
                parents.append(len(nodes) - 2)
                costs.append(best_cost + np.linalg.norm(goal - new_node))
                return self._reconstruct_path(nodes, parents, len(nodes) - 1)
        return None

    def _reconstruct_path(self, nodes: List[np.ndarray], parents: List[int], goal_idx: int) -> List[np.ndarray]:
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        return path

    # ------------------------- A* helpers ------------------------- #
    def _astar(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        mins, maxs = self.bounds
        grid_min = mins
        grid_max = maxs
        res = self.grid_resolution
        grid_shape = np.ceil((grid_max - grid_min) / res).astype(int) + 1

        def to_idx(pt: np.ndarray) -> Tuple[int, int]:
            return tuple(np.clip(((pt - grid_min) / res).astype(int), [0, 0], grid_shape - 1))

        def to_point(idx: Tuple[int, int]) -> np.ndarray:
            return grid_min + np.array(idx, dtype=float) * res

        start_idx = to_idx(start)
        goal_idx = to_idx(goal)

        def free(idx: Tuple[int, int]) -> bool:
            pt = to_point(idx)
            return (
                _within_bounds(pt, self.bounds)
                and not self._point_blocked_by_maze(pt)
                and not self._point_in_obstacle(pt)
            )

        if not free(start_idx) or not free(goal_idx):
            return None

        neighbors = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))
        came_from = {start_idx: None}
        g_score = {start_idx: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = self._astar_reconstruct(came_from, current, to_point)
                # start and goal on the same grid vertex: return a 2-point path (a 1-point path makes the
                # harness's resampling return None and the iteration crash)
                return path * 2 if len(path) == 1 else path

            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    nxt[0] < 0
                    or nxt[0] >= grid_shape[0]
                    or nxt[1] < 0
                    or nxt[1] >= grid_shape[1]
                ):
                    continue
                if not free(nxt):
                    continue
                p_cur, p_nxt = to_point(current), to_point(nxt)
                if self._segment_hits_maze(p_cur, p_nxt) or self._segment_hits_obstacle(p_cur, p_nxt):
                    continue

                step_cost = math.hypot(dx, dy) * res
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(nxt, float('inf')):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f_score = tentative_g + _euclidean(np.array(nxt), np.array(goal_idx))
                    heapq.heappush(open_set, (f_score, nxt))
        return None

    def _astar_reconstruct(self, came_from, current, to_point_fn) -> List[np.ndarray]:
        path = [to_point_fn(current)]
        while current in came_from and came_from[current] is not None:
            current = came_from[current]
            path.append(to_point_fn(current))
        path.reverse()
        return path

