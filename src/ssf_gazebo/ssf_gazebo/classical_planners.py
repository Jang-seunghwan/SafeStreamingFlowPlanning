#!/usr/bin/env python3
"""Classical (model-free) planners: grid-based A* and OMPL RRT*.

Both planners run on an augmented occupancy map: the warehouse PGM plus the
same `WAREHOUSE_OBSTACLES` virtual safety circles used by the CBF stack, so
their plans are scored on the same constraint as the safe diffusion planners.

Each planner produces a list of (x, y) waypoints that the benchmark hands to the
Nav2 FollowPath action (Regulated Pure Pursuit controller).
"""
from __future__ import annotations

import heapq
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ssf_gazebo.cbf import WAREHOUSE_OBSTACLES


# ---------------------------------------------------------------------------
# Warehouse occupancy map loader (shared)
# ---------------------------------------------------------------------------

class _OccupancyMap:
    """Tiny 2-D occupancy grid loaded from the Nav2 PGM/YAML pair, augmented
    with the CBF virtual safety circles so classical planners avoid them too.

    Virtual obstacle format matches `ssf_gazebo.cbf.WAREHOUSE_OBSTACLES`:
        {'center': (cx, cy), 'radius': r, 'order': n}
    The CBF treats h = (dx/r)^n + (dy/r)^n - 1.05 < 0 as unsafe; for the order=2
    case that is a circle of effective radius r * sqrt(1.05).
    """

    def __init__(self, yaml_path: str,
                 virtual_obstacles: Sequence[dict] = WAREHOUSE_OBSTACLES):
        try:
            import yaml
        except ImportError:
            raise RuntimeError('PyYAML required for map loading; pip install pyyaml')
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        self.resolution = float(meta['resolution'])
        self.origin_x, self.origin_y = float(meta['origin'][0]), float(meta['origin'][1])
        self.free_thresh = float(meta.get('free_thresh', 0.25))
        self.negate = int(meta.get('negate', 0)) != 0
        pgm_path = os.path.join(os.path.dirname(yaml_path), meta['image'])

        with open(pgm_path, 'rb') as f:
            magic = f.readline().strip()
            assert magic == b'P5', f"Expected P5 PGM, got {magic!r}"
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()
            self.width, self.height = map(int, line.split())
            self.maxval = int(f.readline().strip())
            data = np.frombuffer(f.read(self.width * self.height), dtype=np.uint8)
        img = data.reshape(self.height, self.width)
        self.img = np.flipud(img)  # world origin bottom-left

        # Pre-compute effective inflated radii for virtual obstacles using the
        # CBF inflation (robust_term=0.05, order=2 → eff = r * sqrt(1.05))
        self.virtual_obstacles = []
        for obs in virtual_obstacles:
            cx, cy = obs['center']
            r = float(obs.get('radius', 1.0))
            n = int(obs.get('order', 2))
            eff = r * (1.05 ** (1.0 / n))
            self.virtual_obstacles.append((float(cx), float(cy), eff))

    # ---- world-frame query --------------------------------------------------
    def _virtual_violates(self, x: float, y: float, robot_radius: float) -> bool:
        for cx, cy, r_eff in self.virtual_obstacles:
            if math.hypot(x - cx, y - cy) < r_eff + robot_radius:
                return True
        return False

    def is_free_xy(self, x: float, y: float, robot_radius: float = 0.25) -> bool:
        """Disc check at (x, y). Returns False if the disc overlaps any
        PGM-occupied cell OR any virtual safety circle."""
        if self._virtual_violates(x, y, robot_radius):
            return False
        r_cells = max(1, int(math.ceil(robot_radius / self.resolution)))
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        if cx - r_cells < 0 or cx + r_cells >= self.width:
            return False
        if cy - r_cells < 0 or cy + r_cells >= self.height:
            return False
        threshold = (1.0 - self.free_thresh) * self.maxval  # default ≈ 191
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy > r_cells * r_cells:
                    continue
                v = int(self.img[cy + dy, cx + dx])
                if self.negate:
                    v = self.maxval - v
                if v <= threshold:
                    return False
        return True

    # ---- grid-frame query (fast for A*) -------------------------------------
    def is_free_cell(self, cx: int, cy: int, robot_radius: float = 0.25) -> bool:
        """Fast cell-frame check used by grid A*."""
        wx = self.origin_x + (cx + 0.5) * self.resolution
        wy = self.origin_y + (cy + 0.5) * self.resolution
        return self.is_free_xy(wx, wy, robot_radius)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        return cx, cy

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        return (self.origin_x + (cx + 0.5) * self.resolution,
                self.origin_y + (cy + 0.5) * self.resolution)


_MAP_CACHE = {}

def get_map(yaml_path: str) -> _OccupancyMap:
    if yaml_path not in _MAP_CACHE:
        _MAP_CACHE[yaml_path] = _OccupancyMap(yaml_path)
    return _MAP_CACHE[yaml_path]


# ---------------------------------------------------------------------------
# RRT* via OMPL (with virtual-obstacle-aware validity checker)
# ---------------------------------------------------------------------------

def plan_rrt_star(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    map_yaml: str,
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 25.0, 20.0),
    time_budget_sec: float = 5.0,
    range_m: float = 0.5,
    robot_radius_m: float = 0.25,
) -> Optional[np.ndarray]:
    """RRT* with a StateValidityChecker that rejects PGM-occupied AND
    virtual-circle-occupied points. Returns (N, 2) numpy or None on failure."""
    from ompl import base as ob
    from ompl import geometric as og

    occ = get_map(map_yaml)

    space = ob.RealVectorStateSpace(2)
    bnds = ob.RealVectorBounds(2)
    bnds.setLow(0, bounds[0]); bnds.setHigh(0, bounds[2])
    bnds.setLow(1, bounds[1]); bnds.setHigh(1, bounds[3])
    space.setBounds(bnds)

    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()

    class _Checker(ob.StateValidityChecker):
        def __init__(self, si, occ, r):
            super().__init__(si)
            self.occ = occ
            self.r = r
        def isValid(self, state):
            return self.occ.is_free_xy(state[0], state[1], self.r)

    ss.setStateValidityChecker(_Checker(si, occ, robot_radius_m))
    si.setStateValidityCheckingResolution(0.01)

    s_start = si.allocState()
    s_start[0], s_start[1] = float(start_xy[0]), float(start_xy[1])
    s_goal = si.allocState()
    s_goal[0], s_goal[1] = float(goal_xy[0]), float(goal_xy[1])

    if not si.isValid(s_start) or not si.isValid(s_goal):
        return None

    ss.setStartAndGoalStates(s_start, s_goal)
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    planner = og.RRTstar(si)
    planner.setRange(range_m)
    ss.setPlanner(planner)

    solved = ss.solve(time_budget_sec)
    if not solved:
        return None
    path = ss.getSolutionPath()
    if path is None or path.getStateCount() < 2:
        return None
    path.interpolate(256)
    return np.array(
        [[path.getState(i)[0], path.getState(i)[1]] for i in range(path.getStateCount())],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Grid A* on the augmented occupancy map
# ---------------------------------------------------------------------------

_NEIGHBORS_8 = (
    (-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356),
    ( 0, -1, 1.0),                       ( 0, 1, 1.0),
    ( 1, -1, 1.41421356), ( 1, 0, 1.0), ( 1, 1, 1.41421356),
)


def plan_a_star(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    map_yaml: str,
    robot_radius_m: float = 0.25,
    max_iter: int = 400000,
    interp_resolution_m: float = 0.05,
) -> Optional[np.ndarray]:
    """8-connected grid A* on the augmented occupancy map. Returns dense
    (M, 2) waypoint array (linearly interpolated at ~5 cm) or None on failure.

    The augmented map is the warehouse PGM intersected with the complement of
    every virtual safety circle in WAREHOUSE_OBSTACLES — same as RRT*.
    """
    occ = get_map(map_yaml)
    sx_c, sy_c = occ.world_to_cell(start_xy[0], start_xy[1])
    gx_c, gy_c = occ.world_to_cell(goal_xy[0],  goal_xy[1])

    if not occ.is_free_cell(sx_c, sy_c, robot_radius_m):
        return None
    if not occ.is_free_cell(gx_c, gy_c, robot_radius_m):
        return None

    def h(cx, cy):  # admissible 8-connected Euclidean heuristic
        return math.hypot(cx - gx_c, cy - gy_c)

    open_pq: List[Tuple[float, int, int, int]] = []
    counter = 0
    heapq.heappush(open_pq, (h(sx_c, sy_c), counter, sx_c, sy_c))
    parent: dict = {(sx_c, sy_c): None}
    g_score = {(sx_c, sy_c): 0.0}
    closed = set()

    iters = 0
    found = False
    while open_pq and iters < max_iter:
        iters += 1
        _, _, x, y = heapq.heappop(open_pq)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        if (x, y) == (gx_c, gy_c):
            found = True
            break
        g_here = g_score[(x, y)]
        for dx, dy, step_cost in _NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            key = (nx, ny)
            if key in closed:
                continue
            if not occ.is_free_cell(nx, ny, robot_radius_m):
                continue
            ng = g_here + step_cost
            if key in g_score and g_score[key] <= ng:
                continue
            g_score[key] = ng
            parent[key] = (x, y)
            counter += 1
            heapq.heappush(open_pq, (ng + h(nx, ny), counter, nx, ny))

    if not found:
        return None

    # Reconstruct cell path
    cell_path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = (gx_c, gy_c)
    while cur is not None:
        cell_path.append(cur)
        cur = parent[cur]
    cell_path.reverse()

    # Convert to world waypoints
    raw = np.array(
        [occ.cell_to_world(cx, cy) for cx, cy in cell_path],
        dtype=np.float32,
    )

    # Linear interpolation at ~interp_resolution_m so RegulatedPurePursuit
    # sees a smooth dense path (same density as RRT* output).
    seg_lens = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    total = float(seg_lens.sum())
    if total < 1e-6:
        return raw
    n_samples = max(2, int(math.ceil(total / interp_resolution_m)))
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    targets = np.linspace(0.0, total, n_samples)
    xs = np.interp(targets, cumlen, raw[:, 0])
    ys = np.interp(targets, cumlen, raw[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)
