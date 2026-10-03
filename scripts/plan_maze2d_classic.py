"""
plan_maze2d_classic.py — A* / RRT* baselines on Maze2D (planner: diffuser/guides/classic_planners.py).

The planner plans a collision-free 2-D path (walls in the MuJoCo frame, obstacles in the CBF / evaluation frame
inflated by --planner_safety_margin); the path is resampled to H waypoints and tracked with the same PD
controller as the other open-loop methods.

Run (CPU is enough):
    CUDA_VISIBLE_DEVICES="" python scripts/plan_maze2d_classic.py --dataset maze2d-umaze-v1 --planner_type astar

Obstacles, horizon and robust_term come from config/maze2d.py ('sfp' plan block plus the per-map override).
Results: <logbase>/<dataset>/plans/<exp_name>/<suffix>/{episodes.csv, summary.json} (diffuser/utils/metrics.py).
"""
import random
import time

import numpy as np
import torch

import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.guides.classic_planners import SafePlanner
from diffuser.models.cbf import is_point_safe
from diffuser.utils.metrics import (obstacle_center_offset, episode_safety, summarize, format_episode,
                                    format_summary, save_results)
from diffuser.utils.trajectory_metrics import acceleration_smoothness


PAIR_SAFETY_MARGIN = 0.01   # start/goal rule, = SAFETY_MARGIN in plan_maze2d_sfp.py
PAIR_MAX_ATTEMPTS = 100

MAZE_BOUNDS_BY_ENV = {
    'maze2d-umaze-v1':  ((0.0, 0.0), (5.0, 5.0)),    # (mins[py, px], maxs[py, px])
    'maze2d-medium-v1': ((0.0, 0.0), (8.0, 8.0)),
    'maze2d-large-v1':  ((0.0, 0.0), (9.0, 12.0)),
}


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'
    method: str = 'sfp'                   # config block for the shared plan defaults
    planner_type: str = 'rrt_star'        # rrt_star | astar
    rrt_step_size: float = 0.5
    rrt_max_nodes: int = 5000
    rrt_rewire_radius: float = 1.0
    astar_grid_res: float = 0.2
    planner_goal_radius: float = 0.3
    planner_safety_margin: float = 0.05
    n_episodes: int = 100                 # number of evaluation episodes (start/goal pairs)


def wall_frame_offset(env):
    """Offset of the MuJoCo wall grid in qpos: wall cell (r, c) = [r-0.5, r+0.5] x [c-0.5, c+0.5] + offset.
    wall_r_c geom sits at world (r+1, c+1); qpos = world - particle body_pos, so offset = geom_pos - body_pos - (r, c)
    (= (-0.2, -0.2) on all three Maze2D maps). Checked to be identical for every wall geom."""
    m = env.unwrapped.model
    body = np.array(m.body_pos[m.body_name2id('particle')][:2], dtype=float)
    offs = []
    for i in range(m.ngeom):
        name = m.geom_id2name(i) or ''
        if name.startswith('wall_'):
            r, c = (int(v) for v in name.split('_')[1:3])
            offs.append(np.array(m.geom_pos[i][:2], dtype=float) - body - np.array([r, c], dtype=float))
    offs = np.array(offs)
    assert len(offs) and np.allclose(offs, offs[0], atol=1e-9), 'inconsistent wall frame'
    return offs[0]


def resample_path_with_velocity(path_xy, n_steps, env_dt=0.01):
    """Densify the planner's 2-D path to `n_steps` waypoints with synthetic
    velocity (forward diff). path_xy shape (N, 2)."""
    if path_xy is None or len(path_xy) < 2:
        return None, None
    pts = np.asarray(path_xy, dtype=np.float32)
    # cumulative arc length
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = s_cum[-1]
    if total < 1e-6:
        # Degenerate path — replicate
        wp_pos = np.tile(pts[-1], (n_steps, 1))
        wp_vel = np.zeros_like(wp_pos)
        return wp_pos, wp_vel
    s_query = np.linspace(0.0, total, n_steps)
    wp_pos = np.stack([np.interp(s_query, s_cum, pts[:, 0]),
                       np.interp(s_query, s_cum, pts[:, 1])], axis=1).astype(np.float32)
    wp_vel = np.zeros_like(wp_pos)
    wp_vel[:-1] = (wp_pos[1:] - wp_pos[:-1]) / env_dt
    wp_vel[-1] = wp_vel[-2]
    return wp_pos, wp_vel


def run_one_episode(env, planner, args):
    """Single episode: draw start/goal, plan, PD-track. Returns the metrics row and the timing."""
    # Start/goal: the same rule as scripts/plan_maze2d_sfp.py -- re-draw until the point is outside every
    # obstacle (margin 0.01 on the barrier with robust_term), up to 100 attempts each.
    center_offset = obstacle_center_offset(args.dataset)
    safe = lambda p: is_point_safe(p, args.obstacles, center_offset, args.robust_term, margin=PAIR_SAFETY_MARGIN)[0]
    for _ in range(PAIR_MAX_ATTEMPTS):
        observation = env.reset()
        if safe(observation[:2]):
            break
    for _ in range(PAIR_MAX_ATTEMPTS):
        env.set_target()
        if safe(np.asarray(env._target)[:2]):
            break
    target = np.array(env._target, dtype=np.float32)
    env.set_state(observation[:2], observation[2:4])

    # PLAN ----------------------------------------------------------------
    plan_start = time.time()
    path, _ = planner.plan(observation[:2].astype(np.float32), target.astype(np.float32), algo=args.planner_type)
    plan_time = time.time() - plan_start
    if path is None:
        # No path found: hold the start
        path = [observation[:2].astype(np.float32)] * 2

    H = args.horizon
    wp_pos, wp_vel = resample_path_with_velocity(np.asarray(path), H, env_dt=0.01)

    # EXECUTE -------------------------------------------------------------
    rollout = [observation.copy()]
    last_obs = observation.copy()
    ctrl_start = time.time()
    reward = 0.0
    for t in range(H):
        action = (wp_pos[t] - last_obs[:2]) + (wp_vel[t] - last_obs[2:])
        next_obs, reward, terminal, _ = env.step(action)
        rollout.append(next_obs.copy())
        if terminal:
            break
        last_obs = next_obs
    ctrl_time = time.time() - ctrl_start

    rollout_arr = np.asarray(rollout, dtype=np.float64)
    row = episode_safety(rollout_arr, reward > 0.95, args.obstacles, args.dataset)
    row['s_smooth'] = float(acceleration_smoothness(
        torch.from_numpy(np.asarray(rollout, dtype=np.float32)).float().unsqueeze(0), action_dim=0)[0])
    return row, plan_time, ctrl_time


def main():
    args = Parser().parse_args('plan')
    if args.planner_type not in ('rrt_star', 'astar'):
        raise SystemExit(f'--planner_type must be one of: rrt_star | astar, got {args.planner_type}')

    env = datasets.load_environment(args.dataset)
    bounds = MAZE_BOUNDS_BY_ENV[args.dataset]
    bounds_xy = (np.array(bounds[0], dtype=np.float32), np.array(bounds[1], dtype=np.float32))

    planner = SafePlanner(
        bounds=bounds_xy,
        obstacles=args.obstacles,
        maze_arr=env.maze_arr,
        step_size=args.rrt_step_size,
        max_nodes=args.rrt_max_nodes,
        rewire_radius=args.rrt_rewire_radius,
        grid_resolution=args.astar_grid_res,
        goal_radius=args.planner_goal_radius,
        safety_margin=args.planner_safety_margin,
        center_offset=obstacle_center_offset(args.dataset),   # obstacle frame = the CBF / evaluation frame
        wall_offset=wall_frame_offset(env),                    # wall frame = the simulated MuJoCo walls
    )
    print(f'[ classic-planner ] dataset={args.dataset}  planner={args.planner_type}  H={args.horizon}  '
          f'obstacles={len(planner.obstacles)}  center_offset={planner.center_offset}  '
          f'safety_margin={planner.safety_margin}  wall_offset={planner.wall_offset.tolist()}')

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    rows, plan_times, ctrl_times = [], [], []
    for episode in range(1, args.n_episodes + 1):
        row, plan_time, ctrl_time = run_one_episode(env, planner, args)
        rows.append(row)
        plan_times.append(plan_time)
        ctrl_times.append(ctrl_time)
        print(format_episode(episode, row))

    summary = summarize(rows)
    t_total = np.array(plan_times) + np.array(ctrl_times)
    summary['timing_ms'] = {
        't_total_mean': float(t_total.mean() * 1000),
        't_total_std': float(t_total.std() * 1000),
        'plan_mean': float(np.mean(plan_times) * 1000),
        'ctrl_mean': float(np.mean(ctrl_times) * 1000),
    }

    print('======================results======================')
    print(format_summary(summary))
    print('[timing, ms] ' + '  '.join(f'{k}: {v:.3f}' for k, v in summary['timing_ms'].items()))
    print('=======================end=========================')
    save_results(args.savepath, rows, summary)


if __name__ == '__main__':
    main()
