"""Evaluation metrics of the Maze2D table, shared by the three harnesses
(scripts/plan_maze2d.py, scripts/plan_maze2d_sfp.py, scripts/plan_maze2d_classic.py).

One definition for every method:

    h0_i(s)    = |(s0 - o0_i) / r0_i|^n_i + |(s1 - o1_i) / r1_i|^n_i - 1      (obstacle i, state s)
    Goal       = harness goal test: reward > 0.95 at the last executed step
    Viol@0     = min over executed states k and obstacles i of h0_i(s_k) < 0
    SafeSucc@0 = Goal and not Viol@0                                         (table column "Succ.")
    Sm         = acceleration smoothness, diffuser/utils/trajectory_metrics.py
    Trap       = the plan's denoising path has a jump (diffuser/utils/local_trap.py; Diffuser / FM families)

The obstacle centre (o0, o1) is the configured centre (cx, cy) mapped to the qpos frame:
o0 = cy + offset, o1 = cx + offset, with offset -0.5 on maze2d-large and -0.7 on umaze/medium
(the same frame as the CBFs in diffuser/models/cbf.py and diffuser/models/cbf_diffuser.py).
Executed states are the initial observation followed by every observation returned by env.step.
"""
import csv
import json
import os

import numpy as np


def obstacle_center_offset(dataset):
    """Offset from the configured obstacle centre to the qpos frame (same rule as the CBFs)."""
    return -0.5 if 'large' in dataset else -0.7


def barrier_h0(states, obstacles, dataset):
    """h0 for every state and obstacle, shape [T, n_obstacles] (no robustness margin)."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
    off = obstacle_center_offset(dataset)
    out = []
    for obs in obstacles:
        cx, cy = obs['center']
        n = obs.get('order', 2)
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))
        d0 = (s[:, 0] - (cy + off)) / ry
        d1 = (s[:, 1] - (cx + off)) / rx
        out.append(np.abs(d0) ** n + np.abs(d1) ** n - 1.0)
    if not out:
        return np.full((s.shape[0], 0), np.inf)
    return np.stack(out, axis=1)


def episode_safety(states, goal_reached, obstacles, dataset):
    """Goal / Viol@0 / SafeSucc@0 of one executed episode.

    Returns a dict with goal (0/1), min_h0 (float, +inf without obstacles), viol0 (0/1), safe_succ0 (0/1).
    """
    h = barrier_h0(states, obstacles, dataset)
    min_h0 = float(h.min()) if h.size else float('inf')
    goal = bool(goal_reached)
    viol0 = min_h0 < 0.0
    return dict(goal=int(goal), min_h0=min_h0, viol0=int(viol0), safe_succ0=int(goal and not viol0))


def summarize(rows):
    """Table columns over episodes. Each row has goal, viol0, safe_succ0, s_smooth and, optionally, trap (0/1).
    Rates are in percent of the number of episodes."""
    n = len(rows)
    pct = (lambda k: 100.0 * k / n) if n else (lambda k: 0.0)
    goal = sum(r['goal'] for r in rows)
    viol0 = sum(r['viol0'] for r in rows)
    safe_succ0 = sum(r['safe_succ0'] for r in rows)
    sm = np.array([r['s_smooth'] for r in rows], dtype=np.float64)
    out = dict(
        n_episodes=n,
        safe_succ0_pct=pct(safe_succ0), safe_succ0_count=safe_succ0,
        viol0_pct=pct(viol0), viol0_count=viol0,
        sm_mean=float(sm.mean()) if n else float('nan'),
        sm_std=float(sm.std()) if n else float('nan'),
        goal_pct=pct(goal), goal_count=goal,
    )
    if n and 'trap' in rows[0]:
        trap = sum(r['trap'] for r in rows)
        out.update(trap_pct=pct(trap), trap_count=trap)
    return out


def format_episode(i, row):
    s = (f"[Episode {i}] Goal: {row['goal']}, Viol@0: {row['viol0']}, SafeSucc@0: {row['safe_succ0']}, "
         f"min_h0: {row['min_h0']:.6f}, Sm: {row['s_smooth']:.3f}")
    if 'trap' in row:
        s += f", Trap: {row['trap']}"
    return s


def format_summary(summary):
    s = (f"[Maze2D table] Succ. (SafeSucc@0): {summary['safe_succ0_pct']:.1f}%  "
         f"Viol@0: {summary['viol0_pct']:.1f}%  "
         f"Sm: {summary['sm_mean']:.3f} ± {summary['sm_std']:.3f}")
    if 'trap_pct' in summary:
        s += f"  Trap: {summary['trap_pct']:.1f}%"
    s += f"  (Goal: {summary['goal_pct']:.1f}%, {summary['n_episodes']} episodes)"
    return s


def save_results(savepath, rows, summary):
    """Writes <savepath>/episodes.csv (one row per episode) and <savepath>/summary.json."""
    os.makedirs(savepath, exist_ok=True)
    with open(os.path.join(savepath, 'episodes.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode'] + list(rows[0].keys()))
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({'episode': i, **{k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()}})
    with open(os.path.join(savepath, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[ utils/metrics ] Saved episodes.csv and summary.json to {savepath}')
