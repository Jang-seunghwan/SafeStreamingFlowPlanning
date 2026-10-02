#!/usr/bin/env python3
"""Aggregate per-trial result JSONs into the Warehouse table (paper Table 5).

Per row:
    Goal    : % of trials that reached the goal (within 0.25 m).
    Viol    : % of trials with min_k min_i h_i(x_k) < 0 over the executed path
              (barrier with kappa = 0, see bench/metrics.py:min_barrier).
    Succ.   : Goal for methods without active safety; Goal AND NOT Viol for the
              safe methods and Diffuser + CG (marked (V)).
    h_min   : mean +- std over trials of the per-trial minimum barrier value.
    Sm.     : mean +- std (ddof 0) of the per-trial cmd_vel acceleration magnitude.
    t_Opt / t_total : optional, read from bench/measure_planning_time.py output
              (safety-correction time and full planning time, seconds).

Usage:
    python -m bench.aggregate --results-dir results/bench_results \
        [--timing-json results/planning_time.json] [--csv results/table5.csv]
Several --results-dir may be given; for the same (planner, pair) a later
directory overrides an earlier one.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os

import numpy as np

from bench.metrics import min_barrier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (row label, planner name) in Table 5 order.
ROWS = [
    ('RRT*', 'rrt_star'),
    ('A*', 'a_star'),
    ('Diffuser', 'diffuser_pd'),
    ('Diffuser + CG', 'diffuser_cg_pd'),
    ('SafeDiffuser', 'safe_diffuser_pd'),
    ('FM', 'cfm_pd'),
    ('SafeFM', 'safe_fm_pd'),
    ('FlowMatcher', 'flow_matcher_pd'),
    ('SafeFlowMatcher', 'safe_flow_matcher_pd'),
    ('StreamingFlow (open-loop)', 'sfp_offline_pd'),
    ('SSF (open-loop)', 'safe_sfp_offline_pd'),
    ('StreamingFlow (closed-loop)', 'sfp_online'),
    ('SSF (closed-loop)', 'safe_sfp_online'),
]
# Rows whose Succ. requires the absence of a safety violation.
VIOLATION_DISCOUNTED = {
    'diffuser_cg_pd', 'safe_diffuser_pd', 'safe_fm_pd', 'safe_flow_matcher_pd',
    'safe_sfp_offline_pd', 'safe_sfp_online',
}
# planning_time.json keys (measure_planning_time.py) for each row.
TIMING_KEY = {
    'rrt_star': 'rrt_star', 'a_star': 'a_star',
    'diffuser_pd': 'diffuser', 'diffuser_cg_pd': 'diffuser_cg', 'safe_diffuser_pd': 'safe_diffuser',
    'cfm_pd': 'cfm', 'safe_fm_pd': 'safe_fm',
    'flow_matcher_pd': 'flow_matcher', 'safe_flow_matcher_pd': 'safe_flow_matcher',
    'sfp_offline_pd': 'sfp_offline', 'safe_sfp_offline_pd': 'safe_sfp_offline',
    'sfp_online': 'sfp_online', 'safe_sfp_online': 'safe_sfp_online',
}


def trial_min_h(trial: dict):
    """Per-trial min barrier. Uses the runner's full-rate value when present,
    otherwise falls back to the stored (every 4th sample) odom path."""
    if trial.get('min_h') is not None:
        return float(trial['min_h'])
    path = trial.get('odom_path_xy') or []
    return min_barrier([p[0] for p in path], [p[1] for p in path])


def load_trials(results_dirs):
    by_planner = {p: {} for _, p in ROWS}
    for d in results_dirs:
        for path in sorted(glob.glob(os.path.join(d, '*_[0-9][0-9][0-9].json'))):
            with open(path) as f:
                t = json.load(f)
            if t.get('planner') in by_planner:
                by_planner[t['planner']][int(t['pair_idx'])] = t
    return by_planner


def summarize(by_planner, timing=None):
    rows = []
    for label, p in ROWS:
        trials = [by_planner[p][i] for i in sorted(by_planner[p])]
        n = len(trials)
        if n == 0:
            continue
        min_h = [trial_min_h(t) for t in trials]
        viol = [h is not None and h < 0.0 for h in min_h]
        goal = [bool(t['success']) for t in trials]
        succ = [g and not v for g, v in zip(goal, viol)] if p in VIOLATION_DISCOUNTED else goal
        hs = np.array([h for h in min_h if h is not None], dtype=float)
        sm = np.array([float(t.get('sm_acc', 0.0)) for t in trials], dtype=float)
        row = {
            'row': label, 'planner': p, 'n': n,
            'succ_pct': 100.0 * sum(succ) / n,
            'goal_pct': 100.0 * sum(goal) / n,
            'viol_pct': 100.0 * sum(viol) / n,
            'h_min_mean': float(hs.mean()) if hs.size else float('nan'),
            'h_min_std': float(hs.std()) if hs.size else float('nan'),
            'sm_mean': float(sm.mean()), 'sm_std': float(sm.std()),
        }
        if timing is not None and TIMING_KEY[p] in timing:
            tm = timing[TIMING_KEY[p]]
            row['t_opt_s'] = tm['t_qp_ms'] / 1000.0
            row['t_total_s'] = tm['t_total_ms'] / 1000.0
            row['t_total_std_s'] = tm.get('t_total_std_ms', 0.0) / 1000.0
        rows.append(row)
    return rows


def print_table(rows):
    has_t = any('t_total_s' in r for r in rows)
    head = (f"{'method':<28} {'n':>3} {'Succ.':>6} {'Goal':>6} {'Viol':>6} "
            f"{'h_min':>16} {'Sm.':>16}")
    if has_t:
        head += f" {'t_Opt(s)':>9} {'t_total(s)':>14}"
    print(head)
    print('-' * len(head))
    for r in rows:
        tag = ' (V)' if r['planner'] in VIOLATION_DISCOUNTED else ''
        line = (f"{r['row'] + tag:<28} {r['n']:>3} {r['succ_pct']:>6.1f} {r['goal_pct']:>6.1f} "
                f"{r['viol_pct']:>6.1f} {r['h_min_mean']:>7.3f} ± {r['h_min_std']:<6.3f} "
                f"{r['sm_mean']:>7.3f} ± {r['sm_std']:<6.3f}")
        if 't_total_s' in r:
            line += f" {r['t_opt_s']:>9.2f} {r['t_total_s']:>6.2f} ± {r['t_total_std_s']:<5.2f}"
        print(line)
    print('(V) = Succ. requires no safety violation (min h >= 0 on every /odom sample).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', nargs='+',
                    default=[os.path.join(REPO, 'results', 'bench_results')])
    ap.add_argument('--timing-json', default=None,
                    help='Output of bench/measure_planning_time.py (adds t_Opt / t_total).')
    ap.add_argument('--csv', default=None, help='Optional path to write the table as CSV.')
    args = ap.parse_args()

    timing = json.load(open(args.timing_json)) if args.timing_json else None
    rows = summarize(load_trials(args.results_dir), timing)
    print_table(rows)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            keys = []
            for r in rows:
                keys += [k for k in r if k not in keys]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'Saved {args.csv}')


if __name__ == '__main__':
    main()
