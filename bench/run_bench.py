#!/usr/bin/env python3
"""Run the benchmark: N pairs x M planners, one runner subprocess per trial.

Assumes the full Gazebo + Nav2 stack is already up (see bench/run_table5.sh).
Each trial is dispatched as a subprocess so a hung trial can be timed-out
without poisoning the parent. Progress is appended to <out-dir>/progress.log.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS_DEFAULT = os.path.join(REPO, 'bench', 'test_pairs.json')
# Table 5 rows (Warehouse navigation).
PLANNERS = [
    'rrt_star', 'a_star',
    'diffuser_pd', 'diffuser_cg_pd', 'safe_diffuser_pd',
    'cfm_pd', 'safe_fm_pd', 'flow_matcher_pd', 'safe_flow_matcher_pd',
    'sfp_offline_pd', 'safe_sfp_offline_pd', 'sfp_online', 'safe_sfp_online',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num-pairs', type=int, default=100,
                   help='Run pairs [start-from, num-pairs).')
    p.add_argument('--start-from', type=int, default=0)
    p.add_argument('--planners', nargs='+', default=PLANNERS)
    p.add_argument('--pairs-json', default=PAIRS_DEFAULT)
    p.add_argument('--models-dir', default=os.path.join(REPO, 'models_h512'),
                   help='Checkpoint directory (diffuser/cfm/sfp *_planner_best.pt).')
    p.add_argument('--seed', type=int, default=42,
                   help='Base seed; each trial uses seed + pair_idx.')
    p.add_argument('--trial-timeout-sec', type=float, default=180.0,
                   help='Hard wall-clock timeout per trial subprocess.')
    p.add_argument('--out-dir', default=os.path.join(REPO, 'results', 'bench_results'),
                   help='Where per-trial JSONs are written (passed to runner).')
    p.add_argument('--rollout-timeout-sec', type=float, default=90.0,
                   help='Per-rollout in-loop timeout forwarded as runner --timeout-sec.')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pairs_json) as f:
        pairs = json.load(f)['pairs']

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'progress.log')

    total = (args.num_pairs - args.start_from) * len(args.planners)
    done = 0
    successes = {p: 0 for p in args.planners}
    overall_start = time.monotonic()
    print(f'Running {total} trials ({args.num_pairs - args.start_from} pairs × {len(args.planners)} planners)', flush=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH', '')}"

    with open(log_path, 'a') as logf:
        logf.write(f'\n===== bench run at {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n')
        logf.flush()

        for pair_idx in range(args.start_from, args.num_pairs):
            for planner in args.planners:
                t0 = time.monotonic()
                cmd = [sys.executable, '-m', 'bench.runner_one_pair',
                       '--pair-idx', str(pair_idx),
                       '--planner', planner,
                       '--pairs-json', args.pairs_json,
                       '--models-dir', args.models_dir,
                       '--seed', str(args.seed),
                       '--out-dir', out_dir,
                       '--timeout-sec', str(args.rollout_timeout_sec)]
                try:
                    ret = subprocess.run(cmd, env=env, cwd=REPO,
                                         capture_output=True, text=True,
                                         timeout=args.trial_timeout_sec)
                    last_line = (ret.stdout.strip().split('\n')[-1] if ret.stdout else '(no stdout)')
                except subprocess.TimeoutExpired:
                    last_line = f'(SUBPROCESS TIMEOUT after {args.trial_timeout_sec}s)'

                done += 1
                trial_time = time.monotonic() - t0
                total_elapsed = time.monotonic() - overall_start
                eta_sec = (total_elapsed / done) * (total - done)

                # Try to read the saved JSON to count success.
                json_path = os.path.join(out_dir, f'{planner}_{pair_idx:03d}.json')
                succ = False
                if os.path.exists(json_path):
                    try:
                        with open(json_path) as jf:
                            succ = bool(json.load(jf).get('success'))
                    except Exception:
                        pass
                if succ:
                    successes[planner] += 1

                msg = (f'[{done:>3}/{total}] pair={pair_idx:>3} planner={planner:>13} '
                       f'trial={trial_time:>5.1f}s  ETA={eta_sec/60:>4.1f}m | '
                       f'{last_line[:120]}')
                print(msg, flush=True)
                logf.write(msg + '\n'); logf.flush()

    elapsed_min = (time.monotonic() - overall_start) / 60.0
    print()
    print(f'=== bench done in {elapsed_min:.1f} min ===')
    print(f'Success counts (out of {args.num_pairs - args.start_from}):')
    for p in args.planners:
        n = args.num_pairs - args.start_from
        print(f'  {p:>13}: {successes[p]:>3}/{n}  ({100*successes[p]/max(n,1):.1f}%)')


if __name__ == '__main__':
    main()
