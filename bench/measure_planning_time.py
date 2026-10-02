#!/usr/bin/env python3
"""Clean planning-time decomposition — simulator-free, single idle GPU.

Reports, per planner:
    t_opt   : base model / search time (no safety)
    t_qp    : safety-correction time (QP shield / CG / HOCBF-QCQP)
    t_total : t_opt + t_qp   (full planning cost)

Method
------
Safety cost is captured by monkey-patching the four GazeboCBF correction
methods to accumulate their wall-time (cbf.py itself is untouched). For one
safe-plan generation we read the accumulated t_qp, and t_opt = t_total - t_qp
— both from the SAME run, so no cross-run subtraction noise.

Base planners: t_qp = 0, t_total = t_opt.
Online SFP   : amortized per control step → t_opt = velocity_net×H,
               t_qp(safe) = HOCBF-QCQP per-step × H.
Classical    : A* grid search / OMPL RRT* solve, on CPU, t_qp = 0.

Everything is serial on one idle GPU with warmup + cuda.synchronize, so no
parallel-contention inflation. Table 5 reports t_Opt = t_qp and t_total.

Usage:  python -m bench.measure_planning_time --num-pairs 15 [--gpu 0]
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import time


# ---- monkey-patch GazeboCBF to accumulate safety-correction wall-time -------
_SAFETY = {'t': 0.0}

def _install_cbf_timers():
    import ssf_gazebo.cbf as cbfmod
    targets = [
        'shield_positions_phys',
        'classifier_guidance_phys',
        'hocbf_qcqp_velocity_filter_phys',
    ]

    def wrap(fn):
        def inner(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return fn(self, *a, **k)
            finally:
                _SAFETY['t'] += time.perf_counter() - t0
        return inner

    for name in targets:
        orig = getattr(cbfmod.GazeboCBF, name)
        setattr(cbfmod.GazeboCBF, name, wrap(orig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=None,
                    help='Sets CUDA_VISIBLE_DEVICES (default: leave the environment as is).')
    ap.add_argument('--num-pairs', type=int, default=15,
                    help='First N pairs of --pairs-json are timed.')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--models-dir', default=None,
                    help='Checkpoint directory (default: models_h512/).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pairs-json', default=None)
    ap.add_argument('--max-linear', type=float, default=0.5,
                    help='Admissible speed set of the safe_sfp_online QCQP (runner --max-linear).')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    import numpy as np
    import torch

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = args.models_dir or os.path.join(REPO, 'models_h512')
    map_yaml = os.path.join(REPO, 'src', 'ssf_gazebo', 'maps', 'ssf_map_warehouse.yaml')
    device = 'cuda'

    _install_cbf_timers()
    from bench.seeding import seed_everything
    seed_everything(args.seed)

    from ssf_gazebo.diffuser_model import DiffuserPlanner
    from ssf_gazebo.sfp_model import load_sfp_checkpoint
    from ssf_gazebo.safe_planners import (
        build_safe_diffuser, build_safe_fm, build_safe_flow_matcher, build_safe_sfp,
    )
    from ssf_gazebo.classical_planners import plan_a_star, plan_rrt_star
    from ssf_gazebo.cbf import build_normalized_cbf

    pairs_json = args.pairs_json or os.path.join(REPO, 'bench', 'test_pairs.json')
    pairs = json.load(open(pairs_json))['pairs'][:args.num_pairs]

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    results = {}

    def record(name, totals, qps, **extra):
        t_total = statistics.mean(totals)
        t_qp = statistics.mean(qps)
        results[name] = {
            't_opt_ms': 1000 * (t_total - t_qp),
            't_qp_ms': 1000 * t_qp,
            't_total_ms': 1000 * t_total,
            't_total_std_ms': 1000 * (statistics.pstdev(totals) if len(totals) > 1 else 0),
            'n': len(totals), **extra,
        }
        r = results[name]
        print(f'  {name:<20} t_opt={r["t_opt_ms"]:>8.1f}  t_qp={r["t_qp_ms"]:>8.1f}  '
              f't_total={r["t_total_ms"]:>8.1f} ms', flush=True)

    # ---- diffusion / flow offline (sample_plan); safe ones accumulate t_qp --
    def run_sample_planner(name, make, warmup_first=True):
        planner = make()
        for k in range(args.warmup):
            try:
                planner.sample_plan(start_state=list(pairs[k % len(pairs)]['start']),
                                    goal_xy=list(pairs[k % len(pairs)]['goal'][:2]), batch_size=1)
            except Exception:
                pass
        sync()
        totals, qps, skipped = [], [], 0
        for p in pairs:
            _SAFETY['t'] = 0.0
            sync(); t0 = time.perf_counter()
            try:
                planner.sample_plan(start_state=list(p['start']),
                                    goal_xy=list(p['goal'][:2]), batch_size=1)
            except Exception:
                skipped += 1
                continue
            sync(); totals.append(time.perf_counter() - t0)
            qps.append(_SAFETY['t'])
        if totals:
            record(name, totals, qps, skipped=skipped)
        else:
            print(f'  {name:<20} ALL {skipped} pairs crashed (QP singular)', flush=True)

    print(f'[planning-time] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "all")} pairs={len(pairs)} warmup={args.warmup}\n', flush=True)

    d_ckpt = os.path.join(models_dir, 'diffuser_planner_best.pt')
    c_ckpt = os.path.join(models_dir, 'cfm_planner_best.pt')
    run_sample_planner('diffuser',          lambda: DiffuserPlanner.load(d_ckpt, device=device))
    run_sample_planner('diffuser_cg',       lambda: build_safe_diffuser(DiffuserPlanner.load(d_ckpt, device=device), safety_method='gd'))
    run_sample_planner('safe_diffuser',     lambda: build_safe_diffuser(DiffuserPlanner.load(d_ckpt, device=device), safety_method='shield'))
    run_sample_planner('cfm',               lambda: DiffuserPlanner.load(c_ckpt, device=device))
    run_sample_planner('safe_fm',           lambda: build_safe_fm(DiffuserPlanner.load(c_ckpt, device=device)))
    run_sample_planner('flow_matcher',      lambda: DiffuserPlanner.load(c_ckpt, device=device, planner_type_override='flow_matcher'))
    run_sample_planner('safe_flow_matcher', lambda: build_safe_flow_matcher(DiffuserPlanner.load(c_ckpt, device=device, planner_type_override='flow_matcher')))

    # ---- SFP offline (rollout); safe one runs HOCBF-QCQP per step -----------
    base_model, norm, cfg = load_sfp_checkpoint(os.path.join(models_dir, 'sfp_planner_best.pt'), device=device)
    base_model.eval()
    H = int(cfg.horizon)

    def run_sfp_rollout(name, model):
        def one(p):
            s = torch.from_numpy(norm.normalize_state(list(p['start']))).float().to(device)
            g = torch.from_numpy(norm.normalize_state([p['goal'][0], p['goal'][1], 0., 0.])).float().to(device)
            with torch.no_grad():
                return model.rollout(s, g, pred_horizon=H)
        for k in range(args.warmup):
            try: one(pairs[k % len(pairs)])
            except Exception: pass
        sync()
        totals, qps, skipped = [], [], 0
        for p in pairs:
            _SAFETY['t'] = 0.0
            sync(); t0 = time.perf_counter()
            try:
                one(p)
            except Exception:
                skipped += 1; continue
            sync()
            totals.append(time.perf_counter() - t0); qps.append(_SAFETY['t'])
        if totals:
            record(name, totals, qps, skipped=skipped)
        else:
            print(f'  {name:<20} ALL {skipped} pairs crashed', flush=True)

    run_sfp_rollout('sfp_offline', base_model)
    run_sfp_rollout('safe_sfp_offline', build_safe_sfp(base_model, norm, cfg).to(device))

    # ---- SFP online: per-step velocity_net (t_opt) + per-step QCQP (t_qp) ---
    cbf = build_normalized_cbf(normalizer=norm, device=torch.device(device), dtype=torch.float32)
    per_model, per_qcqp = [], []
    for p in pairs:
        obs = np.array([p['start'][0], p['start'][1], 0., 0.], np.float32)
        obs_n = norm.normalize_state(obs.tolist())
        cond = np.concatenate([norm.normalize_state([p['start'][0], p['start'][1], 0., 0.]),
                               norm.normalize_state([p['goal'][0], p['goal'][1], 0., 0.])]).astype(np.float32)
        x_in = torch.from_numpy(obs_n).float().to(device).view(1, 1, -1)
        c_t = torch.from_numpy(cond).float().to(device).view(1, -1)
        t_t = torch.tensor([0.5], dtype=torch.float32, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                base_model.velocity_net(sample=x_in, timestep=t_t, global_cond=c_t)
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            base_model.velocity_net(sample=x_in, timestep=t_t, global_cond=c_t)
        sync(); per_model.append(time.perf_counter() - t0)
        # per-step HOCBF-QCQP cost
        p_phys = np.array([p['start'][0], p['start'][1]], np.float32)
        v_pos = np.array([0.3, 0.3], np.float32)
        v_vel = np.array([0.0, 0.0], np.float32)
        t1 = time.perf_counter()
        cbf.hocbf_velocity_filter_phys(p_phys=p_phys, v_pos_phys=v_pos, v_vel_phys=v_vel, dt=0.05,
                                       max_speed=args.max_linear)
        per_qcqp.append(time.perf_counter() - t1)

    ms = statistics.mean(per_model); qs = statistics.mean(per_qcqp)
    results['sfp_online'] = {'t_opt_ms': 1000*ms*H, 't_qp_ms': 0.0, 't_total_ms': 1000*ms*H,
                             'per_step_model_ms': 1000*ms, 'horizon': H, 'n': len(pairs), 'kind': 'online'}
    print(f'  {"sfp_online":<20} t_opt={results["sfp_online"]["t_opt_ms"]:>8.1f}  t_qp={0.0:>8.1f}  '
          f't_total={results["sfp_online"]["t_total_ms"]:>8.1f} ms  (model {1000*ms:.3f} ms/step ×{H})', flush=True)
    results['safe_sfp_online'] = {'t_opt_ms': 1000*ms*H, 't_qp_ms': 1000*qs*H, 't_total_ms': 1000*(ms+qs)*H,
                                  'per_step_model_ms': 1000*ms, 'per_step_qcqp_ms': 1000*qs, 'horizon': H,
                                  'n': len(pairs), 'kind': 'online'}
    print(f'  {"safe_sfp_online":<20} t_opt={results["safe_sfp_online"]["t_opt_ms"]:>8.1f}  '
          f't_qp={results["safe_sfp_online"]["t_qp_ms"]:>8.1f}  t_total={results["safe_sfp_online"]["t_total_ms"]:>8.1f} ms  '
          f'(QCQP {1000*qs:.3f} ms/step ×{H})', flush=True)

    # ---- Classical (CPU planning, no GPU, no QP) ----------------------------
    def run_classical(name, fn):
        # warmup once
        fn((pairs[0]['start'][0], pairs[0]['start'][1]), (pairs[0]['goal'][0], pairs[0]['goal'][1]))
        totals = []
        for p in pairs:
            t0 = time.perf_counter()
            fn((p['start'][0], p['start'][1]), (p['goal'][0], p['goal'][1]))
            totals.append(time.perf_counter() - t0)
        record(name, totals, [0.0] * len(totals), kind='classical')

    run_classical('a_star',   lambda s, g: plan_a_star(s, g, map_yaml=map_yaml, robot_radius_m=0.25))
    run_classical('rrt_star', lambda s, g: plan_rrt_star(s, g, map_yaml=map_yaml, time_budget_sec=5.0, range_m=0.5, robot_radius_m=0.25))

    out = args.out or os.path.join(REPO, 'results', 'planning_time.json')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
