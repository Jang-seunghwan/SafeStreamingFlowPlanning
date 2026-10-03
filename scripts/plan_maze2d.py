"""Maze2D evaluation of the Diffuser / FM / FlowMatcher baseline families (open loop: plan once, then PD-track).

    # Diffuser family (--method base): Diffuser, Diffuser+CG (--safety_method gd), SafeDiffuser (--safety_method invariance)
    python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 --logbase logs --method base \
        --safety_enabled false
    # FM family (--method cfm): FM / SafeFM (--integrator pure), FlowMatcher / SafeFlowMatcher (--integrator pc --one_shot_enabled true)
    python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 --logbase logs --method cfm \
        --safety_enabled false --integrator pure

The flags of every table row are in scripts/run_baselines.sh. The model is read from
<logbase>/<dataset>/{diffusion,cfm}/H{horizon}_T{n_diffusion_steps}/ (latest state_*.pt).
Results: <logbase>/<dataset>/plans/<exp_name>/<suffix>/{episodes.csv, summary.json} (diffuser/utils/metrics.py).
"""
import os
import random
import time
from os.path import join, dirname

import numpy as np
import torch

from diffuser.guides.policies import Policy
from diffuser.utils.metrics import episode_safety, summarize, format_episode, format_summary, save_results
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'
    method: str = 'cfm'
    n_episodes: int = 100      # number of evaluation episodes (start/goal pairs)
    # Note: --safety_enabled, --safety_method, --integrator, --one_shot_enabled, --robust_term,
    # --relax_threshold and --seed are config keys (overridden through add_extras), so they are not
    # declared as typed Tap fields here — read_config() would overwrite typed defaults from config.


#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

seed = int(args.seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(f"[seed] {seed}")
print(f"[config] safety_enabled={args.safety_enabled}, safety_method={args.safety_method}, "
      f"integrator={args.integrator}, one_shot_enabled={args.one_shot_enabled}")

utils.set_device(args.device)

env = datasets.load_environment(args.dataset)
env.seed(seed)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.diffusion
dataset = diffusion_experiment.dataset

renderer = utils.Maze2dRenderer(args.dataset)
renderer.set_obstacles(args.obstacles)

policy = Policy(diffusion, dataset.normalizer, args)

def check_position_safety(pos, obstacles, dataset=''):
    """
    Check if a position is safe (outside all obstacles).
    Returns True if safe, False if inside any obstacle.
    pos: (y, x) position in physical coordinates from env.reset()
    """
    # Physical center = config center + offset (large: -0.5, others: -0.7)
    center_offset = -0.5 if 'large' in dataset else -0.7

    for obs in obstacles:
        cx, cy = obs['center']
        order = obs['order']
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))

        dy = (pos[0] - (cy + center_offset)) / ry
        dx = (pos[1] - (cx + center_offset)) / rx
        if abs(dy)**order + abs(dx)**order - 1 < 0:
            return False  # Inside this obstacle

    return True  # Outside all obstacles

#---------------------------------- main loop ----------------------------------#
rows = []
plan_time_batch = []
MAX_RESAMPLE_ATTEMPTS = 100

for episode in range(1, args.n_episodes + 1):
    print(f"Episode: {episode} / {args.n_episodes}")

    # Start / goal from the env; with the safety layer enabled, re-draw both until outside every obstacle
    resample_count = 0
    while True:
        observation = env.reset()
        env.set_target()
        target = env._target

        if args.safety_enabled and args.obstacles:
            if (check_position_safety(observation[:2], args.obstacles, dataset=args.dataset)
                    and check_position_safety(target, args.obstacles, dataset=args.dataset)):
                break
            resample_count += 1
            if resample_count >= MAX_RESAMPLE_ATTEMPTS:
                print(f"Warning: Could not find safe start-goal pair after {MAX_RESAMPLE_ATTEMPTS} attempts.")
                break
        else:
            break

    if resample_count > 0:
        print(f"  Resampled {resample_count} times to find safe start-goal pair")

    env.set_state(observation[:2], observation[2:4])

    ## condition on the start state and on the goal position (zero goal velocity)
    target = env._target
    cond = {
        0: observation,
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    rollout = [observation.copy()]

    for t in range(diffusion.horizon):

        state = env.state_vector().copy()

        ## plan once at t = 0 (open loop)
        if t == 0:
            plan_start_time = time.time()
            samples, num_trap, iter_time, s_smooth = policy(cond, batch_size=args.batch_size)
            plan_time = time.time() - plan_start_time
            sequence = samples.observations[0]

            # planned trajectory
            fullpath = join(args.savepath, f'results/{episode}.png')
            os.makedirs(dirname(fullpath), exist_ok=True)
            renderer.composite(fullpath, samples.observations, ncol=1)

        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0

        # PD controller tracking the planned states
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

        next_observation, reward, terminal, _ = env.step(action)
        rollout.append(next_observation.copy())

        if terminal:
            break

        observation = next_observation

    #---------------------------------- metrics ----------------------------------#
    is_success = reward > 0.95
    row = episode_safety(np.stack(rollout), is_success, args.obstacles, args.dataset)
    row['s_smooth'] = float(s_smooth.item())   # Sm of the planned sample (normalized coordinates)
    row['trap'] = int(num_trap >= 1)
    rows.append(row)
    print(format_episode(episode, row))
    plan_time_batch.append(plan_time)

#---------------------------------- summary ----------------------------------#
summary = summarize(rows)
summary['timing_ms'] = {
    'plan_total_mean': float(np.mean(plan_time_batch) * 1000),
    'plan_total_std': float(np.std(plan_time_batch) * 1000),
}

print("======================results======================")
print(format_summary(summary))
print('[timing, ms] ' + '  '.join(f'{k}: {v:.3f}' for k, v in summary['timing_ms'].items()))
print("=======================end=========================")
save_results(args.savepath, rows, summary)
