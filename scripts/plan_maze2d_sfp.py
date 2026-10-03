"""Maze2D evaluation of StreamingFlow and Safe Streaming Flow (SSF), open- and closed-loop.

    python scripts/plan_maze2d_sfp.py --config config.maze2d --dataset maze2d-umaze-v1 --logbase logs \
        --method sfp --safety_enabled true --closed_loop true

--safety_enabled false gives StreamingFlow (no safety filter); --closed_loop false runs the open-loop variant
(plan the whole trajectory first, then track it with the PD controller). The checkpoint is read from
<logbase>/<dataset>/sfp/k{k}_s{sigma_train}/sfp_velocity_policy.pt (defaults k 0.1, sigma_train 0.005).
Results: <logbase>/<dataset>/plans/<exp_name>/<suffix>/{episodes.csv, summary.json} (diffuser/utils/metrics.py).
"""
import os
import pickle
import random
import time
from os.path import join, dirname

import numpy as np
import torch

from diffuser.guides.policies import Trajectories
from diffuser.models.sfpd import StreamingFlowPolicyDeterministic
from diffuser.models.cond_unet1D import ConditionalUnet1D
from diffuser.utils.config import resolve_checkpoint_path
from diffuser.utils.metrics import episode_safety, summarize, format_episode, format_summary, save_results
from diffuser.utils.trajectory_metrics import acceleration_smoothness
import diffuser.datasets as datasets
import diffuser.utils as utils

# Start/goal re-draw rule (safety filter enabled): the point must satisfy cbf.is_point_safe(margin)
SAFETY_MARGIN = 0.01
MAX_RESAMPLE_ATTEMPTS = 100


class SFPPlanner:
    """Utility for loading and querying a trained Streaming Flow Policy."""

    def __init__(self, planner_args, device):
        self.args = planner_args
        checkpoint_path = resolve_checkpoint_path(
            self.args.checkpoint,
            self.args.logbase,
            self.args.dataset,
            self.args.diffusion_loadpath,
        )

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except pickle.UnpicklingError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.normalizer = checkpoint['normalizer']
        config = checkpoint['config']

        state_dim = config['state_dim']
        self.pred_horizon = int(config['horizon'])

        # Velocity network with the same architecture as in training
        velocity_net = ConditionalUnet1D(
            input_dim=state_dim,
            global_cond_dim=config['condition_dim'],
            vel_weight=config['vel_weight'],
            fc_timesteps=1,
            horizon=self.pred_horizon,
        ).to(self.device)
        velocity_net.load_state_dict(checkpoint['velocity_state_dict'])

        policy = StreamingFlowPolicyDeterministic(
            velocity_net=velocity_net,
            state_dim=state_dim,
            device=self.device,
            normalizer=self.normalizer,
            args=self.args
        ).to(self.device)

        policy.eval()
        self.policy = policy

    def make_stepper(self, start: np.ndarray, goal: np.ndarray):
        return ClosedLoopStepper(
            policy=self.policy,
            normalizer=self.normalizer,
            device=self.device,
            pred_horizon=self.pred_horizon,
            start=start,
            goal=goal,
        )

    def rollout(self, start: np.ndarray, goal: np.ndarray):
        """Generate a trajectory conditioned on start/goal positions (open loop).

        Args:
            start: Raw (unnormalized) start observation [y, x, vy, vx]
            goal: Raw (unnormalized) goal position [y, x]

        Returns:
            samples: Trajectories with unnormalized observations
            info: dict with the unnormalized filter corrections and the planning time
        """
        goal = np.concatenate([goal, np.array([0.0, 0.0])], axis=0)  # add zero goal velocity
        start_normalized = self.normalizer.normalize(start, 'observations')
        goal_normalized = self.normalizer.normalize(goal, 'observations')

        start = torch.from_numpy(start_normalized).float().unsqueeze(0).to(self.device)
        goal = torch.from_numpy(goal_normalized).float().unsqueeze(0).to(self.device)

        plan_start_time = time.time()
        with torch.no_grad():
            traj_normalized, corrections, timing_info = self.policy.rollout(
                start=start,
                goal=goal,
                pred_horizon=self.pred_horizon,
            )
        plan_total_time = time.time() - plan_start_time
        states_normalized = traj_normalized.detach().cpu().numpy()[0]  # (H, state_dim)
        observations = self.normalizer.unnormalize(states_normalized, 'observations')

        # Unnormalize corrections for visualization
        unnorm_corrections = []
        for corr in corrections:
            unnorm_corr = {}
            for key in ['current', 'before', 'after']:
                val = corr[key].squeeze(0).detach().cpu().numpy()
                unnorm_corr[key] = self.normalizer.unnormalize(val, 'observations')
            unnorm_corrections.append(unnorm_corr)

        actions = np.zeros_like(observations[:, :2])  # dummy actions (not used)
        samples = Trajectories(actions=actions[None], observations=observations[None])
        info = {
            'corrections': unnorm_corrections,
            'plan_total_time': plan_total_time,
            'plan_model_time_avg': timing_info['model_time_avg'],
            'plan_cbf_time_avg': timing_info['cbf_time_avg'],
        }
        return samples, info


class ClosedLoopStepper:
    """Closed-loop: generates the next waypoint from the latest state."""

    def __init__(
        self,
        policy: StreamingFlowPolicyDeterministic,
        normalizer,
        device: torch.device,
        pred_horizon: int,
        start: np.ndarray,
        goal: np.ndarray,
    ):
        self.policy = policy
        self.normalizer = normalizer
        self.device = device
        self.state_dim = policy.state_dim

        start_norm = self.normalizer.normalize(start, 'observations')[:self.state_dim]
        # Pad goal (pos-only) to the full state (pos+vel) with zero velocity
        goal_aug = np.concatenate([goal, np.zeros(self.state_dim - goal.shape[0], dtype=np.float32)], axis=0)
        goal_norm = self.normalizer.normalize(goal_aug, 'observations')[:self.state_dim]

        cond = np.stack([start_norm, goal_norm], axis=0).astype(np.float32)
        cond_tensor = torch.from_numpy(cond).to(self.device)
        self.cond_flat = cond_tensor.unsqueeze(0).flatten(start_dim=1)

        self.t_span = torch.linspace(
            0, 1.0, pred_horizon, device=self.device, dtype=torch.float32
        )
        self.delta_ts = torch.diff(self.t_span)

    def step(self, observation: np.ndarray, step_idx: int) -> tuple:
        """Return the next waypoint (position and velocity) for the current observation.

        Returns:
            waypoint: next waypoint (position and velocity)
            correction_info: dict with 'current', 'before', 'after' (unnormalized) or None
            u_safe_phys: safe action in physical space (from the CBF) or None
            timing_info: dict with 'model_time' and 'cbf_time'
        """
        obs_norm = self.normalizer.normalize(observation, 'observations')[:self.state_dim]
        current = torch.from_numpy(obs_norm.astype(np.float32)).to(self.device)

        # cap the index at the last flow time step
        idx = min(step_idx, self.delta_ts.shape[0] - 1)
        t = self.t_span[idx]
        dt = self.delta_ts[idx]

        sample = current.view(1, 1, -1) # (B, 1, state_dim)
        timestep = t.unsqueeze(0)

        # Model inference timing (cuda sync for accurate GPU timing)
        torch.cuda.synchronize()
        model_start = time.time()
        velocity = self.policy.velocity_net(
            sample=sample,
            timestep=timestep,
            global_cond=self.cond_flat,
        ).squeeze(0).squeeze(0)
        torch.cuda.synchronize()
        model_time = time.time() - model_start

        next_state = current + velocity * dt
        correction_info = None
        u_safe_phys = None  # Safe action from CBF in physical space
        cbf_time = 0.0

        if self.policy.safety_enabled and self.policy.cbf is not None:
            # CBF-QP solving timing (cuda sync for accurate GPU timing)
            torch.cuda.synchronize()
            cbf_start = time.time()
            corrected, _, corr_info = self.policy.cbf.apply(
                current.unsqueeze(0),
                next_state.unsqueeze(0),
                t=t,
            )
            torch.cuda.synchronize()
            cbf_time = time.time() - cbf_start
            next_state = corrected.squeeze(0)

            u_safe_phys = corr_info['u_safe_phys'].squeeze(0).detach().cpu().numpy()

            # Unnormalize correction info for visualization
            correction_info = {
                key: self.normalizer.unnormalize(corr_info[key].squeeze(0).detach().cpu().numpy(), 'observations')
                for key in ['current', 'before', 'after']
            }

        waypoint = self.normalizer.unnormalize(next_state.detach().cpu().numpy(), 'observations')
        timing_info = {'model_time': model_time, 'cbf_time': cbf_time}
        return waypoint, correction_info, u_safe_phys, timing_info


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'
    method: str = 'sfp'
    n_episodes: int = 100      # number of evaluation episodes (start/goal pairs)


#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')
env = datasets.load_environment(args.dataset)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

#---------------------------------- loading ----------------------------------#

sfp_planner = SFPPlanner(args, device)

renderer = utils.Maze2dRenderer(args.dataset)
renderer.set_obstacles(args.obstacles)

set_seed(env, args.seed)

#---------------------------------- main loop ----------------------------------#
rows = []
plan_time_batch, plan_model_time_batch, plan_cbf_time_batch = [], [], []
ctrl_model_time_batch, ctrl_cbf_time_batch = [], []
use_filter = sfp_planner.policy.safety_enabled and sfp_planner.policy.cbf is not None

for episode in range(1, args.n_episodes + 1):
    print(f"\n{'='*50}")
    print(f"Episode: {episode}/{args.n_episodes}  closed_loop: {args.closed_loop}  safety_enabled: {use_filter}")
    print(f"{'='*50}")

    # Start: re-draw until it is outside every obstacle (only when the safety filter is enabled)
    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        observation = env.reset()
        if not use_filter or sfp_planner.policy.cbf.is_point_safe(observation[:2], margin=SAFETY_MARGIN)[0]:
            break
        if attempt == MAX_RESAMPLE_ATTEMPTS - 1:
            print(f"  Warning: Could not find safe start after {MAX_RESAMPLE_ATTEMPTS} attempts")

    env.set_state(observation[0:2], observation[2:4])

    # Goal: same rule
    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        env.set_target()
        if not use_filter or sfp_planner.policy.cbf.is_point_safe(env._target[:2], margin=SAFETY_MARGIN)[0]:
            break
        if attempt == MAX_RESAMPLE_ATTEMPTS - 1:
            print(f"  Warning: Could not find safe goal after {MAX_RESAMPLE_ATTEMPTS} attempts")

    target = env._target
    renderer.goal = np.array(target, copy=True)

    rollout = [observation.copy()]
    control_corrections = []
    ctrl_model_times, ctrl_cbf_times = [], []

    for t in range(args.horizon):  # env steps = plan horizon
        state = env.state_vector().copy()

        if t == 0:
            if args.closed_loop:
                closed_loop_stepper = sfp_planner.make_stepper(start=observation, goal=target)
                plan_time, plan_model_time, plan_cbf_time = 0.0, 0.0, 0.0
            else:
                samples, sfp_info = sfp_planner.rollout(start=observation, goal=target)
                plan_time = sfp_info['plan_total_time']
                plan_model_time = sfp_info['plan_model_time_avg']
                plan_cbf_time = sfp_info['plan_cbf_time_avg']
                sequence = samples.observations[0]

                # planned trajectory (open loop)
                plan_image_path = join(args.savepath, f'plan_results/plan_{episode}.png')
                os.makedirs(dirname(plan_image_path), exist_ok=True)
                renderer.composite(
                    plan_image_path, samples.observations, ncol=1,
                    show_correction_arrows=args.show_correction_arrows,
                    corrections=[sfp_info['corrections']],
                )

        u_safe_action = None
        if args.closed_loop:
            next_waypoint, ctrl_corr_info, u_safe_action, ctrl_timing = closed_loop_stepper.step(observation, t)
            control_corrections.append(ctrl_corr_info)
            ctrl_model_times.append(ctrl_timing['model_time'])
            ctrl_cbf_times.append(ctrl_timing['cbf_time'])
        else:
            if t < len(sequence) - 1:   # within the planned horizon
                next_waypoint = sequence[t+1]
            else:                       # beyond it: hold the last position with zero velocity
                next_waypoint = sequence[-1].copy()
                next_waypoint[2:] = 0

        # Closed loop with the safety filter: apply the filtered action u_safe directly.
        # Otherwise: apply the PD tracking action of the next waypoint of the (filtered) plan.
        if u_safe_action is not None:
            action = u_safe_action
        else:
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

        next_observation, reward, terminal, _ = env.step(action)
        rollout.append(next_observation.copy())

        if terminal:
            break
        observation = next_observation

    # executed trajectory
    control_image_path = join(args.savepath, f'control_results/control_{episode}.png')
    os.makedirs(dirname(control_image_path), exist_ok=True)
    valid_corrections = [c for c in control_corrections if c is not None]
    renderer.composite(
        control_image_path, np.stack(rollout)[None], ncol=1,
        show_correction_arrows=args.show_correction_arrows,
        corrections=[valid_corrections],
    )

    #---------------------------------- metrics ----------------------------------#
    rollout_arr = np.array(rollout)  # [T, 4] = [py, px, vy, vx]
    is_success = reward > 0.95
    row = episode_safety(rollout_arr, is_success, args.obstacles, args.dataset)
    # Sm of the executed trajectory; Trap = 0 by construction (no denoising path)
    row['s_smooth'] = float(acceleration_smoothness(torch.from_numpy(rollout_arr).float().unsqueeze(0), action_dim=0)[0])
    row['trap'] = 0
    rows.append(row)
    print(format_episode(episode, row))

    plan_time_batch.append(plan_time)
    plan_model_time_batch.append(plan_model_time)
    plan_cbf_time_batch.append(plan_cbf_time)
    if ctrl_model_times:
        ctrl_model_time_batch.append(np.mean(ctrl_model_times))
        ctrl_cbf_time_batch.append(np.mean(ctrl_cbf_times))

#---------------------------------- summary ----------------------------------#
summary = summarize(rows)
if args.closed_loop:
    summary['timing_ms'] = {
        'step_model_mean': float(np.mean(ctrl_model_time_batch) * 1000),
        'step_cbf_mean': float(np.mean(ctrl_cbf_time_batch) * 1000),
    }
else:
    summary['timing_ms'] = {
        'plan_total_mean': float(np.mean(plan_time_batch) * 1000),
        'plan_step_model_mean': float(np.mean(plan_model_time_batch) * 1000),
        'plan_step_cbf_mean': float(np.mean(plan_cbf_time_batch) * 1000),
    }

print("======================results======================")
print(format_summary(summary))
print('[timing, ms] ' + '  '.join(f'{k}: {v:.3f}' for k, v in summary['timing_ms'].items()))
print("=======================end=========================")
save_results(args.savepath, rows, summary)
