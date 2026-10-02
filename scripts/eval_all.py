"""
Evaluation harness for the F1TENTH table (one method on one track per call).

Each episode plans a full trajectory from a start to a goal at the start/finish
line (both perturbed by Gaussian noise, see `generate_noisy_pairs`) and then
executes it in the F1TENTH gym with a PD tracking controller.

Methods (config/f1tenth.py METHOD_CONFIG):
  diffuser / diffuser_cg / safediffuser         GaussianDiffusion
  fm / safefm                                   CFM, Euler sampler
  flowmatcher / safeflowmatcher                 CFM, predictor-corrector sampler
  sfp_off (StreamingFlow) / safe_sfp_off (SSF)  streaming flow policy, open loop

Per-episode metrics (rollout = executed gym positions, plan = generated trajectory):
  goal        final rollout position within --goal_threshold of the goal and no collision
  h_min       min over the rollout of h = |dx/rx|^n + |dy/ry|^n - 1        (kappa = 0)
  b_min       same with the robust margin kappa = 0.01
  viol_k0     h_min < 0          succ_k0   = goal and not viol_k0
  viol_k001   b_min < 0          succ_k001 = goal and not viol_k001
  plan_h_min  min of h (kappa = 0) over the plan waypoints; the plan-level mark
              (h_min > 0 on the plan) holds when plan_h_min >= 0 in every episode
  sm          (1/H) sum_t ||a_{t+1} - a_t||, a_t = (p_{t+2} - 2 p_{t+1} + p_t) / dt^2 on plan positions
  t_opt_ms    time spent in the safety layer while generating the plan
  t_total_ms  wall-clock time of the planning call

Usage:
    python scripts/eval_all.py --method safe_sfp_off --track budapest --checkpoint_root checkpoints
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.f1tenth import (
    CHECKPOINT_ROOT,
    METHOD_CONFIG,
    TRACKS,
    TRACK_CONFIGS,
    SAFETY_DEFAULTS,
    PROJECT_ROOT,
    RACETRACKS_DIR,
    TRACK_DIR_MAP,
    TRACK_ID,
    checkpoint_path,
)


# ===================================================================== #
#  Constants
# ===================================================================== #

# Trajectory time step: training data is subsampled 10x from 100Hz -> 10Hz
TRAJ_DT = 0.1
# Gym simulation time step
GYM_DT = 0.01
# Margin (m) of the obstacle-aware PD reference used for the safe rows (see _safe_interpolate)
REF_MARGIN = 0.30


# ===================================================================== #
#  Utility functions (gym support)
# ===================================================================== #

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_initial_heading(track: str) -> float:
    """Compute initial heading at centerline origin from CSV data."""
    dir_name = TRACK_DIR_MAP.get(track, track)
    csv_path = os.path.join(RACETRACKS_DIR, dir_name, f'{dir_name}_centerline.csv')
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    dx = data[1, 0] - data[0, 0]
    dy = data[1, 1] - data[0, 1]
    return float(np.arctan2(dy, dx))


def derive_velocity_from_positions(trajectory: np.ndarray,
                                   traj_dt: float = TRAJ_DT) -> np.ndarray:
    """Replace model velocities with forward-difference velocities (in-place).

    v_i = (p_{i+1} - p_i) / dt  — the velocity needed to go from p_i to p_{i+1}.
    """
    H = len(trajectory)
    if H < 2:
        return trajectory
    pos = trajectory[:, :2]
    vel = np.zeros_like(pos)
    vel[:-1] = (pos[1:] - pos[:-1]) / traj_dt
    vel[-1] = vel[-2] if H > 2 else vel[0]
    trajectory[:, 2:4] = vel
    return trajectory


def interpolate_trajectory(trajectory: np.ndarray, sim_time: float,
                           traj_dt: float = TRAJ_DT) -> np.ndarray:
    """Linearly interpolate trajectory reference at given simulation time."""
    idx_float = sim_time / traj_dt
    i0 = int(idx_float)
    H = len(trajectory)
    if i0 >= H - 1:
        return trajectory[-1].copy()
    alpha = idx_float - i0
    return (1.0 - alpha) * trajectory[i0] + alpha * trajectory[i0 + 1]


def _safe_interpolate(trajectory: np.ndarray, sim_time: float,
                      obstacles: list, traj_dt: float = TRAJ_DT,
                      ref_margin: float = 0.1) -> np.ndarray:
    """Obstacle-aware interpolation: routes reference around obstacles.

    Only activates when the linearly interpolated reference would actually
    penetrate an obstacle (barrier < 0).  Routes the reference along a
    circular arc at radius = r + ref_margin.
    """
    ref = interpolate_trajectory(trajectory, sim_time, traj_dt)

    idx_float = sim_time / traj_dt
    i0 = int(idx_float)
    H = len(trajectory)
    i0 = min(i0, H - 2)
    alpha = idx_float - i0

    for obs in obstacles:
        cx, cy = obs['center']
        n = obs.get('order', 2)
        rx = obs.get('radius_x', obs.get('radius', 0.3))
        ry = obs.get('radius_y', obs.get('radius', 0.3))

        # Compute barrier at interpolated reference
        dx_n = (ref[0] - cx) / rx
        dy_n = (ref[1] - cy) / ry
        b = abs(dx_n)**n + abs(dy_n)**n - 1.0

        if b < 0:
            # Reference is INSIDE the obstacle — arc route around it
            r = max(rx, ry)
            safe_r = r + ref_margin

            a0 = np.arctan2(trajectory[i0, 1] - cy, trajectory[i0, 0] - cx)
            a1 = np.arctan2(trajectory[i0 + 1, 1] - cy, trajectory[i0 + 1, 0] - cx)

            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            angle = a0 + alpha * da

            ref[0] = cx + safe_r * np.cos(angle)
            ref[1] = cy + safe_r * np.sin(angle)

            # Update velocity to be tangent to arc
            speed = np.sqrt(ref[2]**2 + ref[3]**2)
            if speed > 1e-6:
                arc_dir = 1.0 if da >= 0 else -1.0
                ref[2] = arc_dir * (-np.sin(angle)) * speed
                ref[3] = arc_dir * np.cos(angle) * speed

    return ref


# ===================================================================== #
#  PD Tracking Controller
# ===================================================================== #

class PDTrackingController:
    """World-frame PD controller tracking (x, y, vx, vy) trajectories."""

    def __init__(self, Kp_pos=2.0, Kp_heading=1.5, Kd_heading=0.1,
                 Kp_speed=1.0, Kd_speed=0.05, dt=GYM_DT,
                 max_steer=0.4189, min_speed=-5.0, max_speed=20.0):
        self.Kp_pos = Kp_pos
        self.Kp_heading = Kp_heading
        self.Kd_heading = Kd_heading
        self.Kp_speed = Kp_speed
        self.Kd_speed = Kd_speed
        self.dt = dt
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self._prev_e_theta = 0.0
        self._prev_e_speed = 0.0

    def reset(self):
        self._prev_e_theta = 0.0
        self._prev_e_speed = 0.0

    def compute(self, x, y, theta, v_actual, x_ref, y_ref, vx_ref, vy_ref):
        """Compute (steering_angle, speed_cmd) from PD tracking."""
        e_x = x_ref - x
        e_y = y_ref - y
        vx_cmd = vx_ref + self.Kp_pos * e_x
        vy_cmd = vy_ref + self.Kp_pos * e_y

        speed_cmd = np.sqrt(vx_cmd ** 2 + vy_cmd ** 2)
        if speed_cmd < 0.05:
            heading_cmd = theta
        else:
            heading_cmd = np.arctan2(vy_cmd, vx_cmd)

        e_theta = wrap_angle(heading_cmd - theta)
        de_theta = (e_theta - self._prev_e_theta) / self.dt
        self._prev_e_theta = e_theta
        steering = self.Kp_heading * e_theta + self.Kd_heading * de_theta
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))

        e_speed = speed_cmd - v_actual
        de_speed = (e_speed - self._prev_e_speed) / self.dt
        self._prev_e_speed = e_speed
        speed = speed_cmd + self.Kp_speed * e_speed + self.Kd_speed * de_speed
        speed = float(np.clip(speed, self.min_speed, self.max_speed))

        return steering, speed


# ===================================================================== #
#  F1TENTH Gym environment
# ===================================================================== #

def make_gym_env(track: str, gym_dt: float = GYM_DT):
    """Create a single-agent F1TENTH Gym environment."""
    import gymnasium as gym
    import f1tenth_gym  # noqa: F401 – registers the env

    dir_name = TRACK_DIR_MAP.get(track, track)
    env = gym.make(
        'f1tenth-v0',
        config={
            'map': dir_name,
            'num_agents': 1,
            'timestep': gym_dt,
            'ego_idx': 0,
            'integrator': 'rk4',
            'model': 'st',
            'control_input': ['speed', 'steering_angle'],
            'observation_config': {'type': 'original'},
            'reset_config': {'type': 'cl_grid_static'},
        },
    )
    return env


# ===================================================================== #
#  Gym rollout execution
# ===================================================================== #

def execute_in_gym(env, trajectory: np.ndarray,
                   controller: PDTrackingController,
                   initial_heading: float,
                   traj_dt: float = TRAJ_DT,
                   gym_dt: float = GYM_DT,
                   obstacles: list | None = None,
                   ref_margin: float = REF_MARGIN,
                   seed: int | None = None):
    """Execute planned trajectory in F1TENTH gym with PD tracking controller.

    Args:
        env: F1TENTH gym environment.
        trajectory: [H, 4] planned trajectory (x, y, vx, vy).
        controller: PDTrackingController instance.
        initial_heading: heading angle at start position (radians).
        traj_dt: trajectory time step (0.1s for 10Hz).
        gym_dt: gym simulation time step (0.01s for 100Hz).
        obstacles: obstacles for the obstacle-aware reference (safe rows only).
        ref_margin: safety margin for obstacle-aware interpolation (meters).
        seed: seed passed to env.reset (first episode only).

    Returns:
        rollout_positions: list of [x, y] positions from gym execution.
        collision: True if car collided during rollout.
    """
    controller.reset()
    H = len(trajectory)
    total_traj_time = (H - 1) * traj_dt
    max_steps = int(total_traj_time / gym_dt) + 100

    # Reset to trajectory start position
    x0, y0 = float(trajectory[0, 0]), float(trajectory[0, 1])
    initial_pose = np.array([[x0, y0, initial_heading]])
    obs, info = env.reset(seed=seed, options={'poses': initial_pose})

    rollout_positions = [[x0, y0]]
    collision = False

    use_safe_interp = obstacles is not None and len(obstacles) > 0

    for step in range(max_steps):
        sim_time = step * gym_dt
        if sim_time > total_traj_time:
            break

        x = float(obs['poses_x'][0])
        y = float(obs['poses_y'][0])
        theta = float(obs['poses_theta'][0])
        v_body = float(obs['linear_vels_x'][0])

        if obs['collisions'][0] > 0:
            collision = True

        if use_safe_interp:
            ref = _safe_interpolate(trajectory, sim_time, obstacles,
                                    traj_dt, ref_margin)
        else:
            ref = interpolate_trajectory(trajectory, sim_time, traj_dt)
        x_ref, y_ref, vx_ref, vy_ref = ref

        steering, speed = controller.compute(
            x, y, theta, v_body, x_ref, y_ref, vx_ref, vy_ref,
        )

        action = np.array([[steering, speed]])
        obs, reward, done, truncated, info_step = env.step(action)

        x_new = float(obs['poses_x'][0])
        y_new = float(obs['poses_y'][0])
        rollout_positions.append([x_new, y_new])

        if done:
            if obs['collisions'][0] > 0:
                collision = True

    return rollout_positions, collision


# ===================================================================== #
#  Metric helpers
# ===================================================================== #

def min_barrier(xy: np.ndarray, obstacles: list, robust_term: float = 0.0) -> float:
    """Minimum over positions and obstacles of |dx/rx|^n + |dy/ry|^n - (1 + robust_term)."""
    values = []
    px = xy[:, 0]
    py = xy[:, 1]
    for obs in obstacles:
        cx, cy = obs['center']
        n = obs['order']
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))
        dx = (px - cx) / rx
        dy = (py - cy) / ry
        values.append(float(np.min(np.abs(dx) ** n + np.abs(dy) ** n - (1.0 + robust_term))))
    return min(values)


def position_smoothness(positions: np.ndarray, dt: float = TRAJ_DT) -> float:
    """Sm on plan positions: (1/H) * sum_t ||a_{t+1} - a_t||, a_t = (p_{t+2} - 2 p_{t+1} + p_t) / dt^2."""
    p = np.asarray(positions, np.float64)
    a = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt ** 2
    return float(np.linalg.norm(a[1:] - a[:-1], axis=1).sum() / len(p))


# ===================================================================== #
#  Model loading
# ===================================================================== #

def _make_policy_args(safety_enabled: bool, safety_method: str, obstacles: list, config: dict,
                      model_type: str, integrator: str | None = None,
                      n_sampling_steps: int | None = None):
    """Namespace carrying the safety / sampler parameters of one table row.

    n_sampling_steps overrides the number of flow-integration steps K stored in
    the checkpoint config (CFM family, see --k_cfm).
    """

    class PolicyArgs:
        pass

    p = PolicyArgs()
    p.safety_enabled = safety_enabled
    p.safety_method = safety_method
    p.obstacles = obstacles
    for key, value in SAFETY_DEFAULTS.items():
        setattr(p, key, value)

    if model_type in ('diffuser', 'cfm'):
        p.n_diffusion_steps = (n_sampling_steps if n_sampling_steps is not None
                               else config['n_diffusion_steps'])
        p.action_dim = config['action_dim']
        if model_type == 'cfm':
            p.integrator = integrator
    else:
        p.action_dim = 0  # selects the 2nd-order ECBF branch of CBF
    return p


def load_diffuser_policy(ckpt_path: str, device: torch.device, safety_enabled: bool,
                         safety_method: str, obstacles: list, integrator: str | None,
                         k_cfm: int):
    """Load a GaussianDiffusion (integrator None) or CFM policy from a checkpoint.

    The CFM family samples with k_cfm steps; Diffuser uses the K stored in its checkpoint.
    """
    from diffuser.models.temporal import TemporalUnet
    from diffuser.models.diffusion import GaussianDiffusion
    from diffuser.models.cfm import CFM
    from diffuser.guides.policies import Policy

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = checkpoint['normalizer']
    config = checkpoint['config']

    model = TemporalUnet(
        horizon=config['horizon'],
        transition_dim=config['transition_dim'],
        cond_dim=config['observation_dim'],
        dim=32,
        dim_mults=tuple(config['dim_mults']),
        time_scale=config.get('time_scale', 1.0),
    ).to(device)

    use_cfm = integrator is not None
    model_cls = CFM if use_cfm else GaussianDiffusion
    gen_model = model_cls(
        model=model,
        horizon=config['horizon'],
        observation_dim=config['observation_dim'],
        action_dim=config['action_dim'],
        n_timesteps=config['n_diffusion_steps'],
    ).to(device)
    gen_model.load_state_dict(checkpoint['ema'])  # EMA weights

    pargs = _make_policy_args(safety_enabled, safety_method, obstacles, config,
                              'cfm' if use_cfm else 'diffuser', integrator=integrator,
                              n_sampling_steps=k_cfm if use_cfm else None)
    policy = Policy(gen_model, normalizer, pargs)
    return policy, normalizer, config


def load_sfp_policy(ckpt_path: str, device: torch.device, safety_enabled: bool,
                    safety_method: str, obstacles: list):
    """Load a StreamingFlowPolicyDeterministic from a checkpoint."""
    from diffuser.models.cond_unet1D import ConditionalUnet1D
    from diffuser.models.sfpd import StreamingFlowPolicyDeterministic

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = checkpoint['normalizer']
    config = checkpoint['config']

    velocity_net = ConditionalUnet1D(
        input_dim=config.get('state_dim', 4),
        horizon=config['horizon'],
        time_scale=config.get('time_scale', 1.0),
    ).to(device)
    velocity_net.load_state_dict(checkpoint['velocity_state_dict'])

    sfp_args = _make_policy_args(safety_enabled, safety_method, obstacles, config, 'sfp')
    sfp_policy = StreamingFlowPolicyDeterministic(
        velocity_net=velocity_net,
        device=device,
        normalizer=normalizer,
        args=sfp_args,
    ).to(device)
    sfp_policy.eval()
    return sfp_policy, normalizer, config


# ===================================================================== #
#  Planning
# ===================================================================== #

def plan_diffuser_cfm(policy, config, start_obs: np.ndarray, goal_pos: np.ndarray):
    """Plan with a Diffuser / CFM policy.

    Returns: sequence [H, 4] (unnormalized), t_opt (s, safety layer), t_total (s).
    """
    horizon = config['horizon']
    cond = {
        0: start_obs,
        horizon - 1: np.array([*goal_pos[:2], 0.0, 0.0], dtype=np.float32),
    }
    t_start = time.time()
    observations, safety_time_avg = policy(cond, batch_size=1)
    total_s = time.time() - t_start
    # safety_time_avg is the per-step average over the K sampling steps
    return observations[0], safety_time_avg * policy.n_diffusion_steps, total_s


def plan_sfp_open_loop(sfp_policy, normalizer, config, start_obs: np.ndarray,
                       goal_pos: np.ndarray, device: torch.device):
    """Open-loop SFP plan (sfp_off / safe_sfp_off); the ECBF acts inside the rollout.

    Returns: sequence [H, 4] (unnormalized), t_opt (s, safety filter), t_total (s).
    """
    pred_horizon = config['horizon']

    start_norm = normalizer.normalize(start_obs, 'observations')
    goal_aug = np.array([*goal_pos[:2], 0.0, 0.0], dtype=np.float32)
    goal_norm = normalizer.normalize(goal_aug, 'observations')

    start_t = torch.from_numpy(start_norm).float().unsqueeze(0).to(device)
    goal_t = torch.from_numpy(goal_norm).float().unsqueeze(0).to(device)

    t_start = time.time()
    with torch.no_grad():
        traj_norm, cbf_time_avg = sfp_policy.rollout(
            start=start_t, goal=goal_t, pred_horizon=pred_horizon,
        )
    total_s = time.time() - t_start

    states_norm = traj_norm.detach().cpu().numpy()[0]  # [H, 4]
    sequence = normalizer.unnormalize(states_norm, 'observations')
    return sequence, cbf_time_avg * (pred_horizon - 1), total_s


# ===================================================================== #
#  Start-goal pairs and seeding
# ===================================================================== #

def generate_noisy_pairs(n_episodes: int, track: str, sigma: float, seed: int,
                         start_vel=(0.0, 0.0)):
    """Start/goal pairs at the start/finish line (origin) with Gaussian position noise.

    start = (N(0, sigma^2), N(0, sigma^2), start_vel), goal = (N(0, sigma^2), N(0, sigma^2)),
    drawn in the order start, goal for each episode from default_rng([seed, TRACK_ID[track]]).
    The pairs are therefore identical for every method on a track, and the first k
    pairs do not depend on n_episodes.
    """
    rng = np.random.default_rng([seed, TRACK_ID[track]])
    pairs = []
    for _ in range(n_episodes):
        s = rng.normal(0.0, sigma, 2)
        g = rng.normal(0.0, sigma, 2)
        pairs.append({
            'start': [float(s[0]), float(s[1]), float(start_vel[0]), float(start_vel[1])],
            'goal': [float(g[0]), float(g[1])],
        })
    return pairs


def seed_everything(seed: int):
    """Seed python, numpy and torch (CPU and CUDA). Called after the model is built,
    so the sampling noise depends only on the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================================================================== #
#  Main evaluation loop
# ===================================================================== #

def evaluate(args):
    model_type, safety_enabled, integrator, safety_method = METHOD_CONFIG[args.method]
    obstacles = TRACK_CONFIGS[args.track]['obstacles']
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt_path = checkpoint_path(args.method, args.track, args.checkpoint_root)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f'[eval_all] {args.method} on {args.track}  (model {model_type}, integrator {integrator}, '
          f'safety {safety_method})')
    print(f'[eval_all] Checkpoint: {ckpt_path}')

    # Load model
    if model_type == 'sfp':
        sfp_policy, normalizer, config = load_sfp_policy(
            ckpt_path, device, safety_enabled, safety_method, obstacles)
        n_steps = None
    else:
        policy, normalizer, config = load_diffuser_policy(
            ckpt_path, device, safety_enabled, safety_method, obstacles, integrator, args.k_cfm)
        n_steps = policy.n_diffusion_steps
        print(f'[eval_all] Sampling steps K: {n_steps}')

    seed_everything(args.seed)

    start_vel = config.get('start_vel', [0.0, 0.0])
    pairs = generate_noisy_pairs(args.n_episodes, args.track, args.sigma, args.seed,
                                 start_vel=tuple(start_vel))
    print(f'[eval_all] {args.n_episodes} start/goal pairs: sigma={args.sigma} m around the S/F line, '
          f'seed [{args.seed}, {TRACK_ID[args.track]}]')

    initial_heading = get_initial_heading(args.track)
    env = make_gym_env(args.track, gym_dt=GYM_DT)
    controller = PDTrackingController(dt=GYM_DT)

    per_episode = []
    for ep_idx, pair in enumerate(pairs):
        start_obs = np.array(pair['start'], dtype=np.float32)
        goal_pos = np.array(pair['goal'], dtype=np.float32)

        # ── 1. Plan ──
        if model_type == 'sfp':
            sequence, t_opt_s, total_s = plan_sfp_open_loop(
                sfp_policy, normalizer, config, start_obs, goal_pos, device)
        else:
            sequence, t_opt_s, total_s = plan_diffuser_cfm(policy, config, start_obs, goal_pos)

        # ── 2. Plan-level metrics ──
        sm = position_smoothness(sequence[:, :2])
        plan_h_min = min_barrier(np.asarray(sequence[:, :2], np.float64), obstacles)

        # ── 3. Execute in the F1TENTH gym ──
        traj_for_gym = derive_velocity_from_positions(sequence.copy(), traj_dt=TRAJ_DT)
        # Safe rows: the PD reference is re-routed around an obstacle whenever the
        # linearly interpolated reference would lie inside it (mitigates PD lag).
        rollout_positions, collision = execute_in_gym(
            env, traj_for_gym, controller, initial_heading,
            traj_dt=TRAJ_DT, gym_dt=GYM_DT,
            obstacles=obstacles if safety_enabled else None,
            seed=args.seed if ep_idx == 0 else None,
        )

        # ── 4. Rollout metrics ──
        rollout_arr = np.array(rollout_positions, dtype=np.float32)  # [T, 2]
        goal_dist = float(np.linalg.norm(rollout_arr[-1] - goal_pos[:2]))
        goal = (goal_dist < args.goal_threshold) and (not collision)
        b_min = min_barrier(rollout_arr, obstacles, robust_term=SAFETY_DEFAULTS['robust_term'])
        h_min = min_barrier(rollout_arr, obstacles, robust_term=0.0)
        viol_k001 = b_min < 0
        viol_k0 = h_min < 0

        per_episode.append({
            'episode': ep_idx,
            'start': pair['start'],
            'goal': pair['goal'],
            'goal_reached': bool(goal),
            'collision': bool(collision),
            'goal_dist': round(goal_dist, 6),
            'h_min': round(h_min, 6),
            'b_min': round(b_min, 6),
            'viol_k0': bool(viol_k0),
            'viol_k001': bool(viol_k001),
            'succ_k0': bool(goal and not viol_k0),
            'succ_k001': bool(goal and not viol_k001),
            'plan_h_min': round(plan_h_min, 6),
            'sm': round(sm, 6),
            't_opt_ms': round(t_opt_s * 1000.0, 3),
            't_total_ms': round(total_s * 1000.0, 3),
        })
        print(f'  [{ep_idx+1:4d}/{len(pairs)}] goal={goal} col={collision} dg={goal_dist:.3f} '
              f'h_min={h_min:.3f} plan_h_min={plan_h_min:.3f} sm={sm:.3f} t_total={total_s:.2f}s')

    env.close()

    def rate(key):
        return round(float(np.mean([e[key] for e in per_episode])), 4)

    def mean_std(key):
        values = [e[key] for e in per_episode]
        return round(float(np.mean(values)), 6), round(float(np.std(values)), 6)

    plan_n_viol = sum(1 for e in per_episode if e['plan_h_min'] < 0)
    sm_mean, sm_std = mean_std('sm')
    t_opt_mean, t_opt_std = mean_std('t_opt_ms')
    t_total_mean, t_total_std = mean_std('t_total_ms')
    summary = {
        'method': args.method,
        'track': args.track,
        'checkpoint': ckpt_path,
        'n_episodes': len(per_episode),
        'seed': args.seed,
        'sigma_m': args.sigma,
        'sampling_steps_K': n_steps,
        'goal_rate': rate('goal_reached'),
        'viol_k0_rate': rate('viol_k0'),
        'succ_k0_rate': rate('succ_k0'),
        'viol_k001_rate': rate('viol_k001'),
        'succ_k001_rate': rate('succ_k001'),
        'plan_safe': plan_n_viol == 0,
        'plan_n_viol': plan_n_viol,
        'sm_mean': sm_mean,
        'sm_std': sm_std,
        't_opt_s_mean': round(t_opt_mean / 1000.0, 4),
        't_opt_s_std': round(t_opt_std / 1000.0, 4),
        't_total_s_mean': round(t_total_mean / 1000.0, 4),
        't_total_s_std': round(t_total_std / 1000.0, 4),
    }

    out_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.track}.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'per_episode': per_episode}, f, indent=2)

    s = summary
    print('\n' + '=' * 60)
    print(f'  {s["method"]} on {s["track"]}, {s["n_episodes"]} episodes')
    print(f'  Goal:       {s["goal_rate"]:.2%}')
    print(f'  Viol@0:     {s["viol_k0_rate"]:.2%}   Succ@0:    {s["succ_k0_rate"]:.2%}')
    print(f'  Viol@0.01:  {s["viol_k001_rate"]:.2%}   Succ@0.01: {s["succ_k001_rate"]:.2%}')
    print(f'  Plan h>0:   {"yes" if s["plan_safe"] else "no"} ({s["plan_n_viol"]} episodes violate)')
    print(f'  Sm:         {s["sm_mean"]:.4f} +/- {s["sm_std"]:.4f}')
    print(f'  t_Opt:      {s["t_opt_s_mean"]:.3f} +/- {s["t_opt_s_std"]:.3f} s')
    print(f'  t_total:    {s["t_total_s_mean"]:.3f} +/- {s["t_total_s_std"]:.3f} s')
    print(f'  Results:    {out_path}')
    print('=' * 60)
    return summary


# ===================================================================== #
#  CLI
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description='F1TENTH evaluation (planning + gym rollout).')
    parser.add_argument('--track', type=str, required=True, choices=TRACKS)
    parser.add_argument('--method', type=str, required=True, choices=list(METHOD_CONFIG.keys()))
    parser.add_argument('--checkpoint_root', type=str, default=CHECKPOINT_ROOT,
                        help='Directory with the checkpoints in the layout written by the training '
                             'scripts (see config/f1tenth.py checkpoint_path).')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'results'),
                        help='Results are written to <output_dir>/<method>/<track>.json.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seeds python/numpy/torch, the environment, and the start/goal '
                             'noise ([seed, track_id]).')
    parser.add_argument('--sigma', type=float, default=0.02,
                        help='Std (m) of the Gaussian noise on the start and goal positions.')
    parser.add_argument('--k_cfm', type=int, default=512,
                        help='Flow-integration steps K for fm/safefm/flowmatcher/safeflowmatcher '
                             '(Diffuser rows use the K stored in their checkpoint).')
    parser.add_argument('--goal_threshold', type=float, default=1.0,
                        help='Distance threshold (m) for reaching the goal.')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
