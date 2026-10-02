#!/usr/bin/env python3
"""Run ONE planner on ONE (start, goal) pair and save a per-trial result JSON.

Assumes the full Gazebo + map_server + AMCL + Nav2 stack is already running
(see bench/run_table5.sh). bench/run_bench.py loops this script over pairs and
planners.

Usage:
    python3 -m bench.runner_one_pair --pair-idx 0 --planner safe_sfp_online
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from rclpy.node import Node

from bench.util import (
    CmdVelLog,
    OdomIntegrator,
    build_path_msg,
    clear_costmaps,
    dist_xy,
    publish_initialpose,
    send_follow_path,
    stop_robot,
    teleport_robot,
)
from bench.metrics import acceleration_smoothness, min_barrier
from bench.seeding import seed_everything

# Table 5 rows (Warehouse navigation).
PLANNERS = [
    # classical, tracked by Nav2 FollowPath (Regulated Pure Pursuit)
    'rrt_star', 'a_star',
    # offline (trajectory-level) planners, PD-tracked on /cmd_vel
    'diffuser_pd', 'diffuser_cg_pd', 'safe_diffuser_pd',
    'cfm_pd', 'safe_fm_pd', 'flow_matcher_pd', 'safe_flow_matcher_pd',
    'sfp_offline_pd', 'safe_sfp_offline_pd',
    # closed-loop streaming planners (one velocity-field query per control tick)
    'sfp_online', 'safe_sfp_online',
]
# Planners that take an offline plan and PD-track it via /cmd_vel.
PD_TRACK_PLANNERS = (
    'diffuser_pd', 'cfm_pd', 'flow_matcher_pd', 'sfp_offline_pd',
    'safe_diffuser_pd', 'diffuser_cg_pd', 'safe_fm_pd',
    'safe_flow_matcher_pd', 'safe_sfp_offline_pd',
)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PAIRS = os.path.join(REPO, 'bench', 'test_pairs.json')
DEFAULT_MODELS = os.path.join(REPO, 'models_h512')
DEFAULT_OUT = os.path.join(REPO, 'results', 'bench_results')
# Calls to rclpy.spin_once per control tick: drains the callback queue so the
# controller always acts on the latest /odom sample.
_DRAIN_SPINS = 65


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pair-idx', type=int, required=True)
    p.add_argument('--planner', choices=PLANNERS, required=True)
    p.add_argument('--pairs-json', default=DEFAULT_PAIRS)
    p.add_argument('--models-dir', default=DEFAULT_MODELS,
                   help='Directory with diffuser_planner_best.pt, cfm_planner_best.pt, sfp_planner_best.pt.')
    p.add_argument('--seed', type=int, default=42,
                   help='The trial is seeded with seed + pair_idx (planner sampling, RRT*).')
    p.add_argument('--world', default='warehouse')
    p.add_argument('--robot-name', default='robot')
    p.add_argument('--device', default='cuda')
    p.add_argument('--timeout-sec', type=float, default=90.0)
    p.add_argument('--goal-tolerance', type=float, default=0.25)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--amcl-wait-sec', type=float, default=5.0,
                   help='Wait for amcl to converge after publishing initialpose.')
    p.add_argument('--control-rate-hz', type=float, default=20.0,
                   help='Rate of the PD / streaming control loop (learned planners).')
    p.add_argument('--max-linear', type=float, default=0.5,
                   help='cmd_vel speed clamp (m/s); also the admissible set U of the '
                        'safe_sfp_online HOCBF-QCQP filter.')
    p.add_argument('--sfp-kp', type=float, default=2.0,
                   help='PD: P-gain on position error.')
    p.add_argument('--sfp-kd', type=float, default=0.2,
                   help='PD: D-gain on velocity error.')
    return p.parse_args()


def load_pair(pairs_json: str, pair_idx: int):
    with open(pairs_json) as f:
        data = json.load(f)
    if pair_idx < 0 or pair_idx >= len(data['pairs']):
        raise IndexError(f'pair_idx {pair_idx} out of range [0, {len(data["pairs"])})')
    return data['pairs'][pair_idx]


def yaw_from_start_goal(start, goal):
    return math.atan2(goal[1] - start[1], goal[0] - start[0])


def _drain_callbacks(node: Node):
    for _ in range(_DRAIN_SPINS):
        rclpy.spin_once(node, timeout_sec=0.0)


# ---------------------------------------------------------------------------
# Per-planner inference (offline)
# ---------------------------------------------------------------------------

def plan_offline_timed(planner_name, start_state, goal_xy, models_dir, device):
    """Wrap plan_offline with a wall-clock timer. Returns (plan, t_opt_sec)."""
    t0 = time.perf_counter()
    plan = plan_offline(planner_name, start_state, goal_xy, models_dir, device)
    return plan, time.perf_counter() - t0


def plan_offline(planner_name, start_state, goal_xy, models_dir, device):
    """Generate a (H, 4) numpy trajectory for the offline planners (incl. safe variants)."""
    # PD-controller variants share the offline plan with their base planner;
    # strip the suffix so we don't duplicate the generation code.
    if planner_name.endswith('_pd'):
        planner_name = planner_name[:-3]

    from ssf_gazebo.diffuser_model import DiffuserPlanner
    from ssf_gazebo.sfp_model import load_sfp_checkpoint

    # ---- Base (no safety) ---------------------------------------------------
    if planner_name == 'diffuser':
        p = DiffuserPlanner.load(os.path.join(models_dir, 'diffuser_planner_best.pt'), device=device)
        return p.sample_plan(start_state=list(start_state), goal_xy=list(goal_xy), batch_size=1)[0]
    if planner_name == 'cfm':
        p = DiffuserPlanner.load(os.path.join(models_dir, 'cfm_planner_best.pt'), device=device)
        return p.sample_plan(start_state=list(start_state), goal_xy=list(goal_xy), batch_size=1)[0]
    if planner_name == 'flow_matcher':
        p = DiffuserPlanner.load(os.path.join(models_dir, 'cfm_planner_best.pt'),
                                 device=device, planner_type_override='flow_matcher')
        return p.sample_plan(start_state=list(start_state), goal_xy=list(goal_xy), batch_size=1)[0]
    if planner_name == 'sfp_offline':
        model, norm, cfg = load_sfp_checkpoint(os.path.join(models_dir, 'sfp_planner_best.pt'), device=device)
        model.eval()
        start_n = torch.from_numpy(norm.normalize_state(list(start_state))).float().to(device)
        goal_n  = torch.from_numpy(norm.normalize_state([goal_xy[0], goal_xy[1], 0.0, 0.0])).float().to(device)
        with torch.no_grad():
            traj_n = model.rollout(start_n, goal_n, pred_horizon=cfg.horizon)
        return norm.denormalize_trajectory(traj_n.cpu().numpy())

    # ---- Safe variants ------------------------------------------------------
    from ssf_gazebo.safe_planners import (
        build_safe_diffuser, build_safe_fm, build_safe_flow_matcher, build_safe_sfp,
    )

    if planner_name == 'safe_diffuser':
        base = DiffuserPlanner.load(os.path.join(models_dir, 'diffuser_planner_best.pt'), device=device)
        p = build_safe_diffuser(base, safety_method='shield')
        return p.sample_plan(start_state=list(start_state), goal_xy=list(goal_xy), batch_size=1)[0]
    if planner_name == 'diffuser_cg':
        base = DiffuserPlanner.load(os.path.join(models_dir, 'diffuser_planner_best.pt'), device=device)
        p = build_safe_diffuser(base, safety_method='gd')
        return p.sample_plan(start_state=list(start_state), goal_xy=list(goal_xy), batch_size=1)[0]
    if planner_name == 'safe_fm':
        base = DiffuserPlanner.load(os.path.join(models_dir, 'cfm_planner_best.pt'), device=device)
        p = build_safe_fm(base)
        return p.sample_plan(start_state=list(start_state), goal_xy=list(goal_xy), batch_size=1)[0]
    if planner_name == 'safe_flow_matcher':
        base = DiffuserPlanner.load(os.path.join(models_dir, 'cfm_planner_best.pt'),
                                    device=device, planner_type_override='flow_matcher')
        p = build_safe_flow_matcher(base)
        return p.sample_plan(start_state=list(start_state), goal_xy=list(goal_xy), batch_size=1)[0]
    if planner_name == 'safe_sfp_offline':
        base_model, norm, cfg = load_sfp_checkpoint(os.path.join(models_dir, 'sfp_planner_best.pt'), device=device)
        base_model.eval()
        safe_model = build_safe_sfp(base_model, norm, cfg).to(device)
        start_n = torch.from_numpy(norm.normalize_state(list(start_state))).float().to(device)
        goal_n  = torch.from_numpy(norm.normalize_state([goal_xy[0], goal_xy[1], 0.0, 0.0])).float().to(device)
        with torch.no_grad():
            traj_n = safe_model.rollout(start_n, goal_n, pred_horizon=cfg.horizon)
        return norm.denormalize_trajectory(traj_n.cpu().numpy())

    raise ValueError(f'unknown offline planner: {planner_name}')


# ---------------------------------------------------------------------------
# sfp_online inline driver (controller bypass, /cmd_vel direct @ 20Hz)
# ---------------------------------------------------------------------------

def run_sfp_online(node: Node, start, goal, odom: OdomIntegrator, args, extra: dict):
    """SFP online streaming: each control cycle predicts velocity from current
    state and publishes /cmd_vel directly. Bypasses Nav2 controller.

    When args.planner == 'safe_sfp_online', the PD-computed world-frame velocity
    additionally goes through the relative-degree-2 HOCBF-QCQP filter (cbf.py)
    with admissible set U = {||v|| <= max_linear}.

    The loop is timed on the wall clock (one tick = 1/control_rate_hz s); run
    it in the real-time world (warehouse.sdf, real_time_factor 1) so one tick
    also equals 1/control_rate_hz s of simulated time, as in the training data.
    """
    from ssf_gazebo.sfp_model import load_sfp_checkpoint
    model, norm, cfg = load_sfp_checkpoint(
        os.path.join(args.models_dir, 'sfp_planner_best.pt'), device=args.device)
    model.eval()
    horizon = int(cfg.horizon)

    # Optional CBF — only built when running the safe variant.
    cbf = None
    if args.planner == 'safe_sfp_online':
        from ssf_gazebo.cbf import build_normalized_cbf
        cbf = build_normalized_cbf(
            normalizer=norm,
            device=torch.device(args.device),
            dtype=torch.float32,
        )

    cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
    period = 1.0 / args.control_rate_hz

    start_xy = (float(start[0]), float(start[1]))
    goal_xy = (float(goal[0]), float(goal[1]))
    t_begin = time.monotonic()
    step = 0
    success = False
    horizon_exhausted = False
    model_time_sum = 0.0   # cumulative SFP forward time

    # Stuck detection (post-horizon only): if PD-goal-pull can't move the robot
    # > 0.20 m within 10 s, bail out. During SFP horizon we don't check.
    stuck_threshold_m = 0.20
    stuck_window_s = 10.0
    last_progress_pos = (start_xy[0], start_xy[1])
    last_progress_time = time.monotonic()
    stuck = False

    next_tick = time.monotonic()
    while time.monotonic() - t_begin < args.timeout_sec:
        # Sleep until next tick to enforce real control rate.
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        next_tick += period
        _drain_callbacks(node)
        if not odom.have_data:
            continue
        cur = odom.xy
        if cur is None:
            continue
        if dist_xy(cur, goal_xy) < args.goal_tolerance:
            success = True
            break
        # stuck check ONLY after horizon is exhausted (i.e. SFP rollout is
        # done and we're in the goal-direct PD pull phase). During the SFP
        # phase the robot may pause briefly while it follows curvature; that
        # is not a real stuck.
        if horizon_exhausted:
            if dist_xy(cur, last_progress_pos) > stuck_threshold_m:
                last_progress_pos = cur
                last_progress_time = time.monotonic()
            elif time.monotonic() - last_progress_time > stuck_window_s:
                stuck = True
                node.get_logger().info(
                    f'sfp_online stuck — no progress > {stuck_threshold_m}m in {stuck_window_s}s (post-horizon), aborting'
                )
                break
        else:
            # Keep the stuck timer pinned to "now" while SFP is still rolling out.
            last_progress_pos = cur
            last_progress_time = time.monotonic()
        if step >= horizon - 1 and not horizon_exhausted:
            horizon_exhausted = True   # mark, but keep going — clamp t to last valid step
            node.get_logger().info(f'sfp_online horizon exhausted at step {step}; holding t at horizon-1 with PD')

        # Build normalized state — convert body-frame odom velocity to world frame
        # since training rolled out v_x, v_y in world frame.
        yaw_now = odom.yaw
        cy_, sy_ = math.cos(yaw_now), math.sin(yaw_now)
        vx_w = cy_ * odom.log.vx[-1] - sy_ * odom.log.vy[-1]
        vy_w = sy_ * odom.log.vx[-1] + cy_ * odom.log.vy[-1]
        obs = np.array([cur[0], cur[1], vx_w, vy_w], dtype=np.float32)
        start4 = np.array([start_xy[0], start_xy[1], 0.0, 0.0], dtype=np.float32)
        goal4 = np.array([goal_xy[0],  goal_xy[1],  0.0, 0.0], dtype=np.float32)
        obs_n = norm.normalize_state(obs.tolist())
        start_n = norm.normalize_state(start4.tolist())
        goal_n = norm.normalize_state(goal4.tolist())
        cond_flat = np.concatenate([start_n, goal_n]).astype(np.float32)

        # After horizon-1, t is clamped to (horizon-1)/horizon so the velocity field
        # keeps producing "trajectory end" outputs (≈0 velocity), and the PD's
        # position term keeps pulling the robot toward the predicted goal.
        t_step_clamped = min(step, horizon - 1)
        t_now = float(t_step_clamped) / float(horizon)
        with torch.no_grad():
            x_in = torch.from_numpy(obs_n).float().to(args.device).view(1, 1, -1)
            c_t  = torch.from_numpy(cond_flat).float().to(args.device).view(1, -1)
            t_t  = torch.tensor([t_now], dtype=torch.float32, device=args.device)
            _t0 = time.perf_counter()
            v_norm = model.velocity_net(sample=x_in, timestep=t_t, global_cond=c_t)[0, 0].cpu().numpy()
            model_time_sum += time.perf_counter() - _t0

        # Decide PD target. Within SFP horizon: use SFP-predicted next state.
        # After horizon-1 the model output saturates (target ≈ current pos), so
        # fall back to pulling the robot directly toward the user-provided goal.
        if horizon_exhausted:
            target_pos_world = np.array(goal_xy, dtype=np.float32)
            target_vel_world = np.zeros(2, dtype=np.float32)
        else:
            x_next_n = obs_n + v_norm / float(horizon)
            x_next_phys = norm.denormalize_trajectory(x_next_n[None, :])[0]
            target_pos_world = x_next_phys[:2]
            target_vel_world = x_next_phys[2:]

        # PD on world frame: feedforward target_vel + Kp*pos_err + Kd*vel_err
        pos_err = target_pos_world - obs[:2]
        vel_err = target_vel_world - obs[2:]
        v_world = target_vel_world + args.sfp_kp * pos_err + args.sfp_kd * vel_err

        # >>> SAFETY <<< — for safe_sfp_online, run the PD-computed v_world
        # through the HOCBF QCQP velocity filter (RD=2, treats SFP's v_vel as
        # acceleration). v_norm = [v_pos, v_vel] in normalized world frame.
        if cbf is not None:
            dt_pred = 1.0 / args.control_rate_hz
            # v_vel is a rate per unit of normalised flow time (the model steps
            # x + v/H per tick): denormalise with the velocity std, then divide
            # by H * dt to obtain a physical acceleration (m/s^2).
            v_vel_phys = v_norm[2:] * norm.std[2:]
            v_vel_phys = v_vel_phys / (float(horizon) * dt_pred)
            v_safe = cbf.hocbf_velocity_filter_phys(
                p_phys=np.array(cur, dtype=np.float32),
                v_pos_phys=v_world.astype(np.float32),
                v_vel_phys=v_vel_phys.astype(np.float32),
                dt=dt_pred,
                max_speed=args.max_linear,
            )
            v_world = v_safe.detach().cpu().numpy().astype(np.float32)

        # World → body frame (yaw from latest odom).
        yaw = odom.yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        vx_body =  cy * float(v_world[0]) + sy * float(v_world[1])
        vy_body = -sy * float(v_world[0]) + cy * float(v_world[1])

        speed = math.hypot(vx_body, vy_body)
        if speed > args.max_linear:
            vx_body *= args.max_linear / speed
            vy_body *= args.max_linear / speed

        cmd = Twist()
        cmd.linear.x = vx_body
        cmd.linear.y = vy_body
        cmd_pub.publish(cmd)
        step += 1

    stop_robot(node)
    elapsed = time.monotonic() - t_begin
    extra['t_opt_sec'] = float(model_time_sum)
    return success, elapsed, {
        'horizon_exhausted': horizon_exhausted, 'steps': step, 'stuck': stuck,
    }


# ---------------------------------------------------------------------------
# Offline-plan PD tracker (bypass Nav2 controller, drive /cmd_vel directly)
# ---------------------------------------------------------------------------

def run_offline_pd(node: Node, pair, plan: np.ndarray, odom: OdomIntegrator, args, extra: dict):
    """Track an offline-generated (H, ≥2) plan via a world-frame PD law on /cmd_vel.

    Mirrors run_sfp_online's controller: PD targets a waypoint that advances one
    tick per control period (matching the training dt ≈ 1/20 s). After horizon
    exhaustion, target falls back to the user-provided goal so the robot can
    settle in. World-frame Vx/Vy → body-frame via current yaw, then publish.
    """
    cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
    period = 1.0 / args.control_rate_hz

    start_xy = (float(pair['start'][0]), float(pair['start'][1]))
    goal_xy  = (float(pair['goal'][0]),  float(pair['goal'][1]))

    N = int(plan.shape[0])
    has_vel = plan.shape[1] >= 4

    t_begin = time.monotonic()
    step = 0
    success = False
    horizon_exhausted = False

    # Stuck detection (post-horizon only): identical to run_sfp_online.
    stuck_threshold_m = 0.20
    stuck_window_s = 10.0
    last_progress_pos = start_xy
    last_progress_time = time.monotonic()
    stuck = False

    next_tick = time.monotonic()
    while time.monotonic() - t_begin < args.timeout_sec:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        next_tick += period
        _drain_callbacks(node)
        if not odom.have_data:
            continue
        cur = odom.xy
        if cur is None:
            continue
        if dist_xy(cur, goal_xy) < args.goal_tolerance:
            success = True
            break

        if horizon_exhausted:
            if dist_xy(cur, last_progress_pos) > stuck_threshold_m:
                last_progress_pos = cur
                last_progress_time = time.monotonic()
            elif time.monotonic() - last_progress_time > stuck_window_s:
                stuck = True
                node.get_logger().info(
                    f'{args.planner} stuck — no progress > {stuck_threshold_m}m '
                    f'in {stuck_window_s}s (post-horizon), aborting'
                )
                break
        else:
            last_progress_pos = cur
            last_progress_time = time.monotonic()

        # Target = plan[step] until horizon exhausted, then user goal.
        if step >= N:
            if not horizon_exhausted:
                horizon_exhausted = True
                node.get_logger().info(
                    f'{args.planner} horizon exhausted at step {step}; holding target at goal'
                )
            target_pos_world = np.array(goal_xy, dtype=np.float32)
            target_vel_world = np.zeros(2, dtype=np.float32)
        else:
            target_pos_world = plan[step, :2].astype(np.float32)
            target_vel_world = (plan[step, 2:4].astype(np.float32)
                                if has_vel else np.zeros(2, dtype=np.float32))

        # Body-frame odom velocity → world-frame (training was in world frame).
        yaw_now = odom.yaw
        cy_, sy_ = math.cos(yaw_now), math.sin(yaw_now)
        vx_w = cy_ * odom.log.vx[-1] - sy_ * odom.log.vy[-1]
        vy_w = sy_ * odom.log.vx[-1] + cy_ * odom.log.vy[-1]

        pos_err = target_pos_world - np.array(cur, dtype=np.float32)
        vel_err = target_vel_world - np.array([vx_w, vy_w], dtype=np.float32)
        v_world = target_vel_world + args.sfp_kp * pos_err + args.sfp_kd * vel_err

        # World → body
        vx_body =  cy_ * float(v_world[0]) + sy_ * float(v_world[1])
        vy_body = -sy_ * float(v_world[0]) + cy_ * float(v_world[1])

        speed = math.hypot(vx_body, vy_body)
        if speed > args.max_linear:
            vx_body *= args.max_linear / speed
            vy_body *= args.max_linear / speed

        cmd = Twist()
        cmd.linear.x = vx_body
        cmd.linear.y = vy_body
        cmd_pub.publish(cmd)
        step += 1

    stop_robot(node)
    elapsed = time.monotonic() - t_begin
    return success, elapsed, {
        'horizon_exhausted': horizon_exhausted, 'steps': step, 'stuck': stuck,
    }


# ---------------------------------------------------------------------------
# Main trial driver
# ---------------------------------------------------------------------------

def run_trial(args, node: Node, pair, odom: OdomIntegrator, cmdvel: 'CmdVelLog'):
    start = pair['start']                      # [x, y, vx, vy]
    goal = pair['goal']                        # [x, y]
    extra = {}
    seed_everything(args.seed + args.pair_idx)

    # 1) plan first (offline planners) so we can teleport facing the path heading
    plan = None
    online_planners = ('sfp_online', 'safe_sfp_online')
    classical_planners = ('a_star', 'rrt_star')

    if args.planner in online_planners:
        yaw0 = yaw_from_start_goal(start, goal)
    elif args.planner in classical_planners:
        # ---- A* / RRT* — plan first, no model, returns (N, 2) xy ----
        from ssf_gazebo.classical_planners import plan_a_star, plan_rrt_star
        map_yaml = os.path.join(
            REPO, 'src', 'ssf_gazebo', 'maps', f'ssf_map_{args.world}.yaml',
        )
        t_opt_start = time.perf_counter()
        if args.planner == 'a_star':
            xy = plan_a_star((start[0], start[1]), (goal[0], goal[1]),
                             map_yaml=map_yaml, robot_radius_m=0.25)
        else:
            xy = plan_rrt_star((start[0], start[1]), (goal[0], goal[1]), map_yaml=map_yaml,
                               time_budget_sec=5.0, range_m=0.5, robot_radius_m=0.25)
        t_opt = time.perf_counter() - t_opt_start
        extra['t_opt_sec'] = float(t_opt)
        if xy is None or len(xy) < 2:
            node.get_logger().error(f'{args.planner} produced no plan')
            return False, 0.0, {**extra, 'plan_failed': True}
        # Pad to (N, 4) — velocity columns unused by Nav2 follow_path
        plan = np.concatenate([xy, np.zeros((len(xy), 2), dtype=xy.dtype)], axis=1)
        extra['plan_path_len_m'] = float(np.linalg.norm(np.diff(plan[:, :2], axis=0), axis=1).sum())
        extra['plan_goal_err_m'] = float(np.linalg.norm(plan[-1, :2] - np.asarray(goal)))
        N = min(5, len(plan) - 1)
        dx = float(plan[N, 0] - plan[0, 0]); dy = float(plan[N, 1] - plan[0, 1])
        if abs(dx) + abs(dy) > 1e-6:
            yaw0 = math.atan2(dy, dx)
        else:
            yaw0 = yaw_from_start_goal(start, goal)
    else:
        plan, t_opt = plan_offline_timed(args.planner, start, goal, args.models_dir, args.device)
        extra['t_opt_sec'] = float(t_opt)
        extra['plan_path_len_m'] = float(np.linalg.norm(np.diff(plan[:, :2], axis=0), axis=1).sum())
        extra['plan_goal_err_m'] = float(np.linalg.norm(plan[-1, :2] - np.asarray(goal)))
        N = min(5, len(plan) - 1)
        dx = float(plan[N, 0] - plan[0, 0])
        dy = float(plan[N, 1] - plan[0, 1])
        if abs(dx) + abs(dy) > 1e-6:
            yaw0 = math.atan2(dy, dx)
        else:
            yaw0 = yaw_from_start_goal(start, goal)
    extra['teleport_yaw_deg'] = float(math.degrees(yaw0))

    # 2) teleport
    ok = teleport_robot(args.world, args.robot_name, start[0], start[1], z=0.05, yaw=yaw0)
    if not ok:
        node.get_logger().error(f'teleport failed for pair {args.pair_idx}')
        return False, 0.0, {'teleport_failed': True, **extra}
    time.sleep(0.5)   # let physics settle

    # 3) reset amcl
    publish_initialpose(node, start[0], start[1], yaw0)
    time.sleep(args.amcl_wait_sec)

    # 4) clear costmaps
    clear_costmaps(node)

    # 5) snapshot odom + cmd_vel log for this trial
    odom.reset()
    cmdvel.reset()
    rclpy.spin_once(node, timeout_sec=0.2)     # wait for at least one odom sample

    # 6) planner-specific drive
    if args.planner in online_planners:
        success, elapsed, extra_run = run_sfp_online(node, start, goal, odom, args, extra)
        extra.update(extra_run)
    elif args.planner in PD_TRACK_PLANNERS:
        success, elapsed, extra_run = run_offline_pd(node, pair, plan, odom, args, extra)
        extra.update(extra_run)
    else:
        # A* / RRT*: hand the geometric path to the Nav2 FollowPath controller.
        path_msg = build_path_msg(plan[:, :2], node, frame_id='map')
        goal_handle, result_fut = send_follow_path(node, path_msg)
        if goal_handle is None:
            return False, 0.0, {**extra, 'follow_path_rejected': True}

        t_begin = time.monotonic()
        success = False
        goal_xy = (float(goal[0]), float(goal[1]))
        while time.monotonic() - t_begin < args.timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            if odom.xy is not None and dist_xy(odom.xy, goal_xy) < args.goal_tolerance:
                success = True
                break
        elapsed = time.monotonic() - t_begin
        try:
            goal_handle.cancel_goal_async()
        except Exception:
            pass
        stop_robot(node)

    return success, elapsed, extra


def save_result(args, pair, success, elapsed, odom: OdomIntegrator,
                cmdvel: 'CmdVelLog', extra: dict):
    os.makedirs(args.out_dir, exist_ok=True)
    # Decimated odom path (every 4th sample) to keep JSON size manageable.
    odom_xy = list(zip(odom.log.x[::4], odom.log.y[::4]))
    if odom.log.x and (len(odom.log.x) - 1) % 4 != 0:
        odom_xy.append((odom.log.x[-1], odom.log.y[-1]))

    # Compute Sm. (acceleration_smoothness) from the cmd_vel stream.
    sm = acceleration_smoothness(cmdvel.vx, cmdvel.vy, cmdvel.times)
    # Execution-level safety: min barrier value (kappa = 0, nominal circles)
    # over EVERY recorded /odom sample of the trial.
    min_h = min_barrier(odom.log.x, odom.log.y)

    result = {
        'pair_idx': args.pair_idx,
        'planner': args.planner,
        'start': pair['start'],
        'goal': pair['goal'],
        'dist_m_straight': pair.get('dist_m'),
        'seed': args.seed + args.pair_idx,
        # headline metrics
        'success': bool(success),                                  # goal reached
        'sm_acc': float(sm),                                       # Sm.
        't_opt_sec': float(extra.get('t_opt_sec', 0.0)),
        'elapsed_sec': float(elapsed),
        'min_h': min_h,                                            # min_k min_i h_i(x_k)
        'violation': bool(min_h is not None and min_h < 0.0),      # any sample inside a circle
        # auxiliary
        'odom_distance_m': float(odom.log.distance_m),
        'final_xy': list(odom.xy) if odom.xy else None,
        'odom_path_xy': odom_xy,
        'cmd_vel_samples': len(cmdvel.times),
        'extra': extra,
    }
    out = os.path.join(args.out_dir, f'{args.planner}_{args.pair_idx:03d}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    # Compact one-line log
    print(f'[trial] pair={args.pair_idx:03d} planner={args.planner:>13} '
          f'success={success} elapsed={elapsed:.2f}s '
          f'driven={odom.log.distance_m:.2f}m  -> {out}')


def main():
    args = parse_args()
    pair = load_pair(args.pairs_json, args.pair_idx)

    rclpy.init()
    node = Node('runner_one_pair')
    odom = OdomIntegrator(node, '/odom')
    cmdvel = CmdVelLog(node, '/cmd_vel')

    try:
        success, elapsed, extra = run_trial(args, node, pair, odom, cmdvel)
        save_result(args, pair, success, elapsed, odom, cmdvel, extra)
    finally:
        stop_robot(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
