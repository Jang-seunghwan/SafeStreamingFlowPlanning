#!/usr/bin/env python3
"""Safe variants of the base planners — add CBF correction at each step.

Built by the build_safe_* helpers below:
    safe_diffuser       — GaussianDiffusion + shield@every denoising step
    diffuser_cg         — GaussianDiffusion + classifier guidance (∇h push)
    safe_fm             — Flow Matching Euler + shield@every step
    safe_flow_matcher   — Prediction-Correction Euler + shield@every step
    safe_sfp_offline    — SFP rollout + HOCBF-QCQP filter@every Euler step (SSF open-loop)

For safe_sfp_online (SSF closed-loop) we don't subclass — the runner
(bench/runner_one_pair.py) has the inline control loop; safety lives there as
a `cbf.hocbf_velocity_filter_phys(...)` call before /cmd_vel is published.

All safe variants share the same obstacle list (cbf.WAREHOUSE_OBSTACLES) and
the same NormalizedCBF wrapper.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ssf_gazebo.diffuser_model import (
    GaussianDiffusionPlanner,
    FlowMatchingPlanner,
    FlowMatcherPlanner,
    DiffuserPlanner,
    apply_endpoint_conditioning,
)
from ssf_gazebo.sfp_model import StreamingFlowPolicyDeterministic
from ssf_gazebo.cbf import NormalizedCBF, build_normalized_cbf


# ===========================================================================
# Diffusion safe variants
# ===========================================================================

class SafeGaussianDiffusionPlanner(GaussianDiffusionPlanner):
    """Standard DDPM sampling + position-shield correction at each step."""

    def __init__(self, model, config, cbf: NormalizedCBF, safety_method: str = 'shield'):
        super().__init__(model, config)
        self.cbf = cbf
        assert safety_method in ('shield', 'gd')
        self.safety_method = safety_method

    @torch.no_grad()
    def _apply_safety(self, x: Tensor) -> Tensor:
        if self.safety_method == 'shield':
            return self.cbf.shield_state(x)
        if self.safety_method == 'gd':
            # additive gradient push
            return x + self.cbf.classifier_guidance_state(x)
        raise ValueError(f"unknown safety_method '{self.safety_method}'")

    @torch.no_grad()
    def conditional_sample(self, condition, start_state, goal_state, batch_size):
        device = self.betas.device
        x = torch.randn(
            batch_size,
            self.config.horizon,
            self.config.transition_dim,
            device=device,
        )
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        for step in reversed(range(self.config.diffusion_steps)):
            timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            model_mean, _, model_log_variance = self.p_mean_variance(x, condition, timesteps)
            if step == 0:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            x = model_mean + torch.exp(0.5 * model_log_variance) * noise
            x = apply_endpoint_conditioning(x, start_state, goal_state)
            # >>> SAFETY <<< — apply after endpoint conditioning so endpoints
            # remain pinned at start/goal (those should not be projected away).
            x = self._apply_safety(x)
            # endpoints could have been bumped by classifier guidance — re-pin
            x = apply_endpoint_conditioning(x, start_state, goal_state)
        return x


# ===========================================================================
# CFM (Flow Matching) safe variant
# ===========================================================================

class SafeFlowMatchingPlanner(FlowMatchingPlanner):
    """CFM Euler integration + position-shield correction at each step."""

    def __init__(self, model, config, cbf: NormalizedCBF):
        super().__init__(model, config)
        self.cbf = cbf

    @torch.no_grad()
    def conditional_sample(self, condition, start_state, goal_state, batch_size):
        device = next(self.parameters()).device
        x = torch.randn(
            batch_size,
            self.config.horizon,
            self.config.transition_dim,
            device=device,
        )
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        step_count = max(1, int(self.config.diffusion_steps))
        dt = 1.0 / float(step_count)
        for step in range(step_count):
            times = torch.full((batch_size,), step * dt, device=device)
            velocity = self.model(x, condition, self._embedding_times(times))
            x = x + velocity * dt
            x = apply_endpoint_conditioning(x, start_state, goal_state)
            x = self.cbf.shield_state(x)
            x = apply_endpoint_conditioning(x, start_state, goal_state)
        return x


# ===========================================================================
# FlowMatcher (prediction + correction) safe variant
# ===========================================================================

class SafeFlowMatcherPlanner(FlowMatcherPlanner):
    """FlowMatcher P+C sampling + position-shield correction at each step."""

    def __init__(self, model, config, cbf: NormalizedCBF):
        super().__init__(model, config)
        self.cbf = cbf

    @torch.no_grad()
    def conditional_sample(self, condition, start_state, goal_state, batch_size):
        device = next(self.parameters()).device
        x = torch.randn(
            batch_size,
            self.config.horizon,
            self.config.transition_dim,
            device=device,
        )
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        # Prediction stage (one-shot)
        prediction_times = torch.zeros(batch_size, device=device)
        prediction_velocity = self.model(x, condition, self._embedding_times(prediction_times))
        x = x + prediction_velocity
        x = apply_endpoint_conditioning(x, start_state, goal_state)
        x = self.cbf.shield_state(x)
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        # Correction stage
        step_count = max(1, int(self.config.diffusion_steps))
        dt = 1.0 / float(step_count)
        alpha = float(self.config.flow_matcher_alpha)
        for step in range(step_count):
            times = torch.full((batch_size,), step * dt, device=device)
            velocity = self.model(x, condition, self._embedding_times(times))
            scale = (alpha * (1.0 - times)).reshape(batch_size, 1, 1)
            x = x + velocity * scale * dt
            x = apply_endpoint_conditioning(x, start_state, goal_state)
            x = self.cbf.shield_state(x)
            x = apply_endpoint_conditioning(x, start_state, goal_state)
        return x


# ===========================================================================
# SFP offline safe variant
# ===========================================================================

class SafeStreamingFlowPolicy(StreamingFlowPolicyDeterministic):
    """SFP rollout with HOCBF QCQP velocity filter (relative degree 2).

    At each Euler step we call cbf.hocbf_velocity_filter_phys(p, v_pos, v_vel, dt)
    which solves a QCQP for the minimum-norm v_pos correction such that the
    look-ahead position barriers (h(p_{k+1}), h(p_{k+2})) satisfy a class-K
    HOCBF inequality.

    This matches the formulation upstream uses in sfpd.py / cbf.py:apply_discrete
    — the velocity field's v_vel term plays the role of "acceleration" in the
    discrete dynamics, making the constraint on position relative degree 2.

    Normalization handled inside: we pull v_pos / v_vel out in PHYSICAL units,
    run the QCQP, then re-normalize.
    """

    def __init__(self, velocity_net, config, cbf: NormalizedCBF,
                 hocbf_kp: float = 0.5, hocbf_kv: float = 0.3):
        super().__init__(velocity_net, config)
        self.cbf = cbf
        self.hocbf_kp = float(hocbf_kp)
        self.hocbf_kv = float(hocbf_kv)

    @torch.inference_mode()
    def rollout(self, start_state: Tensor, goal_state: Tensor, pred_horizon: int) -> Tensor:
        if pred_horizon < 2:
            raise ValueError('pred_horizon must be >= 2')
        device = next(self.velocity_net.parameters()).device
        start = start_state.reshape(-1).to(device)
        goal = goal_state.reshape(-1).to(device).clone()
        goal[self.config.pos_dim:] = 0.0
        ncond = torch.stack([start, goal], dim=0)
        cond_flat = ncond.unsqueeze(0).flatten(start_dim=1)

        t_span = torch.linspace(0, 1.0, pred_horizon, device=device, dtype=start.dtype)
        delta_ts = torch.diff(t_span)

        cur = start.unsqueeze(0).unsqueeze(0)       # (1, 1, S)
        traj = [cur.squeeze(0).squeeze(0).clone()]
        pos_dim = self.config.pos_dim

        # Pull normalizer scales for de/re-normalization at the safety boundary
        norm = self.cbf.normalizer
        mean = torch.as_tensor(norm.mean, device=device, dtype=start.dtype)
        std = torch.as_tensor(norm.std, device=device, dtype=start.dtype)
        mean_xy = mean[:pos_dim]; std_xy = std[:pos_dim]
        mean_v = mean[pos_dim:]; std_v = std[pos_dim:]

        for i in range(pred_horizon - 1):
            t = t_span[i].unsqueeze(0)
            v = self.velocity_net(sample=cur, timestep=t, global_cond=cond_flat)
            v = v.squeeze(0).squeeze(0)
            cur_flat = cur.squeeze(0).squeeze(0)
            dt_i = float(delta_ts[i].item())

            v_pos_n = v[:pos_dim]                     # normalized
            v_vel_n = v[pos_dim:]
            cur_pos_n = cur_flat[:pos_dim]
            cur_vel_n = cur_flat[pos_dim:]

            # --- denormalize to physical for QCQP ---
            # Position normalization: x_n = (x - mean) / std → δp_n · std = δp_phys
            # Velocity normalization same form, so v_pos_phys = v_pos_n · std_xy (a rate)
            # and v_vel_phys = v_vel_n · std_v.
            cur_pos_phys = cur_pos_n * std_xy + mean_xy
            v_pos_phys = v_pos_n * std_xy        # rate-of-position-change in physical units
            v_vel_phys = v_vel_n * std_v          # rate-of-velocity in physical units

            # --- HOCBF QCQP (returns safe v_pos in physical units) ---
            v_pos_safe_phys = self.cbf.hocbf_velocity_filter_phys(
                cur_pos_phys, v_pos_phys, v_vel_phys, dt=dt_i,
                kp=self.hocbf_kp, kv=self.hocbf_kv,
            ).to(device=device, dtype=start.dtype)

            # --- re-normalize and integrate ---
            v_pos_safe_n = v_pos_safe_phys / std_xy
            next_pos = cur_pos_n + v_pos_safe_n * dt_i
            next_vel = cur_vel_n + v_vel_n * dt_i
            next_state = torch.cat([next_pos, next_vel], dim=-1)
            cur = next_state.unsqueeze(0).unsqueeze(0)
            traj.append(next_state.clone())
        return torch.stack(traj, dim=0)


# ===========================================================================
# Convenience builders — used by runner_one_pair.py
# ===========================================================================

def build_safe_diffuser(base: DiffuserPlanner, safety_method: str = 'shield') -> DiffuserPlanner:
    """Rebuild a DiffuserPlanner with safety applied. Returns a new DiffuserPlanner."""
    cbf = build_normalized_cbf(base.normalizer)
    safe = SafeGaussianDiffusionPlanner(
        base.diffusion.model, base.diffusion.config, cbf, safety_method=safety_method
    )
    safe.load_state_dict(base.diffusion.state_dict())
    safe.to(base.device)
    safe.eval()
    new_planner = DiffuserPlanner.__new__(DiffuserPlanner)
    new_planner.diffusion = safe
    new_planner.normalizer = base.normalizer
    new_planner.device = base.device
    return new_planner


def build_safe_fm(base: DiffuserPlanner) -> DiffuserPlanner:
    cbf = build_normalized_cbf(base.normalizer)
    safe = SafeFlowMatchingPlanner(base.diffusion.model, base.diffusion.config, cbf)
    safe.load_state_dict(base.diffusion.state_dict())
    safe.to(base.device)
    safe.eval()
    new_planner = DiffuserPlanner.__new__(DiffuserPlanner)
    new_planner.diffusion = safe
    new_planner.normalizer = base.normalizer
    new_planner.device = base.device
    return new_planner


def build_safe_flow_matcher(base: DiffuserPlanner) -> DiffuserPlanner:
    cbf = build_normalized_cbf(base.normalizer)
    safe = SafeFlowMatcherPlanner(base.diffusion.model, base.diffusion.config, cbf)
    safe.load_state_dict(base.diffusion.state_dict())
    safe.to(base.device)
    safe.eval()
    new_planner = DiffuserPlanner.__new__(DiffuserPlanner)
    new_planner.diffusion = safe
    new_planner.normalizer = base.normalizer
    new_planner.device = base.device
    return new_planner


def build_safe_sfp(base_model, normalizer, config) -> SafeStreamingFlowPolicy:
    """Returns a SafeStreamingFlowPolicy with the base model's weights."""
    cbf = build_normalized_cbf(normalizer)
    safe = SafeStreamingFlowPolicy(base_model.velocity_net, config, cbf)
    safe.eval()
    return safe
