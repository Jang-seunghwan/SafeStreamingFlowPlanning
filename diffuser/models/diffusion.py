import time

import torch
from torch import nn
from qpth.qp import QPFunction, QPSolvers

from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    WeightedL2,
)


class GaussianDiffusion(nn.Module):
    """Diffuser (x0-prediction, clipped) with optional safety during sampling:
    classifier guidance ('gd', Diffuser+CG) or relaxed invariance QP ('invariance', SafeDiffuser)."""

    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 action_weight=1.0):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        # All schedule buffers are part of the checkpoints' state_dicts.
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Loss (uniform weights; the first action is weighted by action_weight)
        loss_weights = torch.ones(self.horizon, self.transition_dim, dtype=torch.float32)
        loss_weights[0, :self.action_dim] = action_weight
        self.loss_fn = WeightedL2(loss_weights)

        # Safety
        self.safety_enabled = False
        self.safety_method = 'none'  # 'none' | 'gd' | 'invariance'
        self.cbf = None

    # ── forward diffusion ──
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.model(x, cond, t)
        x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """Returns the sample x0 and the average safety-layer time per denoising step."""
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        safety_time = 0

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            if self.safety_enabled and self.cbf is not None:
                safety_start = time.time()
                if self.safety_method == 'gd':
                    x = self.GD(x)
                elif self.safety_method == 'invariance':
                    x = self.invariance_relax(x, x, t=i / self.n_timesteps)
                safety_time += time.time() - safety_start
                x = apply_conditioning(x, cond, self.action_dim)

        return x, safety_time / self.n_timesteps

    # ── Safety methods ──

    @torch.no_grad()
    def invariance_relax(self, x, xp1, t):
        """
        Relaxed Safe Diffuser (ReS-diffuser) — per-obstacle slack variables.

        Each obstacle has its own slack variable in the QP:
            QP dim = 2 + K  (2 position corrections + K slacks)

        Args:
            x:   [1, H, D] current denoised state
            xp1: [1, H, D] proposed next state
            t:   normalized timestep (1.0 → 0.0 during reversed denoising)
        """
        x = x.squeeze(0)      # [H, D]
        xp1 = xp1.squeeze(0)  # [H, D]

        nBatch = x.shape[0]   # H (horizon = number of waypoints)
        ref = xp1 - x

        obstacles = self.cbf.obstacles
        K = len(obstacles)
        offset = self.cbf.center_offset
        robust_term = self.cbf.robust_term
        px = self.cbf.pos_x_idx
        py = self.cbf.pos_y_idx
        pi = px + self.action_dim  # trajectory index for pos_x
        pj = py + self.action_dim  # trajectory index for pos_y

        # Relax on for most of denoising, off for last ~10 raw steps
        if t >= 10.0 / self.n_timesteps:
            sign = 100.0
        else:
            sign = 0.0

        G_list = []
        h_list = []

        for i, obs in enumerate(obstacles):
            cx, cy = obs['center']
            n = obs['order']
            rx = obs.get('radius_x', obs.get('radius', 1.0))
            ry = obs.get('radius_y', obs.get('radius', 1.0))

            # Normalize radius and center to data space
            xr = 2 * rx / (self.cbf.norm_maxs[px] - self.cbf.norm_mins[px])
            yr = 2 * ry / (self.cbf.norm_maxs[py] - self.cbf.norm_mins[py])
            off_x = 2 * (cx + offset - self.cbf.norm_mins[px]) / (
                self.cbf.norm_maxs[px] - self.cbf.norm_mins[px]) - 1
            off_y = 2 * (cy + offset - self.cbf.norm_mins[py]) / (
                self.cbf.norm_maxs[py] - self.cbf.norm_mins[py]) - 1

            dx = (x[:, pi:pi+1] - off_x) / xr
            dy = (x[:, pj:pj+1] - off_y) / yr

            # CBF value: b = dx^n + dy^n - 1 - robust_term
            b_i = dx**n + dy**n - 1 - robust_term

            # Lie derivative (gradient of barrier w.r.t. normalized position)
            Lgbu_x = n * dx**(n - 1) / xr
            Lgbu_y = n * dy**(n - 1) / yr

            # Constraint row: [-Lgbu_x, -Lgbu_y, 0..sign_at_i..0]
            slack_cols = []
            for j in range(K):
                if j == i:
                    slack_cols.append(sign * torch.ones_like(Lgbu_x))
                else:
                    slack_cols.append(torch.zeros_like(Lgbu_x))

            G_row = torch.cat([-Lgbu_x, -Lgbu_y] + slack_cols, dim=1)
            G_list.append(G_row.unsqueeze(1))
            h_list.append(b_i)

        G = torch.cat(G_list, dim=1)   # [H, K, 2+K]
        h = torch.cat(h_list, dim=1)   # [H, K]

        # QP: min ||[u, slack] - [u_ref, 0]||^2  s.t.  G @ [u, slack] <= h
        dim = 2 + K
        q = torch.zeros(nBatch, dim, device=G.device)
        q[:, 0] = -ref[:, pi]
        q[:, 1] = -ref[:, pj]
        Q = torch.eye(dim)
        Q = Q.unsqueeze(0).expand(nBatch, dim, dim).to(G.device)

        e = torch.Tensor()
        try:
            out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)
        except RuntimeError:
            # QP singular — fallback to uncorrected reference
            out = torch.zeros(nBatch, dim, device=G.device)
            out[:, 0] = ref[:, pi]
            out[:, 1] = ref[:, pj]

        rt = xp1.clone()
        rt[:, pi] = x[:, pi] + out[:, 0]
        rt[:, pj] = x[:, pj] + out[:, 1]
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()
    def GD(self, xp10, eps=0.0, lr=0.0002):
        """Classifier guidance: gradient push away from obstacles.

        For each waypoint inside/near an obstacle, compute the barrier gradient
        and push the waypoint outward proportionally.
        """
        xp1 = xp10.clone().squeeze(0)  # [H, D]
        pi = self.cbf.pos_x_idx + self.action_dim
        pj = self.cbf.pos_y_idx + self.action_dim

        for obs in self.cbf.obstacles:
            cx, cy = obs['center']
            rx = obs.get('radius_x', obs.get('radius', 1.0))
            ry = obs.get('radius_y', obs.get('radius', 1.0))
            n = obs['order']

            off_x = 2 * (cx + self.cbf.center_offset - self.cbf.norm_mins[self.cbf.pos_x_idx]) / (
                self.cbf.norm_maxs[self.cbf.pos_x_idx] - self.cbf.norm_mins[self.cbf.pos_x_idx]) - 1
            off_y = 2 * (cy + self.cbf.center_offset - self.cbf.norm_mins[self.cbf.pos_y_idx]) / (
                self.cbf.norm_maxs[self.cbf.pos_y_idx] - self.cbf.norm_mins[self.cbf.pos_y_idx]) - 1

            dx = (xp1[:, pi:pi+1] - off_x) / self.cbf.xr / rx
            dy = (xp1[:, pj:pj+1] - off_y) / self.cbf.yr / ry

            b = torch.abs(dy)**n + torch.abs(dx)**n - 1

            # Gradient of barrier w.r.t. normalized position
            grad_y = n * torch.abs(dy)**(n-1) * torch.sign(dy) / self.cbf.yr / ry
            grad_x = n * torch.abs(dx)**(n-1) * torch.sign(dx) / self.cbf.xr / rx

            # Push waypoints where b < eps
            mask = (b < eps).float()
            xp1[:, pj:pj+1] = xp1[:, pj:pj+1] + mask * lr * grad_y
            xp1[:, pi:pi+1] = xp1[:, pi:pi+1] + mask * lr * grad_x

        return xp1.unsqueeze(0)

    @property
    def device(self):
        return next(self.parameters()).device

    # ── training ──
    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)
        return self.loss_fn(x_recon, x)

    def forward(self, cond, n_diffusion_steps=None):
        """Sample a plan. Returns (x0 [B, H, transition_dim], safety time per step in s)."""
        if n_diffusion_steps is not None:
            self.n_timesteps = int(n_diffusion_steps)
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond)
