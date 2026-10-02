import time

import torch
from torch import nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

from .helpers import (
    cosine_beta_schedule,
    apply_conditioning,
    WeightedL2,
)


class CFM(nn.Module):
    """Conditional flow matching over trajectories with two samplers:
    'pure' (Euler, FM / SafeFM) and 'pc' (one-shot prediction + correction,
    FlowMatcher / SafeFlowMatcher). With safety enabled, every integration
    step is filtered by the CBF-QP (self.cbf)."""

    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 action_weight=1.0):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        # CFM setting
        self.FM = ConditionalFlowMatcher(sigma=0.0)
        # Not used for sampling (explicit Euler loops below); kept because the
        # checkpoints' state_dicts contain its (shared) parameters under node.*.
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # Loss (uniform weights; the first action is weighted by action_weight)
        loss_weights = torch.ones(self.horizon, self.transition_dim, dtype=torch.float32)
        loss_weights[0, :self.action_dim] = action_weight
        self.loss_fn = WeightedL2(loss_weights)

        # Safety
        self.safety_enabled = False
        self.cbf = None

        # Integrator routing: 'pure' (FM) or 'pc' (FlowMatcher)
        self.integrator = 'pc'

        # Number of integration steps K (set per call in forward).
        self.n_timesteps = int(n_timesteps)

        # Diffusion schedule buffers: not used by CFM, but part of the checkpoints' state_dicts.
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    #------------------------------------------ sampling ------------------------------------------#
    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """
        FM (pure integrator): Euler integration with dt = 1/K.
        When safety_enabled=True, CBF correction is applied at each step (SafeFM).
        Returns the sample and the average safety-layer time per step.
        """
        x = torch.randn(shape).to(self.device)
        x = apply_conditioning(x, cond, self.action_dim)

        dt = 1.0 / self.n_timesteps
        safety_time = 0

        for i in range(self.n_timesteps):
            t_now = i * dt
            t_batch = torch.full((x.shape[0],), t_now, device=x.device)
            u_raw = self.model(x, None, t_batch)

            # CBF correction (SafeFM)
            if self.safety_enabled and self.cbf is not None:
                x_next_naive = x + u_raw * dt
                safety_start = time.time()
                x_corr = self.cbf.apply(x, x_next_naive, t=t_now)
                safety_time += time.time() - safety_start
                dx = x_corr - x
            else:
                dx = u_raw * dt

            x = x + dx
            x = apply_conditioning(x, cond, self.action_dim)

        return x, safety_time / self.n_timesteps

    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond):
        """
        FlowMatcher (pc integrator): one Euler prediction step from noise to an
        initial estimate, then K correction steps with step size
        dt_i = z (1 - t_i) / K, z = 2 (K + 1) / K. With safety_enabled=True,
        every correction step is filtered by the CBF-QP (SafeFlowMatcher).
        Returns the sample and the average safety-layer time per correction step.
        """
        n_timesteps = self.n_timesteps
        batch_size = len(cond[0])

        # ================ Prediction Stage ================
        x0_1st_phase = torch.randn(shape).to(self.device)
        x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
        t_pred = torch.linspace(0, 1, 2).to(self.device)[0]
        v_t = self.model(x0_1st_phase, None,
                         torch.full((batch_size,), t_pred, device=x0_1st_phase.device))
        x0_1st_phase = x0_1st_phase + v_t * 1.0
        x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)

        # ================ Correction Stage ================
        x_now = apply_conditioning(x0_1st_phase, cond, self.action_dim)

        T = n_timesteps + 1
        time_list = torch.linspace(0, 1, T).to(self.device)  # uniform schedule on [0, 1]

        safety_time = 0
        z = 2*(n_timesteps+1) / n_timesteps  # scaling of the step size after the one-shot prediction
        for i in range(1, T):
            t_now = time_list[i-1]
            dt = 1 / n_timesteps
            one_minus_t = (n_timesteps - (i-1))/(n_timesteps)
            dt = z*one_minus_t*dt

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt

            # CBF correction
            if self.safety_enabled and self.cbf is not None:
                x_next_naive = x_now + u_raw * dt
                safety_start = time.time()
                x_corr = self.cbf.apply(x_now, x_next_naive, t=t_now)
                safety_time += time.time() - safety_start
                dx = x_corr - x_now
            else:
                dx = u_raw * dt

            x_next = x_now + dx
            x_now = apply_conditioning(x_next, cond, self.action_dim)

        return x_now, safety_time / n_timesteps

    @property
    def device(self):
        """
        Get the device where the model's parameters are allocated
        """
        return next(self.parameters()).device

    #------------------------------------------ training ------------------------------------------#

    def loss(self, x, cond):
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)

        # Generate xt and flow field ut at time t
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        # Apply condition
        xt = apply_conditioning(xt, cond, self.action_dim)

        # Compute vector field
        vt = self.model(xt, cond, t)  # TemporalUnet ignores cond (conditioning by inpainting)

        # Zero out loss at conditioned positions (match diffusion behavior)
        for t_cond in cond:
            vt[:, t_cond, self.action_dim:] = ut[:, t_cond, self.action_dim:]

        return self.loss_fn(vt, ut)

    def forward(self, cond, n_diffusion_steps):
        """Sample a plan with K = n_diffusion_steps integration steps (dt = 1/K;
        K may differ from the value used for training).
        Returns (x1 [B, H, transition_dim], safety time per step in s)."""
        self.n_timesteps = int(n_diffusion_steps)
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        if self.integrator == 'pure':
            return self.p_sample_loop(shape, cond)
        return self.p_sample_loop_ode_planning(shape, cond)
