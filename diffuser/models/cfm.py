import time
import torch
from torch import nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
from .helpers import (
    cosine_beta_schedule,
    apply_conditioning,
    Losses,
)

class CFM(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        # CFM setting
        sigma = 0.0
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        # Not used for sampling (the Euler loops below are); kept because it is part of the module's
        # state_dict (it wraps `model`), so checkpoints load with the same keys.
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # Get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # One-shot initialization
        self.one_shot_enabled = False

        # Safety (set by diffuser/guides/policies.py)
        self.safety_enabled = False
        self.cbf = None
        self.norm_mins = 0
        self.norm_maxs = 0

        # Integrator routing: 'pure' (FM) or 'pc' (FlowMatcher)
        self.integrator = 'pc'

        # Diffusion-schedule buffers: not used by CFM, kept so that the state_dict layout matches the checkpoints
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

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

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#
    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """
        FM (pure integrator): Manual Euler ODE integration.
        When safety_enabled=True, CBF correction is applied at each step (SafeFM).
        Uses the same manual Euler stepping for both plain FM and SafeFM
        to ensure fair comparison (only CBF on/off differs).
        """
        x = torch.randn(shape).to(self.device)
        x = apply_conditioning(x, cond, self.action_dim)

        traj = [x]
        dt = 1.0 / self.n_timesteps
        iter_time = 0
        nn_time = 0
        safety_time = 0

        for i in range(self.n_timesteps):
            iter_start = time.time()
            t_now = i * dt
            t_batch = torch.full((x.shape[0],), t_now, device=x.device)

            nn_start = time.time()
            u_raw = self.model(x, None, t_batch)
            nn_time += time.time() - nn_start

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
            traj.append(x)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)  # [B, T+1, H, D]
        x1 = traj_tensor[:, -1, :, :]
        return x1, traj_tensor, [iter_time / self.n_timesteps, nn_time / self.n_timesteps, safety_time / self.n_timesteps]
    
    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond):
        """
        FlowMatcher (pc integrator): optional one-step prediction stage, then Euler correction stage.
        When safety_enabled=True, CBF correction is applied at each step (SafeFlowMatcher).
        """
        n_timesteps = self.n_timesteps
        pred_n_timesteps = 1  # number of prediction steps
        # ================ Prediction Stage ================
        if self.one_shot_enabled:
            batch_size = len(cond[0])
            x0_1st_phase = torch.randn(shape).to(self.device)
            x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            
            pred_time_list = torch.linspace(0, 1, pred_n_timesteps+1).to(self.device)
            for i in range(pred_n_timesteps):
                t_now = pred_time_list[i]
                dt = 1 / pred_n_timesteps
                v_t = self.model(x0_1st_phase, None, torch.full((batch_size,), t_now, device=x0_1st_phase.device))
                x0_1st_phase = x0_1st_phase + v_t * dt
                x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            x0_2nd_phase = x0_1st_phase
        # ================ Correction Stage ================
        else:
            x0_2nd_phase = torch.randn(shape).to(self.device)

        x0_2nd_phase = apply_conditioning(x0_2nd_phase, cond, self.action_dim)

        T = n_timesteps + 1

        # Uniform scheduling
        time_list = torch.linspace(0, 1, T).to(self.device)  # [0, 1] for uniform scheduling
        
        traj = [x0_2nd_phase]

        iter_time = 0
        nn_time = 0
        safety_time = 0
        z =  2*(n_timesteps+1) / n_timesteps  # for scaling one-shot init velocity
        for i in range(1, T):
            iter_start = time.time()
            t_now = time_list[i-1]
            # define dt based on scheduling
            if self.one_shot_enabled:
                dt = 1/ n_timesteps
                one_minus_t = (n_timesteps - (i-1))/(n_timesteps)
                dt = z*one_minus_t*dt
            else:
                dt = time_list[i] - t_now
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            nn_start = time.time()
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt
            nn_time += time.time() - nn_start

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

            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)  # [B, T, H, D]
        return traj_tensor[:,T-1,:,:], traj_tensor, [iter_time/n_timesteps, nn_time/n_timesteps, safety_time/n_timesteps]

    @torch.no_grad()
    def conditional_sample(self, cond):
        '''
        conditions : [ (time, state), ... ]

        Routing:
            integrator='pure'  → p_sample_loop          (FM / SafeFM)
            integrator='pc'    → p_sample_loop_ode_planning (FlowMatcher / SafeFlowMatcher)
        '''
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)

        if self.integrator == 'pure':
            # FM: pure ODE integration (with optional CBF for SafeFM)
            return self.p_sample_loop(shape, cond)
        else:
            # FlowMatcher: Prediction-Correction structure (with optional CBF for SafeFlowMatcher)
            return self.p_sample_loop_ode_planning(shape, cond)

    @property
    def device(self):
        """
        Get the device where the model's parameters are allocated
        """
        # Assumes the model's parameters are all on the same device.
        return next(self.parameters()).device
    
    #------------------------------------------ training ------------------------------------------#
    
    def loss(self, x, cond):
        x = x.to(self.device)
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)

        # Generate xt and flow field ut at time t
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        # Apply condition
        xt = apply_conditioning(xt, cond, self.action_dim)

        # Compute vector field
        vt = self.model(xt, cond, t) # if there are cond, modify None -> cond

        # Zero out loss at conditioned positions (match diffusion behavior)
        for t_cond, val in cond.items():
            vt[:, t_cond, self.action_dim:] = ut[:, t_cond, self.action_dim:]

        # Compute loss
        loss, info = self.loss_fn(vt, ut)
        
        return loss, info

    def forward(self, cond, n_diffusion_steps):
        self.n_timesteps = int(n_diffusion_steps)
        x1, traj, iter_per_time = self.conditional_sample(cond=cond)
        return x1, traj, iter_per_time
