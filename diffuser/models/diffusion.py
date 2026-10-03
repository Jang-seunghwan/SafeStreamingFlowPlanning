import torch
from torch import nn
import time

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
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

        # Safety (set by diffuser/guides/policies.py)
        self.safety_enabled = False
        self.safety_method = 'invariance'  # 'gd' (Diffuser+CG) | 'invariance' (SafeDiffuser)
        self.cbf = None
        self.norm_mins = 0
        self.norm_maxs = 0

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

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

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
    
    #------------------------------------------ Safety (Only for Sampling) ------------------------------------------#

    def _apply_safety(self, x, xp1, t):
        """Safety correction of one denoising step.

            'gd'         -> GD: classifier guidance / potential-based correction (Diffuser+CG)
            'invariance' -> invariance_relax_cf: Relaxed Safe Diffuser, closed form (SafeDiffuser)
        """
        if not self.safety_enabled:
            return xp1
        if self.safety_method == 'gd':
            return self.GD(x, xp1, eps=0)
        if self.safety_method == 'invariance':
            return self.invariance_relax_cf(x, xp1, t)
        raise ValueError(f'Unknown safety_method: {self.safety_method}')

    @torch.no_grad()
    def GD(self, x0, xp10, eps=0.1):
        """
        Classifier guidance or potential-based method.
        """
        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]

        # normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        # normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        # CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        for k in range(nBatch):
            if b[k, 0] < eps:  # 0, 0.2
                u1 = 0.2/(2*((xp1[k,2:3] - off_y)/yr)/yr)
                u2 = 0.2/(2*((xp1[k,3:4] - off_x)/xr)/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001  # note no 0.1/0.01 for GD, but has for potential
                xp1[k,3] = xp1[k,3] + u2*0.001
            elif b2[k, 0] < eps:  # 0, 0.2
                u1 = 0.2/(4*((xp1[k,2:3] - off_y)/yr)**3/yr)
                u2 = 0.2/(4*((xp1[k,3:4] - off_x)/xr)**3/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001
                xp1[k,3] = xp1[k,3] + u2*0.001

        xp1 = xp1.unsqueeze(0)
        return xp1

    @torch.no_grad()
    def invariance_relax_cf(self, x, xp1, t):
        """
        Relaxed Safe Diffuser (ReS-diffuser), closed-form KKT solution for K <= 2 obstacles.
        """
        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        obstacles = self.cbf.obstacles
        K = len(obstacles)
        offset = self.cbf.center_offset
        robust_term = self.cbf.robust_term

        if t >= 10:
            sign = 100
        else:
            sign = 0

        # Build per-obstacle constraint rows
        G_rows = []
        h_rows = []
        for i, obs in enumerate(obstacles):
            cx, cy = obs['center']
            n = obs['order']
            rx = obs.get('radius_x', obs.get('radius', 1.0))
            ry = obs.get('radius_y', obs.get('radius', 1.0))

            xr = 2 * rx / (self.norm_maxs[1] - self.norm_mins[1])
            yr = 2 * ry / (self.norm_maxs[0] - self.norm_mins[0])
            off_x = 2 * (cx + offset - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
            off_y = 2 * (cy + offset - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1

            dy = (x[:, 2:3] - off_y) / yr
            dx = (x[:, 3:4] - off_x) / xr
            b_i = dy**n + dx**n - 1 - robust_term
            Lgbu1 = n * dy**(n - 1) / yr
            Lgbu2 = n * dx**(n - 1) / xr

            slack_cols = []
            for j in range(K):
                if j == i:
                    slack_cols.append(sign * torch.ones_like(Lgbu1))
                else:
                    slack_cols.append(torch.zeros_like(Lgbu1))

            G_row = torch.cat([-Lgbu1, -Lgbu2] + slack_cols, dim=1)
            G_rows.append(G_row)
            h_rows.append(b_i)

        dim = 2 + K
        q = torch.zeros(nBatch, dim, device=x.device)
        q[:, :2] = -ref[:, 2:4]
        u_bar = -q  # reference point

        if K == 1:
            # Single constraint closed-form: project onto G*u=h if violated
            G0 = G_rows[0]
            h0 = h_rows[0]
            p = h0 - torch.sum(G0 * u_bar, dim=1).unsqueeze(1)  # negative if violated
            g_norm_sq = torch.sum(G0 * G0, dim=1, keepdim=True) + 1e-8
            lam = torch.clamp(p / g_norm_sq, max=0)  # lam <= 0 when violated
            out = u_bar + lam * G0
        elif K == 2:
            # Two-constraint closed-form KKT (original analytical solution)
            G0, G1 = G_rows[0], G_rows[1]
            h0, h1 = h_rows[0], h_rows[1]
            y1_bar = G0
            y2_bar = G1
            p1_bar = h0 - torch.sum(G0 * u_bar, dim=1).unsqueeze(1)
            p2_bar = h1 - torch.sum(G1 * u_bar, dim=1).unsqueeze(1)

            Gm = torch.cat([
                torch.sum(y1_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0),
                torch.sum(y1_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0),
                torch.sum(y2_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0),
                torch.sum(y2_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0),
            ], dim=0)
            w_p1 = torch.clamp(p1_bar, max=0)
            w_p2 = torch.clamp(p2_bar, max=0)
            det = Gm[0] * Gm[3] - Gm[1] * Gm[2] + 1e-6

            lambda1 = torch.where(
                Gm[2] * w_p2 < Gm[3] * p1_bar, torch.zeros_like(p1_bar),
                torch.where(Gm[1] * w_p1 < Gm[0] * p2_bar, w_p1 / Gm[0],
                             torch.clamp(Gm[3] * p1_bar - Gm[2] * p2_bar, max=0) / det))
            lambda2 = torch.where(
                Gm[2] * w_p2 < Gm[3] * p1_bar, w_p2 / Gm[3],
                torch.where(Gm[1] * w_p1 < Gm[0] * p2_bar, torch.zeros_like(p1_bar),
                             torch.clamp(Gm[0] * p2_bar - Gm[1] * p1_bar, max=0) / det))
            out = lambda1 * y1_bar + lambda2 * y2_bar + u_bar
        else:
            raise ValueError(f'invariance_relax_cf supports at most 2 obstacles, got {K}')

        rt = xp1.clone()
        rt[:, 2:4] = x[:, 2:4] + out[:, 0:2]
        rt = rt.unsqueeze(0)
        return rt

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b = x.shape[0]

        nn_start = time.time()
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        nn_elapsed = time.time() - nn_start

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        safety_start = time.time()
        x = self._apply_safety(x, xp1, t)
        safety_elapsed = time.time() - safety_start

        return x, nn_elapsed, safety_elapsed

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """Reverse diffusion with the safety correction at every step.

        Returns the sample, the denoising path [B, n_timesteps + 1, H, D] and the average
        [step, network, safety] time per denoising step.
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        diffusion = [x]

        progress = utils.Progress(self.n_timesteps)
        iter_time = 0
        nn_time = 0
        safety_time = 0
        for i in reversed(range(0, self.n_timesteps)):
            iter_start = time.time()
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, nn_elapsed, safety_elapsed = self.p_sample(x, cond, timesteps)
            nn_time += nn_elapsed
            safety_time += safety_elapsed
            x = apply_conditioning(x, cond, self.action_dim)
            progress.update({'t': i})

            diffusion.append(x)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        progress.close()
        return x, torch.stack(diffusion, dim=1), [iter_time / self.n_timesteps, nn_time / self.n_timesteps, safety_time / self.n_timesteps]

    @torch.no_grad()
    def conditional_sample(self, cond):
        '''
            conditions : [ (time, state), ... ]
        '''
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, n_diffusion_steps):
        self.n_timesteps = int(n_diffusion_steps)
        return self.conditional_sample(cond=cond)
