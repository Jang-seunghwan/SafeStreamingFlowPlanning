import numpy as np
import torch
from torch import nn
import time
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


class GaussianDiffusion(nn.Module):
    '''
        Diffuser with x0-prediction (`predict_epsilon` and `clip_denoised` must be False;
        they are kept as arguments so that saved configs load).
    '''
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=False, predict_epsilon=False,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        assert not predict_epsilon and not clip_denoised
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        # Safety (torso-height ceiling, applied during sampling only)
        self.safety_enabled = False
        self.safety_method = 'invariance'  # 'invariance' (SafeDiffuser QP) or 'cg' (truncate)
        self._safety_time_acc = 0.0  # reset by the evaluator; accumulated by the safety methods
        self.mean = 0  # GaussianNormalizer means (set externally)
        self.std = 0   # GaussianNormalizer stds (set externally)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

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

    #------------------------------------------ sampling ------------------------------------------#

    def q_posterior(self, x_start, x_t, t):

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # the model predicts x0 directly
        x_recon = self.model(x, cond, t)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # ── Locomotion safety methods (from the original SafeDiffuser) ────

    @torch.no_grad()
    def invariance_hopper(self, x, xp1):
        """SafeDiffuser QP for the hopper ceiling constraint (from the original SafeDiffuser)."""
        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        # normalize height: GaussianNormalizer
        height = 1.5
        height = (height - self.mean[0]) / self.std[0]

        # CBF — ceiling
        b = height - x[:, 3:4]
        Lfb = 0
        Lgbu1 = -1 * torch.ones_like(x[:, 3:4])

        G = torch.cat([-Lgbu1], dim=1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k * b

        q = -torch.cat([ref[:, 3:4]], dim=1).to(G.device)
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)

        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()
        rt[:, 3:4] = x[:, 3:4] + out[:, 0:1]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)

    @torch.no_grad()
    def Shield_hopper(self, x0, xp10):
        """Truncate/Shield method for the hopper ceiling constraint (Diffuser+CG baseline,
        from the original SafeDiffuser)."""
        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)
        nBatch = xp1.shape[0]

        # normalize height: GaussianNormalizer
        height = 1.5
        height = (height - self.mean[0]) / self.std[0]

        # ceiling: clip
        b = height - xp1[:, 3:4]
        for k in range(nBatch):
            if b[k, 0] < 0:
                xp1[k, 3] = height

        b = height - xp1[:, 3:4]
        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:, 0])

    def _apply_safety(self, x, xp1):
        """Safety filter on one denoising step x -> xp1 (accumulates its run time)."""
        _ts0 = time.perf_counter()
        if self.safety_method == 'invariance':
            # SafeDiffuser QP (invariance_hopper)
            xp1, _ = self.invariance_hopper(x, xp1)
        elif self.safety_method == 'cg':
            # truncate (Diffuser+CG baseline)
            xp1, _ = self.Shield_hopper(x, xp1)
        self._safety_time_acc += time.perf_counter() - _ts0
        return xp1

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        """Unguided denoising step (used with --guide_scale 0)."""
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        if self.safety_enabled:
            xp1 = self._apply_safety(x, xp1)
        return xp1

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, sample_fn=None, **sample_kwargs):
        """
            Returns the final sample, the denoising chain and [mean time per denoising step].
            With `sample_fn` (value-guided step), the safety filter is applied after it.
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        values = torch.zeros(batch_size, device=device)
        iter_time = 0
        for i in reversed(range(0, self.n_timesteps)):
            iter_start = time.time()
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            if sample_fn is not None:
                x_prev = x
                x, values = sample_fn(self, x, cond, timesteps, **sample_kwargs)
                if self.safety_enabled:
                    x = self._apply_safety(x_prev, x)
            else:
                x = self.p_sample(x, cond, timesteps)

            x = apply_conditioning(x, cond, self.action_dim)
            progress.update({'t': i})

            diffusion.append(x)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        progress.close()

        # sort by values when using guided sampling
        if sample_fn is not None:
            inds = torch.argsort(values, descending=True)
            x = x[inds]
            values = values[inds]

        return x, torch.stack(diffusion, dim=1), [iter_time/self.n_timesteps]

    @torch.no_grad()
    def conditional_sample(self, cond, **kwargs):
        '''
            conditions : { time: state }
        '''
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, **kwargs)

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

        loss, info = self.loss_fn(x_recon, x_start)
        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, **kwargs):
        return self.conditional_sample(cond=cond, **kwargs)
