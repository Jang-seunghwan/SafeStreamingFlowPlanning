import time
import torch
from torch import nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from .helpers import (
    apply_conditioning,
    Losses,
)


class CFM(nn.Module):
    '''
        Conditional flow matching trajectory model.

        Sampling (integrator):
            'pure' -> FM:          Euler integration of the flow, K = n_timesteps steps
            'pc'   -> FlowMatcher: one-shot prediction step, then K correction steps
        With `safety_enabled`, the CBF in `self.cbf` corrects every integration step
        (SafeFM / SafeFlowMatcher). The flow is guided by a value function (`enable_guidance`).

        `clip_denoised` and `predict_epsilon` are unused; they are kept as arguments so that
        saved configs load.
    '''
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=False, predict_epsilon=False,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.n_timesteps = int(n_timesteps)

        # Flow matching with straight (OT) paths, sigma = 0
        self.FM = ConditionalFlowMatcher(sigma=0.0)

        # Get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # Integrator routing: 'pure' (FM) or 'pc' (FlowMatcher)
        self.integrator = 'pc'
        # One-shot prediction stage of FlowMatcher (set together with integrator='pc')
        self.one_shot_enabled = False

        # Safety (CBFLoco, set externally for SafeFM / SafeFlowMatcher)
        self.safety_enabled = False
        self.cbf = None

        # Reward guidance
        self.guidance_enabled = False
        self.guidance_matcher = None
        self.value_guide = None

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

    # ---- Guidance ----

    def enable_guidance(self, value_model, scale=1.0):
        """Enable reward guidance of the flow field with a value function (Cov-G method)."""
        from diffuser.sampling.guides import ValueGuide
        from diffuser.models.guidance_matcher import GuidanceMatcher

        self.guidance_enabled = True
        self.value_guide = ValueGuide(value_model)
        self.guidance_matcher = GuidanceMatcher(scale=scale)

    def _guided_model(self, t, x):
        """Model call with guidance applied."""
        vt = self.model(x, None, t)
        if self.guidance_enabled:
            x = x.detach().requires_grad_()
            with torch.enable_grad():
                x1_pred = x + (1 - t) * vt
                _, grad_v = self.value_guide.gradients(x1_pred, None, t)
            vt = self.guidance_matcher.apply_guidance(vt, grad_v, t)
        return vt

    #------------------------------------------ sampling ------------------------------------------#
    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """
        FM (pure integrator): Euler integration.
        When safety_enabled=True, CBF correction is applied at each step (SafeFM).
        Returns the final sample, the integration path and the per-step timing
        [iteration, network, safety] in seconds.
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
            u_raw = self._guided_model(t_batch, x)
            nn_time += time.time() - nn_start

            # CBF correction (SafeFM)
            if self.safety_enabled:
                x_next_naive = x + u_raw * dt
                safety_start = time.time()
                x_corr, _ = self.cbf.apply(x, x_next_naive)
                safety_time += time.time() - safety_start
                dx = x_corr - x
            else:
                dx = u_raw * dt

            x = x + dx
            x = apply_conditioning(x, cond, self.action_dim)
            traj.append(x)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)
        x1 = traj_tensor[:, -1, :, :]
        return x1, traj_tensor, [iter_time / self.n_timesteps, nn_time / self.n_timesteps, safety_time / self.n_timesteps]

    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond):
        """
        FlowMatcher (prediction-correction integrator).
            Prediction: one Euler step of the flow from noise over the whole interval
                        (one-shot estimate of the trajectory).
            Correction: n_timesteps steps starting from the prediction, with step sizes
                        dt_i = z (1 - t_{i-1}) / n, z = 2 (n + 1) / n.
        When safety_enabled=True, CBF correction is applied at each correction step
        (SafeFlowMatcher). Returns the same tuple as `p_sample_loop`.
        """
        assert self.one_shot_enabled, "FlowMatcher ('pc') uses the one-shot prediction stage"
        n_timesteps = self.n_timesteps
        pred_n_timesteps = 1

        # ================ Prediction Stage ================
        batch_size = len(cond[0])
        x0_1st_phase = torch.randn(shape).to(self.device)
        x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)

        pred_time_list = torch.linspace(0, 1, pred_n_timesteps + 1).to(self.device)
        for i in range(pred_n_timesteps):
            t_now = pred_time_list[i]
            dt_pred = 1 / pred_n_timesteps
            t_batch = torch.full((batch_size,), t_now, device=x0_1st_phase.device)
            v_t = self._guided_model(t_batch, x0_1st_phase)
            x0_1st_phase = x0_1st_phase + v_t * dt_pred
            x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)

        # ================ Correction Stage ================
        x0_2nd_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)

        T = n_timesteps + 1
        time_list = torch.linspace(0, 1, T).to(self.device)

        traj = [x0_2nd_phase]

        iter_time = 0
        nn_time = 0
        safety_time = 0
        z = 2 * (n_timesteps + 1) / n_timesteps  # scales the steps after the one-shot prediction
        for i in range(1, T):
            iter_start = time.time()
            t_now = time_list[i - 1]
            dt = 1 / n_timesteps
            one_minus_t = (n_timesteps - (i - 1)) / n_timesteps
            dt = z * one_minus_t * dt
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)

            nn_start = time.time()
            u_raw = self._guided_model(t_batch, x_now)
            nn_time += time.time() - nn_start

            # CBF correction (SafeFlowMatcher)
            if self.safety_enabled:
                x_next_naive = x_now + u_raw * dt
                safety_start = time.time()
                x_corr, _ = self.cbf.apply(x_now, x_next_naive)
                safety_time += time.time() - safety_start
                dx = x_corr - x_now
            else:
                dx = u_raw * dt

            x_next = x_now + dx
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)
        return traj_tensor[:, T - 1, :, :], traj_tensor, [iter_time / n_timesteps, nn_time / n_timesteps, safety_time / n_timesteps]

    @torch.no_grad()
    def conditional_sample(self, cond):
        '''
        conditions : { time: state }

        Routing:
            integrator='pure'  → p_sample_loop              (FM / SafeFM)
            integrator='pc'    → p_sample_loop_ode_planning (FlowMatcher / SafeFlowMatcher)
        '''
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)

        if self.integrator == 'pure':
            return self.p_sample_loop(shape, cond)
        else:
            return self.p_sample_loop_ode_planning(shape, cond)

    @property
    def device(self):
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
        vt = self.model(xt, cond, t)

        # Zero out loss at conditioned positions (match diffusion behavior)
        for t_cond, val in cond.items():
            vt[:, t_cond, self.action_dim:] = ut[:, t_cond, self.action_dim:]

        # Compute loss
        loss, info = self.loss_fn(vt, ut)

        return loss, info

    def forward(self, cond, verbose=False):
        # returns (x1, integration path, [iteration, network, safety] time per step)
        return self.conditional_sample(cond)


class ValueCFM(CFM):
    """CFM subclass for value function training (locomotion)."""

    def loss(self, x, *args):
        return self.p_losses(x, *args)

    def p_losses(self, x_start, cond, target):
        x_0 = torch.randn_like(x_start)
        t, x_noisy, ut = self.FM.sample_location_and_conditional_flow(x_0, x_start)

        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)
