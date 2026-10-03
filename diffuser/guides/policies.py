"""
Policy wrapper for the Diffuser / FM / FlowMatcher baselines (scripts/plan_maze2d.py).

Uses CBFDiffuser (cbf_diffuser.py — class-K CBF in NORMALIZED space).
Streaming flow / SSF do NOT use this wrapper — they use sfpd.py + cbf.py (DT-HOCBF).

Args wired into the model:
  safety_enabled       global safety toggle
  safety_method        'gd' (Diffuser+CG) | 'invariance' (SafeDiffuser)     (Diffuser only)
  integrator           'pure' (FM) | 'pc' (FlowMatcher)                     (CFM only)
  one_shot_enabled     True for the FlowMatcher one-step prediction stage   (CFM only)
"""
from collections import namedtuple
import torch
import einops
from diffuser.models.cbf_diffuser import CBF
from diffuser.utils.trajectory_metrics import acceleration_smoothness

import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:

    def __init__(self, diffusion_model, normalizer, args):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

        device = next(diffusion_model.parameters()).device
        norm_mins = torch.tensor(normalizer.normalizers['observations'].mins, device=device)
        norm_maxs = torch.tensor(normalizer.normalizers['observations'].maxs, device=device)

        self.diffusion_model.one_shot_enabled = getattr(args, 'one_shot_enabled', False)

        self.diffusion_model.safety_enabled = getattr(args, 'safety_enabled', False)
        self.diffusion_model.cbf = CBF(norm_mins, norm_maxs, args)
        self.n_diffusion_steps = args.n_diffusion_steps

        # FM vs FlowMatcher routing (CFM only): 'pure' or 'pc'
        self.diffusion_model.integrator = getattr(args, 'integrator', 'pc')

        # Safety method routing (Diffuser only): 'gd' | 'invariance'
        self.diffusion_model.safety_method = getattr(args, 'safety_method', 'invariance')

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, batch_size=1):
        """Plan once. Returns (trajectories, num_trap, iter_time, s_smooth):
        trajectories : unnormalized actions / observations of the plan
        num_trap     : number of jumps > 0.2 (normalized position) in the denoising path (Trap column)
        iter_time    : average [step, network, safety] time per denoising / integration step
        s_smooth     : acceleration smoothness of the planned sample (normalized coordinates; Sm column)
        """
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion / ODE process
        self.diffusion_model.norm_mins = self.normalizer.normalizers['observations'].mins
        self.diffusion_model.norm_maxs = self.normalizer.normalizers['observations'].maxs
        sample, diffusion, iter_time = self.diffusion_model(conditions, self.n_diffusion_steps)
        s_smooth = acceleration_smoothness(sample, action_dim=self.action_dim)[0]

        num_trap = utils.local_trap(diffusion, batch_idx=0, n_timesteps=self.n_diffusion_steps-1)

        sample = utils.to_np(sample)

        ## extract action [batch_size x horizon x transition_dim]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations)
        return trajectories, num_trap, iter_time, s_smooth
