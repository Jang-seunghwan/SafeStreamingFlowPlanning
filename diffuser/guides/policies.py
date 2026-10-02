"""
Policy wrapper for Diffuser/CFM inference with the safety layer attached.
"""
import torch
from ..models.cbf import CBF
from ..utils.arrays import to_np


class Policy:
    def __init__(self, diffusion_model, normalizer, args):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

        device = next(diffusion_model.parameters()).device
        self.diffusion_model.safety_enabled = args.safety_enabled
        self.diffusion_model.safety_method = args.safety_method
        if args.safety_enabled:
            norm_mins = torch.tensor(normalizer.normalizers['observations'].mins, device=device)
            norm_maxs = torch.tensor(normalizer.normalizers['observations'].maxs, device=device)
            self.diffusion_model.cbf = CBF(norm_mins, norm_maxs, args)
        self.n_diffusion_steps = args.n_diffusion_steps
        if hasattr(args, 'integrator'):  # CFM family
            self.diffusion_model.integrator = args.integrator

    @property
    def device(self):
        return next(self.diffusion_model.parameters()).device

    def _format_conditions(self, conditions, batch_size):
        normed = {}
        for k, v in conditions.items():
            v_normed = self.normalizer.normalize(v, 'observations')
            v_tensor = torch.from_numpy(v_normed).float().to(self.device)
            if v_tensor.dim() == 1:
                v_tensor = v_tensor.unsqueeze(0).repeat(batch_size, 1)
            normed[k] = v_tensor
        return normed

    def __call__(self, conditions, batch_size=1):
        """Sample a plan. Returns (observations [B, H, 4] unnormalized, safety_time_avg_s)."""
        conditions = self._format_conditions(conditions, batch_size)

        x1, safety_time_avg = self.diffusion_model.forward(
            cond=conditions,
            n_diffusion_steps=self.n_diffusion_steps,
        )

        observations = to_np(x1)[:, :, self.action_dim:]
        obs_unnorm = self.normalizer.unnormalize(observations, 'observations')
        return obs_unnorm, safety_time_avg
