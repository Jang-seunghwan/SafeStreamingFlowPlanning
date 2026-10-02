import torch
import einops

import diffuser.utils as utils


class GuidedPolicy:
    """
    Receding-horizon planning policy: plans a trajectory conditioned on the current
    observation and returns its first action. `sample_kwargs` (value guide, guided
    sampling function and its parameters) are passed through to the sampler.
    """

    def __init__(self, diffusion_model, normalizer, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.sample_kwargs = sample_kwargs

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, batch_size=1, verbose=False):
        """Returns (first action, timing list returned by the sampler)."""
        conditions = self._format_conditions(conditions, batch_size)

        trajectories, _, timing = self.diffusion_model(
            conditions, verbose=verbose, **self.sample_kwargs,
        )

        # extract actions [batch x horizon x action_dim]
        actions = utils.to_np(trajectories)[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        return actions[0, 0], timing
