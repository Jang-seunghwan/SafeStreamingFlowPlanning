from collections import namedtuple
import numpy as np
import torch

from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-expert-v2', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=(), max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True):
        assert not preprocess_fns  # argument kept so that saved configs load
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start + 1):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        # When sfp_fast is enabled and indices are sparse (horizon==max_path_length),
        # inflate length so each epoch has enough gradient steps.
        base_len = len(self.indices)
        if getattr(self, 'sfp_fast', False) and base_len < self.n_episodes * 100:
            return self.n_episodes * 400  # ~same scale as horizon<max_path_length
        return base_len

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx % len(self.indices)]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        if getattr(self, 'sfp_fast', False):
            # SFP training: return only (cond_start, cond_end, x_t, x_t1, seg_idx, horizon)
            # with x = [observation, action], instead of the full trajectory
            horizon = observations.shape[0]
            seg_idx = np.random.randint(0, horizon - 1)
            obs_dim = observations.shape[-1]

            x_t = np.concatenate([observations[seg_idx], actions[seg_idx]], axis=-1).astype(np.float32)
            x_t1 = np.concatenate([observations[seg_idx + 1], actions[seg_idx + 1]], axis=-1).astype(np.float32)

            cond_start = observations[0, :obs_dim].astype(np.float32)
            cond_end = observations[-1, :obs_dim].astype(np.float32)
            return (cond_start, cond_end, x_t, x_t1, seg_idx, horizon)

        # planners: full trajectory [action, observation]
        trajectories = np.concatenate([actions, observations], axis=-1)
        conditions = self.get_conditions(observations)
        batch = Batch(trajectories, conditions)
        return batch

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
