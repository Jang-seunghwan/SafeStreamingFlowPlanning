"""
F1TENTH trajectory datasets.

State space: [x, y, vx_w, vy_w] (4D, world-frame velocity), loaded from a
preprocessed <track>.npz written by scripts/regenerate_processed_data.py.

  - F1tenthSequenceDataset: SFP training (state only, 4D)
  - F1tenthGoalDataset:     Diffuser / CFM training (action + state, 6D)
"""
import os
from collections import namedtuple

import numpy as np
import torch

from .normalization import DatasetNormalizer

Batch = namedtuple('Batch', 'trajectories conditions')


def load_observations(npz_path):
    """Load preprocessed observations [N, H, 4] from an .npz file."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f'{npz_path} not found; run scripts/regenerate_processed_data.py first')
    obs = np.load(npz_path)['observations']
    print(f'  Loaded preprocessed: {npz_path}')
    print(f'  {obs.shape[0]} episodes, H={obs.shape[1]}')
    return obs


class F1tenthSequenceDataset(torch.utils.data.Dataset):
    """
    F1tenth dataset for SFP training.
    Returns 4D state trajectories: [x, y, vx_w, vy_w]
    """

    def __init__(self, npz_path, horizon):
        self.horizon = horizon
        obs_array = load_observations(npz_path)
        assert obs_array.shape[1] == horizon, (obs_array.shape, horizon)

        all_obs = obs_array.reshape(-1, 4)
        all_actions = np.zeros((all_obs.shape[0], 2), dtype=np.float32)
        self.normalizer = DatasetNormalizer(all_obs, all_actions)

        self.observation_dim = 4
        self.action_dim = 2
        self.observations = obs_array
        self.n_episodes = obs_array.shape[0]
        print(f'[ F1tenthSequenceDataset ] {self.n_episodes} episodes (H={horizon})')

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        raw_obs = self.observations[idx].copy()
        normed = self.normalizer.normalize(raw_obs, 'observations')
        conditions = {
            0: normed[0],
            self.horizon - 1: normed[-1],
        }
        return Batch(normed, conditions)


class F1tenthGoalDataset(torch.utils.data.Dataset):
    """
    F1tenth dataset for Diffuser/CFM training.
    Returns 6D trajectories: [action(2), observation(4)]
    """

    def __init__(self, npz_path, horizon):
        self.horizon = horizon
        obs_array = load_observations(npz_path)
        assert obs_array.shape[1] == horizon, (obs_array.shape, horizon)
        n_episodes = obs_array.shape[0]

        # Compute actions: a_t = v_{t+1} - v_t
        vel = obs_array[:, :, 2:4]
        act_array = np.zeros((n_episodes, horizon, 2), dtype=np.float32)
        act_array[:, :-1] = vel[:, 1:] - vel[:, :-1]

        self.normalizer = DatasetNormalizer(obs_array.reshape(-1, 4), act_array.reshape(-1, 2))

        self.observation_dim = 4
        self.action_dim = 2
        self.observations = obs_array
        self.actions = act_array
        self.n_episodes = n_episodes
        print(f'[ F1tenthGoalDataset ] {n_episodes} episodes (H={horizon})')

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        raw_obs = self.observations[idx].copy()
        raw_act = self.actions[idx].copy()

        normed_obs = self.normalizer.normalize(raw_obs, 'observations')
        normed_act = self.normalizer.normalize(raw_act, 'actions')

        trajectories = np.concatenate([normed_act, normed_obs], axis=-1)

        conditions = {
            0: normed_obs[0],
            self.horizon - 1: normed_obs[-1],
        }
        return Batch(trajectories, conditions)
