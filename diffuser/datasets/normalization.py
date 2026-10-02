import numpy as np


class DatasetNormalizer:
    """Per-key normalizers for observations and actions.

    Instances are pickled into every checkpoint, so the module path
    (diffuser.datasets.normalization) and the attributes must not change.
    """

    def __init__(self, observations, actions):
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.normalizers = {
            'observations': LimitsNormalizer(observations),
            'actions': LimitsNormalizer(actions),
        }

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)


class Normalizer:
    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'[ Normalizer ] dim: {self.mins.size}\n    -: '
            f'{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'
        )

    def __call__(self, x):
        return self.normalize(x)


class LimitsNormalizer(Normalizer):
    '''maps [ xmin, xmax ] to [ -1, 1 ]'''
    def normalize(self, x):
        x = (x - self.mins) / (self.maxs - self.mins + 1e-8)
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)
        x = (x + 1) / 2.
        return x * (self.maxs - self.mins) + self.mins
