import numpy as np

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, path_lengths=None):
        dataset = flatten(dataset, path_lengths)

        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')

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

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins
