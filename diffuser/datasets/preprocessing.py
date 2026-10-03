import numpy as np

from .d4rl import load_environment

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn

def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)

#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#

def maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset['observations'][:,:2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        return dataset

    return _fn
