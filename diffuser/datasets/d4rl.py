import os
import collections
import numpy as np
import gym

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def sequence_dataset(env):
    """
    Returns an iterator through the episodes of the D4RL dataset of `env`
    (split at `terminals` / `timeouts`), as dictionaries of arrays with keys
    observations, actions, rewards, terminals, timeouts, ...
    """
    dataset = env.get_dataset()

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        final_timestep = dataset['timeouts'][i]

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)
