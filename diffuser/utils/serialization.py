import os
import pickle
import glob
import torch

from collections import namedtuple

DiffusionExperiment = namedtuple('Diffusion', 'dataset diffusion epoch')

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, epoch='latest'):
    """Load a trained Diffuser / CFM model (EMA weights of state_{epoch}.pt) and its dataset from `loadpath`."""
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')

    dataset = dataset_config()
    model = model_config()
    diffusion = diffusion_config(model)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    data = torch.load(os.path.join(*loadpath, f'state_{epoch}.pt'))
    diffusion.load_state_dict(data['ema'])

    return DiffusionExperiment(dataset, diffusion, epoch)
