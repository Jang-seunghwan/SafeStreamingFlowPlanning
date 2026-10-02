import os
import pickle
import glob


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

def load_experiment(loadbase, dataset, loadpath, epoch='latest'):
    """
        Rebuild dataset / model / trainer from the configs saved by the training script
        in <loadbase>/<dataset>/<loadpath>/ and load the checkpoint `state_<epoch>.pt`.
        Returns (dataset, EMA model).
    """
    path = [loadbase, dataset, loadpath]
    dataset_config = load_config(*path, 'dataset_config.pkl')
    model_config = load_config(*path, 'model_config.pkl')
    diffusion_config = load_config(*path, 'diffusion_config.pkl')
    trainer_config = load_config(*path, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = os.path.join(*path)

    data = dataset_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, data)

    if epoch == 'latest':
        epoch = get_latest_epoch(path)
    print(f'[ utils/serialization ] Loading {os.path.join(*path)} (epoch {epoch})')
    trainer.load(epoch)
    return data, trainer.ema_model
