"""Seed every random number generator the benchmark samples from."""
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python `random`, numpy, torch (CPU and CUDA) and OMPL (RRT*), if installed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        from ompl import util as ou
        ou.RNG.setSeed(seed)
    except ImportError:
        pass
