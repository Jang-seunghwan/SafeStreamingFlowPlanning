import torch


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x
