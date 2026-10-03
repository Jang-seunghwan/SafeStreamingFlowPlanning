import torch


def local_trap(traj_tensor, batch_idx=0, n_timesteps=256):
    """
    Trap metric of the Diffuser / FM baselines: number of consecutive-state jumps larger than
    DIST_THRESHOLD (normalized position) in the plan at denoising-path index `n_timesteps`.

    Args:
        traj_tensor: denoising path, [B, n_steps + 1, H, D] with D = [action(2), obs(4)]
        batch_idx: Batch index to use
        n_timesteps: index into the denoising path
    """
    traj = traj_tensor[batch_idx, n_timesteps, :, 2:4]

    DIST_THRESHOLD = 0.20
    num_trap = 0
    for i in range(1, traj.shape[0]):
        if torch.norm(traj[i] - traj[i-1], p=2).item() > DIST_THRESHOLD:
            num_trap += 1
    return num_trap
