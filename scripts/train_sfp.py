"""
train_sfp.py

Train the hierarchical (pos -> vel -> act) streaming flow velocity network on
hopper-medium-expert-v2. The same model is used by the StreamingFlow and SSF rows.
Defaults (config/locomotion.py:sfp['train']) are the settings used for the paper.

    python scripts/train_sfp.py      # -> logs/hopper-medium-expert-v2/sfp/k2.0_s0.0001/
"""
import os
import copy
import json

import numpy as np
import torch
import diffuser.utils as utils
from torch.utils.data import DataLoader
from diffuser.utils.training import EMA
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

from diffuser.models.cond_unet1D_loco import ConditionalUnet1DLoco
from diffuser.models.sfpd_loco import StreamingFlowPolicyLoco


def make_collate_fn(
    sigma: float,
    k: float,
    pos_dim: int,
    vel_dim: int,
    act_dim: int = 0,
):
    """
    Isotropic Gaussian tube collate for SFP training.
    Each sample: (cond_start, cond_end, x_t, x_t1, seg_idx, horizon)
    """
    sigma = float(sigma)
    k = float(k)
    state_dim = pos_dim + vel_dim + act_dim

    def collate(samples):
        cond_starts = np.stack([s[0] for s in samples], axis=0)
        cond_ends   = np.stack([s[1] for s in samples], axis=0)
        xs          = np.stack([s[2] for s in samples], axis=0)
        x_nexts     = np.stack([s[3] for s in samples], axis=0)
        seg_idxs    = np.array([s[4] for s in samples], dtype=np.int64)
        horizon     = samples[0][5]

        batch_size = len(samples)
        cond = np.stack([cond_starts, cond_ends], axis=1)

        t  = (seg_idxs / (horizon - 1)).astype(np.float32)
        dt = 1.0 / (horizon - 1)
        x_dot = (x_nexts - xs) / dt

        # Gaussian tube around the demonstration: sigma_t = sigma * exp(-k t)
        sigma_t = sigma * np.exp(-k * t).astype(np.float32)
        eps = sigma_t[:, None] * np.random.randn(batch_size, state_dim).astype(np.float32)
        x_train = xs + eps
        v_train = x_dot - k * eps

        x_label = x_train + v_train * dt

        return {
            'cond':    torch.from_numpy(cond),
            'x':       torch.from_numpy(x_train[:, None, :]),
            'x_label': torch.from_numpy(x_label[:, None, :]),
            'v':       torch.from_numpy(v_train[:, None, :]),
            't':       torch.from_numpy(t),
        }

    return collate


def train_sfp(args):
    # Reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
    )

    dataset = dataset_config()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dimension split of the observation: [pos (qpos[1:]), vel (qvel)]
    env_name = args.dataset.lower()
    if 'hopper' in env_name:
        pos_dim = 5; vel_dim = 6
    else:
        raise ValueError(f"Unsupported env: {args.dataset} (this release covers Hopper only)")

    obs_dim = pos_dim + vel_dim
    act_dim = dataset.action_dim

    # fast dataloader: return (cond, x_t, x_t1) with x = [obs, action] instead of full trajectory
    dataset.sfp_fast = True

    num_cond = 2   # conditioning on start and end states
    state_dim = obs_dim + act_dim  # 11 + 3 = 14

    # Conditioning uses obs only (not actions)
    condition_dim = obs_dim * num_cond
    print(f"[ train_sfp ] env={args.dataset}")
    print(f"[ train_sfp ] state_dim={state_dim}, obs_dim={obs_dim}, pos_dim={pos_dim}, vel_dim={vel_dim}, act_dim={act_dim}")
    print(f"[ train_sfp ] condition_dim={condition_dim}")

    # ── 3-tier: pos_net → vel_net → act_net ──
    velocity_net = ConditionalUnet1DLoco(
        pos_dim=pos_dim,
        vel_dim=vel_dim,
        act_dim=act_dim,
        global_cond_dim=condition_dim,
        fc_timesteps=1,
        horizon=args.horizon,
        down_dims_vel=(64, 128),
        down_dims_act=(64, 128),
    ).to(device)

    policy = StreamingFlowPolicyLoco(
        velocity_net=velocity_net,
        pos_dim=pos_dim,
        vel_dim=vel_dim,
        act_dim=act_dim,
        num_cond=num_cond,
        sigma=args.sigma_train,
        k=args.k,
        device=device,
    ).to(device)

    collate_fn = make_collate_fn(
        sigma=args.sigma_train,
        k=args.k,
        pos_dim=pos_dim,
        vel_dim=vel_dim,
        act_dim=act_dim,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 4
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    ema = EMA(args.ema_decay)
    ema_velocity = copy.deepcopy(policy.velocity_net).to(device)

    # Separate learning rates for pos_net, vel_net, act_net
    pos_params = list(policy.velocity_net.pos_net.parameters())
    vel_params = list(policy.velocity_net.vel_net.parameters())
    act_params = list(policy.velocity_net.act_net.parameters())
    shared_ids = {id(p) for p in pos_params + vel_params + act_params}
    shared_params = [p for p in policy.velocity_net.parameters() if id(p) not in shared_ids]
    optimizer = torch.optim.AdamW([
        {'params': shared_params + pos_params, 'lr': args.learning_rate * 5},  # pos: 5e-4
        {'params': vel_params, 'lr': args.learning_rate * 3},                  # vel: 3e-4
        {'params': act_params, 'lr': args.learning_rate * 3},                  # act: 3e-4
    ], weight_decay=args.weight_decay)

    def save_checkpoint(path, epoch_label=None):
        checkpoint = {
            'epoch': epoch_label,
            'velocity_state_dict': policy.velocity_net.state_dict(),
            'config': {
                'dataset': args.dataset,
                'horizon': args.horizon,
                'sigma_train': args.sigma_train,
                'k': args.k,
                'state_dim': state_dim,
                'obs_dim': obs_dim,
                'pos_dim': pos_dim,
                'vel_dim': vel_dim,
                'act_dim': act_dim,
                'condition_dim': condition_dim,
                'is_loco': True,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'ema_decay': args.ema_decay,
                'n_epochs': args.n_epochs,
                'num_workers': args.num_workers,
                'device': args.device,
            },
        }
        torch.save(checkpoint, path)
        print(f"[ train_sfp ] Saved checkpoint to {path}")

    os.makedirs(args.savepath, exist_ok=True)

    # Training history (per-epoch mean losses)
    training_history = {
        'config': {
            'dataset': args.dataset,
            'horizon': args.horizon,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_epochs': args.n_epochs,
            'sigma_train': args.sigma_train,
            'k': args.k,
        },
        'epoch_losses': [],
        'epoch_p_losses': [],
        'epoch_vel_losses': [],
        'epoch_act_losses': [],
    }
    history_path = os.path.join(args.savepath, 'training_history.json')

    def save_history():
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

    policy.train()
    global_step = 0
    total_epochs = args.n_epochs
    with tqdm(range(total_epochs), desc='Epoch') as epoch_bar:
        for epoch_idx in epoch_bar:
            epoch_losses = []
            epoch_p_losses = []
            epoch_vel_losses = []
            epoch_act_losses = []
            with tqdm(dataloader, desc='Batch', leave=False) as batch_bar:
                for batch in batch_bar:
                    loss, p_loss, vel_loss, act_loss = policy.Loss(batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update_model_average(ema_velocity, policy.velocity_net)

                    global_step += 1

                    # Sync to CPU only at log intervals to avoid GPU stalls
                    if global_step % 10 == 0:
                        loss_val = loss.item()
                        p_loss_val = p_loss.item()
                        vel_loss_val = vel_loss.item()
                        act_val = act_loss.item()
                        epoch_losses.append(loss_val)
                        epoch_p_losses.append(p_loss_val)
                        epoch_vel_losses.append(vel_loss_val)
                        epoch_act_losses.append(act_val)

                        batch_bar.set_postfix({
                            "loss": f"{loss_val:.4f}",
                            "p": f"{p_loss_val:.4f}",
                            "vel": f"{vel_loss_val:.4f}",
                            "act": f"{act_val:.4f}",
                        })

            if epoch_losses:
                mean_loss = float(np.mean(epoch_losses))
                mean_p_loss = float(np.mean(epoch_p_losses))
                mean_vel_loss = float(np.mean(epoch_vel_losses))
                mean_act = float(np.mean(epoch_act_losses))

                training_history['epoch_losses'].append(mean_loss)
                training_history['epoch_p_losses'].append(mean_p_loss)
                training_history['epoch_vel_losses'].append(mean_vel_loss)
                training_history['epoch_act_losses'].append(mean_act)

                epoch_bar.set_postfix({
                    "loss": mean_loss, "p": mean_p_loss,
                    "vel": mean_vel_loss, "act": mean_act,
                })

    policy.velocity_net.load_state_dict(ema_velocity.state_dict())

    # Save final training history
    save_history()
    print(f"[ train_sfp ] Saved final training history to {history_path}")

    model_path = os.path.join(args.savepath, args.model_filename)
    save_checkpoint(model_path, epoch_label=total_epochs)
    print(f"[ train_sfp ] Saved SFP velocity model to {model_path}")


if __name__ == '__main__':

    class Parser(utils.Parser):
        dataset: str = 'hopper-medium-expert-v2'
        config: str = 'config.locomotion'
        method: str = 'sfp'

    args = Parser().parse_args('train')
    train_sfp(args)
