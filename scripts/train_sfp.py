"""
Train the streaming flow policy (SFP) used by the StreamingFlow (sfp_off) and SSF
(safe_sfp_off) rows; the safety filter is applied only at evaluation. Defaults are
the settings of the paper models: time scale 256, 2000 epochs, batch 256, lr 1e-4,
k = 0.1, sigma_train = 0.005, data with velocity in m/s.

Usage:
    python scripts/train_sfp.py --track budapest
"""
import os
import sys
import copy
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diffuser.models.cond_unet1D import ConditionalUnet1D
from diffuser.models.sfpd import StreamingFlowPolicyDeterministic
from diffuser.datasets.f1tenth import F1tenthSequenceDataset
from diffuser.utils.training import EMA
from config.f1tenth import CHECKPOINT_ROOT, TRACKS, get_horizon, get_processed_path

VELOCITY_UNITS = 'm_per_s'


def make_collate_fn(sigma, k):
    """Sample one segment (x_i, x_{i+1}) per trajectory at flow time t = i / (H - 1)
    and build the stabilized flow target: positions are perturbed by
    eps ~ N(0, (sigma e^{-k t})^2) and the target velocity is x_dot - k * eps."""
    sigma = float(sigma)
    k = float(k)

    def collate(samples):
        trajectories = np.stack(
            [np.asarray(s.trajectories, dtype=np.float32) for s in samples], axis=0
        )
        batch_size, horizon, state_dim = trajectories.shape
        pos_dim = state_dim // 2
        cond = np.stack([trajectories[:, 0], trajectories[:, -1]], axis=1)

        seg_idx = np.random.randint(0, horizon - 1, size=batch_size, dtype=np.int64)
        t = (seg_idx / (horizon - 1)).astype(np.float32)

        batch_idx = np.arange(batch_size)
        x = trajectories[batch_idx, seg_idx]
        x_next = trajectories[batch_idx, seg_idx + 1]

        dt = 1.0 / (horizon - 1)
        x_dot = (x_next - x) / dt

        exp_term = np.exp(-k * t).astype(np.float32)
        eps_pos = sigma * exp_term[:, None] * np.random.randn(batch_size, pos_dim).astype(np.float32)
        eps_vel = np.zeros((batch_size, pos_dim), dtype=np.float32)

        x_train = x + np.concatenate([eps_pos, eps_vel], axis=-1)
        v_train = x_dot - k * np.concatenate([eps_pos, eps_vel], axis=-1)
        x_label = x_train + v_train * dt

        return {
            'cond': torch.from_numpy(cond),
            'x': torch.from_numpy(x_train[:, None, :]),
            'x_label': torch.from_numpy(x_label[:, None, :]),
            'v': torch.from_numpy(v_train[:, None, :]),
            't': torch.from_numpy(t),
        }

    return collate


def train(args):
    horizon = get_horizon(args.track)
    output_dir = args.output_dir or os.path.join(CHECKPOINT_ROOT, 'sfp', args.track)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = F1tenthSequenceDataset(get_processed_path(args.track, VELOCITY_UNITS), horizon)

    state_dim = 4
    velocity_net = ConditionalUnet1D(
        input_dim=state_dim,
        horizon=horizon,
        time_scale=args.time_scale,
    ).to(device)

    class PolicyArgs:  # safety filter is not used during training
        safety_enabled = False

    policy = StreamingFlowPolicyDeterministic(
        velocity_net=velocity_net,
        device=device,
        normalizer=dataset.normalizer,
        args=PolicyArgs(),
    ).to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(sigma=args.sigma_train, k=args.k),
    )

    ema = EMA(args.ema_decay)
    ema_velocity = copy.deepcopy(velocity_net).to(device)

    # Optimizer with separate LR for pos_net and vel_net
    vel_net_params = list(velocity_net.vel_net.parameters())
    vel_net_ids = {id(p) for p in vel_net_params}
    other_params = [p for p in velocity_net.parameters() if id(p) not in vel_net_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.learning_rate * 5},
        {'params': vel_net_params, 'lr': args.learning_rate * 3},
    ], weight_decay=args.weight_decay)

    os.makedirs(output_dir, exist_ok=True)

    history = {
        'config': vars(args),
        'epoch_losses': [],
        'epoch_p_losses': [],
        'epoch_vel_losses': [],
    }

    steps_per_epoch = len(dataloader)
    total_steps = args.n_epochs * steps_per_epoch
    print(f'[ train_sfp ] Track: {args.track}, H={horizon}; {steps_per_epoch} batches/epoch × '
          f'{args.n_epochs} epochs = {total_steps} gradient steps')

    policy.train()
    global_step = 0

    with tqdm(range(args.n_epochs), desc='Epoch') as epoch_bar:
        for epoch_idx in epoch_bar:
            epoch_losses, epoch_p, epoch_v = [], [], []
            for batch in tqdm(dataloader, desc='Batch', leave=False):
                loss, p_loss, vel_loss = policy.Loss(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update_model_average(ema_velocity, velocity_net)

                global_step += 1
                if global_step % 10 == 0:
                    epoch_losses.append(loss.item())
                    epoch_p.append(p_loss.item())
                    epoch_v.append(vel_loss.item())

            if epoch_losses:
                ml, mp, mv = np.mean(epoch_losses), np.mean(epoch_p), np.mean(epoch_v)
                history['epoch_losses'].append(float(ml))
                history['epoch_p_losses'].append(float(mp))
                history['epoch_vel_losses'].append(float(mv))
                epoch_bar.set_postfix({'loss': f'{ml:.4f}', 'p': f'{mp:.4f}', 'v': f'{mv:.4f}'})

    # Load EMA weights
    velocity_net.load_state_dict(ema_velocity.state_dict())

    # Save
    ckpt_path = os.path.join(output_dir, 'sfp_velocity_policy.pt')
    # Mean start velocity of the training data (start state of the evaluation pairs)
    start_vels = dataset.observations[:, 0, 2:4]
    mean_start_vel = start_vels.mean(axis=0).tolist()

    torch.save({
        'epoch': 'final',
        'velocity_state_dict': velocity_net.state_dict(),
        'config': {
            'horizon': horizon,
            'sigma_train': args.sigma_train,
            'k': args.k,
            'state_dim': state_dim,
            'start_vel': mean_start_vel,
            'time_scale': args.time_scale,
        },
        'normalizer': dataset.normalizer,
    }, ckpt_path)
    print(f'[ train_sfp ] Saved to {ckpt_path}')

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, required=True, choices=TRACKS)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint directory (default: checkpoints/sfp/<track>)')
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--sigma_train', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time_scale', type=float, default=256.0,
                        help='Factor applied to t in [0, 1] before the sinusoidal time embedding.')
    train(parser.parse_args())
