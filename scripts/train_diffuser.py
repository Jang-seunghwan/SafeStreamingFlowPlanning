"""
Train the Diffuser model (GaussianDiffusion) used by the Diffuser, Diffuser+CG and
SafeDiffuser rows. Defaults are the settings of the paper models: K = 512 diffusion
steps, 100k steps, batch 64, lr 2e-4, data with velocity in m per step.

Usage:
    python scripts/train_diffuser.py --track budapest
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

from diffuser.models.temporal import TemporalUnet
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.datasets.f1tenth import F1tenthGoalDataset
from diffuser.utils.training import EMA
from config.f1tenth import CHECKPOINT_ROOT, TRACKS, get_horizon, get_processed_path

VELOCITY_UNITS = 'm_per_step'


def train(args):
    horizon = get_horizon(args.track)
    output_dir = args.output_dir or os.path.join(CHECKPOINT_ROOT, 'diffuser', args.track)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = F1tenthGoalDataset(get_processed_path(args.track, VELOCITY_UNITS), horizon)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    observation_dim = dataset.observation_dim  # 4
    action_dim = dataset.action_dim  # 2
    transition_dim = observation_dim + action_dim  # 6

    model = TemporalUnet(
        horizon=horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim=32,
        dim_mults=tuple(args.dim_mults),
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        action_weight=args.action_weight,
    ).to(device)

    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(diffusion).to(device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[ train_diffuser ] Track: {args.track}, H={horizon}, model params: {n_params:,}')

    os.makedirs(output_dir, exist_ok=True)

    # Training loop — cycle DataLoader until n_train_steps reached
    steps_per_epoch = len(dataloader)
    n_epochs = (args.n_train_steps + steps_per_epoch - 1) // steps_per_epoch
    global_step = 0
    history = {'losses': [], 'config': vars(args)}
    print(f'[ train_diffuser ] {steps_per_epoch} batches/epoch × {n_epochs} epochs → {args.n_train_steps} steps')

    for epoch in range(n_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False)
        for batch in pbar:
            trajectories = batch.trajectories.float().to(device)
            conditions = {k: v.float().to(device) for k, v in batch.conditions.items()}

            loss = diffusion.loss(trajectories, conditions)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ema.update_model_average(ema_model, diffusion)

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if global_step >= args.n_train_steps:
                break

            if global_step % args.save_freq == 0:
                save_checkpoint(diffusion, ema_model, dataset, global_step, args, horizon, output_dir)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0
        history['losses'].append(mean_loss)
        print(f'Epoch {epoch+1}/{n_epochs} | loss: {mean_loss:.6f} | step: {global_step}')

        if global_step >= args.n_train_steps:
            break

    save_checkpoint(diffusion, ema_model, dataset, global_step, args, horizon, output_dir, final=True)

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[ train_diffuser ] Done! Saved to {output_dir}')


def save_checkpoint(model, ema_model, dataset, step, args, horizon, output_dir, final=False):
    suffix = 'final' if final else f'step_{step}'
    path = os.path.join(output_dir, f'checkpoint_{suffix}.pt')
    # Mean start velocity of the training data (start state of the evaluation pairs)
    start_vels = dataset.observations[:, 0, 2:4]
    mean_start_vel = start_vels.mean(axis=0).tolist()

    torch.save({
        'step': step,
        'model': model.state_dict(),
        'ema': ema_model.state_dict(),
        'normalizer': dataset.normalizer,
        'config': {
            'horizon': horizon,
            'n_diffusion_steps': args.n_diffusion_steps,
            'observation_dim': dataset.observation_dim,
            'action_dim': dataset.action_dim,
            'transition_dim': dataset.observation_dim + dataset.action_dim,
            'dim_mults': args.dim_mults,
            'start_vel': mean_start_vel,
        },
    }, path)
    print(f'  Saved checkpoint: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, required=True, choices=TRACKS)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint directory (default: checkpoints/diffuser/<track>)')
    parser.add_argument('--n_diffusion_steps', type=int, default=512)
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--action_weight', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--n_train_steps', type=int, default=100000)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    train(parser.parse_args())
