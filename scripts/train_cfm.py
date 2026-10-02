"""
Train the Conditional Flow Matching (CFM) model. One model serves FM (Euler
sampler) and FlowMatcher (predictor-corrector sampler), with and without the
safety filter. Defaults are the settings of the paper models: time scale 256,
K = 256 at training (evaluation samples with K = 512), 100k steps, batch 64,
lr 2e-4, data with velocity in m per step.

Paper models:
    python scripts/train_cfm.py --track budapest     # FM, FlowMatcher, SafeFM, SafeFlowMatcher (Budapest)
    python scripts/train_cfm.py --track catalunya    # FM, FlowMatcher (Catalunya)
    python scripts/train_cfm.py --track catalunya --velocity_units m_per_s
                                                     # SafeFM, SafeFlowMatcher (Catalunya)
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
from diffuser.models.cfm import CFM
from diffuser.datasets.f1tenth import F1tenthGoalDataset
from diffuser.utils.training import EMA
from config.f1tenth import (
    CHECKPOINT_ROOT, TRACKS, VELOCITY_UNITS, get_horizon, get_processed_path,
)


def train(args):
    horizon = get_horizon(args.track)
    family = 'cfm' if args.velocity_units == 'm_per_step' else 'cfm_m_per_s'
    output_dir = args.output_dir or os.path.join(CHECKPOINT_ROOT, family, args.track)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = F1tenthGoalDataset(get_processed_path(args.track, args.velocity_units), horizon)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    transition_dim = observation_dim + action_dim

    model = TemporalUnet(
        horizon=horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim=32,
        dim_mults=tuple(args.dim_mults),
        time_scale=args.time_scale,
    ).to(device)

    cfm = CFM(
        model=model,
        horizon=horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        action_weight=args.action_weight,
    ).to(device)

    # EMA — build fresh copy (NeuralODE breaks deepcopy)
    ema = EMA(args.ema_decay)
    ema_model = CFM(
        model=copy.deepcopy(model),
        horizon=horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        action_weight=args.action_weight,
    ).to(device)

    optimizer = torch.optim.Adam(cfm.parameters(), lr=args.learning_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[ train_cfm ] Track: {args.track}, H={horizon}, velocity data: {args.velocity_units}, '
          f'model params: {n_params:,}')

    os.makedirs(output_dir, exist_ok=True)

    # Training loop — cycle DataLoader until n_train_steps reached
    steps_per_epoch = len(dataloader)
    n_epochs = (args.n_train_steps + steps_per_epoch - 1) // steps_per_epoch
    global_step = 0
    history = {'losses': [], 'config': vars(args)}
    print(f'[ train_cfm ] {steps_per_epoch} batches/epoch × {n_epochs} epochs → {args.n_train_steps} steps')

    for epoch in range(n_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False)
        for batch in pbar:
            trajectories = batch.trajectories.float().to(device)
            conditions = {k: v.float().to(device) for k, v in batch.conditions.items()}

            loss = cfm.loss(trajectories, conditions)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ema.update_model_average(ema_model, cfm)

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if global_step >= args.n_train_steps:
                break

            if global_step % args.save_freq == 0:
                save_checkpoint(cfm, ema_model, dataset, global_step, args, horizon, output_dir)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0
        history['losses'].append(mean_loss)
        print(f'Epoch {epoch+1}/{n_epochs} | loss: {mean_loss:.6f} | step: {global_step}')

        if global_step >= args.n_train_steps:
            break

    save_checkpoint(cfm, ema_model, dataset, global_step, args, horizon, output_dir, final=True)

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[ train_cfm ] Done! Saved to {output_dir}')


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
            'time_scale': args.time_scale,
        },
    }, path)
    print(f'  Saved checkpoint: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, required=True, choices=TRACKS)
    parser.add_argument('--velocity_units', type=str, default='m_per_step', choices=VELOCITY_UNITS,
                        help='Training data version (see scripts/regenerate_processed_data.py)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint directory (default: checkpoints/cfm/<track>, or '
                             'checkpoints/cfm_m_per_s/<track> with --velocity_units m_per_s)')
    parser.add_argument('--n_diffusion_steps', type=int, default=256)
    parser.add_argument('--time_scale', type=float, default=256.0,
                        help='Factor applied to t in [0, 1] before the sinusoidal time embedding.')
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
