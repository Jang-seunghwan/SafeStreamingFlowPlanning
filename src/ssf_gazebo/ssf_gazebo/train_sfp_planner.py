#!/usr/bin/env python3
"""Train a Streaming Flow Policy (SFP-Deterministic) from rollout CSV logs.

One model serves the rows StreamingFlow (open-loop / closed-loop) and SSF
(open-loop / closed-loop); SSF adds the CBF filter at inference only.

Faithful port of scripts/train_sfp.py (main / Maze2D branch) adapted to the
Gazebo rollout CSV pipeline:

  - Variable-length rollouts (no resampling), sliding-window stride-1 segments,
    episodes shorter than --horizon dropped.  Mirrors
    diffuser/datasets/gazebo.py:GazeboSequenceDataset.
  - make_sfp_collate_fn(sigma, k) tube-perturbs (x, v, x_label, t) per
    upstream train_sfp.py.
  - Two-group AdamW (pos_net 5x lr, vel_net 3x lr), EMA on velocity_net.
  - No gradient clipping (matches upstream).

Defaults are the settings of the paper model (H = 512).
Usage (from the repository root):
    python -m ssf_gazebo.train_sfp_planner
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from ssf_gazebo.diffuser_model import TrajectoryNormalizer
from ssf_gazebo.sfp_model import (
    SFPConfig,
    SFPRolloutDataset,
    StreamingFlowPolicyDeterministic,
    build_sfp,
    load_raw_preprocessed,
    load_raw_rollouts_from_logs,
    make_sfp_collate_fn,
    save_raw_preprocessed,
    save_sfp_checkpoint,
)


# Paths are relative to the working directory (the repository root).
DEFAULT_CHECKPOINT = 'models_h512/sfp_planner.pt'
HISTORY_FILENAME = 'training_history.json'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a Streaming Flow Policy (SFP) planner from rollout CSV logs.'
    )
    parser.add_argument('--logs-dir', default='logs')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='Rolling / final (EMA) checkpoint; the best-validation checkpoint '
                             'is written next to it with a _best suffix.')
    parser.add_argument('--dataset-cache', default='auto',
                        help="Raw-rollout .npz cache shared with Diffuser/FM training. "
                             "'auto' = cache/sfp_raw_all_min{min_episode_length}.npz (built on first use).")
    parser.add_argument('--horizon', type=int, default=512,
                        help='Sliding-window segment length. Episodes shorter than this are dropped.')
    parser.add_argument('--min-episode-length', type=int, default=50,
                        help='Drop CSV rollouts with fewer than this many rows.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps-per-epoch', type=int, default=1500,
                        help='Optimizer steps per epoch (0 = full pass over the dataset).')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Base AdamW lr. pos_net uses 5x, vel_net uses 3x (matches upstream).')
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--ema-decay', type=float, default=0.995)
    parser.add_argument('--sigma-train', type=float, default=0.005,
                        help='σ₀ of the tube perturbation (upstream gazebo config).')
    parser.add_argument('--k', type=float, default=0.1,
                        help='Tube contraction rate σ_t = σ₀·e^{-k t}.')
    parser.add_argument('--samples-per-segment', type=int, default=8,
                        help='Random t samples drawn per trajectory per batch '
                             '(effective batch = batch_size * K).')
    parser.add_argument('--down-dims', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--vel-down-dims', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--diffusion-step-embed-dim', type=int, default=32)
    parser.add_argument('--n-groups', type=int, default=8)
    parser.add_argument('--validation-split', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--save-every', type=int, default=5,
                        help='Rolling checkpoint interval in epochs (overwrites --checkpoint).')
    parser.add_argument('--snapshot-every', type=int, default=20,
                        help='Additional epoch-numbered snapshot every N epochs (0 = none).')
    return parser.parse_args()


def _select_device(arg: str) -> torch.device:
    if arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')
    return torch.device(arg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_cache_path(args: argparse.Namespace) -> str:
    if args.dataset_cache != 'auto':
        return os.path.expanduser(args.dataset_cache)
    return f'cache/sfp_raw_all_min{args.min_episode_length}.npz'


def _ema_update(target: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _run_epoch(
    model: StreamingFlowPolicyDeterministic,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    ema_velocity: torch.nn.Module,
    ema_decay: float,
    steps_per_epoch: int,
) -> Dict[str, float]:
    model.train()
    total = {'loss': 0.0, 'p_loss': 0.0, 'vel_loss': 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, p_loss, vel_loss = model.Loss(batch)
        loss.backward()
        optimizer.step()
        _ema_update(ema_velocity, model.velocity_net, ema_decay)

        bs = int(batch['x'].shape[0])
        total['loss'] += float(loss.detach()) * bs
        total['p_loss'] += float(p_loss.detach()) * bs
        total['vel_loss'] += float(vel_loss.detach()) * bs
        count += bs

        if steps_per_epoch > 0 and step >= steps_per_epoch:
            break

    return {k: v / max(count, 1) for k, v in total.items()}


@torch.no_grad()
def _evaluate(
    model: StreamingFlowPolicyDeterministic,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = {'loss': 0.0, 'p_loss': 0.0, 'vel_loss': 0.0}
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        loss, p_loss, vel_loss = model.Loss(batch)
        bs = int(batch['x'].shape[0])
        total['loss'] += float(loss.detach()) * bs
        total['p_loss'] += float(p_loss.detach()) * bs
        total['vel_loss'] += float(vel_loss.detach()) * bs
        count += bs
    return {k: v / max(count, 1) for k, v in total.items()}


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError('--validation-split must be in [0.0, 1.0).')
    if args.samples_per_segment < 1:
        raise ValueError('--samples-per-segment must be >= 1.')
    _set_seed(args.seed)
    device = _select_device(args.device)

    # ---- Dataset loading (CSV → raw cache → sliding-window segments) ----
    cache_path = _resolve_cache_path(args)
    if os.path.exists(cache_path):
        print(f'Loading preprocessed raw cache: {cache_path}', flush=True)
        observations, path_lengths, goals, stats, _meta, dt_sim = load_raw_preprocessed(cache_path)
    else:
        print(f'Reading rollout CSVs from {os.path.expanduser(args.logs_dir)} ...', flush=True)
        observations, path_lengths, goals, stats, dt_sim = load_raw_rollouts_from_logs(
            logs_dir=args.logs_dir,
            min_episode_length=args.min_episode_length,
            progress_interval=500,
            progress_fn=lambda msg: print(f'  {msg}', flush=True),
        )
        print(f'Saving preprocessed raw cache to {cache_path}', flush=True)
        save_raw_preprocessed(
            cache_path=cache_path,
            observations=observations,
            path_lengths=path_lengths,
            goals=goals,
            stats=stats,
            logs_dir=args.logs_dir,
            min_episode_length=args.min_episode_length,
            dt_sim=dt_sim,
        )

    dt_str = (
        f'{dt_sim:.6f}s ({1.0 / dt_sim:.1f}Hz)' if dt_sim > 0 else 'N/A (no t column)'
    )
    print(
        f'Loaded {stats.rollout_count} rollouts (skipped {stats.skipped_count}). '
        f'lengths: min={int(path_lengths.min())} max={int(path_lengths.max())} '
        f'mean={float(path_lengths.mean()):.1f} | dt_sim={dt_str}',
        flush=True,
    )

    config = SFPConfig(
        horizon=args.horizon,
        sigma_train=args.sigma_train,
        k=args.k,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.down_dims),
        vel_down_dims=tuple(args.vel_down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
    )

    # ---- Normalizer fit on observed (non-padding) timesteps only ----
    flat_obs = np.concatenate(
        [observations[i, :path_lengths[i]] for i in range(len(path_lengths))],
        axis=0,
    )
    normalizer = TrajectoryNormalizer.from_trajectories(flat_obs)

    # ---- Dataset / DataLoader ----
    full_dataset = SFPRolloutDataset(
        observations=observations,
        path_lengths=path_lengths,
        normalizer=normalizer,
        horizon=args.horizon,
    )
    print(f'Sliding-window segments: {len(full_dataset)} (horizon={args.horizon})', flush=True)

    val_size = int(round(args.validation_split * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    if val_size > 0:
        train_set, val_set = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set, val_set = full_dataset, None

    collate_fn = make_sfp_collate_fn(
        sigma=args.sigma_train,
        k=args.k,
        samples_per_segment=args.samples_per_segment,
    )
    common_loader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        common_loader['prefetch_factor'] = 2
    train_loader = DataLoader(train_set, shuffle=True, **common_loader)
    val_loader = DataLoader(val_set, shuffle=False, **common_loader) if val_set is not None else None

    # ---- Model + EMA + optimizer (two-group, matches upstream) ----
    model = build_sfp(config).to(device)
    ema_velocity = copy.deepcopy(model.velocity_net).to(device)
    for p in ema_velocity.parameters():
        p.requires_grad_(False)

    vel_net_params = list(model.velocity_net.vel_net.parameters())
    vel_net_ids = {id(p) for p in vel_net_params}
    other_params = [p for p in model.velocity_net.parameters() if id(p) not in vel_net_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.learning_rate * 5},
        {'params': vel_net_params, 'lr': args.learning_rate * 3},
    ], weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'SFP model parameters: {n_params:,}')
    print(f'Train batches/epoch: {len(train_loader)} | Val batches: '
          f'{len(val_loader) if val_loader else 0}')
    print(f'samples_per_segment: {args.samples_per_segment}  '
          f'(effective batch = {args.batch_size * args.samples_per_segment} '
          f'(x_t, x_{{t+1}}, t) tuples per step)')

    # ---- Train loop ----
    checkpoint_path = os.path.expanduser(args.checkpoint)
    base, ext = os.path.splitext(checkpoint_path)
    best_path = f'{base}_best{ext or ".pt"}'
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)

    history = {
        'config': vars(args),
        'epochs': [],  # list of dicts: {epoch, train_*, val_*}
    }
    history_path = os.path.join(os.path.dirname(checkpoint_path) or '.', HISTORY_FILENAME)

    def _save_history():
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            model, train_loader, device, optimizer,
            ema_velocity=ema_velocity, ema_decay=args.ema_decay,
            steps_per_epoch=args.steps_per_epoch,
        )

        log = (
            f'epoch {epoch:04d} | '
            f'train loss={train_stats["loss"]:.5f} '
            f'p={train_stats["p_loss"]:.5f} '
            f'v={train_stats["vel_loss"]:.5f}'
        )
        epoch_record = {'epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()}}

        if val_loader is not None:
            val_stats = _evaluate(model, val_loader, device)
            log += (
                f' | val loss={val_stats["loss"]:.5f} '
                f'p={val_stats["p_loss"]:.5f} '
                f'v={val_stats["vel_loss"]:.5f}'
            )
            epoch_record.update({f'val_{k}': v for k, v in val_stats.items()})
            if val_stats['loss'] < best_val:
                best_val = val_stats['loss']
                save_sfp_checkpoint(
                    path=best_path, model=model, normalizer=normalizer, config=config,
                    extra={'epoch': epoch, 'val_loss': best_val},
                )
                log += f' | best -> {best_path}'

        print(log, flush=True)
        history['epochs'].append(epoch_record)
        _save_history()

        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_sfp_checkpoint(
                path=checkpoint_path, model=model, normalizer=normalizer, config=config,
                extra={'epoch': epoch},
            )

        if args.snapshot_every > 0 and epoch % args.snapshot_every == 0:
            snapshot_path = checkpoint_path.replace('.pt', f'_epoch_{epoch:04d}.pt')
            save_sfp_checkpoint(
                path=snapshot_path, model=model, normalizer=normalizer, config=config,
                extra={'epoch': epoch},
            )
            print(f'   snapshot -> {snapshot_path}', flush=True)

    # Final save: copy EMA weights into model first (matches upstream)
    model.velocity_net.load_state_dict(ema_velocity.state_dict())
    save_sfp_checkpoint(
        path=checkpoint_path, model=model, normalizer=normalizer, config=config,
        extra={'epoch': args.epochs, 'ema_applied': True},
    )
    print(f'Final EMA-applied SFP checkpoint saved to {checkpoint_path}', flush=True)
    print(f'Training history saved to {history_path}', flush=True)


if __name__ == '__main__':
    main()
