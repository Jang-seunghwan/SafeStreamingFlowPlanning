#!/usr/bin/env python3
"""Train the trajectory-level baselines from rollout CSV logs.

  --planner-type diffusion      -> Diffuser model (rows Diffuser, Diffuser + CG, SafeDiffuser)
  --planner-type flow_matching  -> FM model (rows FM, SafeFM, FlowMatcher, SafeFlowMatcher;
                                   FlowMatcher samples the FM model with its own sampler)

Defaults are the settings of the paper models (H = 512, 256 denoising steps).
Usage (from the repository root):
    python -m ssf_gazebo.train_diffuser_planner --planner-type diffusion
    python -m ssf_gazebo.train_diffuser_planner --planner-type flow_matching
"""
from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from ssf_gazebo.diffuser_model import (
    DiffuserConfig,
    FEATURE_COLUMNS,
    CONDITION_COLUMNS,
    SlidingWindowDiffuserDataset,
    TrajectoryNormalizer,
    build_denoiser,
    build_planner,
    save_checkpoint,
)
from ssf_gazebo.sfp_model import (
    load_raw_preprocessed,
    load_raw_rollouts_from_logs,
    save_raw_preprocessed,
)

# Paths are relative to the working directory (the repository root).
DEFAULT_CHECKPOINTS = {
    'diffusion': 'models_h512/diffuser_planner.pt',
    'flow_matching': 'models_h512/cfm_planner.pt',
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the Diffuser (diffusion) or FM (flow_matching) trajectory planner.'
    )
    parser.add_argument('--planner-type', default='diffusion', choices=['diffusion', 'flow_matching'])
    parser.add_argument('--logs-dir', default='logs', help='Directory containing rollout_*.csv files.')
    parser.add_argument('--checkpoint', default=None,
                        help='Output checkpoint (default: models_h512/diffuser_planner.pt or '
                             'models_h512/cfm_planner.pt). The best-validation checkpoint is '
                             'written next to it with a _best suffix.')
    parser.add_argument('--dataset-cache', default='auto',
                        help="Raw-rollout .npz cache shared with SFP training. 'auto' = "
                             "cache/sfp_raw_all_min{min_episode_length}.npz (built on first use).")
    parser.add_argument('--horizon', type=int, default=512,
                        help='Trajectory horizon (sliding-window segment length).')
    parser.add_argument('--diffusion-steps', type=int, default=256,
                        help='Denoising / flow integration steps (and FM time scale).')
    parser.add_argument('--flow-matcher-alpha', type=float, default=2.0,
                        help='Correction velocity scale of the FlowMatcher sampler (stored in the config).')
    parser.add_argument('--dim', type=int, default=32, help='TemporalUnet base width.')
    parser.add_argument('--dim-mults', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps-per-epoch', type=int, default=1000,
                        help='Optimizer steps per epoch (0 = full pass over the dataset).')
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--ema-decay', type=float, default=0.995)
    parser.add_argument('--validation-split', type=float, default=0.05)
    parser.add_argument('--min-episode-length', type=int, default=50,
                        help='Drop CSV rollouts with fewer than this many rows.')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader worker count.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save an EMA snapshot (and the pending best checkpoint) every N epochs.')
    return parser.parse_args()


def _select_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is false.')
    return torch.device(device_arg)


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


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _clone_state_dict(planner: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in planner.state_dict().items()}


def _save_checkpoint_state(
    checkpoint_path: str,
    model_state_dict: Dict[str, torch.Tensor],
    normalizer: TrajectoryNormalizer,
    config: DiffuserConfig,
    metadata: Dict[str, object],
) -> None:
    checkpoint_path = os.path.expanduser(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
    torch.save(
        {
            'model_state_dict': model_state_dict,
            'config': asdict(config),
            'planner_type': config.planner_type,
            'normalizer': normalizer.to_dict(),
            'metadata': metadata,
        },
        checkpoint_path,
    )


def _ema_update(target: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


def _run_epoch(
    planner: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    ema_planner: Optional[torch.nn.Module] = None,
    ema_decay: float = 0.995,
) -> float:
    planner.train()
    total_loss = 0.0
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = planner.p_losses(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=1.0)
        optimizer.step()
        if ema_planner is not None:
            _ema_update(ema_planner, planner, ema_decay)

        batch_size = int(batch['trajectory'].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        count += batch_size
        if steps_per_epoch > 0 and step >= steps_per_epoch:
            break
    return total_loss / max(count, 1)


@torch.no_grad()
def _evaluate(planner: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    planner.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        loss = planner.p_losses(batch)
        batch_size = int(batch['trajectory'].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return total_loss / max(count, 1)


def main() -> None:
    args = _parse_args()
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINTS[args.planner_type]
    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError('--validation-split must be >= 0.0 and < 1.0.')
    _set_seed(args.seed)
    device = _select_device(args.device)

    config = DiffuserConfig(
        horizon=args.horizon,
        transition_dim=len(FEATURE_COLUMNS),
        diffusion_steps=args.diffusion_steps,
        planner_type=args.planner_type,
        flow_matcher_alpha=args.flow_matcher_alpha,
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        kernel_size=args.kernel_size,
    )

    # Raw-rollout cache (shared with SFP): Diffuser / FM see the same raw 20 Hz
    # trajectories as SFP, so all planners have the same plan duration H * dt.
    cache_path = _resolve_cache_path(args)
    if os.path.exists(cache_path):
        print(f'Loading preprocessed raw cache: {cache_path}', flush=True)
        observations, path_lengths, goals, stats, _, dt_sim = load_raw_preprocessed(cache_path)
    else:
        print(f'Preprocessed raw cache not found, building: {cache_path}', flush=True)
        observations, path_lengths, goals, stats, dt_sim = load_raw_rollouts_from_logs(
            logs_dir=args.logs_dir,
            min_episode_length=args.min_episode_length,
            progress_interval=500,
            progress_fn=lambda message: print(message, flush=True),
        )
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
        print(f'Saved preprocessed raw cache: {cache_path}', flush=True)

    print(
        f'Loaded {stats.rollout_count} rollouts (skipped {stats.skipped_count}). '
        f'lengths: min={int(path_lengths.min())} max={int(path_lengths.max())} '
        f'mean={float(path_lengths.mean()):.1f} | dt_sim={dt_sim:.6f}s '
        f'(plan duration at H={config.horizon}: {(config.horizon-1)*dt_sim:.1f}s)',
        flush=True,
    )

    # Normalizer fit on observed (non-padding) timesteps only
    flat_obs = np.concatenate(
        [observations[i, :path_lengths[i]] for i in range(len(path_lengths))],
        axis=0,
    )
    normalizer = TrajectoryNormalizer.from_trajectories(flat_obs)

    dataset = SlidingWindowDiffuserDataset(
        observations=observations,
        path_lengths=path_lengths,
        normalizer=normalizer,
        horizon=config.horizon,
    )
    print(f'Sliding-window segments: {len(dataset)} (horizon={config.horizon})', flush=True)

    val_count = 0
    if args.validation_split > 0.0:
        val_count = int(round(len(dataset) * args.validation_split))
    if val_count > 0 and len(dataset) > 1:
        val_count = min(max(val_count, 1), len(dataset) - 1)
    elif len(dataset) <= 1:
        val_count = 0
    train_count = len(dataset) - val_count

    generator = torch.Generator().manual_seed(args.seed)
    if val_count > 0:
        train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
        )

    model = build_denoiser(config)
    planner = build_planner(model, config).to(device)
    # Shadow planner for EMA weights (matches upstream maze2d ema_decay=0.995)
    ema_planner = copy.deepcopy(planner).to(device)
    for p in ema_planner.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    checkpoint_path = os.path.expanduser(args.checkpoint)
    root, ext = os.path.splitext(checkpoint_path)
    best_checkpoint_path = f'{root}_best{ext or ".pt"}'
    print(
        f'Training {args.planner_type} planner on {len(dataset)} segments '
        f'({train_count} train, {val_count} val), device={device}, checkpoint={checkpoint_path}',
        flush=True,
    )
    print(f'Features: {", ".join(FEATURE_COLUMNS)}', flush=True)
    print(f'Conditions: {", ".join(CONDITION_COLUMNS)}', flush=True)
    print(f'Normalizer mean={normalizer.mean.tolist()} std={normalizer.std.tolist()}', flush=True)

    metadata = {
        'args': vars(args),
        'config': asdict(config),
        'dataset': {
            'rollout_count': stats.rollout_count,
            'skipped_count': stats.skipped_count,
            'cache_path': cache_path,
        },
        'feature_columns': list(FEATURE_COLUMNS),
        'condition_columns': list(CONDITION_COLUMNS),
    }

    best_val = float('inf')
    best_checkpoint_state = None
    best_checkpoint_metadata = None
    best_checkpoint_epoch = None
    best_checkpoint_pending = False
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            planner=planner,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            steps_per_epoch=args.steps_per_epoch,
            ema_planner=ema_planner,
            ema_decay=args.ema_decay,
        )
        if val_loader is not None:
            val_loss = _evaluate(ema_planner, val_loader, device)
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            print(f'epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}', flush=True)
        else:
            val_loss = None
            improved = epoch == 1
            print(f'epoch={epoch:03d} train_loss={train_loss:.6f}', flush=True)

        metadata['last_epoch'] = epoch
        metadata['last_train_loss'] = train_loss
        metadata['last_val_loss'] = val_loss
        metadata['best_val_loss'] = best_val if val_loader is not None else None

        if improved:
            best_checkpoint_state = _clone_state_dict(ema_planner)
            best_checkpoint_metadata = copy.deepcopy(metadata)
            best_checkpoint_epoch = epoch
            best_checkpoint_pending = True

        if args.save_every > 0 and epoch % args.save_every == 0:
            periodic_path = checkpoint_path.replace('.pt', f'_epoch_{epoch:03d}.pt')
            # Periodic snapshots use EMA weights too
            save_checkpoint(periodic_path, ema_planner, normalizer, config, metadata)
            print(f'Saved periodic checkpoint (EMA): {periodic_path}', flush=True)

            if best_checkpoint_pending:
                _save_checkpoint_state(
                    best_checkpoint_path,
                    best_checkpoint_state,
                    normalizer,
                    config,
                    best_checkpoint_metadata,
                )
                print(
                    f'Saved best checkpoint: {best_checkpoint_path} '
                    f'(epoch={best_checkpoint_epoch:03d})',
                    flush=True,
                )
                best_checkpoint_pending = False

    if best_checkpoint_pending:
        _save_checkpoint_state(
            best_checkpoint_path,
            best_checkpoint_state,
            normalizer,
            config,
            best_checkpoint_metadata,
        )
        print(
            f'Saved best checkpoint: {best_checkpoint_path} '
            f'(epoch={best_checkpoint_epoch:03d})',
            flush=True,
        )

    # Copy EMA weights into the main planner for the final save (matches upstream)
    planner.load_state_dict(ema_planner.state_dict())
    metadata['ema_applied'] = True
    save_checkpoint(checkpoint_path, planner, normalizer, config, metadata)
    print(f'Saved final EMA-applied checkpoint: {checkpoint_path}', flush=True)


if __name__ == '__main__':
    main()
