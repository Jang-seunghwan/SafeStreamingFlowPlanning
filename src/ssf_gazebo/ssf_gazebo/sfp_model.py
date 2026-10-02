#!/usr/bin/env python3
"""Streaming Flow Policy (SFP) for ssf_gazebo.

Faithful port of ("upstream" = the main / Maze2D branch of this repository):
  - diffuser/models/cond_unet1D.py (ConditionalUnet1D + UNet core)
  - diffuser/models/sfpd.py        (StreamingFlowPolicyDeterministic)
  - scripts/train_sfp.py:make_collate_fn  (tube-perturbed sampling)
  - diffuser/datasets/gazebo.py    (sliding-window CSV dataset)

Intentional deltas from upstream:
  1. No pydrake / torchdyn / CBF here (explicit Euler rollout); the safety
     filter lives in cbf.py / safe_planners.py.
  2. Sinusoidal time embedding receives t * horizon (default) instead of raw
     t ∈ [0, 1], matching the diffuser/CFM convention in diffuser_model.py.
     Without this scaling the high-frequency sinusoidal components are idle
     because t never moves them far enough.
  3. No dynamics-residual base (upstream vel_weight=[] for Gazebo): vel_net
     predicts v_vel directly. The rollout CSVs are recorded at ~20 Hz with
     position updates interpolated by scripts/clean_stutter.py.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from ssf_gazebo.diffuser_model import (
    FEATURE_COLUMNS,
    CONDITION_COLUMNS,
    DatasetStats,
    TrajectoryNormalizer,
    _read_rollout_csv,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SFPConfig:
    """Hyperparameters for the SFP velocity field model (paper: H = 512)."""
    horizon: int = 512                                # 2x upstream maze2d
    state_dim: int = len(FEATURE_COLUMNS)             # (x, y, vx, vy) = 4
    num_cond: int = 2                                 # condition on [start, goal]
    pos_dim: int = 2
    sigma_train: float = 0.005                        # gazebo default (config/gazebo.py)
    k: float = 0.1                                    # gazebo default
    diffusion_step_embed_dim: int = 32
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    vel_down_dims: Tuple[int, ...] = (64, 128)
    kernel_size: int = 5
    n_groups: int = 8
    fc_timesteps: int = 1                             # Linear1d up/downsampling on T=1 (upstream default)


# ---------------------------------------------------------------------------
# UNet1D building blocks — port of cond_unet1D.py modules
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Linear1d(nn.Module):
    """Channel-wise Linear over flattened (B, C*T), used as up/downsampling."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, T = x.size()
        x = x.view(B, -1)
        x = self.linear(x)
        x = x.view(B, C, T)
        return x


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish (matches upstream)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Additive conditioning (time_mlp), matches upstream cond_unet1D.py.

    Note: this is *not* FiLM. The upstream upstream comment is explicit:
    "Additive conditioning (time_mlp), NOT FiLM."
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_ch
        self.out_channels = out_ch
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class _Unet1DCore(nn.Module):
    """1D UNet core. Matches upstream cond_unet1D.py:_Unet1DCore.

    - down_modules: len(down_dims) levels, each (resnet, resnet, down).
    - up_modules:   len(down_dims)-1 levels (input-side skip h[0] left unused).
    - down/upsample == Linear1d(fc_timesteps * C).
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        diffusion_step_embed_dim: int,
        down_dims: Sequence[int],
        kernel_size: int,
        n_groups: int,
        fc_timesteps: int,
    ):
        super().__init__()
        all_dims = [input_channels] + list(down_dims)
        start_dim = down_dims[0]

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=diffusion_step_embed_dim,
                kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=diffusion_step_embed_dim,
                kernel_size=kernel_size, n_groups=n_groups),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            downsample = Linear1d(fc_timesteps * dim_out)
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                downsample if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            upsample = Linear1d(fc_timesteps * dim_in)
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                upsample if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, output_channels, 1),
        )

        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, x: Tensor, global_feature: Tensor) -> Tensor:
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Two-headed velocity network (pos_net + vel_net) — port of cond_unet1D.py
# ---------------------------------------------------------------------------

class ConditionalUnet1D(nn.Module):
    """Two-headed velocity net for SFP.

    pos_net : [sample_pos + goal_cond]                  -> v_pos
    vel_net : [sample_pos + sample_vel + next_pos +
               goal_cond + t]                           -> v_vel

    Time handling: the sinusoidal embedding receives t * horizon.  This avoids the "all-frequencies-idle" problem you
    get when t lives in [0, 1] — sin(t * 10000^{i/H}) only covers a tiny phase
    range, wasting the embedding's high-frequency basis.  The raw [0, 1] t is
    still passed through `t_ch` into the vel_net input where it functions as a
    phase indicator, not a position code.
    """
    def __init__(
        self,
        input_dim: int,
        diffusion_step_embed_dim: int = 32,
        down_dims: Sequence[int] = (256, 512, 1024),
        vel_down_dims: Sequence[int] = (64, 128),
        kernel_size: int = 5,
        n_groups: int = 8,
        fc_timesteps: int = 1,
        horizon: int = 256,
    ):
        super().__init__()
        self.state_dim = input_dim
        self.pos_dim = input_dim // 2
        self.vel_dim = input_dim - self.pos_dim
        self.cond_in_dim = self.pos_dim   # goal-pos-only conditioning
        self.horizon = horizon
        self.time_scale = float(horizon)

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # pos_net input: [sample_pos(pos_dim) + goal_cond(cond_in_dim)]
        pos_in_channels = self.pos_dim + self.cond_in_dim
        # vel_net input: sample_pos + sample_vel + next_pos + cond_feat + t
        vel_in_channels = self.pos_dim + self.vel_dim + self.pos_dim + self.cond_in_dim + 1

        self.pos_net = _Unet1DCore(
            input_channels=pos_in_channels,
            output_channels=self.pos_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            fc_timesteps=fc_timesteps,
        )
        self.vel_net = _Unet1DCore(
            input_channels=vel_in_channels,
            output_channels=self.vel_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=list(vel_down_dims),
            kernel_size=kernel_size,
            n_groups=min(n_groups, vel_down_dims[0]),
            fc_timesteps=fc_timesteps,
        )

        print("[ ConditionalUnet1D ] number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())
        ))

    def forward(
        self,
        sample: Tensor,
        timestep: Union[Tensor, float, int],
        global_cond: Tensor,
        is_train: bool = False,
        label: Optional[Tensor] = None,
    ) -> Tensor:
        # (B, T, C) -> (B, C, T)
        sample = sample.moveaxis(-1, -2)

        # 1) timestep -> embedding (scaled to [0, time_scale])
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(sample.device, dtype=torch.float32).expand(sample.shape[0])

        t_scaled = timestep * self.time_scale
        global_feature = self.diffusion_step_encoder(t_scaled)

        # 2) goal-only conditioning (extract goal_pos from global_cond layout
        #    [start_state, goal_state])
        cond_feat = global_cond[:, self.state_dim : self.state_dim + self.pos_dim]

        cond_feat = cond_feat.unsqueeze(-1)
        if cond_feat.shape[-1] != sample.shape[-1]:
            cond_feat = cond_feat.expand(-1, -1, sample.shape[-1])

        # 3) pos_net: predict position velocity
        sample_pos = sample[:, :self.pos_dim, :]
        pos_in = torch.cat([sample_pos, cond_feat], dim=1)
        v_pos = self.pos_net(pos_in, global_feature)

        # 4) velocity flow: GT next_pos during training, prediction during inference
        dt = 1.0 / max(self.horizon - 1, 1)
        sample_vel = sample[:, self.pos_dim:, :]

        if is_train and label is not None:
            gt_next = label.moveaxis(-1, -2)
            next_pos = gt_next[:, :self.pos_dim, :]
        else:
            next_pos = sample_pos + v_pos.detach() * dt

        # t channel is the raw, unscaled [0,1] phase
        t_ch = timestep.view(-1, 1, 1).expand(-1, -1, sample.shape[-1])

        vel_in = torch.cat([sample_pos, sample_vel, next_pos, cond_feat, t_ch], dim=1)
        v_vel = self.vel_net(vel_in, global_feature)

        x = torch.cat([v_pos, v_vel], dim=1)
        return x.moveaxis(-1, -2)


# ---------------------------------------------------------------------------
# Streaming Flow Policy — port of sfpd.py (CBF/torchdyn deps removed)
# ---------------------------------------------------------------------------

class StreamingFlowPolicyDeterministic(nn.Module):
    """SFP-Deterministic with explicit Euler rollout.

    Training loss matches upstream sfpd.Loss (MSE on pos, smooth-L1 on vel).
    Rollout uses next_pos = sample_pos + v_pos * dt and
    next_vel = sample_vel + v_vel * dt.
    """

    def __init__(self, velocity_net: ConditionalUnet1D, config: SFPConfig):
        super().__init__()
        self.velocity_net = velocity_net
        self.config = config
        self.state_dim = config.state_dim
        # Buffers for ckpt portability (match upstream sfpd.py)
        self.register_buffer('num_cond_buf', torch.tensor(config.num_cond, dtype=torch.int32))
        self.register_buffer('sigma_buf', torch.tensor(config.sigma_train, dtype=torch.float32))
        self.register_buffer('k_buf', torch.tensor(config.k, dtype=torch.float32))

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (total, p_loss, vel_loss). Matches upstream sfpd.Loss."""
        device = next(self.parameters()).device

        cond = batch['cond'].to(device, non_blocking=True)
        cond = cond.clone()
        cond[:, 1, self.config.pos_dim:] = 0.0          # zero goal-velocity (upstream)

        x = batch['x'].to(device, non_blocking=True)
        x_label = batch['x_label'].to(device, non_blocking=True)
        v = batch['v'].to(device, non_blocking=True)
        t = batch['t'].to(device, non_blocking=True)

        cond_flat = cond.flatten(start_dim=1)
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=cond_flat,
            is_train=True, label=x_label,
        )

        pos = slice(0, self.config.pos_dim)
        vel = slice(self.config.pos_dim, self.config.state_dim)
        p_loss = nn.functional.mse_loss(v_pred[..., pos], v[..., pos])
        vel_loss = nn.functional.smooth_l1_loss(v_pred[..., vel], v[..., vel], beta=0.1)
        return p_loss + vel_loss, p_loss, vel_loss

    @torch.inference_mode()
    def rollout(
        self,
        start_state: Tensor,
        goal_state: Tensor,
        pred_horizon: int,
    ) -> Tensor:
        """Explicit Euler rollout. Mirrors upstream sfpd.__call__ inference path."""
        if pred_horizon < 2:
            raise ValueError('pred_horizon must be >= 2')

        device = next(self.velocity_net.parameters()).device
        start = start_state.reshape(-1).to(device)
        goal = goal_state.reshape(-1).to(device).clone()
        goal[self.config.pos_dim:] = 0.0
        ncond = torch.stack([start, goal], dim=0)
        cond_flat = ncond.unsqueeze(0).flatten(start_dim=1)

        t_span = torch.linspace(0, 1.0, pred_horizon, device=device, dtype=start.dtype)
        delta_ts = torch.diff(t_span)

        cur = start.unsqueeze(0).unsqueeze(0)       # (1, 1, S)
        traj = [cur.squeeze(0).squeeze(0).clone()]
        pos_dim = self.config.pos_dim

        for i in range(pred_horizon - 1):
            t = t_span[i].unsqueeze(0)
            v = self.velocity_net(sample=cur, timestep=t, global_cond=cond_flat)
            v = v.squeeze(0).squeeze(0)             # (S,)
            cur_flat = cur.squeeze(0).squeeze(0)    # (S,)
            dt_i = delta_ts[i]

            next_pos = cur_flat[:pos_dim] + v[:pos_dim] * dt_i
            next_vel = cur_flat[pos_dim:] + v[pos_dim:] * dt_i
            next_state = torch.cat([next_pos, next_vel], dim=-1)
            cur = next_state.unsqueeze(0).unsqueeze(0)
            traj.append(next_state.clone())

        return torch.stack(traj, dim=0)             # (H, S)


# ---------------------------------------------------------------------------
# Raw (variable-length) rollout loader for sliding-window SFP training
#
# Matches upstream diffuser/datasets/gazebo.py:GazeboSequenceDataset
# (episodes < horizon dropped, stride-1 sliding window).
# ---------------------------------------------------------------------------

def load_raw_rollouts_from_logs(
    logs_dir: str,
    min_episode_length: int = 50,
    progress_interval: int = 0,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetStats, float]:
    """Read rollout_*.csv files preserving their original length.

    Returns:
        observations: (N_episodes, max_T, state_dim) float32, zero-padded after path_length
        path_lengths: (N_episodes,) int64
        goals:        (N_episodes, 2) float32  (goal_x, goal_y)
        stats:        DatasetStats
        dt_sim:       float, mean dt between consecutive rows
                      (averaged over all loaded CSVs; reads the 't' column).
    """
    logs_dir = os.path.expanduser(logs_dir)
    files = sorted(glob.glob(os.path.join(logs_dir, 'rollout_*.csv')))

    trajectories: List[np.ndarray] = []
    goals: List[np.ndarray] = []
    used_files: List[str] = []
    dt_samples: List[float] = []
    skipped = 0

    total_files = len(files)
    for file_index, csv_path in enumerate(files, start=1):
        if progress_interval > 0 and progress_fn is not None:
            if file_index == 1 or file_index % progress_interval == 0 or file_index == total_files:
                progress_fn(f'preprocess {file_index}/{total_files}: {os.path.basename(csv_path)}')

        data = _read_rollout_csv(csv_path)
        if data is None or len(data) < min_episode_length:
            skipped += 1
            continue

        obs = np.stack([data[col] for col in FEATURE_COLUMNS], axis=-1).astype(np.float32)
        if not np.isfinite(obs).all():
            skipped += 1
            continue

        # mean dt for this CSV
        if 't' in data.dtype.names and len(data) >= 2:
            dt_arr = np.diff(data['t'].astype(np.float64))
            if dt_arr.size > 0 and np.isfinite(dt_arr).all() and dt_arr.mean() > 0:
                dt_samples.append(float(dt_arr.mean()))

        goal_xy = np.array([data['goal_x'][0], data['goal_y'][0]], dtype=np.float32)
        trajectories.append(obs)
        goals.append(goal_xy)
        used_files.append(csv_path)

    if not trajectories:
        raise FileNotFoundError(
            f'No usable rollouts in {logs_dir} (min_episode_length={min_episode_length})'
        )

    max_path_length = max(len(t) for t in trajectories)

    n_episodes = len(trajectories)
    state_dim = len(FEATURE_COLUMNS)
    observations = np.zeros((n_episodes, max_path_length, state_dim), dtype=np.float32)
    path_lengths = np.zeros(n_episodes, dtype=np.int64)
    for i, traj in enumerate(trajectories):
        T = min(len(traj), max_path_length)
        observations[i, :T] = traj[:T]
        path_lengths[i] = T

    goals_arr = np.stack(goals, axis=0).astype(np.float32)
    dt_sim = float(np.mean(dt_samples)) if dt_samples else 0.0

    stats = DatasetStats(
        rollout_count=n_episodes,
        skipped_count=skipped,
        source_files=used_files[:20],
    )
    return observations, path_lengths, goals_arr, stats, dt_sim


def save_raw_preprocessed(
    cache_path: str,
    observations: np.ndarray,
    path_lengths: np.ndarray,
    goals: np.ndarray,
    stats: DatasetStats,
    logs_dir: str,
    min_episode_length: int,
    dt_sim: float = 0.0,
) -> None:
    cache_path = os.path.expanduser(cache_path)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    metadata = {
        'logs_dir': os.path.expanduser(logs_dir),
        'min_episode_length': int(min_episode_length),
        'rollout_count': int(stats.rollout_count),
        'skipped_count': int(stats.skipped_count),
        'feature_columns': list(FEATURE_COLUMNS),
        'condition_columns': list(CONDITION_COLUMNS),
        'source_files_preview': stats.source_files,
        'dt_sim': float(dt_sim),
        'kind': 'raw',
    }
    np.savez(
        cache_path,
        observations=observations.astype(np.float32),
        path_lengths=path_lengths.astype(np.int64),
        goals=goals.astype(np.float32),
        metadata=np.array(json.dumps(metadata)),
    )


def load_raw_preprocessed(
    cache_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetStats, Dict[str, object], float]:
    cache_path = os.path.expanduser(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload['metadata']))
        if metadata.get('kind') != 'raw':
            raise ValueError(
                f"Cache at {cache_path} is not a raw-rollout cache (kind={metadata.get('kind')!r})"
            )
        observations = payload['observations'].astype(np.float32)
        path_lengths = payload['path_lengths'].astype(np.int64)
        goals = payload['goals'].astype(np.float32)
    stats = DatasetStats(
        rollout_count=int(metadata.get('rollout_count', len(observations))),
        skipped_count=int(metadata.get('skipped_count', 0)),
        source_files=list(metadata.get('source_files_preview', [])),
    )
    dt_sim = float(metadata.get('dt_sim', 0.0))
    return observations, path_lengths, goals, stats, metadata, dt_sim


# ---------------------------------------------------------------------------
# Sliding-window dataset — port of GazeboSequenceDataset / SequenceDataset
# ---------------------------------------------------------------------------

class SFPRolloutDataset(Dataset):
    """Sliding-window (stride=1) sampler over raw, variable-length rollouts.

    Each `__getitem__` returns one normalized (horizon, state_dim) segment.
    Episodes shorter than `horizon` are dropped entirely (no padding), matching
    GazeboSequenceDataset's behaviour in the upstream repo.
    """

    def __init__(
        self,
        observations: np.ndarray,         # (N, max_T, S) raw float32
        path_lengths: np.ndarray,         # (N,) int64
        normalizer: TrajectoryNormalizer,
        horizon: int,
    ):
        if observations.ndim != 3:
            raise ValueError(f'observations must be (N, max_T, state_dim), got {observations.shape}')
        self.horizon = int(horizon)

        # Normalize once (flatten then reshape) so __getitem__ is just a slice
        n, max_T, sdim = observations.shape
        flat = observations.reshape(-1, sdim)
        normed_flat = normalizer.normalize_trajectory(flat)
        self.observations = torch.from_numpy(normed_flat.reshape(n, max_T, sdim)).float()
        self.path_lengths = np.asarray(path_lengths, dtype=np.int64)

        indices = []
        for i, pl in enumerate(self.path_lengths):
            max_start = int(pl) - horizon
            if max_start < 0:
                continue
            for s in range(max_start + 1):
                indices.append((i, s, s + horizon))
        if not indices:
            raise ValueError(
                f"No valid sliding-window segments: all episodes shorter than horizon={horizon}"
            )
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tensor:
        i, s, e = self.indices[idx]
        return self.observations[int(i), int(s):int(e)]


# ---------------------------------------------------------------------------
# Collate — port of train_sfp.py:make_collate_fn (tube perturbation)
# ---------------------------------------------------------------------------

def make_sfp_collate_fn(
    sigma: float = 0.0,
    k: float = 0.0,
    samples_per_segment: int = 1,
):
    """Sample (x, x_label, v, t, cond) with tube noise on the position.

    Matches upstream scripts/train_sfp.py:make_collate_fn, plus
    an explicit `samples_per_segment` knob:

    Why samples_per_segment > 1:
      SFP labels are pairs of consecutive waypoints, i.e. one (x_t, x_{t+1})
      per segment per step.  A segment of length H carries H-1 such pairs but
      upstream only consumes one (random t) per training step.  Setting
      samples_per_segment=K replicates each segment K times and samples K
      different random t's, so each gradient step sees K different t values
      per trajectory — drastically increasing t-coverage without rereading
      data.  Effective batch becomes B*K; expect higher GPU memory.

    `sigma=0` disables the tube perturbation (deterministic mode).
    """
    sigma_f = float(sigma)
    k_f = float(k)
    K = max(1, int(samples_per_segment))

    def collate(batch: Sequence[Tensor]) -> Dict[str, Tensor]:
        trajs = torch.stack(list(batch), dim=0).numpy()             # (B, H, S)
        if trajs.shape[1] < 2:
            raise ValueError('Trajectories must contain at least two states.')

        if K > 1:
            trajs = np.repeat(trajs, K, axis=0)                     # (B*K, H, S)

        batch_size, horizon, state_dim = trajs.shape
        pos_dim = state_dim // 2
        cond = np.stack([trajs[:, 0], trajs[:, -1]], axis=1)        # (B*K, 2, S)

        seg_idx = np.random.randint(0, horizon - 1, size=batch_size, dtype=np.int64)
        t = (seg_idx / (horizon - 1)).astype(np.float32)

        batch_idx = np.arange(batch_size)
        x = trajs[batch_idx, seg_idx]
        x_next = trajs[batch_idx, seg_idx + 1]
        dt = 1.0 / (horizon - 1)
        x_dot = (x_next - x) / dt

        # σ_t = σ₀ · e^{-k t}; position-only perturbation (no velocity coupling)
        exp_term = np.exp(-k_f * t).astype(np.float32)
        eps_pos = sigma_f * exp_term[:, None] * np.random.randn(batch_size, pos_dim).astype(np.float32)
        eps_vel = np.zeros((batch_size, pos_dim), dtype=np.float32)

        sampled_error = np.concatenate([eps_pos, eps_vel], axis=-1)
        x_train = x + sampled_error
        v_train = x_dot - k_f * sampled_error
        x_label = x_train + v_train * dt

        return {
            'cond': torch.from_numpy(cond),
            'x':       torch.from_numpy(x_train[:, None, :]),
            'x_label': torch.from_numpy(x_label[:, None, :]),
            'v':       torch.from_numpy(v_train[:, None, :]),
            't':       torch.from_numpy(t),
        }

    return collate


# ---------------------------------------------------------------------------
# Builders & checkpoint utils
# ---------------------------------------------------------------------------

def build_sfp(config: SFPConfig) -> StreamingFlowPolicyDeterministic:
    velocity_net = ConditionalUnet1D(
        input_dim=config.state_dim,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        down_dims=tuple(config.down_dims),
        vel_down_dims=tuple(config.vel_down_dims),
        kernel_size=config.kernel_size,
        n_groups=config.n_groups,
        fc_timesteps=config.fc_timesteps,
        horizon=config.horizon,
    )
    return StreamingFlowPolicyDeterministic(velocity_net=velocity_net, config=config)


def save_sfp_checkpoint(
    path: str,
    model: StreamingFlowPolicyDeterministic,
    normalizer: TrajectoryNormalizer,
    config: SFPConfig,
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        'model_state_dict': model.state_dict(),
        'config': {
            'horizon': config.horizon,
            'state_dim': config.state_dim,
            'num_cond': config.num_cond,
            'pos_dim': config.pos_dim,
            'sigma_train': config.sigma_train,
            'k': config.k,
            'diffusion_step_embed_dim': config.diffusion_step_embed_dim,
            'down_dims': list(config.down_dims),
            'vel_down_dims': list(config.vel_down_dims),
            'kernel_size': config.kernel_size,
            'n_groups': config.n_groups,
            'fc_timesteps': config.fc_timesteps,
        },
        'normalizer': normalizer.to_dict(),
        'feature_columns': list(FEATURE_COLUMNS),
        'condition_columns': list(CONDITION_COLUMNS),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_sfp_checkpoint(
    path: str,
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[StreamingFlowPolicyDeterministic, TrajectoryNormalizer, SFPConfig]:
    payload = torch.load(path, map_location=device)
    cfg_dict = payload['config']
    config = SFPConfig(
        horizon=int(cfg_dict['horizon']),
        state_dim=int(cfg_dict['state_dim']),
        num_cond=int(cfg_dict['num_cond']),
        pos_dim=int(cfg_dict['pos_dim']),
        sigma_train=float(cfg_dict['sigma_train']),
        k=float(cfg_dict['k']),
        diffusion_step_embed_dim=int(cfg_dict['diffusion_step_embed_dim']),
        down_dims=tuple(cfg_dict['down_dims']),
        vel_down_dims=tuple(cfg_dict['vel_down_dims']),
        kernel_size=int(cfg_dict['kernel_size']),
        n_groups=int(cfg_dict['n_groups']),
        fc_timesteps=int(cfg_dict['fc_timesteps']),
    )
    model = build_sfp(config).to(device)
    model.load_state_dict(payload['model_state_dict'])
    normalizer = TrajectoryNormalizer.from_dict(payload['normalizer'])
    return model, normalizer, config
