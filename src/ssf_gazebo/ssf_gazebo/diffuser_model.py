#!/usr/bin/env python3
"""Trajectory-level baselines for the warehouse benchmark.

TemporalUnet backbone (port of diffuser/models/temporal.py, main branch) with
three samplers sharing it: DDPM (Diffuser), conditional flow matching (FM) and
the FlowMatcher prediction-correction sampler. Also the shared normalizer,
sliding-window dataset and checkpoint helpers.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


FEATURE_COLUMNS = ('x', 'y', 'v_x', 'v_y')
CONDITION_COLUMNS = ('start_x', 'start_y', 'goal_x', 'goal_y')


@dataclass
class DiffuserConfig:
    """Faithful to diffuser/models/temporal.py:TemporalUnet (main branch).

    horizon=512 doubles maze2d-large (384) to fit the average Gazebo episode.
    dim_mults=(1, 4, 8) matches the maze2d.py 'base' config.
    """
    horizon: int = 512
    transition_dim: int = 4                              # (x, y, vx, vy)
    diffusion_steps: int = 256                           # n_diffusion_steps in maze2d.py
    planner_type: str = 'diffusion'
    flow_matcher_alpha: float = 2.0
    dim: int = 32                                        # base feature dim (matches upstream)
    dim_mults: Tuple[int, ...] = (1, 4, 8)               # matches maze2d.py
    kernel_size: int = 5
    beta_schedule: str = 'cosine'


@dataclass
class DatasetStats:
    rollout_count: int
    skipped_count: int
    source_files: List[str]


class TrajectoryNormalizer:
    """Limits-based normalizer matching upstream LimitsNormalizer (maps to [-1, 1]).

    Internally stored as (center, half_range) — kept under the legacy field
    names (`mean`, `std`) so checkpoint serialization stays the same:
        normalized = (x - center) / half_range
    where center = (max + min) / 2 and half_range = (max - min) / 2.

    This pairing matches the residual-base derivation
    `w = pos_range / (dt_sim * vel_range)` used in upstream maze2d train_sfp.py.
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        # `mean` here is the center of the data range, `std` is the half-range.
        # Field names preserved for backward-compat with existing checkpoints.
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.maximum(self.std, 1e-6)

    @classmethod
    def from_trajectories(cls, trajectories: np.ndarray) -> 'TrajectoryNormalizer':
        flat = trajectories.reshape(-1, trajectories.shape[-1])
        mins = flat.min(axis=0)
        maxs = flat.max(axis=0)
        center = (maxs + mins) / 2.0
        half_range = (maxs - mins) / 2.0
        return cls(mean=center, std=half_range)

    @classmethod
    def from_dict(cls, payload: Dict[str, Sequence[float]]) -> 'TrajectoryNormalizer':
        return cls(mean=payload['mean'], std=payload['std'])

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            'mean': self.mean.astype(float).tolist(),
            'std': self.std.astype(float).tolist(),
            'feature_columns': list(FEATURE_COLUMNS),
        }

    def normalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        return ((trajectory - self.mean) / self.std).astype(np.float32)

    def denormalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        return (trajectory * self.std + self.mean).astype(np.float32)

    def normalize_state(self, state: Sequence[float]) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        return ((state_array - self.mean) / self.std).astype(np.float32)

    def normalize_condition(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> np.ndarray:
        start = np.asarray(start_xy, dtype=np.float32)
        goal = np.asarray(goal_xy, dtype=np.float32)
        mean_xy = self.mean[:2]
        std_xy = self.std[:2]
        start_norm = (start - mean_xy) / std_xy
        goal_norm = (goal - mean_xy) / std_xy
        return np.concatenate([start_norm, goal_norm]).astype(np.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


def extract(values: Tensor, timesteps: Tensor, broadcast_shape: Sequence[int]) -> Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(broadcast_shape) - 1)))


def apply_endpoint_conditioning(x: Tensor, start_state: Tensor, goal_state: Tensor) -> Tensor:
    x = x.clone()
    x[:, 0, :] = start_state
    x[:, -1, :] = goal_state
    return x


def _read_rollout_csv(csv_path: str) -> Optional[np.ndarray]:
    try:
        data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=np.float32)
    except Exception:
        return None

    if data.dtype.names is None:
        return None

    missing = set(FEATURE_COLUMNS + CONDITION_COLUMNS).difference(data.dtype.names)
    if missing:
        return None

    data = np.atleast_1d(data)
    if len(data) < 2:
        return None
    return data


class SlidingWindowDiffuserDataset(Dataset):
    """Sliding-window (stride=1) sampler over raw, variable-length rollouts.

    Each segment is a (horizon, state_dim) slice of a single episode (no
    resampling of whole episodes to a fixed horizon). This dataset:
      - Preserves the raw 20 Hz sampling rate (dt_sim ≈ 0.05 s).
      - Conditions on SEGMENT endpoints, not the original episode goal —
        same convention as upstream `GoalDataset.get_conditions()` and SFP.
      - One CSV produces ~(path_length - horizon) segments.

    Required for semantic equivalence with SFP's plans: at inference both
    produce `horizon * dt_sim = 25.6 s` (for H=512) of real wall-clock plan,
    instead of "normalized [0,1]" time that resampling produced.
    """

    def __init__(
        self,
        observations: np.ndarray,   # (N, max_T, S) raw float32
        path_lengths: np.ndarray,   # (N,) int64
        normalizer: 'TrajectoryNormalizer',
        horizon: int,
    ):
        if observations.ndim != 3:
            raise ValueError(f'observations must be (N, max_T, S), got {observations.shape}')
        self.horizon = int(horizon)

        n, max_T, sdim = observations.shape
        normed_flat = normalizer.normalize_trajectory(observations.reshape(-1, sdim))
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

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        i, s, e = self.indices[idx]
        segment = self.observations[int(i), int(s):int(e)]                  # (H, S)
        start_state = segment[0].clone()
        goal_state = segment[-1].clone()
        goal_state[2:] = 0.0                                                 # zero goal velocity (SFP convention)
        # Condition: [start_xy, goal_xy] (4D) — matches the original Gazebo training schema
        condition = torch.cat([start_state[:2], goal_state[:2]])
        return {
            'trajectory': segment,
            'condition': condition,
            'start_state': start_state,
            'goal_state': goal_state,
        }


# ---------------------------------------------------------------------------
# Backbone — faithful port of diffuser/models/temporal.py
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Matches diffuser/models/helpers.py:SinusoidalPosEmb."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm(8) -> Mish.  Matches upstream helpers.py."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """Additive time conditioning (NOT FiLM), matches upstream temporal.py."""
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, kernel_size: int = 5):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # x: (B, C, T); t: (B, embed_dim)
        time_feat = self.time_mlp(t).unsqueeze(-1)            # (B, out_ch, 1)
        out = self.blocks[0](x) + time_feat
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    """Faithful port of diffuser/models/temporal.py:TemporalUnet.

    Differences from upstream:
      - The `cond` argument is accepted for signature compatibility but
        ignored.  Gazebo conditions on (start, goal) by clamping x[:, 0, :]
        and x[:, -1, :] via `apply_endpoint_conditioning` BEFORE every
        forward pass; the model itself receives only x and the diffusion
        timestep, exactly like upstream.
      - We use einops via plain `permute` to avoid the extra einops dep
        (einops is upstream but not strictly needed).
    """
    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 32,
        dim_mults: Sequence[int] = (1, 4, 8),
        kernel_size: int = 5,
    ):
        super().__init__()
        dims = [transition_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        num_resolutions = len(in_out)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, kernel_size=kernel_size),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, kernel_size=kernel_size),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, kernel_size=kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, kernel_size=kernel_size)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, kernel_size=kernel_size),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, kernel_size=kernel_size),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x: Tensor, condition: Optional[Tensor], timesteps: Tensor) -> Tensor:
        # x: (B, H, transition_dim) -> (B, T, H) so 1d convs run over the time axis
        x = x.permute(0, 2, 1)
        t = self.time_mlp(timesteps)
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        x = self.final_conv(x)
        return x.permute(0, 2, 1)


def build_denoiser(config: DiffuserConfig) -> nn.Module:
    return TemporalUnet(
        horizon=config.horizon,
        transition_dim=config.transition_dim,
        dim=config.dim,
        dim_mults=tuple(config.dim_mults),
        kernel_size=config.kernel_size,
    )


class GaussianDiffusionPlanner(nn.Module):
    def __init__(self, model: nn.Module, config: DiffuserConfig):
        super().__init__()
        self.model = model
        self.config = config

        if config.beta_schedule != 'cosine':
            raise ValueError(f'Unknown beta schedule: {config.beta_schedule}')
        betas = cosine_beta_schedule(config.diffusion_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def q_posterior(self, x_start: Tensor, x_t: Tensor, timesteps: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, timesteps, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x: Tensor, condition: Tensor, timesteps: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicted_noise = self.model(x, condition, timesteps)
        x_recon = self.predict_start_from_noise(x, timesteps, predicted_noise).clamp(-5.0, 5.0)
        return self.q_posterior(x_recon, x, timesteps)

    def p_losses(self, batch: Dict[str, Tensor]) -> Tensor:
        x_start = batch['trajectory']
        condition = batch['condition']
        start_state = batch['start_state']
        goal_state = batch['goal_state']

        batch_size = x_start.shape[0]
        timesteps = torch.randint(0, self.config.diffusion_steps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        x_noisy = apply_endpoint_conditioning(x_noisy, start_state, goal_state)

        predicted_noise = self.model(x_noisy, condition, timesteps)
        loss_mask = torch.ones_like(noise)
        loss_mask[:, 0, :] = 0.0
        loss_mask[:, -1, :] = 0.0
        squared_error = (predicted_noise - noise) ** 2
        return (squared_error * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def conditional_sample(
        self,
        condition: Tensor,
        start_state: Tensor,
        goal_state: Tensor,
        batch_size: int,
    ) -> Tensor:
        device = self.betas.device
        x = torch.randn(
            batch_size,
            self.config.horizon,
            self.config.transition_dim,
            device=device,
        )
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        for step in reversed(range(self.config.diffusion_steps)):
            timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            model_mean, _, model_log_variance = self.p_mean_variance(x, condition, timesteps)
            if step == 0:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            x = model_mean + torch.exp(0.5 * model_log_variance) * noise
            x = apply_endpoint_conditioning(x, start_state, goal_state)
        return x


class FlowMatchingPlanner(nn.Module):
    def __init__(self, model: nn.Module, config: DiffuserConfig):
        super().__init__()
        self.model = model
        self.config = config

    def _embedding_times(self, times: Tensor) -> Tensor:
        return times * float(self.config.diffusion_steps)

    def p_losses(self, batch: Dict[str, Tensor]) -> Tensor:
        x_target = batch['trajectory']
        condition = batch['condition']
        start_state = batch['start_state']
        goal_state = batch['goal_state']

        batch_size = x_target.shape[0]
        x_noise = torch.randn_like(x_target)
        x_noise = apply_endpoint_conditioning(x_noise, start_state, goal_state)

        times = torch.rand(batch_size, device=x_target.device)
        broadcast_times = times.reshape(batch_size, 1, 1)
        x_t = (1.0 - broadcast_times) * x_noise + broadcast_times * x_target
        x_t = apply_endpoint_conditioning(x_t, start_state, goal_state)

        target_velocity = x_target - x_noise
        predicted_velocity = self.model(x_t, condition, self._embedding_times(times))

        loss_mask = torch.ones_like(target_velocity)
        loss_mask[:, 0, :] = 0.0
        loss_mask[:, -1, :] = 0.0
        squared_error = (predicted_velocity - target_velocity) ** 2
        return (squared_error * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def conditional_sample(
        self,
        condition: Tensor,
        start_state: Tensor,
        goal_state: Tensor,
        batch_size: int,
    ) -> Tensor:
        device = next(self.parameters()).device
        x = torch.randn(
            batch_size,
            self.config.horizon,
            self.config.transition_dim,
            device=device,
        )
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        step_count = max(1, int(self.config.diffusion_steps))
        dt = 1.0 / float(step_count)
        for step in range(step_count):
            times = torch.full((batch_size,), step * dt, device=device)
            velocity = self.model(x, condition, self._embedding_times(times))
            x = x + velocity * dt
            x = apply_endpoint_conditioning(x, start_state, goal_state)
        return x


class FlowMatcherPlanner(FlowMatchingPlanner):
    @torch.no_grad()
    def conditional_sample(
        self,
        condition: Tensor,
        start_state: Tensor,
        goal_state: Tensor,
        batch_size: int,
    ) -> Tensor:
        device = next(self.parameters()).device
        x = torch.randn(
            batch_size,
            self.config.horizon,
            self.config.transition_dim,
            device=device,
        )
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        prediction_times = torch.zeros(batch_size, device=device)
        prediction_velocity = self.model(x, condition, self._embedding_times(prediction_times))
        x = x + prediction_velocity
        x = apply_endpoint_conditioning(x, start_state, goal_state)

        step_count = max(1, int(self.config.diffusion_steps))
        dt = 1.0 / float(step_count)
        alpha = float(self.config.flow_matcher_alpha)
        for step in range(step_count):
            times = torch.full((batch_size,), step * dt, device=device)
            velocity = self.model(x, condition, self._embedding_times(times))
            scale = (alpha * (1.0 - times)).reshape(batch_size, 1, 1)
            x = x + velocity * scale * dt
            x = apply_endpoint_conditioning(x, start_state, goal_state)
        return x


def build_planner(model: nn.Module, config: DiffuserConfig) -> nn.Module:
    if config.planner_type == 'diffusion':
        return GaussianDiffusionPlanner(model, config)
    if config.planner_type == 'flow_matching':
        return FlowMatchingPlanner(model, config)
    if config.planner_type == 'flow_matcher':
        return FlowMatcherPlanner(model, config)
    raise ValueError(f'Unknown planner type: {config.planner_type}')


class DiffuserPlanner:
    def __init__(self, diffusion: nn.Module, normalizer: TrajectoryNormalizer, device: str):
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.diffusion.to(self.device)
        self.diffusion.eval()

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: str = 'cpu',
        planner_type_override: Optional[str] = None,
    ) -> 'DiffuserPlanner':
        checkpoint_path = os.path.expanduser(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_payload = dict(checkpoint['config'])

        if planner_type_override:
            config_payload['planner_type'] = planner_type_override
        config = DiffuserConfig(**config_payload)
        normalizer = TrajectoryNormalizer.from_dict(checkpoint['normalizer'])
        model = build_denoiser(config)
        diffusion = build_planner(model, config)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        return cls(diffusion=diffusion, normalizer=normalizer, device=device)

    def sample_plan(
        self,
        start_state: Sequence[float],
        goal_xy: Sequence[float],
        batch_size: int = 1,
    ) -> np.ndarray:
        start_state_np = self.normalizer.normalize_state(start_state)
        goal_state_np = self.normalizer.normalize_state([goal_xy[0], goal_xy[1], 0.0, 0.0])
        condition_np = self.normalizer.normalize_condition(
            start_state[:2],
            goal_xy,
        )

        start_state_tensor = torch.from_numpy(start_state_np).float().to(self.device).unsqueeze(0).repeat(batch_size, 1)
        goal_state_tensor = torch.from_numpy(goal_state_np).float().to(self.device).unsqueeze(0).repeat(batch_size, 1)
        condition_tensor = torch.from_numpy(condition_np).float().to(self.device).unsqueeze(0).repeat(batch_size, 1)

        normalized = self.diffusion.conditional_sample(
            condition=condition_tensor,
            start_state=start_state_tensor,
            goal_state=goal_state_tensor,
            batch_size=batch_size,
        )
        plans = normalized.detach().cpu().numpy()
        return np.stack([self.normalizer.denormalize_trajectory(plan) for plan in plans], axis=0)



def save_checkpoint(
    checkpoint_path: str,
    diffusion: nn.Module,
    normalizer: TrajectoryNormalizer,
    config: DiffuserConfig,
    metadata: Dict[str, object],
) -> None:
    checkpoint_path = os.path.expanduser(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
    torch.save(
        {
            'model_state_dict': diffusion.state_dict(),
            'config': asdict(config),
            'planner_type': config.planner_type,
            'normalizer': normalizer.to_dict(),
            'metadata': metadata,
        },
        checkpoint_path,
    )
