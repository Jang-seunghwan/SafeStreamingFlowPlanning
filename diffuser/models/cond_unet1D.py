"""
cond_unet1D.py

Goal-only conditioning + detached v_pos -> vel_net.

vel_net predicts vel correction WITH sample_vel input.
vel_next is converted to v_vel for the flow ODE output.

During inference rollout, vel is REPLACED by vel_next each step (no accumulation):
  state_next = state + v * dt
  vel_next_state = vel + v_vel * dt = vel + (vel_next - vel)*(H-1)*(1/(H-1)) = vel_next

Key design:
  - pos_net: predicts v_pos (position flow velocity)
  - vel_net: predicts vel correction (residual on top of dynamics-based v_vel_base)
  - Input to vel_net: [sample_pos, sample_vel, next_pos(detached), goal, t] = 9ch
  - Output conversion: v_vel = v_vel_base + v_vel_correction, v_vel_base = (delta_pos * vel_weight - vel) / dt
"""
from typing import Union
import math
import torch
from torch import Tensor
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Linear1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, T = x.size()
        x = x.view(B, -1)
        x = self.linear(x)
        x = x.view(B, C, T)
        return x


class Conv1dBlock(nn.Module):
    '''Conv1d --> GroupNorm --> Mish'''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Additive conditioning (time_mlp), NOT FiLM."""
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels
        self.out_channels = out_channels
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class _Unet1DCore(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        diffusion_step_embed_dim: int,
        down_dims,
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
                downsample if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            upsample = Linear1d(fc_timesteps * dim_in)
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                upsample if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
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

        x = self.final_conv(x)
        return x


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        vel_weight,
        diffusion_step_embed_dim=32,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        fc_timesteps: int = 1,
        horizon: int = 384,
    ):
        """
        Args:
            input_dim: state dimension (e.g. 4 for [y, x, vy, vx])
            global_cond_dim: dim of [start, goal] flattened (only goal_pos is used)
            vel_weight: dynamics weight of the v_vel residual base, w = pos_range / (dt_sim * vel_range)
                        per position axis (computed from the dataset normalizer in scripts/train_sfp.py)
            diffusion_step_embed_dim: timestep embedding dim
            down_dims: channel sizes per UNet level for pos_net
            kernel_size: conv kernel size
            n_groups: GroupNorm groups
            fc_timesteps: number of time steps per sample (1: one state per sample; Linear1d up/down)
            horizon: trajectory horizon
        """
        super().__init__()
        self.state_dim = input_dim
        self.pos_dim = input_dim // 2
        self.vel_dim = input_dim - self.pos_dim
        self.cond_in_dim = self.pos_dim   # <<< goal-only (2), NOT start+goal (4)
        self.horizon = horizon
        # Weight for the v_vel residual base
        self.register_buffer('vel_weight', torch.tensor(vel_weight, dtype=torch.float32))

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # pos_net: [sample_pos(2) + goal_cond(2)] = 4ch -> 2ch
        pos_in_channels = self.pos_dim + self.cond_in_dim
        # vel_net: [sample_pos(2) + sample_vel(2) + next_pos(2) + goal_cond(2) + t(1)] -> 2ch (v_vel)
        vel_in_channels = self.pos_dim + self.vel_dim + self.pos_dim + self.cond_in_dim + 1

        self.pos_net = _Unet1DCore(
            input_channels=pos_in_channels,
            output_channels=self.pos_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            fc_timesteps=fc_timesteps,
        )
        # vel_net: smaller UNet for nonlinear dynamics (with timestep conditioning)
        self.vel_net = _Unet1DCore(
            input_channels=vel_in_channels,
            output_channels=self.vel_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=[64, 128],
            kernel_size=kernel_size,
            n_groups=min(n_groups, 64),
            fc_timesteps=fc_timesteps,
        )

        self.diffusion_step_encoder = diffusion_step_encoder

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None,
            **kwargs):           # <<< absorbs is_train, label from sfpd.Loss() compat
        """
        sample: (B, T, input_dim)
        timestep: (B,) or int
        global_cond: (B, global_cond_dim)  -- only goal_pos extracted
        output: (B, T, input_dim)
        """
        # (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)

        # 1. timestep encoding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        # 2. goal-only conditioning: global_cond layout is [start_state(state_dim), goal_state(state_dim)]
        goal_pos = global_cond[:, self.state_dim : self.state_dim + self.pos_dim]
        cond_feat = goal_pos   # (B, 2) -- goal position only

        # broadcast cond to temporal dim
        cond_feat = cond_feat.unsqueeze(-1)                          # (B, 2, 1)
        if cond_feat.shape[-1] != sample.shape[-1]:
            cond_feat = cond_feat.expand(-1, -1, sample.shape[-1])   # (B, 2, T)

        # 3. pos_net: predict position velocity
        sample_pos = sample[:, :self.pos_dim, :]                     # (B, 2, T)
        pos_in = torch.cat([sample_pos, cond_feat], dim=1)           # (B, 4, T)
        v_pos = self.pos_net(pos_in, global_feature)                 # (B, 2, T)

        # 4. velocity flow velocity (flow matching: vel + v_vel * dt = vel_next)
        dt = 1.0 / (self.horizon - 1)
        sample_vel = sample[:, self.pos_dim:, :]                                        # (B, 2, T)

        # Training: use GT next_pos; Inference: use pos_net prediction
        if kwargs.get('is_train', False) and 'label' in kwargs:
            gt_next = kwargs['label'].moveaxis(-1, -2)                                   # (B, C, T)
            next_pos = gt_next[:, :self.pos_dim, :]                                      # (B, 2, T)
        else:
            next_pos = (sample_pos + v_pos.detach() * dt)                                # (B, 2, T)

        t_ch = timesteps.float().view(-1, 1, 1).expand(-1, -1, sample.shape[-1])         # (B, 1, T)
        vel_in = torch.cat([sample_pos, sample_vel, next_pos, cond_feat, t_ch], dim=1)   # (B, 9, T)

        # Residual: dynamics-based v_vel base + learned correction
        delta_pos = next_pos - sample_pos                                            # (B, 2, T)
        w = self.vel_weight.view(1, 2, 1)
        v_vel_base = (delta_pos * w - sample_vel) / dt                               # (B, 2, T)
        v_vel_correction = self.vel_net(vel_in, global_feature)                      # (B, 2, T)
        v_vel = v_vel_base + v_vel_correction                                        # (B, 2, T)
        # base-only velocity, for logging the vel_loss without the learned correction (training only)
        if kwargs.get('is_train', False):
            self._v_vel_base = v_vel_base.moveaxis(-1, -2)                           # (B, T, 2)

        x = torch.cat([v_pos, v_vel], dim=1)                                            # (B, 4, T)

        # (B,C,T) -> (B,T,C)
        x = x.moveaxis(-1, -2)
        return x
