"""
cond_unet1D.py: velocity network of the streaming flow policy.

Goal-only conditioning; the velocity sub-network sees the detached position flow.

  - pos_net: predicts v_pos (position flow velocity) from [sample_pos, goal]
  - vel_net: predicts v_vel (velocity flow velocity) from
             [sample_pos, sample_vel, next_pos (detached), goal, t] = 9 channels
During inference, next_pos = sample_pos + v_pos * dt; during training it is the
ground-truth next position (label).
"""
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
            downsample = Linear1d(dim_out)
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
            upsample = Linear1d(dim_in)
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
        horizon,
        diffusion_step_embed_dim=32,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        time_scale: float = 1.0,
    ):
        """
        Args:
            input_dim: state dimension (4 for [x, y, vx, vy])
            horizon: trajectory horizon H (flow step dt = 1 / (H - 1))
            diffusion_step_embed_dim: timestep embedding dim
            down_dims: channel sizes per UNet level for pos_net
            kernel_size: conv kernel size
            n_groups: GroupNorm groups
            time_scale: factor applied to t in [0, 1] before the sinusoidal embedding
        """
        super().__init__()
        self.state_dim = input_dim
        self.pos_dim = input_dim // 2
        self.vel_dim = input_dim - self.pos_dim
        self.cond_in_dim = self.pos_dim   # goal-only (2), NOT start+goal (4)
        self.horizon = horizon

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # pos_net: [sample_pos(2) + goal_cond(2)] = 4ch -> 2ch
        pos_in_channels = self.pos_dim + self.cond_in_dim
        # vel_net: [sample_pos(2) + sample_vel(2) + next_pos(2) + goal_cond(2) + t(1)] = 9ch -> 2ch (v_vel)
        vel_in_channels = self.pos_dim + self.vel_dim + self.pos_dim + self.cond_in_dim + 1

        self.pos_net = _Unet1DCore(
            input_channels=pos_in_channels,
            output_channels=self.pos_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        # vel_net: smaller UNet for nonlinear dynamics (with timestep conditioning)
        self.vel_net = _Unet1DCore(
            input_channels=vel_in_channels,
            output_channels=self.vel_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=[64, 128],
            kernel_size=kernel_size,
            n_groups=min(n_groups, 64),
        )

        self.diffusion_step_encoder = diffusion_step_encoder

        # Plain attribute (not a buffer, not in the state_dict); the value is stored
        # in the checkpoint config and passed to the constructor at evaluation.
        self.time_scale = float(time_scale)

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, sample, timestep, global_cond, is_train=False, label=None):
        """
        sample:      (B, T, input_dim) state
        timestep:    (B,) flow time t in [0, 1]
        global_cond: (B, 2 * input_dim) = [start_state, goal_state]; only the goal position is used
        is_train / label: during training the ground-truth next state (B, T, input_dim)
                          replaces the predicted next position fed to vel_net
        output:      (B, T, input_dim) flow velocity [v_pos, v_vel]
        """
        # (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)

        # 1. timestep encoding (t is scaled before the sinusoidal embedding)
        timesteps = timestep.expand(sample.shape[0])
        timesteps_scaled = timesteps.to(dtype=torch.float32) * self.time_scale
        global_feature = self.diffusion_step_encoder(timesteps_scaled)

        # 2. goal-only conditioning: goal_pos = global_cond[:, state_dim : state_dim + pos_dim]
        cond_feat = global_cond[:, self.state_dim : self.state_dim + self.pos_dim]   # (B, 2)

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
        if is_train and label is not None:
            next_pos = label.moveaxis(-1, -2)[:, :self.pos_dim, :]                       # (B, 2, T)
        else:
            next_pos = (sample_pos + v_pos.detach() * dt)                                # (B, 2, T)

        t_ch = timesteps.float().view(-1, 1, 1).expand(-1, -1, sample.shape[-1])         # (B, 1, T)
        vel_in = torch.cat([sample_pos, sample_vel, next_pos, cond_feat, t_ch], dim=1)   # (B, 9, T)
        v_vel = self.vel_net(vel_in, global_feature)                                     # (B, 2, T)

        x = torch.cat([v_pos, v_vel], dim=1)                                            # (B, 4, T)

        # (B,C,T) -> (B,T,C)
        x = x.moveaxis(-1, -2)
        return x
