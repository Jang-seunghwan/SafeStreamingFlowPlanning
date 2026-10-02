"""
cond_unet1D_loco.py

3-tier hierarchical velocity network for locomotion:
    pos_net:  pos(5) + cond → v_pos(5)
    vel_net:  pos(5) + vel(6) + next_pos(5) + cond + t → v_vel(6)
    act_net:  pos(5) + vel(6) + act(3) + next_pos(5) + next_vel(6) + cond + t → v_act(3)

Each tier's prediction is detached before feeding to the next tier,
enabling stable hierarchical learning: structure → dynamics → control.

Hopper: pos_dim=5 (qpos[1:]), vel_dim=6 (qvel), act_dim=3
"""
import torch
import torch.nn as nn

from .cond_unet1D import SinusoidalPosEmb, _Unet1DCore


class ConditionalUnet1DLoco(nn.Module):
    def __init__(self,
        pos_dim: int,
        vel_dim: int,
        act_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 32,
        down_dims_pos=(256, 512, 1024),
        down_dims_vel=(64, 128),
        down_dims_act=(64, 128),
        kernel_size: int = 5,
        n_groups: int = 8,
        fc_timesteps: int = 1,
        horizon: int = 1000,
        time_scale: float = 20.0,
    ):
        """
        Args:
            pos_dim: position state dimensions (hopper=5)
            vel_dim: velocity state dimensions (hopper=6)
            act_dim: action dimensions (hopper=3)
            global_cond_dim: [start_state, end_state] flattened dim; the position part of
                             the end-state slot is fed to every net as conditioning
            time_scale: FM t in [0,1] is multiplied by this before the
                        sinusoidal embedding so mode separation across t
                        matches diffusion-style integer T (= n_diffusion_steps, 20).
        """
        super().__init__()
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim
        self.act_dim = act_dim
        self.state_dim = pos_dim + vel_dim + act_dim
        self.obs_dim = pos_dim + vel_dim
        self.horizon = horizon
        self.time_scale = float(time_scale)

        # Conditioning: extract end-state observation from global_cond
        # global_cond = [start_obs(obs_dim), end_obs(obs_dim)]
        # The position part of end_obs is used as conditioning
        self.cond_in_dim = pos_dim

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # ── Tier 1: pos_net ──
        # input:  [pos(pos_dim) + cond(cond_in_dim)]
        # output: v_pos(pos_dim)
        pos_in_ch = pos_dim + self.cond_in_dim
        self.pos_net = _Unet1DCore(
            input_channels=pos_in_ch,
            output_channels=pos_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=list(down_dims_pos),
            kernel_size=kernel_size,
            n_groups=n_groups,
            fc_timesteps=fc_timesteps,
        )

        # ── Tier 2: vel_net ──
        # input:  [pos + vel + next_pos(detached) + cond + t]
        # output: v_vel(vel_dim)
        vel_in_ch = pos_dim + vel_dim + pos_dim + self.cond_in_dim + 1
        self.vel_net = _Unet1DCore(
            input_channels=vel_in_ch,
            output_channels=vel_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=list(down_dims_vel),
            kernel_size=kernel_size,
            n_groups=min(n_groups, 64),
            fc_timesteps=fc_timesteps,
        )

        # ── Tier 3: act_net ──
        # input:  [pos + vel + act + next_pos(det) + next_vel(det) + cond + t]
        # output: v_act(act_dim)
        act_in_ch = pos_dim + vel_dim + act_dim + pos_dim + vel_dim + self.cond_in_dim + 1
        self.act_net = _Unet1DCore(
            input_channels=act_in_ch,
            output_channels=act_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=list(down_dims_act),
            kernel_size=kernel_size,
            n_groups=min(n_groups, 64),
            fc_timesteps=fc_timesteps,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ ConditionalUnet1DLoco ] params: {n_params:e}")
        print(f"  pos_net: {sum(p.numel() for p in self.pos_net.parameters()):e}")
        print(f"  vel_net: {sum(p.numel() for p in self.vel_net.parameters()):e}")
        print(f"  act_net: {sum(p.numel() for p in self.act_net.parameters()):e}")

    def forward(self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            global_cond: torch.Tensor,
            **kwargs):
        """
        Args:
            sample: (B, T, state_dim) where state_dim = pos_dim + vel_dim + act_dim
            timestep: (B,) or (1,) flow time in [0, 1]
            global_cond: (B, global_cond_dim)
            kwargs:
                is_train: bool - use GT labels for next-state computation
                label: (B, T, state_dim) - GT next state

        Returns:
            (B, T, state_dim) velocity field [v_pos, v_vel, v_act]
        """
        # (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)
        B, C, T = sample.shape

        # ── Timestep encoding ──
        timesteps = timestep.expand(B)

        # FM t in [0,1] -> [0, time_scale] so SinusoidalPosEmb frequency band
        # spans enough range for mode separation (matches diffusion T).
        timesteps_emb_in = timesteps.float() * self.time_scale
        global_feature = self.diffusion_step_encoder(timesteps_emb_in)

        # ── Conditioning ──
        # global_cond: [start_obs(obs_dim), end_obs(obs_dim)]; end-state position as conditioning
        cond_feat = global_cond[:, self.obs_dim: self.obs_dim + self.cond_in_dim]

        cond_feat = cond_feat.unsqueeze(-1).expand(-1, -1, T)  # (B, cond_in_dim, T)

        # ── Split input ──
        sample_pos = sample[:, :self.pos_dim, :]                                    # (B, pos_dim, T)
        sample_vel = sample[:, self.pos_dim:self.pos_dim + self.vel_dim, :]         # (B, vel_dim, T)
        sample_act = sample[:, self.pos_dim + self.vel_dim:, :]                     # (B, act_dim, T)

        dt = 1.0 / (self.horizon - 1)
        t_ch = timesteps.float().view(-1, 1, 1).expand(-1, -1, T)                  # (B, 1, T)

        is_train = kwargs.get('is_train', False)

        # ═══════════════════════════════════════════
        # Tier 1: pos_net → v_pos
        # ═══════════════════════════════════════════
        pos_in = torch.cat([sample_pos, cond_feat], dim=1)
        v_pos = self.pos_net(pos_in, global_feature)                                # (B, pos_dim, T)

        # Compute next_pos (detached from pos_net for vel_net stability)
        if is_train and 'label' in kwargs:
            gt_next = kwargs['label'].moveaxis(-1, -2)
            next_pos = gt_next[:, :self.pos_dim, :]
        else:
            next_pos = sample_pos + v_pos.detach() * dt

        # ═══════════════════════════════════════════
        # Tier 2: vel_net → v_vel
        # ═══════════════════════════════════════════
        vel_in = torch.cat([sample_pos, sample_vel, next_pos, cond_feat, t_ch], dim=1)
        v_vel = self.vel_net(vel_in, global_feature)                                # (B, vel_dim, T)

        # Compute next_vel (detached from vel_net for act_net stability)
        if is_train and 'label' in kwargs:
            next_vel = gt_next[:, self.pos_dim:self.pos_dim + self.vel_dim, :]
        else:
            next_vel = sample_vel + v_vel.detach() * dt

        # ═══════════════════════════════════════════
        # Tier 3: act_net → v_act
        # ═══════════════════════════════════════════
        act_in = torch.cat([
            sample_pos, sample_vel, sample_act,
            next_pos, next_vel,
            cond_feat, t_ch,
        ], dim=1)
        v_act = self.act_net(act_in, global_feature)                                # (B, act_dim, T)

        # ── Combine ──
        x = torch.cat([v_pos, v_vel, v_act], dim=1)                                # (B, state_dim, T)

        # (B,C,T) -> (B,T,C)
        return x.moveaxis(-1, -2)
