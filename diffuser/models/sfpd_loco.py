"""
sfpd_loco.py

Locomotion SFP policy with 3-tier hierarchical velocity network.
Trajectory = [obs(11), action(3)] = 14 dims for hopper.

Tier 1 (pos):    structure — where the body should be
Tier 2 (vel):    dynamics — how fast joints are moving
Tier 3 (action): control — what torques to apply

This module holds the training loss; the streaming rollout used at evaluation
(one velocity-network forward and one Euler step per env step) is in scripts/eval_sfp.py.
"""
from typing import Dict
import torch
from torch import Tensor
import torch.nn as nn


class StreamingFlowPolicyLoco(nn.Module):
    def __init__(self,
                 velocity_net: nn.Module,
                 pos_dim: int,
                 vel_dim: int,
                 act_dim: int,
                 num_cond: int = 2,
                 sigma: float = 0.0,
                 k: float = 0.0,
                 device: torch.device = 'cuda',
        ):
        super().__init__()
        self.velocity_net = velocity_net
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim
        self.act_dim = act_dim
        self.obs_dim = pos_dim + vel_dim
        self.state_dim = pos_dim + vel_dim + act_dim
        self.device = device

        self.register_buffer('num_cond', torch.tensor(num_cond, dtype=torch.int32))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.register_buffer('k', torch.tensor(k, dtype=torch.float32))

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        3-tier loss: p_loss + vel_loss + act_loss
        """
        cond = batch['cond'].to(self.device, non_blocking=True)
        x = batch['x'].to(self.device, non_blocking=True)
        x_label = batch['x_label'].to(self.device, non_blocking=True)
        v = batch['v'].to(self.device, non_blocking=True)
        t = batch['t'].to(self.device, non_blocking=True)

        # Zero out non-position dims of end conditioning
        # (locomotion has no "goal", but keep end obs position for structure)
        cond[:, 1, self.pos_dim:] = 0.0

        cond_flat = cond.flatten(start_dim=1)
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=cond_flat,
            is_train=True, label=x_label,
        )

        pd = self.pos_dim
        vd = self.vel_dim

        # Tier 1: position velocity
        p_loss = nn.functional.mse_loss(v_pred[..., :pd], v[..., :pd])

        # Tier 2: velocity velocity (smooth L1 for robustness)
        vel_loss = nn.functional.smooth_l1_loss(
            v_pred[..., pd:pd+vd], v[..., pd:pd+vd], beta=0.1)

        # Tier 3: action velocity
        act_loss = nn.functional.smooth_l1_loss(
            v_pred[..., pd+vd:], v[..., pd+vd:], beta=0.1)

        total_loss = p_loss + vel_loss + act_loss

        return total_loss, p_loss, vel_loss, act_loss
