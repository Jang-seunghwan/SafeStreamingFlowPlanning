from typing import Dict
import time

import torch
from torch import Tensor
import torch.nn as nn

from .cbf import CBF


class StreamingFlowPolicyDeterministic(nn.Module):
    def __init__(self,
                 velocity_net: nn.Module,
                 device: torch.device = 'cuda',
                 normalizer=None,
                 args=None
        ):
        """
        Args:
            velocity_net (nn.Module): velocity network (ConditionalUnet1D)
            device (torch.device): device
            normalizer: dataset normalizer (used to build the safety filter)
            args: policy arguments; args.safety_enabled attaches the 2nd-order ECBF (SSF)
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.device = device

        # Safety
        self.cbf = None
        self.safety_enabled = args.safety_enabled
        if self.safety_enabled:
            obs_norm = normalizer.normalizers['observations']
            norm_mins = torch.tensor(obs_norm.mins, device=device)
            norm_maxs = torch.tensor(obs_norm.maxs, device=device)
            self.cbf = CBF(norm_mins, norm_maxs, args)

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]):
        """
        Flow-matching loss on (state, velocity) pairs produced by the collate
        function of scripts/train_sfp.py: MSE on the position components and
        smooth L1 on the velocity components of the predicted flow.
        Returns (loss, p_loss, vel_loss).
        """
        # device transfer (non_blocking for CPU-GPU overlap with pin_memory)
        cond = batch['cond'].to(self.device, non_blocking=True)
        cond[:,1,2:] = 0.0
        x = batch['x'].to(self.device, non_blocking=True)
        x_label = batch['x_label'].to(self.device, non_blocking=True)
        v = batch['v'].to(self.device, non_blocking=True)
        t = batch['t'].to(self.device, non_blocking=True)

        cond_flat = cond.flatten(start_dim=1)
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=cond_flat, is_train=True, label=x_label
        )
        # p_loss: position flow velocity
        p_loss = nn.functional.mse_loss(v_pred[..., :2], v[..., :2])

        # vel_loss: smooth L1 on v_vel (flow matching style)
        vel_loss = nn.functional.smooth_l1_loss(v_pred[..., 2:], v[..., 2:], beta=0.1)

        return (p_loss + vel_loss), p_loss, vel_loss

    @torch.inference_mode()
    def rollout(self, start: Tensor, goal: Tensor, pred_horizon: int):
        """Generate a trajectory from start to goal by integrating the flow over
        t in [0, 1] with pred_horizon - 1 Euler steps (one step per waypoint).
        With safety enabled, every step is filtered by the ECBF and the
        corrected state is fed back to the velocity network.

        Args:
            start (Tensor, shape=(1, STATE_DIM)): normalized start state
            goal (Tensor, shape=(1, STATE_DIM)): normalized goal state
            pred_horizon (int): number of waypoints

        Returns:
            Tensor (shape=(1, pred_horizon, STATE_DIM)): normalized trajectory
            cbf_time_avg: average safety-filter time per step (s)
        """
        ncond = torch.stack([start[0], goal[0]], dim=0)        # (2, STATE_DIM)
        cond_flat = ncond.unsqueeze(0).flatten(start_dim=1)   # (1, 2 * STATE_DIM)

        x0 = ncond[0, :]  # start state

        # Integration time steps.
        t_span = torch.linspace(0, 1.0, pred_horizon, device=self.device, dtype=x0.dtype)
        delta_ts = torch.diff(t_span)
        current_state = x0.unsqueeze(0)  # (1, STATE_DIM)
        traj = [current_state]
        cbf_times = []

        for i in range(pred_horizon - 1):
            t = t_span[i]
            velocity = self.velocity_net(
                sample=current_state.unsqueeze(0),
                timestep=t.repeat(1),
                global_cond=cond_flat,
            ).flatten().unsqueeze(0)  # (1, STATE_DIM)

            dt = delta_ts[i]
            next_state = current_state + velocity * dt

            if self.safety_enabled:
                cbf_start = time.time()
                corrected = self.cbf.apply(current_state, next_state)
                cbf_times.append(time.time() - cbf_start)
                traj.append(corrected.clone())
                current_state = corrected  # feedback: corrected state -> next velocity_net input
            else:
                traj.append(next_state.clone())
                current_state = next_state

        trajectory = torch.stack(traj, dim=1)
        cbf_time_avg = sum(cbf_times) / len(cbf_times) if cbf_times else 0.0
        return trajectory, cbf_time_avg
