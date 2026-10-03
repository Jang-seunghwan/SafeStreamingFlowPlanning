from typing import Dict
import time
import torch
from torch import Tensor
import torch.nn as nn

from diffuser.models.cbf import CBF

class StreamingFlowPolicyDeterministic(nn.Module):
    def __init__(self,
                 velocity_net: nn.Module,
                 state_dim: int,
                 device: torch.device = 'cuda',
                 normalizer=None,
                 args=None
        ):
        """
        Args:
            velocity_net (nn.Module): velocity network (diffuser/models/cond_unet1D.py)
            state_dim (int): state dimension
            device (torch.device): device
            normalizer: dataset normalizer (needed by the safety filter)
            args: run arguments (safety_enabled and the CBF parameters)

        The training noise (sigma_0 * exp(-k t)) is applied in scripts/train_sfp.py (make_collate_fn).
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.state_dim = state_dim
        self.device = device

        # Safety filter (SSF): discrete-time HOCBF, diffuser/models/cbf.py
        self.cbf = None
        self.safety_enabled = args.safety_enabled
        if self.safety_enabled:
            norm_mins = torch.tensor(normalizer.normalizers['observations'].mins, device=device)
            norm_maxs = torch.tensor(normalizer.normalizers['observations'].maxs, device=device)
            self.cbf = CBF(norm_mins, norm_maxs, args)

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        vel_net predicts vel_next (absolute) WITHOUT sample_vel input.
        forward() returns vel_next directly during training (bypassing v_vel).
        vel_loss = MSE(vel_next_pred, vel_next_gt) — direct gradient to vel_net.
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

        # vel_loss of the dynamics-based base alone (without the learned correction), logged only
        vel_loss_base = nn.functional.smooth_l1_loss(self.velocity_net._v_vel_base, v[..., 2:], beta=0.1)

        return (p_loss + vel_loss), p_loss, vel_loss, vel_loss_base

    @torch.inference_mode()
    def __call__(self,
                 ncond: Tensor,
                 pred_horizon: int,
    ) -> Tensor:
        """
        Args:
            ncond (Tensor, shape=(NUM_COND, STATE_DIM)): normalized conditioning states
            pred_horizon (int): number of states to predict (≥1)

        Returns:
            Tensor (shape=(1, PRED_HORIZON, STATE_DIM)): predicted states
            corrections: list of correction_info dicts (for visualization)
        """
        if pred_horizon < 1:
            raise ValueError("pred_horizon must be ≥ 1.")

        cond_flat = ncond.unsqueeze(0).flatten(start_dim=1)  # (1, NUM_COND * STATE_DIM)

        x0 = ncond[0, :]  # deterministic inference: the integration starts at the conditioning start state

        # Integration time steps.
        t_span = torch.linspace(0, 1.0, pred_horizon, device=self.device, dtype=x0.dtype)
        delta_ts = torch.diff(t_span)
        current_state = x0.unsqueeze(0)  # (1, STATE_DIM)
        traj = [current_state]
        corrections = []  # Store correction info for visualization

        # Timing tracking
        model_times = []
        cbf_times = []

        for i in range(pred_horizon - 1):
            t = t_span[i]

            # Model inference timing (cuda sync for accurate GPU timing)
            torch.cuda.synchronize()
            model_start = time.time()
            velocity = StreamingVelocityField(self.velocity_net, cond_flat).forward(t, current_state.squeeze(0)).unsqueeze(0)  # (1, STATE_DIM)
            torch.cuda.synchronize()
            model_times.append(time.time() - model_start)

            dt = delta_ts[i]

            # ── Build next state ──
            # Position from the position model (flow matching)
            next_pos = current_state[:, :2] + velocity[:, :2] * dt  # (1, 2)

            # Velocity from the maze dynamics: v = delta_pos * vel_weight. The velocity output of the
            # model (velocity[:, 2:]) is not used to build the open-loop plan.
            delta_pos = velocity[:, :2] * dt  # (1, 2)
            vel_w = self.velocity_net.vel_weight.unsqueeze(0)  # (1, 2)
            next_vel = delta_pos * vel_w  # (1, 2)

            next_state = torch.cat([next_pos, next_vel], dim=-1)  # (1, 4)

            if self.safety_enabled and self.cbf is not None:
                # CBF-QP solving timing (cuda sync for accurate GPU timing)
                torch.cuda.synchronize()
                cbf_start = time.time()
                current_state, _, correction_info = self.cbf.apply(current_state, next_state, t=t)
                torch.cuda.synchronize()
                cbf_times.append(time.time() - cbf_start)

                corrections.append(correction_info)
            else:
                current_state = next_state
            traj.append(current_state.clone())

        # Store timing info
        timing_info = {
            'model_time_avg': sum(model_times) / len(model_times) if model_times else 0.0,
            'cbf_time_avg': sum(cbf_times) / len(cbf_times) if cbf_times else 0.0,
        }

        return torch.stack(traj, dim=1), corrections, timing_info

    @torch.inference_mode()
    def rollout(self, start: Tensor, goal: Tensor, pred_horizon: int) -> Tensor:
        """Generate trajectory from start to goal positions.

        Args:
            start (Tensor, shape=(1, STATE_DIM)): start state
            goal (Tensor, shape=(1, STATE_DIM)): goal state
            pred_horizon (int): number of waypoints to predict

        Returns:
            Tensor (shape=(1, pred_horizon, STATE_DIM)): predicted trajectory
            corrections: list of correction_info dicts (for visualization)
            timing_info: dict with model and CBF timing
        """
        # Stack start and goal as conditioning
        ncond = torch.stack([start[0], goal[0]], dim=0)  # (2, STATE_DIM)

        # Generate trajectory using __call__
        trajectory, corrections, timing_info = self(
            ncond=ncond,
            pred_horizon=pred_horizon,
        )  # (1, pred_horizon, STATE_DIM)
        return trajectory, corrections, timing_info

class StreamingVelocityField (nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model: nn.Module, cond: Tensor):
        super().__init__()
        self.model = model
        self.cond = cond

    def forward(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            t (Tensor, shape=(,), dtype=float): time
            x (Tensor, shape=(STATE_DIM,), dtype=float): position

        Returns:
            Tensor (shape=(STATE_DIM,), dtype=float): velocity
        """
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, STATE_DIM)
        v: Tensor = self.model(
            sample=x,
            timestep=t.repeat(x.shape[0]),
            global_cond=self.cond,
        )  # (1, 1, STATE_DIM)
        v = v.flatten()  # (STATE_DIM,)
        return v
