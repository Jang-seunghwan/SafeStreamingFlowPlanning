import torch


class CBFLoco:
    """
    CBF safety filter applied to the flow-matching sampling steps (SafeFM / SafeFlowMatcher).

    Height constraint: h(x) = height_limit - x[:, height_idx]
    (hopper: height_idx = 3, the first observation dimension in the [action, observation] layout).
    Closed-form solution of the CBF-QP, in normalized space (LimitsNormalizer).
    """

    def __init__(self, height, norm_mins, norm_maxs, height_idx=3, epsilon=1.0, rho=0.99):
        """
        Args:
            height: physical height limit (1.5 for hopper)
            norm_mins: normalizer min values of the observations
            norm_maxs: normalizer max values of the observations
            height_idx: index of the torso height in the trajectory state
            epsilon: CBF class-K parameter
            rho: CBF finite-time exponent
        """
        self.epsilon = epsilon
        self.rho = rho
        self.height_idx = height_idx

        norm_mins = torch.tensor(norm_mins, dtype=torch.float32)
        norm_maxs = torch.tensor(norm_maxs, dtype=torch.float32)

        # Normalize the height limit to [-1, 1] using LimitsNormalizer convention
        # LimitsNormalizer: x_norm = 2 * (x - min) / (max - min) - 1
        obs_min = norm_mins[0]  # first observation dim (torso height)
        obs_max = norm_maxs[0]
        self.height_normalized = 2 * (height - obs_min) / (obs_max - obs_min) - 1

        self._correction_count = 0

    @torch.no_grad()
    def apply(self, x_now, x_next_naive):
        """Apply the safety filter to the proposed integration step.

        Args:
            x_now: current trajectory [B, H, D] in normalized space
            x_next_naive: proposed next trajectory [B, H, D] in normalized space

        Returns:
            x_corrected: corrected trajectory [B, H, D]
            info: dict (min barrier value, whether a correction was applied)
        """
        return self._apply_closed_form(x_now, x_next_naive)

    @torch.no_grad()
    def _apply_closed_form(self, x_now, x_next_naive):
        """Closed-form CBF correction for 1D height constraint.

        Two constraints (upper ceiling + lower floor):
            h_0(x) = height_normalized - x[:, height_idx]  (ceiling)
            h_1(x) = x[:, height_idx] + offset             (floor, generous bound)

        Closed-form solution from SafeDiffuser's invariance_*_cf methods.
        """
        x = x_now.squeeze(0)   # [H, D]
        xp1 = x_next_naive.squeeze(0)

        ref = xp1 - x
        idx = self.height_idx

        # Barrier values
        b0 = self.height_normalized - x[:, idx:idx+1]  # ceiling constraint
        Lfb = 0

        # Constraint 0: ceiling  (G0 * u <= h0)
        Lgbu1 = -1 * torch.ones_like(x[:, idx:idx+1])
        G0 = torch.cat([-Lgbu1], dim=1)
        h0 = Lfb + self.epsilon * torch.sign(b0) * torch.abs(b0) ** self.rho

        # Constraint 1: floor  (G1 * u <= h1)
        b1 = x[:, idx:idx+1] + 10  # generous lower bound
        Lgbu1_floor = 1 * torch.ones_like(x[:, idx:idx+1])
        G1 = torch.cat([-Lgbu1_floor], dim=1)
        h1 = Lfb + self.epsilon * torch.sign(b1) * torch.abs(b1) ** self.rho

        # Nominal reference
        q = -torch.cat([ref[:, idx:idx+1]], dim=1).to(G0.device)

        # Closed-form dual solution
        y1_bar = 1 * G0
        y2_bar = 1 * G1
        u_bar = -1 * q
        p1_bar = h0 - torch.sum(G0 * u_bar, dim=1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1 * u_bar, dim=1).unsqueeze(1)

        G = torch.cat([
            torch.sum(y1_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0),
            torch.sum(y1_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0),
            torch.sum(y2_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0),
            torch.sum(y2_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0),
        ], dim=0)

        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G: 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(
            G[2] * w_p2_bar < G[3] * p1_bar,
            torch.zeros_like(p1_bar),
            torch.where(
                G[1] * w_p1_bar < G[0] * p2_bar,
                w_p1_bar / G[0],
                torch.clamp(G[3] * p1_bar - G[2] * p2_bar, max=0) / (G[0] * G[3] - G[1] * G[2])
            )
        )

        lambda2 = torch.where(
            G[2] * w_p2_bar < G[3] * p1_bar,
            w_p2_bar / G[3],
            torch.where(
                G[1] * w_p1_bar < G[0] * p2_bar,
                torch.zeros_like(p1_bar),
                torch.clamp(G[0] * p2_bar - G[1] * p1_bar, max=0) / (G[0] * G[3] - G[1] * G[2])
            )
        )

        out = lambda1 * y1_bar + lambda2 * y2_bar + u_bar

        # Apply correction only to the height dimension
        rt = xp1.clone()
        rt[:, idx:idx+1] = x[:, idx:idx+1] + out[:, 0:1]
        rt = rt.unsqueeze(0)

        b_min = torch.min(b0)
        corrected = (lambda1.abs().sum() + lambda2.abs().sum()) > 1e-8
        if corrected:
            self._correction_count += 1

        info = {'b_min': b_min, 'corrected': corrected}
        return rt, info
