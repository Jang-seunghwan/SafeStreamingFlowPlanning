"""
Class-K CBF of the SafeFM / SafeFlowMatcher baselines (relaxed CBF, closed-form solution), in NORMALIZED space.
SafeDiffuser uses the obstacles / centre offset / robust_term of this object in diffusion.py.

Different from `cbf.py` (SSF), which works in PHYSICAL space with a discrete-time HOCBF.
The state layout is [action(2) + obs(4)] as used by Diffuser / CFM.

apply(x, xp1, t) -> x_next.unsqueeze(0)
"""
import torch
import math


class CBF:
    def __init__(self, norm_mins, norm_maxs, args):
        self.device = norm_mins.device
        self.norm_mins = norm_mins
        self.norm_maxs = norm_maxs
        self.obstacles = args.obstacles
        self.action_dim = 2

        # Parameters of the finite-time class-K function
        self.alpha = args.eps
        self.rho = args.rho

        self.robust_term = args.robust_term
        self.relax_threshold = args.relax_threshold

        # Obstacle centre in the observation frame = configured centre + offset (-0.5 large, -0.7 umaze / medium)
        self.dataset = getattr(args, 'dataset', '')
        self.center_offset = -0.5 if 'large' in self.dataset else -0.7

        # Precompute normalization factors
        self.xr = 2 / (self.norm_maxs[1] - self.norm_mins[1])
        self.yr = 2 / (self.norm_maxs[0] - self.norm_mins[0])

    @torch.no_grad()
    def compute_single_constraint(self, x, obs, t=None):
        cx, cy = obs['center']
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))
        off_x = 2 * (cx + self.center_offset - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2 * (cy + self.center_offset - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        dx = (x[:,3:4] - off_x) / self.xr / rx
        dy = (x[:,2:3] - off_y) / self.yr / ry
        order = obs['order']

        # Lie derivative (chain rule: d/dx_norm [(x-off)/xr/rx]^n = n*[...]^(n-1) / xr / rx)
        L1 = order * dy**(order-1) / self.yr / ry
        L2 = order * dx**(order-1) / self.xr / rx

        # Finite-time parameter
        alpha = self.alpha
        rho = self.rho
        delta = self.robust_term

        # Relaxation weight, active for t <= relax_threshold (exponential drop near the threshold)
        if t <= self.relax_threshold:
            ratio = t / self.relax_threshold
            sign = 200.0 * (1 - math.exp(3 * (ratio - 1)))
        else:
            sign = 0.0

        rx = sign * torch.ones_like(L1)
        G = torch.cat([-L1, -L2, rx], dim=1).unsqueeze(1)

        b = dy**order + dx**order - (1 + delta) **order
        finite_time_term = torch.sign(b) * torch.abs(b)**rho
        h = alpha * finite_time_term
        return G, h

    def solve_closed_form(self, u_ref, G, h):
        """
        Closed-form solution for 1 or 2 constraints.
        G: [B, num_obs, dim], h: [B, num_obs]
        """
        u = u_ref[:, 2:4]   # [B, 2] desired move
        u_relax = torch.zeros_like(u[:, :1])
        u_bar = torch.cat([u, u_relax], dim=1)

        num_obs = G.shape[1]

        if num_obs == 1:
            # Single constraint: project onto half-space Gu <= h
            G0 = G[:, 0, :]         # [B, dim]
            h0 = h[:, 0:1]          # [B, 1]
            p = h0 - torch.sum(G0 * u_bar, dim=1, keepdim=True)  # violation margin
            G_norm_sq = torch.sum(G0 * G0, dim=1, keepdim=True) + 1e-6
            lam = torch.clamp(p, max=0) / G_norm_sq
            out = u_bar + lam * G0
        elif num_obs >= 2:
            G0 = G[:, 0, :]         # [B, dim]
            G1 = G[:, 1, :]         # [B, dim]
            h0 = h[:, 0:1]          # [B, 1]
            h1 = h[:, 1:2]          # [B, 1]

            y1_bar = G0
            y2_bar = G1

            p1_bar = h0 - torch.sum(G0 * u_bar, dim=1, keepdim=True)
            p2_bar = h1 - torch.sum(G1 * u_bar, dim=1, keepdim=True)

            G_mat = torch.cat([
                torch.sum(y1_bar * y1_bar, dim=1, keepdim=True).unsqueeze(0),
                torch.sum(y1_bar * y2_bar, dim=1, keepdim=True).unsqueeze(0),
                torch.sum(y2_bar * y1_bar, dim=1, keepdim=True).unsqueeze(0),
                torch.sum(y2_bar * y2_bar, dim=1, keepdim=True).unsqueeze(0),
            ], dim=0)  # shape: [4, B, 1]

            w_p1_bar = torch.clamp(p1_bar, max=0)
            w_p2_bar = torch.clamp(p2_bar, max=0)

            lambda1 = torch.where(
                G_mat[2] * w_p2_bar < G_mat[3] * p1_bar,
                torch.zeros_like(p1_bar),
                torch.where(
                    G_mat[1] * w_p1_bar < G_mat[0] * p2_bar,
                    w_p1_bar / G_mat[0],
                    torch.clamp(
                        G_mat[3] * p1_bar - G_mat[2] * p2_bar,
                        max=0
                    ) / (G_mat[0] * G_mat[3] - G_mat[1] * G_mat[2] + 1e-6)
                )
            )

            lambda2 = torch.where(
                G_mat[2] * w_p2_bar < G_mat[3] * p1_bar,
                w_p2_bar / G_mat[3],
                torch.where(
                    G_mat[1] * w_p1_bar < G_mat[0] * p2_bar,
                    torch.zeros_like(p1_bar),
                    torch.clamp(
                        G_mat[0] * p2_bar - G_mat[1] * p1_bar,
                        max=0
                    ) / (G_mat[0] * G_mat[3] - G_mat[1] * G_mat[2] + 1e-6)
                )
            )

            out = u_bar + lambda1 * y1_bar + lambda2 * y2_bar
        else:
            out = u_bar

        return out

    @torch.no_grad()
    def apply(self, x, xp1, t=None): # x = now, xp1 = next state
        # remove the leading batch-of-1 dim
        x   = x.squeeze(0)    # [B, state_dim]
        xp1 = xp1.squeeze(0)  # [B, state_dim]

        # desired increment
        ref = xp1 - x         # [B, state_dim]

        G_list, h_list = [], []
        for obs in self.obstacles:
            G_i, h_i = self.compute_single_constraint(x, obs, t)
            G_list.append(G_i)
            h_list.append(h_i)
        # if you have no obstacles, just apply the reference control
        if not G_list:
            out = ref[:,2:4]
        else:
            G = torch.cat(G_list, dim=1)  # [B, num_obs, dim]
            h = torch.cat(h_list, dim=1)  # [B, num_obs]
            out = self.solve_closed_form(ref, G, h)
        # rebuild the next-state
        rt = xp1.clone()
        rt[:,2:4] = x[:,2:4] + out[:, :2]
        return rt.unsqueeze(0)
