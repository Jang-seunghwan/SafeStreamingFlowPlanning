"""
CBF (Control Barrier Function) safety filters.

Two modes, selected by args.action_dim:
  - action_dim == 0 (SSF): 2nd-order ECBF with double-integrator dynamics in
    physical space, sub-stepped at the simulator rate. A QP finds the
    minimum-norm safe acceleration; the resulting position replaces the
    proposed position of the next state.
  - action_dim > 0 (SafeFM / SafeFlowMatcher; obstacle geometry for Diffuser+CG
    and SafeDiffuser): relaxed 1st-order CBF-QP in normalized space.
"""
import math

import torch
from qpth.qp import QPFunction, QPSolvers


class CBF:
    def __init__(self, norm_mins, norm_maxs, args):
        self.device = norm_mins.device
        self.norm_mins = norm_mins
        self.norm_maxs = norm_maxs
        self.obstacles = args.obstacles
        self.action_dim = args.action_dim

        # Position indices in state vector (after action dims)
        self.pos_x_idx = args.pos_x_idx
        self.pos_y_idx = args.pos_y_idx
        self.center_offset = args.center_offset
        self.robust_term = args.robust_term

        if self.action_dim == 0:
            self._setup_sfp_cbf(args)
        else:
            self._setup_first_order(args)

    # ================================================================
    # Setup
    # ================================================================

    def _setup_sfp_cbf(self, args):
        """2nd-order ECBF for SFP with double integrator dynamics.

        System: ṗ = v, v̇ = u (no damping)
        Velocity from forward-diff of trajectory positions.
        ECBF pole placement: (s + α)² → α₀ = α², α₁ = 2α
        """
        self.alpha_cbf = args.alpha_cbf
        self.alpha_0 = self.alpha_cbf ** 2
        self.alpha_1 = 2 * self.alpha_cbf
        self.dt_traj = 0.1   # physical time between waypoints (10 Hz)
        self.n_sub = args.n_sub_cbf  # sub-steps per flow step
        self.u_max = args.u_max_cbf
        self.ecbf_activation_threshold = args.ecbf_threshold

    def _setup_first_order(self, args):
        """Relaxed 1st-order CBF for diffuser/CFM (normalized space)."""
        self.alpha = args.eps
        self.rho = args.rho
        self.relax_threshold = args.relax_threshold
        self.xr = 2 / (self.norm_maxs[self.pos_x_idx] - self.norm_mins[self.pos_x_idx])
        self.yr = 2 / (self.norm_maxs[self.pos_y_idx] - self.norm_mins[self.pos_y_idx])

    # ================================================================
    # 2nd-order ECBF (SSF), physical space
    # ================================================================

    def denormalize(self, x_norm):
        """Normalized [-1,1] → physical coordinates."""
        return (x_norm + 1) * (self.norm_maxs - self.norm_mins) / 2 + self.norm_mins

    def ecbf_constraint_sfp(self, px, py, vx, vy, obs):
        """2nd-order ECBF constraint for double integrator.

        System: ṗ = v, v̇ = u  (γ=1, β=0)

        b(p) = (dx/rx)^n + (dy/ry)^n - (1+δ)
        Lf_b  = ∇b · v
        Lf²_b = v^T H_b v        (β=0)
        LgLf_b = ∇b              (γ=1)

        Exact ECBF:        -LgLf_b · u  ≤  Lf²_b + α₁·Lf_b + α₀·b
        Implemented here:  -LgLf_b · u  ≤          α₁·Lf_b + α₀·b    (Lf²_b omitted, see below)

        Args:
            px, py: current position (physical) [B]
            vx, vy: forward-diff velocity (physical, m/s) [B]
            obs: obstacle dict
        Returns:
            G [B,1,2], h [B,1], b [B]
        """
        cx, cy = obs['center']
        n = obs['order']
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))
        obs_px = cx + self.center_offset
        obs_py = cy + self.center_offset

        dx = (px - obs_px) / rx
        dy = (py - obs_py) / ry

        b = dx**n + dy**n - (1.0 + self.robust_term)

        # ∇b
        db_dpx = n * dx**(n - 1) / rx
        db_dpy = n * dy**(n - 1) / ry

        # Lf_b = ∇b · v
        Lf_b = db_dpx * vx + db_dpy * vy

        # The Lf²_b = v^T H_b v term of the exact ECBF is omitted by design.
        # For the circular (n = 2, rx = ry = r) obstacles used here
        # Lf²_b = 2|v|^2/r^2 >= 0, so omitting it gives a stricter constraint
        # than the exact ECBF. With the term kept, a state moving tangentially
        # to the obstacle (Lf_b = 0) satisfies the constraint with u = 0 whenever
        # 2|v|^2/r^2 >= α₀(1 + δ), even inside the obstacle (b >= -(1 + δ));
        # for α = 10, δ = 0.01, r = 0.3 m this holds for |v| >= 2.1 m/s.
        # Implemented constraint: -∇b · u ≤ α₁·Lf_b + α₀·b
        G = -torch.stack([db_dpx, db_dpy], dim=1).unsqueeze(1)        # [B, 1, 2]
        h = (self.alpha_1 * Lf_b + self.alpha_0 * b).unsqueeze(1)     # [B, 1]

        return G, h, b

    @torch.no_grad()
    def solve_qp_ecbf(self, u_nom, G, h):
        """min ||u - u_nom||²  s.t.  G·u ≤ h  and  |u| ≤ u_max"""
        B = u_nom.shape[0]
        dtype = u_nom.dtype
        device = u_nom.device

        Q = 2.0 * torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(B, 2, 2)
        q = -2.0 * u_nom

        # Box constraints
        G_box = torch.tensor([[ 1.,  0.],
                              [-1.,  0.],
                              [ 0.,  1.],
                              [ 0., -1.]], device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 2)
        h_box = torch.full((B, 4), self.u_max, device=device, dtype=dtype)

        G_all = torch.cat([G, G_box], dim=1)
        h_all = torch.cat([h, h_box], dim=1)

        e = torch.empty(0, device=device, dtype=dtype)
        try:
            u = QPFunction(
                eps=1e-12, verbose=-1, maxIter=30,
                solver=QPSolvers.PDIPM_BATCHED, check_Q_spd=True
            )(Q, q, G_all, h_all, e, e)
        except RuntimeError:
            # Solver failure: fall back to the (clipped) nominal input.
            u = torch.clamp(u_nom, -self.u_max, self.u_max)
        return u

    @torch.no_grad()
    def _apply_sfp_cbf(self, x, next_x):
        """ECBF safety filter for SFP with sub-stepping.

        Each flow step (dt_traj=0.1s) is subdivided into N_sub sub-steps
        (dt_sub=0.01s, the simulator rate), so the constraint is enforced
        at the simulator resolution within each 0.1 s trajectory step.

        At each sub-step, the ECBF constraints of the obstacles whose barrier
        is below the activation threshold are evaluated and a QP finds the
        minimum-norm safe acceleration (nominal u = 0). The position reached
        by the sub-stepped double integrator replaces the proposed position;
        the velocity channel keeps the proposed velocity.

        Args:
            x: current state (normalized) [1, state_dim]
            next_x: proposed next state (normalized) [1, state_dim]
        Returns:
            corrected next state (normalized) [1, state_dim]
        """
        xi, yi = self.pos_x_idx, self.pos_y_idx

        # Denormalize positions to physical space
        x_phys = self.denormalize(x)
        next_phys = self.denormalize(next_x)

        px_start = x_phys[:, xi]
        py_start = x_phys[:, yi]
        px_end = next_phys[:, xi]
        py_end = next_phys[:, yi]

        # Forward-diff velocity over the full step (physical m/s)
        dt = self.dt_traj
        vx_full = (px_end - px_start) / dt
        vy_full = (py_end - py_start) / dt

        # If every obstacle is far away (barrier above threshold), keep the proposed state.
        any_nearby = False
        for obs in self.obstacles:
            _, _, b_i = self.ecbf_constraint_sfp(px_start, py_start, vx_full, vy_full, obs)
            if b_i.min() < self.ecbf_activation_threshold:
                any_nearby = True
        if not any_nearby:
            return next_x

        # --- Sub-stepping ECBF ---
        N_sub = self.n_sub
        dt_sub = dt / N_sub
        B = px_start.shape[0]
        u_nom = torch.zeros(B, 2, device=self.device, dtype=px_start.dtype)

        # Initialize sub-step state
        p_x = px_start.clone()
        p_y = py_start.clone()
        v_x = vx_full.clone()
        v_y = vy_full.clone()

        for _ in range(N_sub):
            # Build ECBF constraints at current sub-step state
            G_list, h_list = [], []
            for obs in self.obstacles:
                G_i, h_i, b_i = self.ecbf_constraint_sfp(p_x, p_y, v_x, v_y, obs)
                if b_i.min() < self.ecbf_activation_threshold:
                    G_list.append(G_i)
                    h_list.append(h_i)

            if len(G_list) > 0:
                G = torch.cat(G_list, dim=1)
                h = torch.cat(h_list, dim=1)
                u_opt = self.solve_qp_ecbf(u_nom, G, h)
            else:
                u_opt = u_nom

            # Semi-implicit Euler sub-step
            v_x = v_x + u_opt[:, 0] * dt_sub
            v_y = v_y + u_opt[:, 1] * dt_sub
            p_x = p_x + v_x * dt_sub
            p_y = p_y + v_y * dt_sub

        # Corrected state in normalized space (LimitsNormalizer formula for the position)
        corrected = next_x.clone()
        corrected[:, xi] = 2 * (p_x - self.norm_mins[xi]) / (self.norm_maxs[xi] - self.norm_mins[xi]) - 1
        corrected[:, yi] = 2 * (p_y - self.norm_mins[yi]) / (self.norm_maxs[yi] - self.norm_mins[yi]) - 1
        return corrected

    # ================================================================
    # Relaxed 1st-order CBF (SafeFM / SafeFlowMatcher), normalized space
    # ================================================================

    @torch.no_grad()
    def compute_single_constraint(self, x, obs, t):
        cx, cy = obs['center']
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))
        pi = self.pos_x_idx + self.action_dim
        pj = self.pos_y_idx + self.action_dim
        off_x = 2 * (cx + self.center_offset - self.norm_mins[self.pos_x_idx]) / (self.norm_maxs[self.pos_x_idx] - self.norm_mins[self.pos_x_idx]) - 1
        off_y = 2 * (cy + self.center_offset - self.norm_mins[self.pos_y_idx]) / (self.norm_maxs[self.pos_y_idx] - self.norm_mins[self.pos_y_idx]) - 1
        dx = (x[:, pi:pi+1] - off_x) / self.xr / rx
        dy = (x[:, pj:pj+1] - off_y) / self.yr / ry
        order = obs['order']

        L1 = order * dy**(order-1) / self.yr / ry
        L2 = order * dx**(order-1) / self.xr / rx

        # Relaxation weight on the shared slack: large early in the flow, 0 after relax_threshold.
        if t <= self.relax_threshold:
            ratio = t / self.relax_threshold
            sign = 200.0 * (1 - math.exp(3 * (ratio - 1)))
        else:
            sign = 0.0
        rx_t = sign * torch.ones_like(L1)
        G = torch.cat([-L1, -L2, rx_t], dim=1).unsqueeze(1)
        b = dy**order + dx**order - (1 + self.robust_term)**order
        finite_time_term = torch.sign(b) * torch.abs(b)**self.rho
        h = self.alpha * finite_time_term
        return G, h

    @torch.no_grad()
    def solve_qp(self, u_ref, G, h):
        pi = self.pos_x_idx + self.action_dim
        pj = self.pos_y_idx + self.action_dim
        q_u = -torch.stack([u_ref[:, pj], u_ref[:, pi]], dim=1)
        q_r = torch.zeros_like(q_u[:, :1])
        q = 2 * torch.cat([q_u, q_r], dim=1)
        Q = 2 * torch.eye(3, device=self.device).unsqueeze(0).expand(u_ref.size(0), 3, 3)
        e = torch.empty(0, device=self.device)
        try:
            out = QPFunction(
                eps=1e-12, verbose=-1, notImprovedLim=10, maxIter=20,
                solver=QPSolvers.PDIPM_BATCHED, check_Q_spd=True
            )(Q, q, G, h, e, e)
        except RuntimeError:
            out = self.solve_closed_form(u_ref, G, h)
        return out

    def solve_closed_form(self, u_ref, G, h):
        """Closed-form fallback for the relaxed QP (one or two active constraints)."""
        pi = self.pos_x_idx + self.action_dim
        pj = self.pos_y_idx + self.action_dim
        u = torch.stack([u_ref[:, pj], u_ref[:, pi]], dim=1)
        u_relax = torch.zeros_like(u[:, :1])
        u_bar = torch.cat([u, u_relax], dim=1)

        num_obs = G.shape[1]

        if num_obs == 1:
            G0 = G[:, 0, :]
            h0 = h[:, 0:1]
            p = h0 - torch.sum(G0 * u_bar, dim=1, keepdim=True)
            G_norm_sq = torch.sum(G0 * G0, dim=1, keepdim=True) + 1e-6
            lam = torch.clamp(p, max=0) / G_norm_sq
            out = u_bar + lam * G0
        else:
            G0, G1 = G[:, 0, :], G[:, 1, :]
            h0, h1 = h[:, 0:1], h[:, 1:2]

            p1 = h0 - torch.sum(G0 * u_bar, dim=1, keepdim=True)
            p2 = h1 - torch.sum(G1 * u_bar, dim=1, keepdim=True)

            g11 = torch.sum(G0 * G0, dim=1, keepdim=True)
            g12 = torch.sum(G0 * G1, dim=1, keepdim=True)
            g21 = g12
            g22 = torch.sum(G1 * G1, dim=1, keepdim=True)

            wp1 = torch.clamp(p1, max=0)
            wp2 = torch.clamp(p2, max=0)

            det = g11 * g22 - g12 * g21 + 1e-6

            lambda1 = torch.where(
                g21 * wp2 < g22 * p1,
                torch.zeros_like(p1),
                torch.where(
                    g12 * wp1 < g11 * p2,
                    wp1 / g11,
                    torch.clamp(g22 * p1 - g21 * p2, max=0) / det
                )
            )

            lambda2 = torch.where(
                g21 * wp2 < g22 * p1,
                wp2 / g22,
                torch.where(
                    g12 * wp1 < g11 * p2,
                    torch.zeros_like(p1),
                    torch.clamp(g11 * p2 - g12 * p1, max=0) / det
                )
            )

            out = u_bar + lambda1 * G0 + lambda2 * G1
        return out

    @torch.no_grad()
    def _apply_first_order(self, x, xp1, t):
        """Relaxed CBF-QP on the step x -> xp1 of a [1, H, D] sample."""
        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)
        ref = xp1 - x
        G_list, h_list = [], []
        for obs in self.obstacles:
            G_i, h_i = self.compute_single_constraint(x, obs, t)
            G_list.append(G_i)
            h_list.append(h_i)
        pi = self.pos_x_idx + self.action_dim
        pj = self.pos_y_idx + self.action_dim
        G = torch.cat(G_list, dim=1)
        h = torch.cat(h_list, dim=1)
        out = self.solve_qp(ref, G, h)
        rt = xp1.clone()
        rt[:, pj] = x[:, pj] + out[:, 0]
        rt[:, pi] = x[:, pi] + out[:, 1]
        return rt.unsqueeze(0)

    # ================================================================
    # Routing
    # ================================================================

    @torch.no_grad()
    def apply(self, x, xp1, t=None):
        """Safe version of the step x -> xp1 (same shape as xp1)."""
        if self.action_dim == 0:
            return self._apply_sfp_cbf(x, xp1)
        return self._apply_first_order(x, xp1, t)
