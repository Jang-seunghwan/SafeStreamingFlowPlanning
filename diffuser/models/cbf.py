import torch
import numpy as np
from scipy.optimize import minimize


def _scalar(v):
    """Convert tensor or python number to plain float."""
    return v.item() if isinstance(v, torch.Tensor) else float(v)


class CBF:
    def __init__(self, norm_mins, norm_maxs, args):
        self.device = norm_mins.device
        self.norm_mins = norm_mins
        self.norm_maxs = norm_maxs
        self.obstacles = args.obstacles

        # Environment-specific center offset:
        # Large: render_offset=0.2 => physical = center - 0.5
        # Medium/Umaze: render_offset=0.0 => physical = center - 0.7
        self.dataset = getattr(args, 'dataset', '')
        self.center_offset = -0.5 if 'large' in self.dataset else -0.7

        # system params (MuJoCo point mass of Maze2D)
        self.mass = 4.1887902
        self.gear = 100.0
        self.gamma = self.gear / self.mass  # γ = gear/m

        self.damping = 1.0
        self.beta = self.damping / self.mass  # β = d/m

        self.dt_env = 0.01  # environment dt

        self.robust_term = args.robust_term

        # ---- Discrete-time HOCBF parameters ----
        # Class-K function coefficients (linear: α(s) = k·s)
        #   ψ_0 = h(x_k)
        #   ψ_1 = h(x_{k+1}) - (1-kv)·h(x_k)
        #   ψ_2 = h(x_{k+2}) + (kv+kp-2)·h(x_{k+1}) + (1-kv)(1-kp)·h(x_k) ≥ 0
        # Roots of char. eq.: z = (1-kv), (1-kp) ∈ [0,1)
        # Constraint: 0 < kv ≤ kp ≤ 1
        self.kv_hocbf = args.kv_hocbf
        self.kp_hocbf = args.kp_hocbf

    # ==================================================================
    #  Shared utilities
    # ==================================================================

    def denormalize(self, x_norm):
        """Convert normalized state [-1, 1] to physical state."""
        return (x_norm + 1) * (self.norm_maxs - self.norm_mins) / 2 + self.norm_mins

    def normalize(self, x_phys):
        """Convert physical state to normalized state [-1, 1]."""
        return 2 * (x_phys - self.norm_mins) / (self.norm_maxs - self.norm_mins) - 1

    def step_dynamics_phys(self, x_phys, u):
        """Semi-implicit Euler dynamics in PHYSICAL space (MuJoCo style).

        MuJoCo: v_next first, then p_next = p + dt*v_next.

        Args:
            x_phys: physical state [B, 4] = [py, px, vy, vx]
            u: action [B, 2] = [uy, ux]
        """
        dt = self.dt_env
        vy_next = x_phys[:, 2] * (1.0 - self.beta * dt) + self.gamma * dt * u[:, 0]
        vx_next = x_phys[:, 3] * (1.0 - self.beta * dt) + self.gamma * dt * u[:, 1]
        # MuJoCo clips velocity to [-5, 5]
        vy_next = vy_next.clamp(-5.0, 5.0)
        vx_next = vx_next.clamp(-5.0, 5.0)
        py_next = x_phys[:, 0] + dt * vy_next
        px_next = x_phys[:, 1] + dt * vx_next
        return torch.stack([py_next, px_next, vy_next, vx_next], dim=1)

    def _barrier_value(self, py, px, obs):
        """Compute barrier h(p) in physical coordinates.

        h(p) = ((py-obs_py)/ry)^n + ((px-obs_px)/rx)^n - (1+δ)
        Works with both tensors and numpy scalars.
        """
        cx, cy = obs['center']
        n = obs['order']
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))
        obs_py = cy + self.center_offset
        obs_px = cx + self.center_offset
        dy = (py - obs_py) / ry
        dx = (px - obs_px) / rx
        return dy**n + dx**n - (1.0 + self.robust_term)

    def _make_correction_info(self, x, x_next_nominal, x_next, u_nom, u_opt):
        return {
            'current': x.clone(),
            'before': x_next_nominal.clone(),
            'after': x_next.clone(),
            'u_nom': u_nom.clone(),
            'u_opt': u_opt.clone(),
            'u_safe_phys': u_opt.clone(),
        }

    # ==================================================================
    #  DISCRETE-TIME HOCBF  (Semi-implicit Euler / MuJoCo-consistent)
    #
    #  Reference: Xiong et al., "Discrete-Time Control Barrier Function"
    #
    #  Semi-implicit Euler (matches MuJoCo and step_dynamics_phys):
    #    v_{k+1} = v_k·d + γ·dt·u           where d = 1 - β·dt
    #    p_{k+1} = p_k + dt·v_{k+1}          (uses NEW velocity → depends on u)
    #
    #  Step k+1→k+2 (no action at k+1):
    #    v_{k+2} = v_{k+1}·d
    #    p_{k+2} = p_{k+1} + dt·v_{k+2}
    #
    #  Both h(p_{k+1}) and h(p_{k+2}) are quadratic in u for n=2
    #  → combined ψ_2 is still a QCQP.
    #
    #  HOCBF constraint (relative degree 2):
    #    ψ_2 = h(p_{k+2}) + (kv+kp-2)·h(p_{k+1}) + (1-kv)(1-kp)·h(p_k) ≥ 0
    # ==================================================================

    def _dt_hocbf_qcqp_data(self, x_phys, obs):
        """Build QCQP constraint data for one obstacle.

        Semi-implicit Euler (MuJoCo-consistent):
            d  = 1 - β·dt
            c1 = γ·dt²                        (u coeff for p_{k+1})
            c2 = c1 + γ·dt²·(1+d)             (TOTAL u coeff for p_{k+2}, accumulated)

            p_{k+1} = (p_k + dt·v_k·d) + c1·u   (linear in u)
            p_{k+2} = (p_k + dt·v_k·d·(1+d)) + c2·u   (linear in u)

        For n=2:
            h(p_{k+1}) = P1·u² + q1·u + r1     (quadratic in u)
            h(p_{k+2}) = P2·u² + q2·u + r2     (quadratic in u)

        Combined ψ_2 = (P2 + α·P1)·u² + (q2 + α·q1)·u + (r2 + α·r1 + β_h·h_k)
            where α = kv+kp-2, β_h = (1-kv)(1-kp)

        Returns: (P_yy, P_xx, qy, qx, s_val, h_k, h1_constraint)
            All values are plain Python floats (extracted from B=1 tensors).
            h1_constraint = (P1_yy, P1_xx, q1_y, q1_x, r1) for h(p_{k+1}) ≥ 0.
        """
        dt = self.dt_env
        py, px = x_phys[:, 0], x_phys[:, 1]
        vy, vx = x_phys[:, 2], x_phys[:, 3]

        cx, cy = obs['center']
        n = obs['order']
        assert n == 2, f"DT-HOCBF QCQP requires order=2, got {n}"
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))
        obs_py = cy + self.center_offset
        obs_px = cx + self.center_offset

        # h(p_k) — current barrier value (no u dependence)
        h_k = self._barrier_value(py, px, obs)

        # Velocity decay and u-coefficients
        d = 1.0 - self.beta * dt          # velocity decay per step
        c1 = self.gamma * dt * dt          # u coeff for p_{k+1}
        # p_{k+2} accumulates u from both steps: c1 (from p_{k+1}) + c2_step (new)
        c2_step = c1 * (1.0 + d)          # additional u coeff at step k+2
        c2 = c1 + c2_step                 # TOTAL u coeff for p_{k+2}

        # ---- h(p_{k+1}): semi-implicit p_{k+1} = p_k + dt·v_{k+1} ----
        # Constant part (no u): a1 = p_k + dt·v_k·d
        ay1 = py + dt * vy * d
        ax1 = px + dt * vx * d
        Ay1 = ay1 - obs_py
        Ax1 = ax1 - obs_px

        P1_yy = c1**2 / ry**2
        P1_xx = c1**2 / rx**2
        q1_y = 2.0 * c1 * Ay1 / ry**2
        q1_x = 2.0 * c1 * Ax1 / rx**2
        r1 = Ay1**2 / ry**2 + Ax1**2 / rx**2 - (1.0 + self.robust_term)

        # ---- h(p_{k+2}): p_{k+2} = p_{k+1} + dt·v_{k+2}, no action at k+1 ----
        # Constant part (no u): a2 = p_k + dt·v_k·d·(1+d)
        ay2 = py + dt * vy * d * (1.0 + d)
        ax2 = px + dt * vx * d * (1.0 + d)
        Ay2 = ay2 - obs_py
        Ax2 = ax2 - obs_px

        P2_yy = c2**2 / ry**2
        P2_xx = c2**2 / rx**2
        q2_y = 2.0 * c2 * Ay2 / ry**2
        q2_x = 2.0 * c2 * Ax2 / rx**2
        r2 = Ay2**2 / ry**2 + Ax2**2 / rx**2 - (1.0 + self.robust_term)

        # ---- Combined ψ_2 coefficients ----
        # ψ_2 = h(p_{k+2}) + α·h(p_{k+1}) + β_h·h(p_k) ≥ 0
        kv, kp = self.kv_hocbf, self.kp_hocbf
        alpha = kv + kp - 2.0              # coefficient for h(p_{k+1})
        beta_h = (1.0 - kv) * (1.0 - kp)  # coefficient for h(p_k)

        P_yy = P2_yy + alpha * P1_yy      # float
        P_xx = P2_xx + alpha * P1_xx       # float
        qy = q2_y + alpha * q1_y           # tensor
        qx = q2_x + alpha * q1_x           # tensor
        s = r2 + alpha * r1 + beta_h * h_k # tensor

        # Convert to plain floats for scipy (assumes B=1)
        # Also return h(p_{k+1}) ≥ margin constraint for direct safety enforcement
        # Margin must exceed SAFETY_MARGIN (0.01) used for violation counting,
        # plus solver tolerance buffer
        h1_margin = 0.015
        h1_constraint = (P1_yy, P1_xx, _scalar(q1_y), _scalar(q1_x), _scalar(r1) - h1_margin)
        return (P_yy, P_xx, _scalar(qy), _scalar(qx), _scalar(s), h_k, h1_constraint)

    @torch.no_grad()
    def solve_qcqp(self, u_nom_np, constraints_data):
        """Solve QCQP via scipy SLSQP.

        min  ||u - u_nom||²
        s.t. P_yy·uy² + P_xx·ux² + qy·uy + qx·ux + s ≥ 0   ∀ obstacle
             -1 ≤ u ≤ 1                                       (actuator limits)

        Box constraint matches MuJoCo's ctrlrange clipping.
        """
        def objective(u):
            return float(np.sum((u - u_nom_np) ** 2))

        def objective_jac(u):
            return 2.0 * (u - u_nom_np)

        cons = []
        for (Pyy, Pxx, qy, qx, sc) in constraints_data:
            # closure captures by default arg
            def _con(u, Pyy=Pyy, Pxx=Pxx, qy=qy, qx=qx, sc=sc):
                return Pyy * u[0]**2 + Pxx * u[1]**2 + qy * u[0] + qx * u[1] + sc

            def _jac(u, Pyy=Pyy, Pxx=Pxx, qy=qy, qx=qx):
                return np.array([2.0 * Pyy * u[0] + qy, 2.0 * Pxx * u[1] + qx])

            cons.append({'type': 'ineq', 'fun': _con, 'jac': _jac})

        # Box constraint: u_safe ∈ [-1, 1] (MuJoCo actuator limits).
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        u0 = np.clip(u_nom_np, -1.0, 1.0)

        result = minimize(
            objective, u0, jac=objective_jac,
            constraints=cons, method='SLSQP', bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-10},
        )

        # Check infeasibility: evaluate ψ₂ at solution
        u_sol = result.x
        for i, (Pyy, Pxx, qy, qx, sc) in enumerate(constraints_data):
            psi2 = Pyy * u_sol[0]**2 + Pxx * u_sol[1]**2 + qy * u_sol[0] + qx * u_sol[1] + sc
            if psi2 < -1e-6:
                print(f"[HOCBF INFEASIBLE] obs {i}: ψ₂={psi2:.6f}, "
                      f"u_sol=[{u_sol[0]:.4f}, {u_sol[1]:.4f}], "
                      f"u_nom=[{u_nom_np[0]:.4f}, {u_nom_np[1]:.4f}], "
                      f"success={result.success}")

        return u_sol

    @torch.no_grad()
    def apply(self, x, next_x, t=None):
        """Apply the discrete-time HOCBF safety filter (QCQP) to one step.

        Uses semi-implicit Euler (MuJoCo-consistent) for both HOCBF constraint
        and state propagation — no dynamics mismatch.
        """
        x_phys = self.denormalize(x)
        next_x_phys = self.denormalize(next_x)

        # PD control → nominal action
        p = x_phys[:, :2]
        v = x_phys[:, 2:]
        p_des = next_x_phys[:, :2]
        v_des = next_x_phys[:, 2:]
        u_nom = (p_des - p) + (v_des - v)

        # Nominal next state (semi-implicit Euler / MuJoCo)
        x_next_nominal_phys = self.step_dynamics_phys(x_phys, u_nom)

        # Build QCQP constraints
        qcqp_data = []       # list of (Pyy, Pxx, qy, qx, s)  -- floats
        safe_list = []        # list of h_k tensors
        b_threshold = float('inf')  # always constrain all obstacles

        for obs in self.obstacles:
            Pyy, Pxx, qy, qx, s, h_k, h1_con = self._dt_hocbf_qcqp_data(x_phys, obs)
            safe_list.append(h_k)
            if _scalar(h_k.min()) < b_threshold:
                # ψ₂ ≥ 0 constraint (HOCBF 2-step invariance)
                qcqp_data.append((Pyy, Pxx, qy, qx, s))
                # h(p_{k+1}) ≥ 0 constraint (direct next-step safety)
                qcqp_data.append(h1_con)

        # Clip u_nom to actuator limits (MuJoCo clips, so HOCBF must too)
        u_nom_clipped = u_nom.clamp(-1.0, 1.0)

        # No nearby obstacles → use clipped nominal
        if len(qcqp_data) == 0:
            x_next_clipped_phys = self.step_dynamics_phys(x_phys, u_nom_clipped)
            x_next_clipped = self.normalize(x_next_clipped_phys)
            x_next_nominal = self.normalize(x_next_nominal_phys)
            return x_next_clipped, safe_list, self._make_correction_info(
                x, x_next_nominal, x_next_clipped, u_nom, u_nom_clipped)

        # Check if clipped nominal u already satisfies all constraints
        u_nom_np = u_nom_clipped[0].detach().cpu().numpy().astype(np.float64)
        violated = False
        for (Pyy, Pxx, qy, qx, sc) in qcqp_data:
            if Pyy * u_nom_np[0]**2 + Pxx * u_nom_np[1]**2 + qy * u_nom_np[0] + qx * u_nom_np[1] + sc < 0:
                violated = True
                break

        if not violated:
            u_opt = u_nom_clipped.clone()
        else:
            u_opt_np = self.solve_qcqp(u_nom_np, qcqp_data)
            u_opt = torch.tensor(u_opt_np, dtype=u_nom.dtype, device=u_nom.device).unsqueeze(0)

        # Propagate (semi-implicit Euler / MuJoCo)
        x_next_phys = self.step_dynamics_phys(x_phys, u_opt)
        x_next = self.normalize(x_next_phys)
        x_next_nominal = self.normalize(x_next_nominal_phys)

        return x_next, safe_list, self._make_correction_info(
            x, x_next_nominal, x_next, u_nom, u_opt)

    def is_point_safe(self, pos_phys, margin=0.01):
        """Check if a single point is safe (CBF > margin for all obstacles), see is_point_safe below."""
        return is_point_safe(pos_phys, self.obstacles, self.center_offset, self.robust_term, margin)


def is_point_safe(pos_phys, obstacles, center_offset, robust_term, margin=0.01):
    """Start/goal rule of the harnesses: h(p) = ((py-oy)/ry)^n + ((px-ox)/rx)^n - (1 + robust_term) > margin
    for every obstacle (obstacle centre = configured centre + center_offset).

    Args:
        pos_phys: physical position [py, px] (numpy array or list)

    Returns:
        is_safe: True if safe, False if violating any obstacle
        cbf_values: list of CBF values for each obstacle
    """
    py, px = pos_phys[0], pos_phys[1]
    cbf_values = []

    for obs in obstacles:
        cx, cy = obs['center']
        n = obs['order']
        rx = obs.get('radius_x', obs.get('radius', 1.0))
        ry = obs.get('radius_y', obs.get('radius', 1.0))

        obs_py = cy + center_offset
        obs_px = cx + center_offset

        dy = (py - obs_py) / ry
        dx = (px - obs_px) / rx
        cbf_value = dy**n + dx**n - (1.0 + robust_term)
        cbf_values.append(cbf_value)

    is_safe = all(v > margin for v in cbf_values)
    return is_safe, cbf_values
