#!/usr/bin/env python3
"""Gazebo-warehouse CBF (Control Barrier Function) for SAFE planner variants.

QP-based safety filter — solves a small per-step quadratic program to find
the minimum-norm position correction that satisfies a first-order safety
constraint for every obstacle:

    min_δ  ‖δ‖²
    s.t.   ∇h_i(x)·δ + h_i(x) + κ ≥ 0     ∀ obstacle i

where h_i is the barrier function of the i-th obstacle and κ is an optional
robust-margin offset (`robust_term` in this code).  After solving, the
returned safe position is `x + δ`.  Same first-order form upstream uses in
`diffuser/models/diffusion.py:invariance_cf` and `cbf.py:solve_qp_u`, but here
we use position (not actuator command) as the decision variable since our
mecanum robot is velocity-controlled directly.

Backends:
  - qpth.qp.QPFunction  : batched GPU QP (used for trajectory-level
    correction, where we have (B*H) independent 2-D QPs)
  - scipy.optimize SLSQP: the relative-degree-2 HOCBF-QCQP of the SSF
    variants (one small QCQP per streaming step / control tick).

Correction modes:
  1. shield_state(x_norm)                — trajectory-step position fix (QP)
     (SafeDiffuser, SafeFM, SafeFlowMatcher)
  2. classifier_guidance_state(x_norm)   — additive ∇h push, no QP (Diffuser + CG)
  3. hocbf_velocity_filter_phys(...)     — RD=2 HOCBF-QCQP on the velocity (SSF)

Obstacle list format (matches upstream maze2d.py):
    obstacles = [
        {'center': (cx, cy), 'radius': r, 'order': 2},
        ...
    ]
Only n=2 (circle / ellipse) is supported by the QP linearization (the
gradient form below specializes to that).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor
from qpth.qp import QPFunction, QPSolvers


# ---------------------------------------------------------------------------
# Default warehouse obstacles — abstract safety circles in the high-traffic
# central corridor (placed on the line-density heatmap of trajectories).
# ---------------------------------------------------------------------------
WAREHOUSE_OBSTACLES: List[dict] = [
    {'center': (8.5,  10.0), 'radius': 1.5, 'order': 2},
    {'center': (16.0, 11.0), 'radius': 1.5, 'order': 2},
]


# ---------------------------------------------------------------------------
# Barrier function & gradient — physical xy in, broadcast-safe.
# ---------------------------------------------------------------------------

def barrier_value(px: Tensor, py: Tensor, obs: dict, robust_term: float = 0.05) -> Tensor:
    cx, cy = obs['center']
    n = obs.get('order', 2)
    rx = obs.get('radius_x', obs.get('radius', 1.0))
    ry = obs.get('radius_y', obs.get('radius', 1.0))
    dx = (px - cx) / rx
    dy = (py - cy) / ry
    return dx ** n + dy ** n - (1.0 + robust_term)


def barrier_gradient(px: Tensor, py: Tensor, obs: dict) -> Tuple[Tensor, Tensor]:
    cx, cy = obs['center']
    n = obs.get('order', 2)
    rx = obs.get('radius_x', obs.get('radius', 1.0))
    ry = obs.get('radius_y', obs.get('radius', 1.0))
    gx = (n / rx) * ((px - cx) / rx) ** (n - 1)
    gy = (n / ry) * ((py - cy) / ry) ** (n - 1)
    return gx, gy


# ---------------------------------------------------------------------------
# Core QP — minimal-norm position correction to keep every barrier non-negative
# (linearized about the current position).
#
# For each sample we solve:
#     min ‖δ‖²    s.t.    -G·δ ≤ h        (G = ∇h, h = h(x) — linearized)
#
# Batched form: (B, 2, 2) Q = 2·I, (B, 2) q = 0, (B, K, 2) G, (B, K) h.
# qpth solves min ½xᵀQx + qᵀx  s.t.  G x ≤ h  so we negate the RHS.
# ---------------------------------------------------------------------------

class _QPSolver:
    """Wraps qpth.QPFunction with a persistent (Q, q, e) since they never change."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._qp = QPFunction(
            eps=1e-9, verbose=-1, maxIter=30,
            solver=QPSolvers.PDIPM_BATCHED, check_Q_spd=False,
        )
        # Q = 2I (size 2), q = 0, no equality constraints
        self.Q = 2.0 * torch.eye(2, device=device, dtype=dtype)
        self.zero_q = torch.zeros(2, device=device, dtype=dtype)
        self.empty_e = torch.empty(0, device=device, dtype=dtype)

    def solve(self, G: Tensor, h: Tensor) -> Tensor:
        """G: (B, K, 2),  h: (B, K).  Returns δ: (B, 2)."""
        B = G.shape[0]
        Q = self.Q.unsqueeze(0).expand(B, -1, -1).contiguous()
        q = self.zero_q.unsqueeze(0).expand(B, -1).contiguous()
        return self._qp(Q, q, G, h, self.empty_e, self.empty_e)


# ---------------------------------------------------------------------------
# GazeboCBF — physical xy in, physical xy out.
# ---------------------------------------------------------------------------

class GazeboCBF:
    def __init__(
        self,
        obstacles: Sequence[dict] = WAREHOUSE_OBSTACLES,
        robust_term: float = 0.05,
        guidance_scale: float = 0.5,
        guidance_threshold: float = 0.10,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32,
    ):
        self.obstacles: List[dict] = [dict(o) for o in obstacles]
        self.robust_term = float(robust_term)
        self.guidance_scale = float(guidance_scale)
        self.guidance_threshold = float(guidance_threshold)
        self.device = device
        self.dtype = dtype
        self._solver = _QPSolver(device, dtype)

    # ---------------- Build linearized constraints ------------------

    def _build_constraints(self, px: Tensor, py: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (G, h) where G·δ ≤ h enforces h_i(x) + ∇h_i·δ ≥ 0  ∀ i.

        Linearization: h(x+δ) ≈ h(x) + ∇h·δ ≥ 0  →  -∇h·δ ≤ h(x)
        Robust margin handled inside `barrier_value`.

        Inputs: px, py shape (B,)   →   G: (B, K, 2),  h: (B, K)
        """
        Gs, hs = [], []
        for obs in self.obstacles:
            gx, gy = barrier_gradient(px, py, obs)            # (B,)
            h_i = barrier_value(px, py, obs, self.robust_term)  # (B,)
            G_i = torch.stack([-gx, -gy], dim=-1)              # (B, 2)
            Gs.append(G_i.unsqueeze(1))
            hs.append(h_i.unsqueeze(1))
        G = torch.cat(Gs, dim=1)   # (B, K, 2)
        h = torch.cat(hs, dim=1)   # (B, K)
        return G, h

    # ---------------- QP-based shield ------------------

    @torch.no_grad()
    def shield_positions_phys(self, xy_phys: Tensor) -> Tensor:
        """QP shield: project xy_phys into the linearized safe set.

        For points strictly safe (h>0 for all obstacles) δ=0 and they're
        unchanged.  For violating points δ moves them onto the (linearized)
        boundary with minimum L2 step.

        Input: (..., 2)  Output: same shape.
        """
        orig_shape = xy_phys.shape
        flat = xy_phys.reshape(-1, 2)
        px, py = flat[:, 0], flat[:, 1]

        G, h = self._build_constraints(px, py)

        # If every sample is already strictly safe we can skip the QP altogether.
        # h here is h(x), constraint is -∇h·δ ≤ h. Strict-safety = h ≥ 0 already.
        already_safe_mask = (h >= 0).all(dim=1)
        if already_safe_mask.all():
            return xy_phys

        # Cast everything to the solver device/dtype
        G_s = G.to(self.device, self.dtype).contiguous()
        h_s = h.to(self.device, self.dtype).contiguous()

        delta = self._solver.solve(G_s, h_s)   # (B, 2)
        delta = delta.to(xy_phys.device, xy_phys.dtype)
        delta = torch.where(
            already_safe_mask.unsqueeze(-1).to(delta.device), torch.zeros_like(delta), delta,
        )
        return (flat + delta).reshape(orig_shape)

    # ---------------- Classifier guidance (no QP — pure gradient push) ------

    @torch.no_grad()
    def classifier_guidance_phys(self, xy_phys: Tensor) -> Tensor:
        dxy = torch.zeros_like(xy_phys)
        px = xy_phys[..., 0]
        py = xy_phys[..., 1]
        for obs in self.obstacles:
            h = barrier_value(px, py, obs, self.robust_term)
            gx, gy = barrier_gradient(px, py, obs)
            grad_sq = (gx * gx + gy * gy).clamp_min(1e-8)
            active = (h < self.guidance_threshold).float()
            mag = active * self.guidance_scale / grad_sq
            dxy = dxy.clone()
            dxy[..., 0] = dxy[..., 0] + mag * gx
            dxy[..., 1] = dxy[..., 1] + mag * gy
        return dxy

    # ----------------------------------------------------------------------
    # SFP-specific: HOCBF QCQP (relative degree 2)
    #
    # SFP's velocity_net outputs a velocity field — `v_pos` advances position,
    # `v_vel` advances velocity.  Treating `v_vel` (the "acceleration" term)
    # as the control input gives the constraint h(p(v_vel)) relative degree 2:
    #     v_vel → v → p → h(p)
    # which mirrors upstream maze2d (force → velocity → position).
    #
    # Discrete dynamics used here:
    #     v_{k+1}   = v_k + v_vel · dt
    #     p_{k+1}   = p_k + v_pos · dt        (one-step look-ahead)
    #     v_{k+2}   = v_{k+1} + v_vel · dt    (assume v_vel constant)
    #     p_{k+2}   = p_{k+1} + (v_pos + v_vel·dt) · dt
    #
    # Decision: u ∈ R² = v_pos correction (Δv_pos).  We KEEP v_vel fixed at the
    # model's prediction (only velocity-output gets safety-filtered).  This is
    # because the model's v_vel is the instantaneous acceleration command and
    # v_pos is what immediately moves the robot — correcting v_pos is the
    # control authority that's directly actuated.
    #
    # HOCBF constraint (linear class-K with parameters kp, kv ∈ (0, 1]):
    #     ψ₀ = h(p_k)
    #     ψ₁ = h(p_{k+1}) - (1-kp)·h(p_k)                 ≥ 0
    #     ψ₂ = h(p_{k+2}) - (2-kp-kv)·h(p_{k+1}) + (1-kp)(1-kv)·h(p_k) ≥ 0
    # (Equivalent rearrangement of upstream `cbf.py`.)
    #
    # For n=2 (circle) both h(p_{k+1}) and h(p_{k+2}) are *quadratic* in u
    # → ψ₂ is quadratic → the inequality is a QCQP constraint.  Solver:
    # scipy SLSQP (same as upstream `solve_qcqp`).  One QCQP per planner
    # tick.  No batching (CBF is called once per state in SFP, not per
    # trajectory point).
    # ----------------------------------------------------------------------

    def _hocbf_qcqp_constraints(
        self,
        p_phys: np.ndarray,
        v_pos_nom: np.ndarray,
        v_vel: np.ndarray,
        dt: float,
        kp: float,
        kv: float,
    ):
        """Return list of (quad_coef_matrix, lin_coef_vector, const) constraints,
        one per obstacle.  Each constraint is:
            uᵀ · Q · u + l · u + c ≥ 0    (ψ₂ ≥ 0)
        with u = v_pos correction (2-vector).  All inputs are plain numpy.
        """
        p = p_phys.astype(np.float64)
        v_nom = v_pos_nom.astype(np.float64)
        a = v_vel.astype(np.float64)

        # Look-ahead positions BEFORE correction (constants under u)
        p_k = p
        # p_{k+1} = p + (v_nom + u) · dt = (p + v_nom·dt) + u·dt
        p_k1_const = p + v_nom * dt
        # p_{k+2} = p_{k+1} + (v_nom + u + v_vel·dt) · dt
        #         = (p + 2·v_nom·dt + v_vel·dt²) + 2·u·dt
        p_k2_const = p + 2.0 * v_nom * dt + a * dt * dt

        cons = []
        for obs in self.obstacles:
            cx, cy = obs['center']
            n = obs.get('order', 2)
            assert n == 2, 'HOCBF QCQP only implemented for n=2 (circular barriers)'
            rx = float(obs.get('radius_x', obs.get('radius', 1.0)))
            ry = float(obs.get('radius_y', obs.get('radius', 1.0)))
            rt = self.robust_term

            # h(p_k) — pure constant
            d0x = (p_k[0] - cx) / rx; d0y = (p_k[1] - cy) / ry
            h_k = d0x * d0x + d0y * d0y - (1.0 + rt)

            # h(p_{k+1}) = ((a1x + u_x·dt/rx)² + (a1y + u_y·dt/ry)²) - (1+rt)
            #            = (dt/rx)²·u_x² + (dt/ry)²·u_y² + 2·a1x·(dt/rx²)·u_x
            #              + 2·a1y·(dt/ry²)·u_y + (a1x² + a1y² - (1+rt))
            a1x = (p_k1_const[0] - cx) / rx
            a1y = (p_k1_const[1] - cy) / ry
            Q1 = np.array([[(dt / rx) ** 2, 0.0],
                           [0.0,            (dt / ry) ** 2]])
            l1 = np.array([2.0 * a1x * dt / rx,
                           2.0 * a1y * dt / ry])
            c1 = a1x * a1x + a1y * a1y - (1.0 + rt)
            # h(p_{k+1}) = u^T Q1 u + l1·u + c1

            # h(p_{k+2}) — same form, u coeff is 2·dt (not dt)
            a2x = (p_k2_const[0] - cx) / rx
            a2y = (p_k2_const[1] - cy) / ry
            Q2 = np.array([[(2.0 * dt / rx) ** 2, 0.0],
                           [0.0,                  (2.0 * dt / ry) ** 2]])
            l2 = np.array([2.0 * a2x * (2.0 * dt) / rx,
                           2.0 * a2y * (2.0 * dt) / ry])
            c2 = a2x * a2x + a2y * a2y - (1.0 + rt)

            # ψ₂ = h(p_{k+2}) - (2-kp-kv)·h(p_{k+1}) + (1-kp)(1-kv)·h(p_k) ≥ 0
            alpha = -(2.0 - kp - kv)             # coefficient on h(p_{k+1})
            beta = (1.0 - kp) * (1.0 - kv)        # coefficient on h(p_k)
            Q_psi = Q2 + alpha * Q1                # h(p_k) has no u dep
            l_psi = l2 + alpha * l1
            c_psi = c2 + alpha * c1 + beta * h_k

            cons.append((Q_psi, l_psi, c_psi))
        return cons

    @torch.no_grad()
    def hocbf_qcqp_velocity_filter_phys(
        self,
        p_phys,
        v_pos_phys,
        v_vel_phys,
        dt: float,
        kp: float = 0.5,
        kv: float = 0.3,
        max_speed: float = None,
    ):
        """Apply HOCBF QCQP correction to v_pos (single sample).

        Inputs may be torch tensors or numpy arrays of shape (2,).
        Returns corrected v_pos as a torch tensor on the same device/dtype.

        Solved by scipy SLSQP — runs on CPU but ~1-3ms for our 2-D + 2 obstacles.

        max_speed: if given, the admissible set U = {v : ||v|| <= max_speed} is
        part of the problem (paper Eq. 18: u in U). The nominal is first
        clipped to U, then ||v_nom + u||^2 <= max_speed^2 is a constraint of
        the QCQP. The closed-loop SSF runner passes its speed clamp here;
        None (no admissible-set constraint) is used by the open-loop
        SafeStreamingFlowPolicy rollout.
        """
        # to numpy
        def _np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().reshape(-1).astype(np.float64)
            return np.asarray(x, dtype=np.float64).reshape(-1)
        p_np = _np(p_phys)
        v_pos_np = _np(v_pos_phys)
        v_vel_np = _np(v_vel_phys)
        if max_speed is not None:
            _sp = float(np.linalg.norm(v_pos_np))
            if _sp > max_speed:
                v_pos_np = v_pos_np * (max_speed / _sp)

        cons_data = self._hocbf_qcqp_constraints(p_np, v_pos_np, v_vel_np, dt, kp, kv)

        # Quick feasibility shortcut: if all constraints already satisfied at u=0
        if all((c >= 0) for (_, _, c) in cons_data):
            if isinstance(v_pos_phys, torch.Tensor) and max_speed is None:
                return v_pos_phys
            return torch.from_numpy(v_pos_np).to(self.device, self.dtype)

        # SLSQP setup — decision variable is u (correction to v_pos)
        def obj(u):
            return float(np.sum(u * u))
        def obj_jac(u):
            return 2.0 * u
        cons_dicts = []
        for Q, l, c in cons_data:
            cons_dicts.append({
                'type': 'ineq',
                'fun': (lambda u, Q=Q, l=l, c=c: float(u @ Q @ u + l @ u + c)),
                'jac': (lambda u, Q=Q, l=l: 2.0 * Q @ u + l),
            })
        if max_speed is not None:
            cons_dicts.append({
                'type': 'ineq',
                'fun': (lambda u, v=v_pos_np: float(max_speed ** 2 - (v + u) @ (v + u))),
                'jac': (lambda u, v=v_pos_np: -2.0 * (v + u)),
            })
        res = minimize(
            obj, np.zeros(2), jac=obj_jac,
            constraints=cons_dicts, method='SLSQP',
            options={'maxiter': 30, 'ftol': 1e-9},
        )
        u_safe = res.x
        v_pos_safe = v_pos_np + u_safe

        if isinstance(v_pos_phys, torch.Tensor):
            return torch.from_numpy(v_pos_safe).to(v_pos_phys.device, v_pos_phys.dtype)
        return torch.from_numpy(v_pos_safe).to(self.device, self.dtype)


# ---------------------------------------------------------------------------
# Normalizer-aware wrappers — planners stay in normalized space.
# ---------------------------------------------------------------------------

class NormalizedCBF:
    def __init__(self, cbf: GazeboCBF, normalizer):
        self.cbf = cbf
        self.normalizer = normalizer

    @torch.no_grad()
    def shield_state(self, x_norm: Tensor) -> Tensor:
        mean = torch.as_tensor(self.normalizer.mean, device=x_norm.device, dtype=x_norm.dtype)
        std = torch.as_tensor(self.normalizer.std, device=x_norm.device, dtype=x_norm.dtype)
        xy_phys = x_norm[..., :2] * std[:2] + mean[:2]
        xy_phys_safe = self.cbf.shield_positions_phys(xy_phys)
        xy_norm_safe = (xy_phys_safe - mean[:2]) / std[:2]
        out = x_norm.clone()
        out[..., :2] = xy_norm_safe
        return out

    @torch.no_grad()
    def classifier_guidance_state(self, x_norm: Tensor) -> Tensor:
        mean = torch.as_tensor(self.normalizer.mean, device=x_norm.device, dtype=x_norm.dtype)
        std = torch.as_tensor(self.normalizer.std, device=x_norm.device, dtype=x_norm.dtype)
        xy_phys = x_norm[..., :2] * std[:2] + mean[:2]
        dxy_phys = self.cbf.classifier_guidance_phys(xy_phys)
        dxy_norm = dxy_phys / std[:2]
        out = torch.zeros_like(x_norm)
        out[..., :2] = dxy_norm
        return out

    @torch.no_grad()
    def hocbf_velocity_filter_phys(
        self, p_phys, v_pos_phys, v_vel_phys, dt: float,
        kp: float = 0.5, kv: float = 0.3, max_speed: float = None,
    ):
        """RD=2 HOCBF QCQP velocity filter for SFP (uses scipy SLSQP)."""
        return self.cbf.hocbf_qcqp_velocity_filter_phys(
            p_phys, v_pos_phys, v_vel_phys, dt, kp=kp, kv=kv, max_speed=max_speed,
        )


def build_normalized_cbf(
    normalizer,
    obstacles: Sequence[dict] = WAREHOUSE_OBSTACLES,
    robust_term: float = 0.05,
    guidance_scale: float = 0.5,
    guidance_threshold: float = 0.10,
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
) -> NormalizedCBF:
    cbf = GazeboCBF(
        obstacles=obstacles,
        robust_term=robust_term,
        guidance_scale=guidance_scale,
        guidance_threshold=guidance_threshold,
        device=device,
        dtype=dtype,
    )
    return NormalizedCBF(cbf, normalizer)
