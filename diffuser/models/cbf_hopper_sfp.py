"""
CBF safety filter for the SSF Hopper torso-height ceiling constraint.

CBFHopperHOCBF: discrete-time first-order HOCBF with a QP (SLSQP) on the action,
using the velocity-augmented barrier
    h(x) = z_max - rootz - beta * rootz_vel - delta
"""
import numpy as np
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════
#  Discrete-time 1st-order HOCBF-QP
# ═══════════════════════════════════════════════════════════════

class CBFHopperHOCBF:
    """Discrete-time first-order HOCBF with QP solver (SLSQP).

    Barrier (velocity-augmented, relative degree 1 w.r.t. u):
        h(x) = z_max - rootz - beta * rootz_vel - delta

    Discrete-time HOCBF condition:
        psi_1 = h(x_{k+1}) - h(x_k) + k1 * h(x_k) >= 0

    Expanding with Euler dynamics:
        h(x_{k+1}) - h(x_k) = -dt * rootz_vel - beta * dt * rootz_acc(u)

    So:
        psi_1(u) = -dt * zdot - beta * dt * zddot(u) + k1 * h  >= 0

    zddot(u) is linearized via MuJoCo mj_forward + finite-difference Jacobian:
        zddot(u) ≈ a0 + J * (u - u_nom)

    QP:  min  ||u - u_nom||^2
         s.t. psi_1(u) >= 0       (linear in u after linearization)
              u_lo <= u <= u_hi    (actuator bounds)

    Flight phase: J ≈ 0 → QP infeasible → passthrough nominal.
    """

    def __init__(self, env, z_max=1.5, beta=0.08, delta=0.01,
                 k1=0.15, eps_jac=1e-3, activation_threshold=0.5):
        """
        Args:
            env: gym env (needs env.unwrapped.sim for MuJoCo dynamics)
            z_max: ceiling height constraint
            beta: velocity lookahead in barrier
            delta: safety margin offset
            k1: class-K gain (0 < k1 < 1). Smaller = more conservative.
            eps_jac: finite-difference step for Jacobian
            activation_threshold: skip QP when h > this value
        """
        self.sim = env.unwrapped.sim
        self.dt = env.unwrapped.dt  # frame_skip * model.opt.timestep
        self.z_max = z_max
        self.beta = beta
        self.delta = delta
        self.k1 = k1
        self.eps_jac = eps_jac
        self.activation_threshold = activation_threshold

        self.u_lo = env.action_space.low.copy()
        self.u_hi = env.action_space.high.copy()
        self.act_dim = len(self.u_lo)

    def barrier(self, rootz, rootz_vel):
        """h(x) = z_max - rootz - beta * rootz_vel - delta"""
        return self.z_max - rootz - self.beta * rootz_vel - self.delta

    # ── MuJoCo dynamics ──────────────────────────────────────

    def _rootz_acc(self, action):
        """Compute rootz acceleration via MuJoCo forward dynamics.

        sim.forward() calls mj_forward: computes qacc from current
        (qpos, qvel, ctrl) without advancing the simulation state.
        """
        self.sim.data.ctrl[:] = action
        self.sim.forward()
        return float(self.sim.data.qacc[1])  # rootz = qpos[1]

    def _linearize_acc(self, u_nom):
        """Linearize zddot(u) ≈ a0 + J @ (u - u_nom).

        Returns (a0, J) where a0 = zddot(u_nom), J = d(zddot)/du.
        Uses forward finite differences (act_dim + 1 mj_forward calls).
        """
        ctrl_backup = self.sim.data.ctrl.copy()
        a0 = self._rootz_acc(u_nom)
        J = np.zeros(self.act_dim)
        for i in range(self.act_dim):
            u_pert = u_nom.copy()
            u_pert[i] += self.eps_jac
            # Clamp to action bounds
            u_pert[i] = min(u_pert[i], self.u_hi[i])
            actual_eps = u_pert[i] - u_nom[i]
            if abs(actual_eps) < 1e-8:
                # At upper bound → use backward diff
                u_pert[i] = u_nom[i] - self.eps_jac
                actual_eps = u_pert[i] - u_nom[i]
            a_pert = self._rootz_acc(u_pert)
            J[i] = (a_pert - a0) / actual_eps
        self.sim.data.ctrl[:] = ctrl_backup
        return a0, J

    # ── QP solve ─────────────────────────────────────────────

    def _solve_qp(self, u_nom, const, A):
        """Solve: min ||u - u_nom||^2  s.t. const + A @ (u - u_nom) >= 0.

        Returns (u_safe, feasible).
        """
        bounds = list(zip(self.u_lo, self.u_hi))

        def objective(u):
            return float(np.sum((u - u_nom) ** 2))

        def objective_jac(u):
            return 2.0 * (u - u_nom)

        def con_fun(u):
            return const + A @ (u - u_nom)

        def con_jac(_u):
            return A

        cons = [{'type': 'ineq', 'fun': con_fun, 'jac': con_jac}]

        res = minimize(
            objective, np.clip(u_nom, self.u_lo, self.u_hi),
            jac=objective_jac,
            constraints=cons,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-10},
        )
        # Verify feasibility at solution
        feasible = con_fun(res.x) >= -1e-6
        return res.x, feasible

    # ── Main entry point ─────────────────────────────────────

    def apply(self, obs, action):
        """Apply HOCBF-QP safety filter.

        Args:
            obs: physical observation (11-dim numpy), unnormalized
            action: physical action (3-dim numpy), unnormalized

        Returns:
            action_safe: filtered action (3-dim numpy)
        """
        rootz = obs[0]
        rootz_vel = obs[6]
        h = self.barrier(rootz, rootz_vel)

        # Early exit: well within safe region → skip QP
        if h > self.activation_threshold:
            return action.copy()

        # Only filter when ascending (descent is naturally safe)
        if rootz_vel <= 0:
            return action.copy()

        # ── Linearize dynamics ──
        u_nom = action.copy()
        a0, J = self._linearize_acc(u_nom)

        # ── Build HOCBF constraint ──
        # psi_1(u) = const + A @ (u - u_nom) >= 0
        const = -self.dt * rootz_vel - self.beta * self.dt * a0 + self.k1 * h
        A = -self.beta * self.dt * J  # d(psi_1)/du

        # Nominal already satisfies → no correction
        if const >= 0:
            return action.copy()

        # No control authority (flight phase: J ≈ 0 → A ≈ 0)
        if np.linalg.norm(A) < 1e-10:
            return action.copy()

        # ── Solve QP ──
        u_safe, feasible = self._solve_qp(u_nom, const, A)

        if not feasible:
            return action.copy()  # passthrough nominal

        return u_safe
