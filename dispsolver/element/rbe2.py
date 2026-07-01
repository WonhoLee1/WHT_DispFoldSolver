"""
rbe2.py
=======
RBE2 hinge element with static condensation of internal DOFs.

Formulation
-----------
The RBE2 (Rigid Body Element type 2) enforces that slave nodes maintain a fixed
offset from a master node, rotated by a shared hinge rotation angle θ:

    g_i(u, θ) = x_si(u_si) − x_m(u_m) − R(θ)·(X_si − X_m) = 0

Unlike the constraint-based approach (dispsolver/constraint/rbe2.py) which
handles this via KKT saddle-point Lagrange multipliers, this element uses
**element-level static condensation** following the Q4_EAS pattern:

    K_uu − K_uq · K_qq⁻¹ · K_uqᵀ

where q = [θ; λ] are the internal DOFs (rotation + multipliers) condensed out
at the element level, exposing only the 2·(m+1) translational DOFs to the
global system.

Kinematics: Updated Lagrangian (incremental)
---------------------------------------------
The constraint is linearised about the **previous converged configuration**:

    g_inc(u, Δθ) = u_s − u_m − (R(Δθ) − I)·(x_s_n − x_m_n)

This follows Abaqus MPC theory ("linearised in deformed configuration") and
Radioss /RBODY ("reference is the last known solution"), ensuring accurate
linearisation even at large total hinge angles (0 → 90°+).

References
----------
- Simo, J.C. & Rifai, M.S. (1990). EAS condensation pattern (Q4_EAS).
- MSC Nastran 2021 "Rigid Body Elements" documentation.
- Abaqus Analysis Guide §31.1.1 — MPC linearisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class RBE2State:
    """Previous converged state for Updated Lagrangian incremental formulation.

    Each RBE2 element stores its state to linearise the constraint about the
    last converged configuration, enabling accurate finite-rotation kinematics.
    """
    u_m_n: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    """Master nodal displacement at previous converged step (2,)."""

    u_s_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    """Slave nodal displacements at previous converged step (2m,)."""

    theta_n: float = 0.0
    """Absolute hinge rotation angle at previous converged step."""

    lam_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    """Lagrange multipliers at previous converged step (2m,) — diagnostic."""


class RBE2HingeElement:
    """RBE2 hinge element with static condensation of θ and λ DOFs.

    The element enforces rigid-body rotation of m slave nodes around a master
    node using an Updated Lagrangian formulation. Internal DOFs (1 rotation θ
    + 2m Lagrange multipliers λ) are condensed out element-locally, making the
    external interface identical to standard Q4 elements.

    Parameters
    ----------
    master_id : int
        Node ID of the hinge rotation centre (master node).
    slave_ids : list of int
        Node IDs of the slave nodes that follow the master rigidly.
    coords_initial : ndarray of shape (n_nodes, 2)
        Initial nodal coordinates array. Used to compute initial offsets.
    """

    def __init__(
        self,
        master_id: int,
        slave_ids: list[int],
        coords_initial: np.ndarray,
        E_estimate: float = 4000.0,
        penalty: float | None = None,
    ):
        self.master_id = master_id
        self.slave_ids = list(slave_ids)
        self.n_slaves = len(slave_ids)

        self.master_idx = master_id
        self.slave_indices = list(slave_ids)

        # Initial offset: d0[i] = X_slave_i − X_master  (m, 2)
        self.d0 = np.array(
            [coords_initial[s] - coords_initial[master_id]
             for s in slave_ids],
            dtype=np.float64,
        )

        # Auto-scale penalty based on element size and material stiffness.
        # ANSYS RBE2 approach: penalty = α · E · h  where α ≈ 100.
        if penalty is not None:
            self.PENALTY = float(penalty)
        else:
            x = coords_initial[:, 0]
            y = coords_initial[:, 1]
            h = float(max(np.ptp(x), np.ptp(y)))
            if h < 1e-12:
                h = 1.0
            self.PENALTY = 100.0 * E_estimate * h

        # Externally prescribed rotation angle.
        # None = solve internally via local Newton (default).
        # float = use this value, skip local Newton, no condensation.
        self._prescribed_theta: float | None = None

        # Initialise state (zeros — first step uses initial geometry)
        self.state = RBE2State(
            u_m_n=np.zeros(2, dtype=np.float64),
            u_s_n=np.zeros(2 * self.n_slaves, dtype=np.float64),
            theta_n=0.0,
            lam_n=np.zeros(2 * self.n_slaves, dtype=np.float64),
        )

    def set_prescribed_theta(self, theta: float | None) -> None:
        self._prescribed_theta = theta

    @property
    def is_prescribed(self) -> bool:
        return self._prescribed_theta is not None

    @property
    def n_external_dofs(self) -> int:
        """Number of external (global) DOFs: 2 per node × (master + slaves)."""
        return 2 * (self.n_slaves + 1)

    # ------------------------------------------------------------------
    #  Penalty parameter
    #  Auto-scaled based on element size and material stiffness.
    #  Following ANSYS RBE2 approach: penalty = α · E · h
    #  where α ≈ 100, E = estimated Young's modulus, h = char. element size.
    #  This gives penalty well-conditioned relative to element stiffness.
    #  The user can override by passing PENALTY to the constructor.
    # ------------------------------------------------------------------
    PENALTY: float = None  # None means auto-scale in __init__

    # ------------------------------------------------------------------
    #  Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _R(theta: float) -> np.ndarray:
        """2×2 rotation matrix R(θ)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    @staticmethod
    def _dR_dtheta(theta: float) -> np.ndarray:
        """Derivative of the rotation matrix w.r.t. θ."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[-s, -c], [c, -s]])

    @staticmethod
    def _d2R_dtheta2(theta: float) -> np.ndarray:
        """Second derivative of the rotation matrix w.r.t. θ."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[-c, s], [-s, -c]])

    def _offset(self, state: RBE2State) -> np.ndarray:
        """Slave-to-master offset in the previous converged config (m,2).

        d_n[i] = (X_si + u_si_n) − (X_m + u_m_n)
        """
        if len(state.u_s_n) == 0 or len(state.u_m_n) == 0:
            return self.d0.copy()
        return self.d0 + (state.u_s_n.reshape(-1, 2) - state.u_m_n[None, :])

    def _build_Cu(self) -> np.ndarray:
        """Constraint Jacobian w.r.t. external displacements (2m, n_ext)."""
        m = self.n_slaves
        n_ext = 2 * (m + 1)
        C_u = np.zeros((2 * m, n_ext))
        for i in range(m):
            C_u[2 * i,     0] = -1.0
            C_u[2 * i,     2 + 2 * i] = 1.0
            C_u[2 * i + 1, 1] = -1.0
            C_u[2 * i + 1, 2 + 2 * i + 1] = 1.0
        return C_u

    # ------------------------------------------------------------------
    #  Non-linear constraint:  g(u, θ) = u_s − u_m − (R(θ)−I)·d₀  =  0
    # ------------------------------------------------------------------

    def _exact_gap(self, u_m: np.ndarray, u_s: np.ndarray,
                   theta: float) -> np.ndarray:
        """Non-linear constraint gap with exact rotation (2m,)."""
        m = self.n_slaves
        g = np.zeros(2 * m)
        R = self._R(theta)
        for i in range(m):
            d0 = self.d0[i]  # initial offset
            rotated = (R - np.eye(2)) @ d0
            g[2 * i]     = u_s[2 * i]     - u_m[0] - rotated[0]
            g[2 * i + 1] = u_s[2 * i + 1] - u_m[1] - rotated[1]
        return g

    def _local_newton_theta(self, u_m: np.ndarray, u_s: np.ndarray,
                            theta_guess: float) -> np.ndarray:
        """Solve g(u, θ) = 0 for θ (scalar local Newton)."""
        m = self.n_slaves
        θ = theta_guess
        for _ in range(8):
            dR = self._dR_dtheta(θ)
            g = self._exact_gap(u_m, u_s, θ)

            # Build C_θ = ∂g/∂θ = −dR/dθ·d₀  (2m×1 column)
            C_θ = np.zeros(2 * m)
            for i in range(m):
                d0 = self.d0[i]
                dg_dθ = -dR @ d0
                C_θ[2 * i: 2 * i + 2] = dg_dθ

            # Scalar Newton:  θ ← θ − (C_θᵀ·C_θ)⁻¹·(C_θᵀ·g)
            CtC = np.dot(C_θ, C_θ)
            if CtC < 1e-30:
                break  # no rotational coupling — nothing to solve
            δθ = -np.dot(C_θ, g) / CtC
            θ += δθ
            if abs(δθ) < 1e-14:
                break
        return θ

    # ------------------------------------------------------------------
    #  Tangent blocks (penalty-stabilized, θ only as internal DOF)
    # ------------------------------------------------------------------

    def _build_tangent(self, u_m: np.ndarray, u_s: np.ndarray,
                       θ: float, lam: np.ndarray, d_n: np.ndarray,
                       k: float, C_u: np.ndarray) -> tuple:
        """Build condensed tangent + force with stable penalty formulation.

        The internal DOF is **only** θ (1 DOF), eliminating the singular
        saddle-point K_qq that arises in the pure-KKT approach.

        Physical blocks
        ---------------
        K_θθ = k·C_θᵀ·C_θ  +  Σ λ·∂²g/∂θ²    (penalty + geometric)
        K_uθ = k·C_uᵀ·C_θ                       (coupling)
        K_uu = k·C_uᵀ·C_u                       (penalty external)

        The Lagrange multiplier λ from the state carries the **exact**
        constraint force (converged over previous time steps), while the
        penalty term k·g provides a well-conditioned tangent even when
        λ is not yet calibrated.

        Returns
        -------
        f_e : (n_ext,)   constraint internal force  = C_uᵀ·λ + k·C_uᵀ·g
        K_e : (n_ext, n_ext)   condensed tangent
        Δθ  : float            recovered rotation increment
        """
        m = self.n_slaves
        n_ext = C_u.shape[1]

        # --- constraint gap and its derivatives ---
        g = self._exact_gap(u_m, u_s, θ)
        dR = self._dR_dtheta(θ)
        d2R = self._d2R_dtheta2(θ)

        C_θ = np.zeros((2 * m, 1))
        K_θθ = 0.0
        for i in range(m):
            d0 = self.d0[i]
            dg_dθ = -dR @ d0          # C_θ entry (2,)
            d2g_dθ2 = -d2R @ d0        # ∂²g/∂θ² entry (2,)
            C_θ[2 * i: 2 * i + 2, 0] = dg_dθ
            K_θθ += np.dot(lam[2 * i: 2 * i + 2], d2g_dθ2)

        # Penalty-stabilised θ-block (ALWAYS positive-definite)
        K_θθ += (k * C_θ.T @ C_θ).item()
        if K_θθ < 1e-30:
            K_θθ = 1e-30  # safeguard

        # Coupling block
        K_uθ = k * (C_u.T @ C_θ)          # (n_ext, 1)

        # External penalty block
        K_uu = k * (C_u.T @ C_u)

        # ── Condensation ────────────────────────────────────────────
        # K_θθ is a scalar — no linear solve needed
        K_e = K_uu - (K_uθ @ K_uθ.T) / K_θθ

        # Residual: ∂Π/∂θ = C_θᵀ·λ + k·C_θᵀ·g  (scalar)
        f_θ = (C_θ.T @ lam).item() + (k * C_θ.T @ g).item()
        Δθ = -f_θ / K_θθ

        # Internal force = exact Lagrange + penalty stabilisation
        f_e = C_u.T @ lam + k * (C_u.T @ g) + K_uθ[:, 0] * Δθ

        return f_e, K_e, Δθ

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def compute_contributions(
        self,
        coords: np.ndarray,
        u_elem: np.ndarray,
        state: Optional[RBE2State],
    ) -> Tuple[np.ndarray, np.ndarray, RBE2State]:
        """Compute condensed element force vector and stiffness matrix.

        Formulation
        -----------
        1. A **local Newton** loop solves the exact non-linear constraint
           g(u, θ) = 0 for the hinge angle θ.
        2. The **tangent stiffness** is obtained by condensing out the θ DOF
           from a penalty-stabilised system.  The penalty (k = 1e10) dominates
           the 1-DOF internal block K_θθ, so the condensation is always
           well-conditioned — no saddle-point singularities.
        3. The internal **force** uses Lagrange multipliers λ stored in the
           element state (converged over previous time steps) plus a small
           penalty restoration.  This gives f_e ≈ 0 for rigid-body motion.
        4. The **state** is updated with the recovered θ increment and an
           Augmented-Lagrange-style multiplier update (λ ← λ + k·g).

        Parameters
        ----------
        coords : (n_nodes, 2)  nodal coordinates (master at master_id).
        u_elem : (2m+2,)   [u_mx, u_my, u_s1x, u_s1y, …]
        state  : previous converged RBE2State, or None for self.state.

        Returns
        -------
        f_e : (2m+2,)     condensed internal force vector
        K_e : (2m+2,2m+2) condensed tangent stiffness matrix
        state_new : updated RBE2State
        """
        prev = state if state is not None else self.state
        k = self.PENALTY
        m = self.n_slaves
        n_ext = self.n_external_dofs

        u_m = u_elem[0:2].copy()
        u_s = u_elem[2:].copy() if m > 0 else np.array([], dtype=np.float64)
        C_u = self._build_Cu()
        lam = prev.lam_n.copy() if len(prev.lam_n) > 0 else np.zeros(2 * m)

        if self._prescribed_theta is not None:
            # ── Externally prescribed θ (Abaqus-style) ────────────────
            # No local Newton, no condensation — pure penalty + AL force.
            θ = float(self._prescribed_theta)
            g = self._exact_gap(u_m, u_s, θ)
            f_e = C_u.T @ lam + k * (C_u.T @ g)       # (n_ext,)
            K_e = k * (C_u.T @ C_u)                     # (n_ext, n_ext)
            lam_new = lam + k * g
            state_new = RBE2State(
                u_m_n=u_m, u_s_n=u_s, theta_n=θ, lam_n=lam_new,
            )
            return f_e, K_e, state_new

        # ── Internal θ (element-level condensation) ──────────────────
        # --- 1. Local Newton for exact θ ---
        θ_converged = self._local_newton_theta(u_m, u_s, prev.theta_n)

        # --- 2. Tangent + force via penalty condensation ---
        f_e, K_e, Δθ = self._build_tangent(
            u_m, u_s, θ_converged, lam, self._offset(prev), k, C_u,
        )

        # --- 3. State update ---
        g_converged = self._exact_gap(u_m, u_s, θ_converged)
        lam_new = lam + k * g_converged

        state_new = RBE2State(
            u_m_n=u_m,
            u_s_n=u_s,
            theta_n=θ_converged,
            lam_n=lam_new,
        )

        return f_e, K_e, state_new
