"""
q4_plastic_numba.py
===================
Numba JIT J2 plasticity return mapping — multiplicative decomposition,
logarithmic (Hencky) strain, tanh-blended yield surface.

Pure functions, no class instances. State layout: [F_p_inv_00, F_p_inv_01,
F_p_inv_10, F_p_inv_11, eqps] (5 scalars).

Direct port of `dispsolver/material/plastic_jax.py` — same algorithm,
numpy/numba ops replace jax.numpy.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from .._jit_cache import njit_cached
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:

    @njit_cached(fastmath=True)
    def _embed_3d(F_2d: np.ndarray) -> np.ndarray:
        """Embed 2x2 deformation gradient into 3x3."""
        F3 = np.eye(3, dtype=np.float64)
        for i in range(2):
            for j in range(2):
                F3[i, j] = F_2d[i, j]
        return F3

    @njit_cached(fastmath=True)
    def _extract_22(M: np.ndarray) -> np.ndarray:
        """Extract upper-left 2x2 from 3x3."""
        out = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                out[i, j] = M[i, j]
        return out

    @njit_cached(fastmath=True)
    def pk2_voigt_j2_numba(
        F_2d: np.ndarray,
        state: np.ndarray,
        lam: float,
        mu: float,
        sigma_y0: float,
        H: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """PK2 stress + updated state for finite-strain J2 plasticity.

        Parameters
        ----------
        F_2d : (2, 2) deformation gradient
        state : (5,) [F_p_inv_00, F_p_inv_01, F_p_inv_10, F_p_inv_11, eqps]
        lam, mu : float — Lamé parameters
        sigma_y0, H : float — yield stress, hardening modulus

        Returns
        -------
        S_voigt : (3,) PK2 stress [S_xx, S_yy, S_xy]
        state_new : (5,) updated state
        """
        F_3d = _embed_3d(F_2d)
        J_F = np.linalg.det(F_3d)

        # --- trial elastic state ---
        F_p_inv_2d = state[:4].reshape(2, 2)
        det_Fp_inv = F_p_inv_2d[0, 0] * F_p_inv_2d[1, 1] - F_p_inv_2d[0, 1] * F_p_inv_2d[1, 0]
        Fp_inv_3d = np.eye(3, dtype=np.float64)
        for i in range(2):
            for j in range(2):
                Fp_inv_3d[i, j] = F_p_inv_2d[i, j]
        safe_det = max(abs(det_Fp_inv), 1e-30)
        Fp_inv_3d[2, 2] = 1.0 / safe_det

        F_e_tr = F_3d @ Fp_inv_3d
        b_e_tr = F_e_tr @ F_e_tr.T

        # --- inversion guard ---
        bad = (abs(J_F) < 1e-6) or (not _all_finite(b_e_tr))
        if bad:
            F_e_tr_safe = np.eye(3, dtype=np.float64)
            b_e_tr_safe = np.eye(3, dtype=np.float64)
        else:
            F_e_tr_safe = F_e_tr
            b_e_tr_safe = b_e_tr

        # --- spectral decomposition ---
        w, v = np.linalg.eigh(b_e_tr_safe)
        lambda_sq = np.maximum(w, 1e-30)
        eps_log = 0.5 * np.log(lambda_sq)

        tr_eps = eps_log[0] + eps_log[1] + eps_log[2]
        tau_a = lam * tr_eps + 2.0 * mu * eps_log
        p = np.mean(tau_a)
        s_a = tau_a - p
        s_norm = np.sqrt(np.sum(s_a ** 2))
        q_tr = np.sqrt(1.5) * s_norm

        eqps = state[4]
        sigma_y = sigma_y0 + H * eqps

        # --- smooth yield surface (tanh blend) ---
        epsilon_s = max(1e-3 * sigma_y, 1e-30)
        xi = (q_tr - sigma_y) / epsilon_s
        alpha_s = 0.5 * (1.0 + np.tanh(xi))

        safe_q = q_tr if q_tr > 1e-30 else 1.0
        n_a = 1.5 * s_a / safe_q
        dgamma = alpha_s * (q_tr - sigma_y) / (3.0 * mu + H)

        s_a_new = s_a - 2.0 * mu * dgamma * n_a
        tau_a_plastic = s_a_new + p
        tau_a_final = alpha_s * tau_a_plastic + (1.0 - alpha_s) * tau_a

        eps_e_new = eps_log - dgamma * n_a
        lambda_new = np.exp(eps_e_new)
        lambda_tr = np.sqrt(lambda_sq)
        lambda_final = alpha_s * lambda_new + (1.0 - alpha_s) * lambda_tr

        # --- PK2 stress ---
        tau_tensor = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for a in range(3):
                    tau_tensor[i, j] += tau_a_final[a] * v[i, a] * v[j, a]

        F3_safe = np.eye(3, dtype=np.float64) if bad else F_3d
        F3_inv = np.linalg.inv(F3_safe)
        S3 = F3_inv @ tau_tensor @ F3_inv.T
        S_voigt = np.array([S3[0, 0], S3[1, 1], S3[0, 1]])

        # --- update state ---
        lambda_tr_safe = np.maximum(lambda_tr, 1e-30)
        V_e_tr_inv = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for a in range(3):
                    V_e_tr_inv[i, j] += (1.0 / lambda_tr_safe[a]) * v[i, a] * v[j, a]

        R_e_tr = V_e_tr_inv @ F_e_tr_safe

        V_e_new = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for a in range(3):
                    V_e_new[i, j] += lambda_final[a] * v[i, a] * v[j, a]

        F_e_new = V_e_new @ R_e_tr
        F_p_inv_new = np.linalg.solve(F3_safe, F_e_new)
        eqps_new = eqps + dgamma

        state_new = np.empty(5, dtype=np.float64)
        state_new[0] = F_p_inv_new[0, 0]
        state_new[1] = F_p_inv_new[0, 1]
        state_new[2] = F_p_inv_new[1, 0]
        state_new[3] = F_p_inv_new[1, 1]
        state_new[4] = eqps_new

        return S_voigt, state_new

    @njit_cached(fastmath=True)
    def _all_finite(x: np.ndarray) -> bool:
        """Check all elements are finite."""
        for val in x.flat:
            if not np.isfinite(val):
                return False
        return True

    @njit_cached(fastmath=True)
    def stress_and_tangent_j2_numba(
        F_2d: np.ndarray,
        state: np.ndarray,
        lam: float,
        mu: float,
        sigma_y0: float,
        H: float,
        h: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PK2 stress + consistent tangent C = dS/dE via forward FD.

        Uses the same finite-difference perturbation along 3 Voigt
        directions as q4_eas._stress_and_tangent.

        Returns
        -------
        S_v : (3,) PK2 stress Voigt
        C_v : (3, 3) tangent modulus
        state_new : (5,) updated internal state
        """
        # Inversion guard
        if abs(np.linalg.det(F_2d)) <= 1e-8:
            C_iso = np.array([
                [lam + 2.0 * mu, lam, 0.0],
                [lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, mu],
            ], dtype=np.float64)
            state_copy = state.copy()
            return np.zeros(3), C_iso, state_copy

        S0, state_new = pk2_voigt_j2_numba(F_2d, state, lam, mu, sigma_y0, H)

        if not _all_finite(S0):
            C_iso = np.array([
                [lam + 2.0 * mu, lam, 0.0],
                [lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, mu],
            ], dtype=np.float64)
            state_copy = state.copy()
            return np.zeros(3), C_iso, state_copy

        FinvT = np.linalg.inv(F_2d).T

        # Three Voigt perturbation directions
        dE_dir = (
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        )

        C = np.zeros((3, 3), dtype=np.float64)
        for m in range(3):
            dF = FinvT @ dE_dir[m]
            Sp, _ = pk2_voigt_j2_numba(F_2d + h * dF, state, lam, mu, sigma_y0, H)
            for i in range(3):
                C[i, m] = (Sp[i] - S0[i]) / h

        # Symmetrize (minor symmetry)
        for i in range(3):
            for j in range(3):
                if j > i:
                    avg = 0.5 * (C[i, j] + C[j, i])
                    C[i, j] = avg
                    C[j, i] = avg

        return S0, C, state_new

else:
    def pk2_voigt_j2_numba(*args, **kwargs):
        raise ImportError("Numba not installed. Run `pip install numba` to use Numba backend.")

    def stress_and_tangent_j2_numba(*args, **kwargs):
        raise ImportError("Numba not installed. Run `pip install numba` to use Numba backend.")
