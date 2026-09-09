"""
q4_eas_numba.py
===============
Numba JIT EAS Q4 element with finite-strain J2 plasticity — internal
Newton for EAS parameters (alpha) with damped line search.

Direct port of `q4_eas.py`'s `compute_eas_j2_contributions`, calls
the Numba J2 kernel from `q4_plastic_numba.py` instead of a Python
class method.

Returns
-------
f_e       : (8,)   condensed internal force
K_e       : (8,8)  condensed tangent
alpha_new : (4,)   converged EAS parameters
state_new : (4,5)  updated per-GP material state (or zeros)
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
    from .q4_plastic_numba import pk2_voigt_j2_numba, stress_and_tangent_j2_numba

    _GP2_VALS = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
    _W2_VALS = np.array([1.0, 1.0], dtype=np.float64)  # product of 1D weights = 1.0

    @njit_cached(fastmath=True)
    def _sd(xi: float, eta: float):
        """Shape function derivatives."""
        dN_dxi = np.array([
            -0.25 * (1.0 - eta),
             0.25 * (1.0 - eta),
             0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ], dtype=np.float64)
        dN_deta = np.array([
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
             0.25 * (1.0 + xi),
             0.25 * (1.0 - xi),
        ], dtype=np.float64)
        return dN_dxi, dN_deta

    @njit_cached(fastmath=True)
    def _jacobian(xi: float, eta: float, coords: np.ndarray):
        """Jacobian, det(J), inv(J)."""
        dN_dxi, dN_deta = _sd(xi, eta)
        J = np.zeros((2, 2), dtype=np.float64)
        for i in range(4):
            J[0, 0] += dN_dxi[i] * coords[i, 0]
            J[0, 1] += dN_dxi[i] * coords[i, 1]
            J[1, 0] += dN_deta[i] * coords[i, 0]
            J[1, 1] += dN_deta[i] * coords[i, 1]
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=np.float64) / detJ
        return J, detJ, invJ

    @njit_cached(fastmath=True)
    def _enhanced_grad_modes(xi: float, eta: float, detJ: float,
                             J0: np.ndarray, detJ0: float):
        """4 enhanced deformation-gradient modes (4, 2, 2).

        Fenh_k = (detJ0/detJ) * D_k @ J0^{-T} -- TRANSPOSE FIX
        2026-09-08, see q4_eas_jax.py::_enhanced_grad_modes for the
        derivation and measured impact (invisible for axis-aligned
        elements since J0 is then diagonal; up to ~10x bending error
        once the UL reference frame rotates off-axis).
        """
        J0inv = np.linalg.inv(J0)
        s = detJ0 / detJ
        Dk = [
            np.array([[xi, 0.0], [0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, eta], [0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0], [xi, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0], [0.0, eta]], dtype=np.float64),
        ]
        modes = [s * (D @ J0inv.T) for D in Dk]
        return modes

    @njit_cached(fastmath=True)
    def _voigt_sym(P: np.ndarray) -> np.ndarray:
        """Voigt [xx, yy, xy] of sym(P) for 2x2 P."""
        return np.array([P[0, 0], P[1, 1], P[0, 1] + P[1, 0]], dtype=np.float64)

    @njit_cached(fastmath=True)
    def _BL_columns(Ft: np.ndarray, gX: np.ndarray, gY: np.ndarray) -> np.ndarray:
        """Total-Lagrangian strain-displacement operator B_L (3x8)."""
        F11, F12 = Ft[0, 0], Ft[0, 1]
        F21, F22 = Ft[1, 0], Ft[1, 1]
        B = np.zeros((3, 8), dtype=np.float64)
        for a in range(4):
            gx, gy = gX[a], gY[a]
            B[0, 2 * a] = F11 * gx
            B[1, 2 * a] = F12 * gy
            B[2, 2 * a] = F11 * gy + F12 * gx
            B[0, 2 * a + 1] = F21 * gx
            B[1, 2 * a + 1] = F22 * gy
            B[2, 2 * a + 1] = F21 * gy + F22 * gx
        return B

    @njit_cached(fastmath=True)
    def compute_eas_j2_contributions_numba_status(
        coords: np.ndarray,       # (4,2)
        u_elem: np.ndarray,       # (8,)
        alpha: np.ndarray,        # (4,) EAS params (warm start)
        state_elem: np.ndarray,   # (4,5) per-GP state or zeros
        lam: float,
        mu: float,
        sigma_y0: float,
        H: float,
        thickness: float = 1.0,
        max_local_iter: int = 12,
        local_tol: float = 1e-11,
        F_n: np.ndarray = None,  # (4,2,2) or ndarray with shape (0,)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """EAS Q4 element with J2 plasticity, fully Numba-compiled.

        Parameters
        ----------
        coords : (4,2) nodal coordinates
        u_elem : (8,) element displacements
        alpha : (4,) EAS parameters
        state_elem : (4,5) per-GP state
        lam, mu, sigma_y0, H : J2 material parameters
        thickness : float
        max_local_iter : int — max Newton iterations for alpha
        local_tol : float — convergence tolerance for alpha residual
        F_n : (4,2,2) or None — UL mode total F at last converged step

        Returns
        -------
        f_e : (8,) force
        K_e : (8,8) tangent
        alpha_new : (4,) converged alpha
        state_new : (4,5) updated state
        """
        n_gp = 4
        J0, detJ0, invJ0 = _jacobian(0.0, 0.0, coords)

        # Pre-compute per-GP reference geometry (independent of alpha)
        gp_geom_xi = np.empty(n_gp, dtype=np.float64)
        gp_geom_eta = np.empty(n_gp, dtype=np.float64)
        gp_geom_gX = np.empty((n_gp, 4), dtype=np.float64)
        gp_geom_gY = np.empty((n_gp, 4), dtype=np.float64)
        gp_geom_Fenh = np.empty((n_gp, 4, 2, 2), dtype=np.float64)
        gp_geom_w = np.empty(n_gp, dtype=np.float64)
        gp_geom_Fc = np.empty((n_gp, 2, 2), dtype=np.float64)

        gp_points_xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0),
                                 -1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
        gp_points_eta = np.array([-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0),
                                   1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)

        for k in range(n_gp):
            xi = gp_points_xi[k]
            eta = gp_points_eta[k]
            _, detJ, invJ = _jacobian(xi, eta, coords)
            dN_dxi, dN_deta = _sd(xi, eta)
            gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
            Fenh_list = _enhanced_grad_modes(xi, eta, detJ, J0, detJ0)

            gp_geom_xi[k] = xi
            gp_geom_eta[k] = eta
            for i in range(4):
                gp_geom_gX[k, i] = gX[i]
                gp_geom_gY[k, i] = gY[i]
                for a in range(2):
                    for b in range(2):
                        gp_geom_Fenh[k, i, a, b] = Fenh_list[i][a, b]

            gp_geom_w[k] = detJ * 1.0 * thickness  # W2[k] = 1.0 for 2-point product

            # Compatible deformation gradient
            ux = u_elem[0::2]
            uy = u_elem[1::2]
            Hc = np.zeros((2, 2), dtype=np.float64)
            for i in range(4):
                Hc[0, 0] += ux[i] * gX[i]
                Hc[0, 1] += ux[i] * gY[i]
                Hc[1, 0] += uy[i] * gX[i]
                Hc[1, 1] += uy[i] * gY[i]
            Fc = np.eye(2) + Hc

            if F_n is not None and F_n.shape[0] == n_gp:
                Fc = Fc @ F_n[k]

            gp_geom_Fc[k] = Fc

        # --- Newton for alpha ---
        alpha_curr = alpha.copy()
        state_new = np.zeros((n_gp, 5), dtype=np.float64)
        converged_local = False
        da_inf = np.inf

        for _ in range(max_local_iter):
            f_a = np.zeros(4, dtype=np.float64)
            K_aa = np.zeros((4, 4), dtype=np.float64)
            for k in range(n_gp):
                _, _, gX, gY, Fenh, w, Fc = (
                    gp_geom_xi[k], gp_geom_eta[k], gp_geom_gX[k], gp_geom_gY[k],
                    gp_geom_Fenh[k], gp_geom_w[k], gp_geom_Fc[k]
                )
                # Total deformation gradient with EAS enhancement
                Ft = Fc.copy()
                for j in range(4):
                    Ft += alpha_curr[j] * Fenh[j]

                sg = state_elem[k] if state_elem.shape == (n_gp, 5) else None
                F2d = Ft[:2, :2]
                S_v, C_v, _ = stress_and_tangent_j2_numba(F2d, sg if sg is not None else np.zeros(5), lam, mu, sigma_y0, H)

                # G matrix
                G = np.zeros((3, 4), dtype=np.float64)
                for j in range(4):
                    GT_Fenh = Ft.T @ Fenh[j]
                    gvec = _voigt_sym(GT_Fenh)
                    for i in range(3):
                        G[i, j] = gvec[i]

                # f_a
                for i in range(3):
                    for j in range(4):
                        f_a[j] += G[i, j] * S_v[i] * w

                # K_aa (material + geometric) -- material term was a
                # multiply-by-0.0 placeholder before this fix; the inner
                # alpha-Newton tangent was missing the material stiffness
                # entirely, converging alpha to a wrong stationary point.
                for a in range(4):
                    for b in range(4):
                        P = 0.5 * (Fenh[a].T @ Fenh[b] + Fenh[b].T @ Fenh[a])
                        Kmat = 0.0
                        for i in range(3):
                            for j in range(3):
                                Kmat += G[i, a] * C_v[i, j] * G[j, b]
                        K_aa[a, b] += Kmat * w
                        # geometric part
                        St00 = S_v[0]
                        St11 = S_v[1]
                        St01 = S_v[2]
                        Kgeo = St00 * P[0, 0] + St11 * P[1, 1] + 2.0 * St01 * P[0, 1]
                        K_aa[a, b] += Kgeo * w

            fn = np.linalg.norm(f_a)
            if fn < local_tol:
                converged_local = True
                break

            # Regularized solve
            trace_K = (K_aa[0, 0] + K_aa[1, 1] + K_aa[2, 2] + K_aa[3, 3]) / 4.0
            reg = 1e-12 * (trace_K + 1e-30)
            for i in range(4):
                K_aa[i, i] += reg
            dalpha = np.linalg.solve(K_aa, -f_a)

            # Line search
            ls = 1.0
            alpha_try = alpha_curr + ls * dalpha
            # Evaluate at trial
            f_a_try = np.zeros(4, dtype=np.float64)
            for k in range(n_gp):
                _, _, _, _, Fenh, w, Fc = (
                    gp_geom_xi[k], gp_geom_eta[k], gp_geom_gX[k], gp_geom_gY[k],
                    gp_geom_Fenh[k], gp_geom_w[k], gp_geom_Fc[k]
                )
                Ft = Fc.copy()
                for j in range(4):
                    Ft += alpha_try[j] * Fenh[j]
                sg = state_elem[k] if state_elem.shape == (n_gp, 5) else None
                F2d = Ft[:2, :2]
                S_v, _, _ = stress_and_tangent_j2_numba(F2d, sg if sg is not None else np.zeros(5), lam, mu, sigma_y0, H)
                for j in range(4):
                    GT_Fenh = Ft.T @ Fenh[j]
                    gvec = _voigt_sym(GT_Fenh)
                    for i in range(3):
                        f_a_try[j] += gvec[i] * S_v[i] * w

            fn_try = np.linalg.norm(f_a_try)
            while ls > 1e-3 and fn_try >= fn:
                ls *= 0.5
                alpha_try = alpha_curr + ls * dalpha
                f_a_try[:] = 0.0
                for k in range(n_gp):
                    _, _, _, _, Fenh, w, Fc = (
                        gp_geom_xi[k], gp_geom_eta[k], gp_geom_gX[k], gp_geom_gY[k],
                        gp_geom_Fenh[k], gp_geom_w[k], gp_geom_Fc[k]
                    )
                    Ft = Fc.copy()
                    for j in range(4):
                        Ft += alpha_try[j] * Fenh[j]
                    sg = state_elem[k] if state_elem.shape == (n_gp, 5) else None
                    F2d = Ft[:2, :2]
                    S_v, _, _ = stress_and_tangent_j2_numba(F2d, sg if sg is not None else np.zeros(5), lam, mu, sigma_y0, H)
                    for j in range(4):
                        GT_Fenh = Ft.T @ Fenh[j]
                        gvec = _voigt_sym(GT_Fenh)
                        for i in range(3):
                            f_a_try[j] += gvec[i] * S_v[i] * w
                fn_try = np.linalg.norm(f_a_try)

            alpha_curr = alpha_try
            da_inf = 0.0
            for i in range(4):
                d_i = abs(ls * dalpha[i])
                if d_i > da_inf:
                    da_inf = d_i

        # --- Final assembly at converged alpha ---
        K_uu = np.zeros((8, 8), dtype=np.float64)
        K_ua = np.zeros((8, 4), dtype=np.float64)
        K_aa = np.zeros((4, 4), dtype=np.float64)
        f_u = np.zeros(8, dtype=np.float64)
        f_a = np.zeros(4, dtype=np.float64)

        for k in range(n_gp):
            _, _, gX, gY, Fenh, w, Fc = (
                gp_geom_xi[k], gp_geom_eta[k], gp_geom_gX[k], gp_geom_gY[k],
                gp_geom_Fenh[k], gp_geom_w[k], gp_geom_Fc[k]
            )
            Ft = Fc.copy()
            for j in range(4):
                Ft += alpha_curr[j] * Fenh[j]

            sg = state_elem[k] if state_elem.shape == (n_gp, 5) else None
            F2d = Ft[:2, :2]
            S_v, C_v, sg_new = stress_and_tangent_j2_numba(
                F2d, sg if sg is not None else np.zeros(5), lam, mu, sigma_y0, H
            )

            for i in range(5):
                state_new[k, i] = sg_new[i]

            BL = _BL_columns(Ft, gX, gY)

            # G matrix
            G = np.zeros((3, 4), dtype=np.float64)
            for j in range(4):
                GT_Fenh = Ft.T @ Fenh[j]
                gvec = _voigt_sym(GT_Fenh)
                for i in range(3):
                    G[i, j] = gvec[i]

            # f_u
            for i in range(8):
                for j in range(3):
                    f_u[i] += BL[j, i] * S_v[j] * w

            # f_a
            for i in range(3):
                for j in range(4):
                    f_a[j] += G[i, j] * S_v[i] * w

            # Stiffness blocks
            St = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]], dtype=np.float64)

            # K_uu = BL^T C BL + geo
            K_uu += (BL.T @ C_v @ BL) * w
            # geometric K_uu
            grad_N = np.zeros((4, 2), dtype=np.float64)
            for i in range(4):
                grad_N[i, 0] = gX[i]
                grad_N[i, 1] = gY[i]
            gamma = grad_N @ St @ grad_N.T
            for i in range(4):
                for j in range(4):
                    K_uu[2*i, 2*j] += gamma[i, j] * w
                    K_uu[2*i+1, 2*j+1] += gamma[i, j] * w

            # K_ua = BL^T C_v G + geo_ua
            K_ua += (BL.T @ C_v @ G) * w
            # geometric K_ua
            for a in range(4):
                for i_ in range(2):
                    ei = np.zeros(2, dtype=np.float64)
                    ei[i_] = 1.0
                    for kk in range(4):
                        Pr = np.outer(grad_N[a], Fenh[kk][i_, :])
                        Ps = 0.5 * (Pr + Pr.T)
                        Kgeo_ua_val = (St[0, 0] * Ps[0, 0] + St[1, 1] * Ps[1, 1]
                                      + 2.0 * St[0, 1] * Ps[0, 1])
                        K_ua[2*a + i_, kk] += Kgeo_ua_val * w

            # K_aa = G^T C_v G + geo_aa
            K_aa += (G.T @ C_v @ G) * w
            for a in range(4):
                for b in range(4):
                    P = 0.5 * (Fenh[a].T @ Fenh[b] + Fenh[b].T @ Fenh[a])
                    Kgeo_aa = St[0, 0] * P[0, 0] + St[1, 1] * P[1, 1] + 2.0 * St[0, 1] * P[0, 1]
                    K_aa[a, b] += Kgeo_aa * w

        # Condensation
        trace_Kaa = (K_aa[0, 0] + K_aa[1, 1] + K_aa[2, 2] + K_aa[3, 3]) / 4.0
        reg_aa = 1e-10 * (trace_Kaa + 1e-30)
        for i in range(4):
            K_aa[i, i] += reg_aa
        K_aa_inv_KuaT = np.linalg.solve(K_aa, K_ua.T)
        K_e = K_uu - K_ua @ K_aa_inv_KuaT
        f_e = f_u - K_ua @ np.linalg.solve(K_aa, f_a)

        # Element-local convergence status (see the JAX sibling): 0.0 when the
        # alpha-Newton met |f_alpha| < local_tol, otherwise the last accepted
        # ||d(alpha)||_inf, or inf on a non-finite result. Never clamped --
        # non-convergence is reported and the solver cuts the increment back.
        alpha_status = 0.0 if converged_local else da_inf
        for i in range(8):
            if not np.isfinite(f_e[i]):
                alpha_status = np.inf
        for i in range(4):
            if not np.isfinite(alpha_curr[i]):
                alpha_status = np.inf

        return f_e, K_e, alpha_curr, state_new, alpha_status

    @njit_cached(fastmath=True)
    def compute_eas_j2_contributions_numba(
        coords: np.ndarray, u_elem: np.ndarray, alpha: np.ndarray,
        state_elem: np.ndarray, lam: float, mu: float, sigma_y0: float,
        H: float, thickness: float = 1.0, max_local_iter: int = 12,
        local_tol: float = 1e-11, F_n: np.ndarray = None,
    ):
        """Back-compatible 4-tuple wrapper (drops the trailing `status`)."""
        f_e, K_e, a, s_e, _st = compute_eas_j2_contributions_numba_status(
            coords, u_elem, alpha, state_elem, lam, mu, sigma_y0, H,
            thickness, max_local_iter, local_tol, F_n)
        return f_e, K_e, a, s_e

    @njit_cached(fastmath=True, parallel=True)
    def assemble_q4_eas_j2_batch_numba(
        elem_coords: np.ndarray,   # (N, 4, 2)
        u_elems: np.ndarray,       # (N, 8)
        alpha_elems: np.ndarray,   # (N, 4) warm-start EAS params
        state_elems: np.ndarray,   # (N, 4, 5)
        lam: float, mu: float, sigma_y0: float, H: float,
        thicknesses: np.ndarray,   # (N,)
    ):
        """Multi-threaded Numba batch assembly for N Q4_EAS + J2 elements.

        TL mode only (F_n_gps not threaded through -- matches this
        codebase's Numba precedent of not covering UL mode, see q4_numba.py).
        """
        n_elems = elem_coords.shape[0]
        f_all = np.empty((n_elems, 8), dtype=np.float64)
        K_all = np.empty((n_elems, 8, 8), dtype=np.float64)
        alpha_all = np.empty((n_elems, 4), dtype=np.float64)
        state_all = np.empty((n_elems, 4, 5), dtype=np.float64)
        status_all = np.empty(n_elems, dtype=np.float64)

        for e in numba.prange(n_elems):
            f_e, K_e, a_e, s_e, st_e = compute_eas_j2_contributions_numba_status(
                elem_coords[e], u_elems[e], alpha_elems[e], state_elems[e],
                lam, mu, sigma_y0, H, thicknesses[e],
            )
            f_all[e] = f_e
            K_all[e] = K_e
            alpha_all[e] = a_e
            state_all[e] = s_e
            status_all[e] = st_e

        return f_all, K_all, alpha_all, state_all, status_all

else:
    def compute_eas_j2_contributions_numba(*args, **kwargs):
        raise ImportError("Numba is not installed. Run `pip install numba` to use Numba backend.")

    def compute_eas_j2_contributions_numba_status(*args, **kwargs):
        raise ImportError("Numba is not installed. Run `pip install numba` to use Numba backend.")

    def assemble_q4_eas_j2_batch_numba(*args, **kwargs):
        raise ImportError("Numba is not installed. Run `pip install numba` to use Numba backend.")
