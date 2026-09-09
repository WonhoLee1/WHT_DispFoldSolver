"""
q4_visco_hybrid_reduced_numba.py
==================================
Numba port of `q4_visco_hybrid_reduced_jax.compute_single_reduced_hybrid_jax`
(CPE4RH): 1-point reduced integration + Flanagan-Belytschko hourglass
stabilization + element-constant hybrid pressure, on the finite-strain
Arruda-Boyce/Neo-Hookean/Yeoh + Prony viscoelastic (PSA) material.

See the JAX file's module docstring for the full formulation rationale and
the scope note on why this is built (Abaqus element-name completeness,
verified by patch test) independent of whether any mesh in this repo
actually selects it (`dev_log/session_20260730_reduced_integration_failure.md`
already found 1-point reduced integration + hourglass control is a poor
fit for THIS repo's specific one-element-through-thickness PSA free span).

Tangent strategy
-----------------
Unlike the JAX version (which gets an exact analytic tangent for the
material part via `jax.jacobian`, since the closed-form pressure leaves no
inner unknown to condense), the Numba material kernel (`_simo_pk2_numba`)
returns stress only -- same limitation documented in
`q4_visco_hybrid_simo_numba.py`. `K_mat` is therefore a forward-difference
Jacobian of `f_mat` here, exactly like every other Numba viscoelastic
kernel in this codebase. `K_hg` (hourglass) is closed-form/exact (it is
the literal Hessian of a quadratic potential, `f_hg = dPi_hg/du`), added
on top of the FD `K_mat`, matching `q4_reduced_jax.py`'s split.

Zero-guards applied preemptively
---------------------------------
This file is written AFTER (and incorporating the lesson from) three
same-class bugs found earlier the same session in the sibling Numba
viscoelastic kernels: an unguarded `detJ` division in the shared
`_grads_numba`, an unguarded `detJ0` in `q4_visco_eas_numba.py`'s enhanced-
mode Jacobian, and `_simo_pk2_numba` itself using a stale volumetric law
with an unclamped `J ** (-2/3)` that raised `ZeroDivisionError` under
`fastmath=True` for an exact-zero `J` reachable at a rejected line-search
trial state. All three are already fixed in the shared helpers this file
imports (`_grads_numba`, `_simo_pk2_numba`), so this element inherits the
fix rather than repeating the bug -- no additional per-file guard is
needed here beyond what the shared helpers already provide.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from .._jit_cache import njit_cached
    _HAS_NUMBA = True
except ImportError:      # pragma: no cover
    _HAS_NUMBA = False

if _HAS_NUMBA:
    from .q4_visco_hybrid_simo_numba import (
        _grads_numba, _F_at_numba, _BL_columns_numba, _simo_pk2_numba,
    )

    # Natural-coordinate hourglass shape vector -- identical to
    # q4_reduced_jax.py / q4_visco_hybrid_reduced_jax.py.
    _HG_SHAPE = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    _ALPHA_HG = 0.05

    @njit_cached(fastmath=True)
    def _reduced_hybrid_force_numba(
        base_code: int, coords: np.ndarray, u_elem: np.ndarray,
        state_elem0: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float, F_n0: np.ndarray,
    ):
        """f_mat(8,) + kinematics, single centroid GP. Returns
        (f_mat, h_new, F(2,2), w0, gX(4,), gY(4,))."""
        gX, gY, detJ0 = _grads_numba(0.0, 0.0, coords)
        w0 = detJ0 * 4.0 * thickness

        F_inc = _F_at_numba(gX, gY, u_elem)
        F = F_inc @ F_n0
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

        # Closed-form hybrid pressure -- exact at a single sample point
        # (see q4_visco_hybrid_reduced_jax.py's module docstring: the
        # usual CPE4H volume average degenerates to the sample itself).
        p = kappa * (J - 1.0)

        S_v, h_new = _simo_pk2_numba(base_code, F, state_elem0, 0.0,
                                     bparams, g_i, tau_i, g_inf, dt)

        C11 = F[0, 0] * F[0, 0] + F[1, 0] * F[1, 0]
        C12 = F[0, 0] * F[0, 1] + F[1, 0] * F[1, 1]
        C22 = F[0, 1] * F[0, 1] + F[1, 1] * F[1, 1]
        detC = C11 * C22 - C12 * C12 + 1e-15
        Ci11 = C22 / detC
        Ci22 = C11 / detC
        Ci12 = -C12 / detC
        S0 = S_v[0] + p * J * Ci11
        S1 = S_v[1] + p * J * Ci22
        S2 = S_v[2] + p * J * Ci12

        # Work-conjugacy push-forward (dev_log/plan_abaqus_element_
        # consolidation_20260908.md finding F4 -- this element inherited
        # the bug at construction time from the established UL pattern,
        # ported here from the JAX fix): (S0,S1,S2) is referred to the
        # ORIGINAL config (built from the TOTAL F); BL(F_inc)/w0 below
        # are step-n quantities. Push forward: Sn = F_n0 @ S @ F_n0.T /
        # det(F_n0) (no-op when F_n0=I, i.e. TL mode unchanged).
        detFn0 = F_n0[0, 0] * F_n0[1, 1] - F_n0[0, 1] * F_n0[1, 0]
        if abs(detFn0) < 1e-30:
            detFn0 = 1e-30
        T00 = F_n0[0, 0] * S0 + F_n0[0, 1] * S2
        T01 = F_n0[0, 0] * S2 + F_n0[0, 1] * S1
        T10 = F_n0[1, 0] * S0 + F_n0[1, 1] * S2
        T11 = F_n0[1, 0] * S2 + F_n0[1, 1] * S1
        Sn0 = (T00 * F_n0[0, 0] + T01 * F_n0[0, 1]) / detFn0
        Sn1 = (T10 * F_n0[1, 0] + T11 * F_n0[1, 1]) / detFn0
        Sn2 = (T00 * F_n0[1, 0] + T01 * F_n0[1, 1]) / detFn0

        BL = _BL_columns_numba(F_inc, gX, gY)
        f_mat = np.zeros(8, dtype=np.float64)
        for i in range(8):
            f_mat[i] = (BL[0, i] * Sn0 + BL[1, i] * Sn1 + BL[2, i] * Sn2) * w0

        return f_mat, h_new, F, w0, gX, gY

    @njit_cached(fastmath=True)
    def compute_single_reduced_hybrid_numba(
        base_code: int, coords: np.ndarray, u_elem: np.ndarray,
        state_elem: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float, F_n: np.ndarray, h: float = 1e-6,
        alpha_hg: float = _ALPHA_HG,
    ):
        """CPE4RH. Returns (f_e(8,), K_e(8,8), state_new(4,n), F_n_new(4,2,2)).

        `state_elem`/`F_n` (4,...): only slot 0 is physically meaningful
        (single centroid GP); the result is broadcast into all 4 slots for
        array-shape compatibility with the solver's uniform per-element
        storage, matching `q4_reduced_jax.py`'s established convention.
        """
        f0, h_new, F, w0, gX, gY = _reduced_hybrid_force_numba(
            base_code, coords, u_elem, state_elem[0], kappa, bparams,
            g_i, tau_i, g_inf, dt, thickness, F_n[0])

        K_mat = np.zeros((8, 8), dtype=np.float64)
        for j in range(8):
            u_pert = u_elem.copy()
            u_pert[j] += h
            f_pert, _, _, _, _, _ = _reduced_hybrid_force_numba(
                base_code, coords, u_pert, state_elem[0], kappa, bparams,
                g_i, tau_i, g_inf, dt, thickness, F_n[0])
            for i in range(8):
                K_mat[i, j] = (f_pert[i] - f0[i]) / h

        n_state = state_elem.shape[1]
        state_new = np.zeros((4, n_state), dtype=np.float64)
        for k in range(4):
            for q in range(n_state):
                state_new[k, q] = h_new[q]
        F_n_new = np.zeros((4, 2, 2), dtype=np.float64)
        for k in range(4):
            F_n_new[k, 0, 0] = F[0, 0]; F_n_new[k, 0, 1] = F[0, 1]
            F_n_new[k, 1, 0] = F[1, 0]; F_n_new[k, 1, 1] = F[1, 1]

        # ---- Hourglass stabilization (Flanagan-Belytschko), exact/closed
        # form -- see q4_visco_hybrid_reduced_jax.py's module docstring.
        x_nodes = coords[:, 0]
        y_nodes = coords[:, 1]
        hx = 0.0; hy = 0.0
        for a in range(4):
            hx += _HG_SHAPE[a] * x_nodes[a]
            hy += _HG_SHAPE[a] * y_nodes[a]
        gamma = np.zeros(4, dtype=np.float64)
        for a in range(4):
            gamma[a] = _HG_SHAPE[a] - hx * gX[a] - hy * gY[a]

        ux = u_elem[0::2]
        uy = u_elem[1::2]
        qx = 0.0; qy = 0.0
        for a in range(4):
            qx += gamma[a] * ux[a]
            qy += gamma[a] * uy[a]
        gg_sum = 0.0
        for a in range(4):
            gg_sum += gX[a] * gX[a] + gY[a] * gY[a]
        k_hg = alpha_hg * (kappa + 2.0 * bparams[0]) * w0 * gg_sum

        f_e = f0.copy()
        for a in range(4):
            f_e[2 * a] += k_hg * qx * gamma[a]
            f_e[2 * a + 1] += k_hg * qy * gamma[a]

        K_e = K_mat.copy()
        for a in range(4):
            for b in range(4):
                gg = gamma[a] * gamma[b]
                K_e[2 * a, 2 * b] += k_hg * gg
                K_e[2 * a + 1, 2 * b + 1] += k_hg * gg

        # Same NaN/Inf zero-guard every other kernel in this codebase has.
        bad = False
        for i in range(8):
            if not np.isfinite(f_e[i]):
                bad = True
        for i in range(8):
            for j in range(8):
                if not np.isfinite(K_e[i, j]):
                    bad = True
        if bad:
            f_e = np.zeros(8, dtype=np.float64)
            K_e = np.zeros((8, 8), dtype=np.float64)

        return f_e, K_e, state_new, F_n_new

    @njit_cached(fastmath=True, parallel=True)
    def assemble_reduced_hybrid_batch_numba(
        base_code: int,
        elem_coords: np.ndarray,   # (N, 4, 2)
        u_elems: np.ndarray,       # (N, 8)
        state_elems: np.ndarray,   # (N, 4, n_state)
        kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thicknesses: np.ndarray,   # (N,)
        F_n: np.ndarray,           # (N, 4, 2, 2)
    ):
        n = elem_coords.shape[0]
        n_state = state_elems.shape[2]
        f_all = np.empty((n, 8), dtype=np.float64)
        K_all = np.empty((n, 8, 8), dtype=np.float64)
        state_all = np.empty((n, 4, n_state), dtype=np.float64)
        Fn_all = np.empty((n, 4, 2, 2), dtype=np.float64)
        for e in numba.prange(n):
            f_e, K_e, s_e, fn_e = compute_single_reduced_hybrid_numba(
                base_code, elem_coords[e], u_elems[e], state_elems[e],
                kappa, bparams, g_i, tau_i, g_inf, dt, thicknesses[e], F_n[e])
            f_all[e] = f_e; K_all[e] = K_e
            state_all[e] = s_e; Fn_all[e] = fn_e
        return f_all, K_all, state_all, Fn_all

else:      # pragma: no cover
    def compute_single_reduced_hybrid_numba(*a, **k):
        raise ImportError("Numba is not installed. Run pip install numba.")

    def assemble_reduced_hybrid_batch_numba(*a, **k):
        raise ImportError("Numba is not installed. Run pip install numba.")
