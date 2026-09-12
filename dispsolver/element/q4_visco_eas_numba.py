"""
q4_visco_eas_numba.py
======================
Numba port of `q4_visco_eas_jax.compute_single_eas` (CPE4I): Total-
Lagrangian finite-strain viscoelastic (Arruda-Boyce/Neo-Hookean/Yeoh + Prony)
element with Simo-Rifai/Simo-Armero EAS-4 incompatible modes.

Tangent strategy
-----------------
The Numba viscoelastic material kernel (`_simo_pk2_numba`, reused from
`q4_visco_hybrid_simo_numba`) returns stress only, no analytic tangent
(documented there: a material-only tangent gave large error for this
near-incompressible material, so this codebase's Numba viscoelastic path is
FD-tangent throughout). The alpha condensation therefore uses a nested FD:
an inner fixed-iteration Newton on the 4 enhanced parameters (its own 4x4
Jacobian by FD of the enhanced residual), producing a condensed force at
each of the 8 outer perturbations of u; differencing those 8+1 condensed
forces gives the outer 8x8 tangent, which already contains the
`-K_ua K_aa^-1 K_au` term because alpha is re-equilibrated at every
perturbed u (same pattern as `q4_visco_hybrid_up_numba`'s closed-form
pressure, just iterative instead of closed-form since alpha has no closed
form).

alpha bound: a p-norm guard (identity to O((a/ALPHA_MAX)^8) near zero),
matching the fix applied to the JAX kernel after review found the original
tanh guard biases the solution -- see `q4_visco_eas_jax.py`.

CANARY: a pure rigid rotation must give exactly zero force (F = R means
C = I means isotropic S = 0). This is the exact class of bug found and
fixed in `q4_corotational_sri_j2_numba.py` (a transposed rotation matrix
produced spurious strain growing linearly in theta) -- verified below
before this kernel was wired into the solver dispatch.
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

    # Element-local enhanced-mode Newton: convergence is measured on the LAST
    # accepted correction ||d(alpha)||_inf (alpha is dimensionless, so an
    # absolute tolerance is unit/mesh independent). The former `_ALPHA_MAX`
    # magnitude clamp is REMOVED here -- see the removal note in
    # q4_eas_jax.py and dev_log/eas_stabilization_modernization_20260909.md.
    # Non-convergence is now REPORTED (the kernels' trailing `status` return)
    # and turned into a global increment cutback by DynamicSolver, which is
    # what a commercial solver does with a non-converged element-local
    # algorithm -- it is never silently clamped and continued.
    _ALPHA_CONV_TOL = 1e-8
    _N_ALPHA_IT = 5        # legacy fixed count, superseded by _ALPHA_MAX_IT
    # Line-search damped element-local Newton -- mirrors the JAX kernel
    # q4_visco_eas_jax.py exactly (same step-length set, same tolerance, same
    # iteration cap) so the two backends solve the SAME local problem and can
    # be compared element-for-element (scratch/tier1_numba_jax_tilted.py).
    # See that module's header for the measurement that motivated it and the
    # Pfefferkorn et al. (IJNME 2021) reference.
    _ALPHA_MAX_IT = 12
    _LS_STEPS = np.array([1.0, 0.5, 0.25, 0.1])
    # sqrt(machine epsilon) -- the forward-difference step scale of
    # Dennis & Schnabel (1983) sec 5.4. See compute_single_eas_numba_status.
    _SQRT_EPS = 1.4901161193847656e-08

    @njit_cached(fastmath=True)
    def _enh_modes_numba(coords: np.ndarray):
        """Per-GP enhanced modes (4gp, 4modes, 2, 2), Simo-Armero Q1/E4."""
        dN_dxi0 = np.array([-0.25, 0.25, 0.25, -0.25])
        dN_deta0 = np.array([-0.25, -0.25, 0.25, 0.25])
        J0_11 = np.dot(dN_dxi0, coords[:, 0]); J0_12 = np.dot(dN_dxi0, coords[:, 1])
        J0_21 = np.dot(dN_deta0, coords[:, 0]); J0_22 = np.dot(dN_deta0, coords[:, 1])
        detJ0 = J0_11 * J0_22 - J0_12 * J0_21
        # Same zero-guard as _grads_numba (q4_visco_hybrid_simo_numba.py) --
        # this is the reference-config centroid Jacobian, computed
        # independently of that shared helper, so it needs its own guard.
        if abs(detJ0) < 1e-30:
            detJ0 = 1e-30
        i11 = J0_22 / detJ0; i12 = -J0_12 / detJ0
        i21 = -J0_21 / detJ0; i22 = J0_11 / detJ0

        gp3 = 1.0 / np.sqrt(3.0)
        gps = np.array([[-gp3, -gp3], [gp3, -gp3], [gp3, gp3], [-gp3, gp3]])
        Fenh = np.zeros((4, 4, 2, 2), dtype=np.float64)
        for gp in range(4):
            xi = gps[gp, 0]; eta = gps[gp, 1]
            _, _, detJ = _grads_numba(xi, eta, coords)
            s = detJ0 / detJ
            # D_k(xi,eta) @ J0^-T, D = {[[xi,0],[0,0]], [[0,eta],[0,0]],
            #                            [[0,0],[xi,0]], [[0,0],[0,eta]]}
            # TRANSPOSE FIX 2026-09-08 (was J0^-1) -- see
            # q4_eas_jax.py::_enhanced_grad_modes for the derivation and
            # measured impact. With inv(J0) = [[i11,i12],[i21,i22]], the
            # transpose swaps i12 <-> i21 in the expansion below; for an
            # axis-aligned element i12 = i21 = 0, which is exactly why
            # this was invisible to every existing test.
            Fenh[gp, 0, 0, 0] = xi * i11;  Fenh[gp, 0, 0, 1] = xi * i21
            Fenh[gp, 1, 0, 0] = eta * i12; Fenh[gp, 1, 0, 1] = eta * i22
            Fenh[gp, 2, 1, 0] = xi * i11;  Fenh[gp, 2, 1, 1] = xi * i21
            Fenh[gp, 3, 1, 0] = eta * i12; Fenh[gp, 3, 1, 1] = eta * i22
            for m in range(4):
                Fenh[gp, m, 0, 0] *= s; Fenh[gp, m, 0, 1] *= s
                Fenh[gp, m, 1, 0] *= s; Fenh[gp, m, 1, 1] *= s
        return Fenh

    @njit_cached(fastmath=True)
    def _residuals_eas_numba(base_code: int, u_elem: np.ndarray, alpha: np.ndarray,
                             coords: np.ndarray, state_elem: np.ndarray,
                             kappa: float, bparams: np.ndarray,
                             g_i: np.ndarray, tau_i: np.ndarray, g_inf: float,
                             dt: float, thickness: float, Fenh: np.ndarray,
                             F_n: np.ndarray):
        """F_n (4,2,2): total deformation gradient at the last converged step,
        per GP -- Updated-Lagrangian. `coords` is then the last converged
        configuration and `u_elem` the incremental displacement, so
        `F_inc @ F_n[gp]` is the total gradient before the enhancement is
        added. Identity F_n reproduces Total-Lagrangian behaviour."""
        f_u = np.zeros(8, dtype=np.float64)
        f_a = np.zeros(4, dtype=np.float64)
        n_state = state_elem.shape[1]
        state_new = np.zeros((4, n_state), dtype=np.float64)
        F_n_new = np.zeros((4, 2, 2), dtype=np.float64)

        gp3 = 1.0 / np.sqrt(3.0)
        gps = np.array([[-gp3, -gp3], [gp3, -gp3], [gp3, gp3], [-gp3, gp3]])
        for gp in range(4):
            gX, gY, detJ = _grads_numba(gps[gp, 0], gps[gp, 1], coords)
            # 2026-09-12: ported from q4_visco_eas_jax._residuals (finding B2).
            # The enhancement belongs in the INCREMENTAL frame, where every
            # other operator of this element already lives -- `Fenh` is built
            # from `coords` (= config n), `BL` differentiates the incremental
            # displacement, and `w = detJ` is the config-n volume element.
            # This kernel used to add it to the TOTAL gradient
            # (`F_c = F_inc @ F_n` first, enhancement on top), which mixes
            # frames and costs the element its variational consistency.
            F_inc_c = _F_at_numba(gX, gY, u_elem)
            F_inc = F_inc_c.copy()
            for m in range(4):
                F_inc[0, 0] += alpha[m] * Fenh[gp, m, 0, 0]
                F_inc[0, 1] += alpha[m] * Fenh[gp, m, 0, 1]
                F_inc[1, 0] += alpha[m] * Fenh[gp, m, 1, 0]
                F_inc[1, 1] += alpha[m] * Fenh[gp, m, 1, 1]
            F_e = F_inc @ F_n[gp]

            S_v, h_new = _simo_pk2_numba(base_code, F_e, state_elem[gp], kappa,
                                         bparams, g_i, tau_i, g_inf, dt)
            for q in range(n_state):
                state_new[gp, q] = h_new[q]

            # Work-conjugacy push-forward (dev_log/plan_abaqus_element_
            # consolidation_20260908.md finding F4; same bug/fix as the
            # JAX counterpart q4_visco_eas_jax.py's `_residuals`): S_v is
            # PK2 referred to the ORIGINAL config (from the TOTAL F_e);
            # BL(F_inc)/w(detJ) below are step-n quantities. Push forward:
            # S_n = F_n @ S_v @ F_n.T / det(F_n) (identity, i.e. no-op,
            # when F_n = I -- TL mode unchanged by this fix).
            #
            # 2026-09-12 (B2): `f_a` used to contract the UN-pushed `S_v`
            # while `f_u` used `S_n`, so the two were gradients of two
            # different functionals and `K_au = K_ua^T` -- which
            # `compute_single_eas_numba_status`'s condensation ASSUMES --
            # was false. Both now contract the SAME `S_n` against operators
            # built from the SAME enhanced `F_inc`.
            Fn_gp = F_n[gp]
            detFn = Fn_gp[0, 0] * Fn_gp[1, 1] - Fn_gp[0, 1] * Fn_gp[1, 0]
            if abs(detFn) < 1e-30:
                detFn = 1e-30
            S0_00 = S_v[0]; S0_11 = S_v[1]; S0_01 = S_v[2]
            # Sn = Fn @ [[S00,S01],[S01,S11]] @ Fn.T / detFn
            T00 = Fn_gp[0, 0] * S0_00 + Fn_gp[0, 1] * S0_01
            T01 = Fn_gp[0, 0] * S0_01 + Fn_gp[0, 1] * S0_11
            T10 = Fn_gp[1, 0] * S0_00 + Fn_gp[1, 1] * S0_01
            T11 = Fn_gp[1, 0] * S0_01 + Fn_gp[1, 1] * S0_11
            Sn00 = (T00 * Fn_gp[0, 0] + T01 * Fn_gp[0, 1]) / detFn
            Sn11 = (T10 * Fn_gp[1, 0] + T11 * Fn_gp[1, 1]) / detFn
            Sn01 = (T00 * Fn_gp[1, 0] + T01 * Fn_gp[1, 1]) / detFn
            S_n0 = Sn00; S_n1 = Sn11; S_n2 = Sn01

            # B_L and G are both built from the ENHANCED incremental gradient:
            # dE_inc = sym(F_inc^T dF_c) du + sym(F_inc^T Fenh_j) dalpha.
            # `BL` used to be built from the COMPATIBLE gradient, which
            # silently dropped the alpha-dependent half of dE_inc/du. That
            # half does NOT vanish in Total Lagrangian, so it was the whole
            # of this lowering's 1.61e-03 contract-C10 disagreement with JAX
            # at F_n = I (alpha itself converged bit-identically in both).
            BL = _BL_columns_numba(F_inc, gX, gY)
            w = detJ * thickness
            for i in range(8):
                acc = S_n0 * BL[0, i] + S_n1 * BL[1, i] + S_n2 * BL[2, i]
                f_u[i] += acc * w

            for m in range(4):
                Fm = Fenh[gp, m]
                Ft00 = F_inc[0, 0] * Fm[0, 0] + F_inc[1, 0] * Fm[1, 0]
                Ft01 = F_inc[0, 0] * Fm[0, 1] + F_inc[1, 0] * Fm[1, 1]
                Ft10 = F_inc[0, 1] * Fm[0, 0] + F_inc[1, 1] * Fm[1, 0]
                Ft11 = F_inc[0, 1] * Fm[0, 1] + F_inc[1, 1] * Fm[1, 1]
                Gm0 = Ft00; Gm1 = Ft11; Gm2 = Ft01 + Ft10
                f_a[m] += (Gm0 * S_n0 + Gm1 * S_n1 + Gm2 * S_n2) * w
            F_n_new[gp, 0, 0] = F_e[0, 0]; F_n_new[gp, 0, 1] = F_e[0, 1]
            F_n_new[gp, 1, 0] = F_e[1, 0]; F_n_new[gp, 1, 1] = F_e[1, 1]
        return f_u, f_a, state_new, F_n_new

    @njit_cached(fastmath=True)
    def compute_single_eas_numba_status(
        base_code: int, coords: np.ndarray, u_elem: np.ndarray, alpha0: np.ndarray,
        state_elem: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float, F_n: np.ndarray, h: float = 0.0,
    ):
        """Returns (f_e(8,), K_e(8,8), alpha_new(4,), state_new(4,n_state),
        F_n_new(4,2,2), status()).

        Step size (2026-09-12, finding B4). The `K_e` sweep differentiates
        w.r.t. `u_elem`, which carries LENGTH units, so a fixed absolute `h`
        gives a truncation error scaling as `h / L_elem` -- it doubles on
        every uniform refinement. Contract C11 measured exactly that on this
        kernel the moment its force agreed with JAX and stopped masking it:
        6.63e-06 / 3.31e-06 / 1.66e-06 / 8.28e-07 as the element grows
        x1/x2/x4/x8. `h = 0.0` now selects the per-column Dennis & Schnabel
        (1983) sec 5.4 step `h_j = sqrt(eps_mach) * max(|u_j|, L_elem)`.

        The `alpha` sweeps keep a step of their own: `alpha` is DIMENSIONLESS
        (it is added straight into the deformation gradient), so it has no
        mesh-size dependence to correct -- only the same sqrt(eps)-relative
        conditioning, via `max(|alpha_j|, 1)`.

        `status` is ||d(alpha)||_inf of the last accepted element-local
        Newton correction (0 = converged), or inf when the local solve or the
        condensed force/tangent went non-finite. The solver turns a value
        above `eas_local_tol` into an increment cutback.

        `F_n` (4,2,2): Updated-Lagrangian total F at the last converged step,
        per GP. Pass a tile of identity for Total-Lagrangian behaviour.
        """
        Fenh = _enh_modes_numba(coords)

        # Element characteristic length: the longer diagonal, a
        # rotation-invariant size measure (a bounding-box extent is not).
        d1 = np.sqrt((coords[2, 0] - coords[0, 0]) ** 2
                     + (coords[2, 1] - coords[0, 1]) ** 2)
        d2 = np.sqrt((coords[3, 0] - coords[1, 0]) ** 2
                     + (coords[3, 1] - coords[1, 1]) ** 2)
        L_elem = max(d1, d2)
        if L_elem < 1e-30:
            L_elem = 1e-30

        def _h_alpha(a_j):
            if h > 0.0:
                return h
            aj = abs(a_j)
            return _SQRT_EPS * (aj if aj > 1.0 else 1.0)

        def _solve_alpha(u, a_start):
            """Returns (alpha, da_inf) -- da_inf is ||d(alpha)||_inf of the
            last accepted correction, or inf if the iterate went non-finite."""
            a = a_start.copy()
            da_inf = 1.0
            for _ in range(_ALPHA_MAX_IT):
                if da_inf <= _ALPHA_CONV_TOL or not np.isfinite(da_inf):
                    break
                f_u0, f_a0, _, _ = _residuals_eas_numba(
                    base_code, u, a, coords, state_elem, kappa, bparams,
                    g_i, tau_i, g_inf, dt, thickness, Fenh, F_n)
                K_aa = np.zeros((4, 4), dtype=np.float64)
                for j in range(4):
                    hj = _h_alpha(a[j])
                    ap = a.copy(); ap[j] += hj
                    _, f_ap, _, _ = _residuals_eas_numba(
                        base_code, u, ap, coords, state_elem, kappa, bparams,
                        g_i, tau_i, g_inf, dt, thickness, Fenh, F_n)
                    for i in range(4):
                        K_aa[i, j] = (f_ap[i] - f_a0[i]) / hj
                tr = (K_aa[0, 0] + K_aa[1, 1] + K_aa[2, 2] + K_aa[3, 3]) / 4.0
                reg = 1e-10 * (abs(tr) + 1e-30)
                for i in range(4):
                    K_aa[i, i] += reg
                da = np.linalg.solve(K_aa, -f_a0)
                # Backtracking line search on |f_alpha| over the same fixed
                # step-length set the JAX kernel uses.
                best_n = np.inf
                best_ls = 0.0
                for q in range(_LS_STEPS.shape[0]):
                    ls = _LS_STEPS[q]
                    a_try = a + ls * da
                    _, f_try, _, _ = _residuals_eas_numba(
                        base_code, u, a_try, coords, state_elem, kappa, bparams,
                        g_i, tau_i, g_inf, dt, thickness, Fenh, F_n)
                    nrm = 0.0
                    finite = True
                    for i in range(4):
                        if not np.isfinite(f_try[i]):
                            finite = False
                        nrm += f_try[i] * f_try[i]
                    nrm = np.sqrt(nrm)
                    if finite and nrm < best_n:
                        best_n = nrm
                        best_ls = ls
                if not np.isfinite(best_n):
                    # Every trial step is non-finite: freeze alpha at its last
                    # good value and latch the failure; the solver cuts the
                    # increment back.
                    da_inf = np.inf
                    break
                ok = True
                dmax = 0.0
                for i in range(4):
                    v = a[i] + best_ls * da[i]
                    if not np.isfinite(v):
                        ok = False
                    if abs(best_ls * da[i]) > dmax:
                        dmax = abs(best_ls * da[i])
                if ok:
                    for i in range(4):
                        a[i] = a[i] + best_ls * da[i]
                    da_inf = dmax
                else:
                    da_inf = np.inf
                    break
            return a, da_inf

        a_conv, alpha_status = _solve_alpha(u_elem, alpha0)
        f_u0, f_a0, state_new, F_n_new = _residuals_eas_numba(
            base_code, u_elem, a_conv, coords, state_elem, kappa, bparams,
            g_i, tau_i, g_inf, dt, thickness, Fenh, F_n)

        K_ua = np.zeros((8, 4), dtype=np.float64)
        K_aa = np.zeros((4, 4), dtype=np.float64)
        for j in range(4):
            hj = _h_alpha(a_conv[j])
            ap = a_conv.copy(); ap[j] += hj
            f_up, f_ap, _, _ = _residuals_eas_numba(
                base_code, u_elem, ap, coords, state_elem, kappa, bparams,
                g_i, tau_i, g_inf, dt, thickness, Fenh, F_n)
            for i in range(8):
                K_ua[i, j] = (f_up[i] - f_u0[i]) / hj
            for i in range(4):
                K_aa[i, j] = (f_ap[i] - f_a0[i]) / hj
        tr = (K_aa[0, 0] + K_aa[1, 1] + K_aa[2, 2] + K_aa[3, 3]) / 4.0
        reg = 1e-10 * (abs(tr) + 1e-30)
        for i in range(4):
            K_aa[i, i] += reg
        f_e = f_u0 - K_ua @ np.linalg.solve(K_aa, f_a0)

        K_e = np.zeros((8, 8), dtype=np.float64)
        for j in range(8):
            if h > 0.0:
                hj = h
            else:
                uj = abs(u_elem[j])
                hj = _SQRT_EPS * (uj if uj > L_elem else L_elem)
            up = u_elem.copy(); up[j] += hj
            a_p, _ = _solve_alpha(up, a_conv)
            f_up, f_ap, _, _ = _residuals_eas_numba(
                base_code, up, a_p, coords, state_elem, kappa, bparams,
                g_i, tau_i, g_inf, dt, thickness, Fenh, F_n)
            for i in range(8):
                K_e[i, j] = (f_up[i] - f_e[i]) / hj

        # Same NaN/Inf zero-guard every other kernel in this codebase has
        # (JAX kernels: `where(isnan(f)|isnan(K), 0, f)`) -- a trial state
        # the line search hasn't rejected yet can still be locally
        # degenerate; return finite zeros for the outer inversion-check /
        # line-search to reject, instead of letting a huge-but-finite (or
        # NaN) value propagate or an exception escape the prange loop
        # (found 2026-09-08: this exact scenario crashed the whole process
        # via a Cinv near-singularity, fixed separately in
        # q4_visco_hybrid_simo_numba's _simo_pk2_numba -- this guard is the
        # second, independent layer of defense).
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
            alpha_status = np.inf

        return f_e, K_e, a_conv, state_new, F_n_new, alpha_status

    @njit_cached(fastmath=True)
    def compute_single_eas_numba(
        base_code: int, coords: np.ndarray, u_elem: np.ndarray, alpha0: np.ndarray,
        state_elem: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float, F_n: np.ndarray, h: float = 0.0,
    ):
        """Back-compatible 5-tuple wrapper around
        `compute_single_eas_numba_status` (drops the trailing `status`)."""
        f_e, K_e, a, s_e, fn_e, _st = compute_single_eas_numba_status(
            base_code, coords, u_elem, alpha0, state_elem, kappa, bparams,
            g_i, tau_i, g_inf, dt, thickness, F_n, h)
        return f_e, K_e, a, s_e, fn_e

    @njit_cached(fastmath=True, parallel=True)
    def assemble_eas_visco_batch_numba(
        base_code: int,
        elem_coords: np.ndarray, u_elems: np.ndarray, alpha_elems: np.ndarray,
        state_elems: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thicknesses: np.ndarray, F_n: np.ndarray,
    ):
        n = elem_coords.shape[0]
        ns = state_elems.shape[2]
        f_all = np.empty((n, 8), dtype=np.float64)
        K_all = np.empty((n, 8, 8), dtype=np.float64)
        a_all = np.empty((n, 4), dtype=np.float64)
        s_all = np.empty((n, 4, ns), dtype=np.float64)
        Fn_all = np.empty((n, 4, 2, 2), dtype=np.float64)
        st_all = np.empty(n, dtype=np.float64)
        for e in numba.prange(n):
            f_e, K_e, a_e, s_e, fn_e, st_e = compute_single_eas_numba_status(
                base_code, elem_coords[e], u_elems[e], alpha_elems[e],
                state_elems[e], kappa, bparams, g_i, tau_i, g_inf, dt,
                thicknesses[e], F_n[e])
            f_all[e] = f_e; K_all[e] = K_e; a_all[e] = a_e; s_all[e] = s_e
            Fn_all[e] = fn_e; st_all[e] = st_e
        return f_all, K_all, a_all, s_all, Fn_all, st_all

else:      # pragma: no cover
    def compute_single_eas_numba(*a, **k):
        raise ImportError("Numba is not installed. Run pip install numba.")

    def compute_single_eas_numba_status(*a, **k):
        raise ImportError("Numba is not installed. Run pip install numba.")

    def assemble_eas_visco_batch_numba(*a, **k):
        raise ImportError("Numba is not installed. Run pip install numba.")
