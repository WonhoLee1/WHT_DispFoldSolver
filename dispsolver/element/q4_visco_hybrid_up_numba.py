"""
q4_visco_hybrid_up_numba.py
===========================
Numba port of the CPE4H element (`q4_visco_eas_jax.compute_single_hybrid`
with `use_eas=False`): 4-node plane-strain quad, **true mixed u-p hybrid**
(independent element-constant pressure) on the finite-strain Simo
viscoelastic constitutive law, with **Updated-Lagrangian** support.

Why a Numba port
----------------
Per AGENTS.md's JIT strategy the JAX kernels are the R&D path and Numba is
the production path. With the refined 2-row PSA mesh (20k elements) the JAX
version spends minutes in `jax.jit` tracing before the first increment, which
makes iteration on the model impractical.

Pressure solve is closed form, not iterative
--------------------------------------------
The perturbed-Lagrangian pressure residual

    r_p(u, p) = int (J - 1) dV - (p / K) * V

is **linear in p**, so for a given displacement state

    p = K * ( int (J - 1) dV ) / V

exactly, in one shot. Recomputing `p` from that formula inside every
finite-difference perturbation makes the resulting 8x8 FD Jacobian the
**already-condensed** tangent `K_uu - K_up K_pp^-1 K_pu` -- no explicit
condensation step, and the same 8 residual evaluations per element the
existing `q4_visco_hybrid_simo_numba` kernel already costs.

The tangent is a full forward-difference Jacobian of the internal force, the
FD analogue of the JAX reference's `jax.jacobian`, for the reason documented
in `q4_visco_hybrid_simo_numba`: for this near-incompressible Arruda-Boyce
material the geometric/coupling terms are not small and a material-only
tangent is not accurate enough.

Updated Lagrangian
------------------
With `F_n` supplied, `coords` is the last converged configuration and
`u_elem` the incremental displacement, so the gradient computed here is
`F_inc` and the total deformation gradient is `F_inc @ F_n`. Passing
identity `F_n` reproduces Total-Lagrangian behaviour.
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
        _SQRT_EPS,
    )

    _GP = np.array([[-0.5773502691896258, -0.5773502691896258],
                    [0.5773502691896258, -0.5773502691896258],
                    [0.5773502691896258, 0.5773502691896258],
                    [-0.5773502691896258, 0.5773502691896258]])

    @njit_cached(fastmath=True)
    def _internal_force_up_numba(
        base_code: int, u_elem: np.ndarray, coords: np.ndarray,
        state_elem: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float, F_n: np.ndarray,
    ):
        """(f_int(8,), state_new, F_n_new, p) for the mixed u-p element."""
        n_state = state_elem.shape[1]
        state_new = np.zeros((4, n_state), dtype=np.float64)
        F_n_new = np.zeros((4, 2, 2), dtype=np.float64)

        # ---- pass 1: closed-form element pressure ----------------------
        vol = 0.0
        int_Jm1 = 0.0
        F_all = np.zeros((4, 2, 2), dtype=np.float64)
        Finc_all = np.zeros((4, 2, 2), dtype=np.float64)
        w_all = np.zeros(4, dtype=np.float64)
        gX_all = np.zeros((4, 4), dtype=np.float64)
        gY_all = np.zeros((4, 4), dtype=np.float64)
        for gp in range(4):
            gX, gY, detJ = _grads_numba(_GP[gp, 0], _GP[gp, 1], coords)
            w = detJ * thickness            # 2x2 Gauss weight is 1.0 each
            F_inc = _F_at_numba(gX, gY, u_elem)
            F = np.dot(F_inc, F_n[gp])
            J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
            int_Jm1 += (J - 1.0) * w
            vol += w
            for a in range(2):
                for b in range(2):
                    F_all[gp, a, b] = F[a, b]
                    Finc_all[gp, a, b] = F_inc[a, b]
            for a in range(4):
                gX_all[gp, a] = gX[a]
                gY_all[gp, a] = gY[a]
            w_all[gp] = w
        p = kappa * int_Jm1 / max(vol, 1e-30)

        # ---- pass 2: internal force at that pressure -------------------
        f_int = np.zeros(8, dtype=np.float64)
        for gp in range(4):
            F = F_all[gp]
            J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
            # deviatoric/viscoelastic response only: kappa = 0 switches off
            # the kernel's own volumetric term (and its distortion barrier)
            S_v, h_new = _simo_pk2_numba(base_code, F, state_elem[gp], 0.0,
                                         bparams, g_i, tau_i, g_inf, dt)
            # hybrid volumetric stress p J C^-1 (2D Voigt [11, 22, 12])
            C11 = F[0, 0] * F[0, 0] + F[1, 0] * F[1, 0]
            C12 = F[0, 0] * F[0, 1] + F[1, 0] * F[1, 1]
            C22 = F[0, 1] * F[0, 1] + F[1, 1] * F[1, 1]
            detC = C11 * C22 - C12 * C12
            if abs(detC) < 1e-30:
                detC = 1e-30
            Ci11 = C22 / detC
            Ci22 = C11 / detC
            Ci12 = -C12 / detC
            S0 = S_v[0] + p * J * Ci11
            S1 = S_v[1] + p * J * Ci22
            S2 = S_v[2] + p * J * Ci12

            # Work-conjugacy push-forward (dev_log/plan_abaqus_element_
            # consolidation_20260908.md finding F4; same fix as CPE4I's
            # numba port): (S0,S1,S2) is referred to the ORIGINAL config
            # (built from the TOTAL F); BL(F_inc)/w below are step-n
            # quantities. Push forward: Sn = Fn @ S @ Fn.T / det(Fn)
            # (no-op when Fn=I, i.e. TL mode unchanged by this fix).
            Fn_gp = F_n[gp]
            detFn = Fn_gp[0, 0] * Fn_gp[1, 1] - Fn_gp[0, 1] * Fn_gp[1, 0]
            if abs(detFn) < 1e-30:
                detFn = 1e-30
            T00 = Fn_gp[0, 0] * S0 + Fn_gp[0, 1] * S2
            T01 = Fn_gp[0, 0] * S2 + Fn_gp[0, 1] * S1
            T10 = Fn_gp[1, 0] * S0 + Fn_gp[1, 1] * S2
            T11 = Fn_gp[1, 0] * S2 + Fn_gp[1, 1] * S1
            Sn0 = (T00 * Fn_gp[0, 0] + T01 * Fn_gp[0, 1]) / detFn
            Sn1 = (T10 * Fn_gp[1, 0] + T11 * Fn_gp[1, 1]) / detFn
            Sn2 = (T00 * Fn_gp[1, 0] + T01 * Fn_gp[1, 1]) / detFn

            BL = _BL_columns_numba(Finc_all[gp], gX_all[gp], gY_all[gp])
            for i in range(8):
                f_int[i] += (BL[0, i] * Sn0 + BL[1, i] * Sn1 + BL[2, i] * Sn2) * w_all[gp]

            for k in range(h_new.shape[0]):
                state_new[gp, k] = h_new[k]
            for a in range(2):
                for b in range(2):
                    F_n_new[gp, a, b] = F[a, b]

        return f_int, state_new, F_n_new, p

    @njit_cached(fastmath=True)
    def compute_visco_hybrid_up_single_numba(
        base_code: int, coords: np.ndarray, u_elem: np.ndarray,
        state_elem: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float, F_n: np.ndarray, h: float = 0.0,
    ):
        """CPE4H. Returns (f_e(8,), K_e(8,8), state_new, F_n_new).

        `K_e` is the CONDENSED tangent: the pressure is recomputed from its
        closed form inside every perturbation, so the FD Jacobian already
        contains `- K_up K_pp^-1 K_pu`.

        Step size (2026-09-12, finding B4 -- the fix `q4_visco_hybrid_simo_numba`
        already carries, applied here). `h` used to default to a FIXED
        ABSOLUTE 1e-6. A forward difference's truncation error relative to
        `K` scales as `h / L_elem`, so a fixed absolute step means the error
        DOUBLES on every uniform mesh refinement. Contract C11 measured
        exactly that signature on this kernel: 2.61e-06 / 1.31e-06 / 6.54e-07
        / 3.27e-07 as the element grows x1/x2/x4/x8 -- a factor-8 spread
        across a factor-8 size sweep, i.e. error exactly inverse in element
        size, while `f_int` agreed with JAX throughout.

        `h = 0.0` now means "choose per column", the standard forward-
        difference step of Dennis & Schnabel, *Numerical Methods for
        Unconstrained Optimization and Nonlinear Equations* (1983) sec 5.4:
        ``h_j = sqrt(eps_mach) * max(|u_j|, L_elem)``, which balances
        truncation against roundoff at ~sqrt(eps) RELATIVE, independently of
        mesh size and units. Pass an explicit positive `h` for the old
        fixed-step behaviour.
        """
        f0, state_new, F_n_new, _p = _internal_force_up_numba(
            base_code, u_elem, coords, state_elem, kappa, bparams,
            g_i, tau_i, g_inf, dt, thickness, F_n)

        # Element characteristic length: the longer diagonal, a
        # rotation-invariant size measure (a bounding-box extent is not).
        d1 = np.sqrt((coords[2, 0] - coords[0, 0]) ** 2
                     + (coords[2, 1] - coords[0, 1]) ** 2)
        d2 = np.sqrt((coords[3, 0] - coords[1, 0]) ** 2
                     + (coords[3, 1] - coords[1, 1]) ** 2)
        L_elem = max(d1, d2)
        if L_elem < 1e-30:
            L_elem = 1e-30

        K_e = np.zeros((8, 8), dtype=np.float64)
        for j in range(8):
            if h > 0.0:
                hj = h
            else:
                uj = abs(u_elem[j])
                hj = _SQRT_EPS * (uj if uj > L_elem else L_elem)
            u_pert = u_elem.copy()
            u_pert[j] += hj
            f_pert, _s, _f, _pp = _internal_force_up_numba(
                base_code, u_pert, coords, state_elem, kappa, bparams,
                g_i, tau_i, g_inf, dt, thickness, F_n)
            for i in range(8):
                K_e[i, j] = (f_pert[i] - f0[i]) / hj

        # Same NaN/Inf zero-guard every other kernel in this codebase has;
        # see q4_visco_eas_numba.py's identical guard for the full rationale
        # (a line-search trial state can be locally degenerate before the
        # outer inversion check rejects it).
        bad = False
        for i in range(8):
            if not np.isfinite(f0[i]):
                bad = True
        for i in range(8):
            for j in range(8):
                if not np.isfinite(K_e[i, j]):
                    bad = True
        if bad:
            f0 = np.zeros(8, dtype=np.float64)
            K_e = np.zeros((8, 8), dtype=np.float64)

        return f0, K_e, state_new, F_n_new

    @njit_cached(fastmath=True, parallel=True)
    def assemble_visco_hybrid_up_batch_numba(
        base_code: int,
        elem_coords: np.ndarray,   # (N, 4, 2)  reference (UL: last converged)
        u_elems: np.ndarray,       # (N, 8)     UL: incremental displacement
        state_elems: np.ndarray,   # (N, 4, n_state)
        kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thicknesses: np.ndarray,   # (N,)
        F_n: np.ndarray,           # (N, 4, 2, 2)
    ):
        n_elems = elem_coords.shape[0]
        n_state = state_elems.shape[2]
        f_all = np.empty((n_elems, 8), dtype=np.float64)
        K_all = np.empty((n_elems, 8, 8), dtype=np.float64)
        state_all = np.empty((n_elems, 4, n_state), dtype=np.float64)
        Fn_all = np.empty((n_elems, 4, 2, 2), dtype=np.float64)

        for e in numba.prange(n_elems):
            f_e, K_e, s_e, fn_e = compute_visco_hybrid_up_single_numba(
                base_code, elem_coords[e], u_elems[e], state_elems[e],
                kappa, bparams, g_i, tau_i, g_inf, dt, thicknesses[e], F_n[e])
            f_all[e] = f_e
            K_all[e] = K_e
            state_all[e] = s_e
            Fn_all[e] = fn_e

        return f_all, K_all, state_all, Fn_all

else:      # pragma: no cover
    def compute_visco_hybrid_up_single_numba(*a, **k):
        raise ImportError("Numba is not installed. Run `pip install numba`.")

    def assemble_visco_hybrid_up_batch_numba(*a, **k):
        raise ImportError("Numba is not installed. Run `pip install numba`.")
