"""
q4_visco_hybrid_simo_numba.py
==============================
Numba JIT port of `q4_visco_simo_fs_jax.py` — finite-strain F-bar
viscoelastic Q4 element with pluggable isochoric ground state (Neo-Hookean,
Yeoh, Arruda-Boyce -- all Ibar1-based) and Simo (1987) Prony/WLF overstress
relaxation. Used for PSA (Q4_UP element_type + ViscoelasticMaterial).

Ogden is intentionally NOT ported here: it requires principal-stretch
(eigenvalue) decomposition rather than the shared Ibar1-invariant formula
the other three bases use, which reintroduces the eigh-based NaN-gradient
risk class documented in AGENTS.md §4.3 (plastic_jax.py's spectral J2
return mapping hits this under jax.jacobian at repeated eigenvalues; the
same mechanism would hit a Numba spectral tangent under central FD at
degenerate stretch states). If Ogden is needed later, give it its own
dedicated kernel/state layout rather than forcing it through this
Ibar1-only dispatch.

Tangent: full 8-DOF forward-difference Jacobian of the internal force
w.r.t. u_elem -- the FD analogue of the JAX reference's exact
jax.jacobian(f_int)(u_elem). A material-only 3-direction strain FD
(q4_plastic_numba.py's convention, cheaper) was tried first and gave
~50% relative error vs the JAX reference for this element: the F-bar
sqrt(J0/J) coupling and geometric stiffness are NOT small for this
near-incompressible Arruda-Boyce material, unlike Q4_EAS/Q4_COROTATIONAL
where the same style of material-only approximation is an established,
proven-acceptable modified-Newton precedent (AGENTS.md §4.4). Verified
against the JAX reference to ~8.6e-7 relative error (all three bases).

base_code (int, not str -- Numba njit dispatches faster and more
reliably on an int than a Python string compare):
    0 = neohookean   bparams = [mu]
    1 = yeoh         bparams = [c1, c2, c3]
    2 = arruda       bparams = [mu, lambda_m]
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

    _AB_C = np.array([0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0,
                       519.0 / 673750.0], dtype=np.float64)

    _GP2 = np.array([
        [-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
        [ 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
        [ 1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
        [-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
    ], dtype=np.float64)

    @njit_cached(fastmath=True)
    def _W1_numba(base_code: int, I1b: float, bparams: np.ndarray) -> float:
        """First invariant derivative of the isochoric strain-energy density."""
        if base_code == 0:  # neohookean
            return 0.5 * bparams[0]
        if base_code == 1:  # yeoh
            x = I1b - 3.0
            return bparams[0] + 2.0 * bparams[1] * x + 3.0 * bparams[2] * x * x
        # base_code == 2: arruda
        mu = bparams[0]
        lm = bparams[1]
        s = 0.0
        for i in range(5):
            s += (i + 1) * _AB_C[i] / lm ** (2 * i) * I1b ** i
        return mu * s

    @njit_cached(fastmath=True)
    def _sd_numba(xi: float, eta: float):
        dN_dxi = np.array([
            -0.25 * (1.0 - eta), 0.25 * (1.0 - eta),
             0.25 * (1.0 + eta), -0.25 * (1.0 + eta),
        ], dtype=np.float64)
        dN_deta = np.array([
            -0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
             0.25 * (1.0 + xi), 0.25 * (1.0 - xi),
        ], dtype=np.float64)
        return dN_dxi, dN_deta

    @njit_cached(fastmath=True)
    def _grads_numba(xi: float, eta: float, coords: np.ndarray):
        dN_dxi, dN_deta = _sd_numba(xi, eta)
        J = np.zeros((2, 2), dtype=np.float64)
        for i in range(4):
            J[0, 0] += dN_dxi[i] * coords[i, 0]
            J[0, 1] += dN_dxi[i] * coords[i, 1]
            J[1, 0] += dN_deta[i] * coords[i, 0]
            J[1, 1] += dN_deta[i] * coords[i, 1]
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        # Guard against exact zero: fastmath=True's relaxed reassociation
        # can round a near-degenerate (but strictly nonzero) Jacobian to
        # exactly 0.0 for a highly distorted line-search TRIAL element
        # geometry (rejected afterwards, but evaluated first) -- an
        # unguarded division then raises Numba's ZeroDivisionError and
        # kills the whole prange loop instead of letting the caller's
        # NaN/Inf output guard or the outer line search reject the state.
        # Same convention as q4_corotational_sri_j2_numba.py's _grads_sri.
        # Found 2026-09-08: this crashed CPE4I production at step 6 (4.25%
        # fold) with an identical signature AFTER the _simo_pk2_numba Cinv
        # fix -- that fix was real but this was the second, separate gap.
        if abs(detJ) < 1e-30:
            detJ = 1e-30
        invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]],
                        dtype=np.float64) / detJ
        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        return gX, gY, detJ

    @njit_cached(fastmath=True)
    def _F_at_numba(gX: np.ndarray, gY: np.ndarray, u_elem: np.ndarray) -> np.ndarray:
        ux = u_elem[0::2]
        uy = u_elem[1::2]
        H = np.zeros((2, 2), dtype=np.float64)
        for i in range(4):
            H[0, 0] += ux[i] * gX[i]
            H[0, 1] += ux[i] * gY[i]
            H[1, 0] += uy[i] * gX[i]
            H[1, 1] += uy[i] * gY[i]
        return np.eye(2, dtype=np.float64) + H

    @njit_cached(fastmath=True)
    def _BL_columns_numba(Ft: np.ndarray, gX: np.ndarray, gY: np.ndarray) -> np.ndarray:
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
    def _voigt6_to_tensor_numba(v6: np.ndarray) -> np.ndarray:
        T = np.zeros((3, 3), dtype=np.float64)
        T[0, 0] = v6[0]; T[1, 1] = v6[1]; T[2, 2] = v6[2]
        T[0, 1] = v6[3]; T[1, 0] = v6[3]
        T[0, 2] = v6[4]; T[2, 0] = v6[4]
        T[1, 2] = v6[5]; T[2, 1] = v6[5]
        return T

    @njit_cached(fastmath=True)
    def _tensor_to_voigt6_numba(T: np.ndarray) -> np.ndarray:
        return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]],
                        dtype=np.float64)

    @njit_cached(fastmath=True)
    def _simo_pk2_numba(
        base_code: int, Fbar2: np.ndarray, h_prev_flat: np.ndarray,
        kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        distortion_j_crit: float = 0.0,
    ):
        """Flory split + pluggable isochoric ground state + Simo overstress.

        Direct port of q4_visco_simo_fs_jax._simo_pk2.
        """
        M = g_i.shape[0]
        F3 = np.eye(3, dtype=np.float64)
        F3[0, 0] = Fbar2[0, 0]; F3[0, 1] = Fbar2[0, 1]
        F3[1, 0] = Fbar2[1, 0]; F3[1, 1] = Fbar2[1, 1]
        C = F3.T @ F3
        J = np.linalg.det(F3)
        # JAX reference (q4_visco_simo_fs_jax._simo_pk2) regularises this
        # inverse with +1e-15*eye(3) precisely because a near-degenerate
        # trial state during line search (Newton proposes a bad du before
        # the inversion check rejects it) can make C singular. This Numba
        # port had dropped that regularisation, so the same trial state
        # that JAX handles gracefully raised a hard exception here instead
        # (found 2026-09-08 via a real production cutback -- crashed the
        # whole process rather than letting the line search reject the
        # trial and backtrack, the way every other kernel's NaN-guard does).
        Cinv = np.linalg.inv(C + 1e-15 * np.eye(3, dtype=np.float64))
        I1 = C[0, 0] + C[1, 1] + C[2, 2]

        # Classical Simo & Armero (1992) / Holzapfel (2000) volumetric PK2:
        # S_vol = 0.5*kappa*(J - 1/J)*Cinv -- smooth hydrostatic barrier as
        # J -> 0, plus optional distortion control. This Numba port had
        # instead used an older logarithmic (Simo & Hughes) S_vol = kappa*
        # lnJ*Cinv with NO J-clamp on the two `J ** (-2/3)` terms below --
        # the two volumetric laws only agree to O((J-1)^2), so this was
        # silently wrong away from J~1, and the unclamped fractional power
        # of an exactly-zero J (reachable at a rejected line-search TRIAL
        # state, not just a converged one) raises Python/Numba
        # ZeroDivisionError for a non-integer exponent on a zero base --
        # the actual root cause of the 2026-09-08 CPE4I production crash
        # (the _grads_numba detJ guard added earlier the same day was a
        # real, separate gap, but insufficient on its own). Fixed by
        # porting q4_visco_simo_fs_jax._simo_pk2 exactly, term for term.
        J_safe = max(J, 1e-4)
        vol_factor = 0.5 * kappa * (J_safe - 1.0 / J_safe)
        S_vol = vol_factor * Cinv

        if distortion_j_crit > 0.0:
            denom = max(distortion_j_crit, 1e-12)
            distortion = max(0.0, (distortion_j_crit - J) / denom)
        else:
            distortion = 0.0
        S_distort_factor = (5000.0 * kappa) * (distortion ** 3)
        S_vol = S_vol + S_distort_factor * Cinv

        I1b = (J_safe ** (-2.0 / 3.0)) * I1
        I1b_safe = I1b
        if I1b_safe < 3.0:
            I1b_safe = 3.0
        elif I1b_safe > 50.0:
            I1b_safe = 50.0
        W1 = _W1_numba(base_code, I1b_safe, bparams)
        S_iso = np.zeros((3, 3), dtype=np.float64)
        factor = 2.0 * W1 * (J_safe ** (-2.0 / 3.0))
        eye3 = np.eye(3, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                S_iso[i, j] = factor * (eye3[i, j] - (I1 / 3.0) * Cinv[i, j])

        h_prev = np.zeros((M + 1, 3, 3), dtype=np.float64)
        for i in range(M + 1):
            h_prev[i] = _voigt6_to_tensor_numba(h_prev_flat[6 * i:6 * i + 6])
        S_iso_prev = h_prev[M]
        dS = S_iso - S_iso_prev

        S_eff = S_vol + g_inf * S_iso
        h_new = np.zeros((M + 1, 3, 3), dtype=np.float64)
        for i in range(M):
            ratio = dt / max(tau_i[i], 1e-30)
            if ratio < 1e-12:
                beta_i = 1.0 - ratio
            else:
                beta_i = np.exp(-ratio)
            if ratio < 1e-12:
                gamma_i = 1.0
            else:
                gamma_i = (1.0 - beta_i) / ratio
            h_i = beta_i * h_prev[i] + g_i[i] * gamma_i * dS
            S_eff = S_eff + h_i
            h_new[i] = h_i
        h_new[M] = S_iso

        h_new_flat = np.empty(6 * (M + 1), dtype=np.float64)
        for i in range(M + 1):
            h_new_flat[6 * i:6 * i + 6] = _tensor_to_voigt6_numba(h_new[i])

        S_voigt = np.array([S_eff[0, 0], S_eff[1, 1], S_eff[0, 1]], dtype=np.float64)
        return S_voigt, h_new_flat

    @njit_cached(fastmath=True)
    def _internal_force_numba(
        base_code: int, u_elem: np.ndarray, coords: np.ndarray,
        state_elem: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float,
    ):
        """F-bar internal force + updated state. Direct port of
        q4_visco_simo_fs_jax._internal_force."""
        gX0, gY0, _ = _grads_numba(0.0, 0.0, coords)
        F0 = _F_at_numba(gX0, gY0, u_elem)
        J0 = F0[0, 0] * F0[1, 1] - F0[0, 1] * F0[1, 0]

        n_state = state_elem.shape[1]
        f_int = np.zeros(8, dtype=np.float64)
        state_new = np.zeros((4, n_state), dtype=np.float64)

        for gp in range(4):
            xi, eta = _GP2[gp, 0], _GP2[gp, 1]
            gX, gY, detJ = _grads_numba(xi, eta, coords)
            w = detJ * thickness
            F = _F_at_numba(gX, gY, u_elem)
            J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
            Fbar = F * np.sqrt(max(J0 / J, 1e-12))

            S_v, h_new = _simo_pk2_numba(
                base_code, Fbar, state_elem[gp], kappa, bparams,
                g_i, tau_i, g_inf, dt,
            )
            BL = _BL_columns_numba(Fbar, gX, gY)
            for i in range(8):
                for j in range(3):
                    f_int[i] += BL[j, i] * S_v[j] * w
            state_new[gp] = h_new

        return f_int, state_new

    @njit_cached(fastmath=True)
    def compute_visco_hybrid_simo_single_numba(
        base_code: int, coords: np.ndarray, u_elem: np.ndarray,
        state_elem: np.ndarray, kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thickness: float, h: float = 1e-6,
    ):
        """F-bar Arruda-Boyce/Yeoh/Neo-Hookean + Prony/WLF Q4 hybrid element.

        Returns (f_e(8,), K_e(8,8), state_new(4,n_state)). Tangent is a
        full 8-DOF forward-difference Jacobian of the internal force w.r.t.
        u_elem -- the FD analogue of the JAX reference's exact
        jax.jacobian(f_int)(u_elem), NOT a material-only tangent. A
        material-only (3-direction strain FD) tangent was tried first and
        gave ~50% relative error vs the JAX reference here (unlike
        Q4_EAS/Q4_COROTATIONAL's ~0.2% for the same style of approximation
        elsewhere in this codebase) -- the F-bar sqrt(J0/J) coupling and
        geometric stiffness are NOT small for this near-incompressible
        Arruda-Boyce material, so they cannot be dropped the way the
        codebase's established modified-Newton precedent (AGENTS.md §4.4)
        drops them for the plasticity-based elements.
        """
        f0, state_new = _internal_force_numba(
            base_code, u_elem, coords, state_elem, kappa, bparams,
            g_i, tau_i, g_inf, dt, thickness,
        )

        K_e = np.zeros((8, 8), dtype=np.float64)
        for j in range(8):
            u_pert = u_elem.copy()
            u_pert[j] += h
            f_pert, _ = _internal_force_numba(
                base_code, u_pert, coords, state_elem, kappa, bparams,
                g_i, tau_i, g_inf, dt, thickness,
            )
            for i in range(8):
                K_e[i, j] = (f_pert[i] - f0[i]) / h

        return f0, K_e, state_new

    @njit_cached(fastmath=True, parallel=True)
    def assemble_visco_hybrid_simo_batch_numba(
        base_code: int,
        elem_coords: np.ndarray,   # (N, 4, 2)
        u_elems: np.ndarray,       # (N, 8)
        state_elems: np.ndarray,   # (N, 4, n_state)
        kappa: float, bparams: np.ndarray,
        g_i: np.ndarray, tau_i: np.ndarray, g_inf: float, dt: float,
        thicknesses: np.ndarray,   # (N,)
    ):
        """Multi-threaded Numba batch assembly, N Q4_UP + ViscoelasticMaterial
        (Neo-Hookean/Yeoh/Arruda-Boyce base) elements."""
        n_elems = elem_coords.shape[0]
        n_state = state_elems.shape[2]
        f_all = np.empty((n_elems, 8), dtype=np.float64)
        K_all = np.empty((n_elems, 8, 8), dtype=np.float64)
        state_all = np.empty((n_elems, 4, n_state), dtype=np.float64)

        for e in numba.prange(n_elems):
            f_e, K_e, s_e = compute_visco_hybrid_simo_single_numba(
                base_code, elem_coords[e], u_elems[e], state_elems[e],
                kappa, bparams, g_i, tau_i, g_inf, dt, thicknesses[e],
            )
            f_all[e] = f_e
            K_all[e] = K_e
            state_all[e] = s_e

        return f_all, K_all, state_all

else:
    def compute_visco_hybrid_simo_single_numba(*args, **kwargs):
        raise ImportError("Numba is not installed. Run `pip install numba` to use Numba backend.")

    def assemble_visco_hybrid_simo_batch_numba(*args, **kwargs):
        raise ImportError("Numba is not installed. Run `pip install numba` to use Numba backend.")
