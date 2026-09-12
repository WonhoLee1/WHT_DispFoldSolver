"""
q4_corotational_sri_j2_numba.py
===============================
Numba port of `q4_sri_jax.compute_corotational_sri_j2_contributions_jax`
(`Q4_COROTATIONAL_SRI`): co-rotational Q4 with selective reduced integration
and finite-strain J2 plasticity.

Why
---
This is the element the display's PET and GLASS layers use, i.e. the large
majority of elements in the folding model (~2/3 of a 20k-element mesh), and
it had no Numba path at all -- the only `q4_sri_numba` assembler present is a
small-strain LINEAR-ELASTIC one with no plasticity, no state variables and no
rotation of the material state, so wiring that in would silently drop
plasticity (the AGENTS.md 4.2 "element type quietly ignored" failure mode).

Formulation (identical to the JAX reference, term for term)
-----------------------------------------------------------
* Co-rotational frame: R from the deformed edge vectors, local deformational
  displacement `u_l = coords_curr @ R - coords`, then
  `f_g = T8 f_l`, `K_g = T8 K_l T8^T`.
* SRI kinematics: the normal terms of F are sampled at the Gauss point and
  the two shear terms at the element centroid (`_F_sri`), and `B_L` is the
  exact dE/du of that same mixed measure -- sampling only B at the centroid
  while feeding the material a full Gauss-point F makes the residual
  non-integrable, which the JAX reference documents explicitly.
* Material: `stress_and_tangent_j2_numba` (PK2 + consistent tangent by
  forward FD in three strain directions), the same routine the other Numba
  element kernels use.

`F_n_gps` is accepted for signature parity with the JAX kernel and, exactly
as there, is not used: this element carries large rotation through the
co-rotational frame rather than an updated reference configuration.
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
    from .q4_plastic_numba import stress_and_tangent_j2_numba

    _INV_S3 = 1.0 / np.sqrt(3.0)
    _SRI_GP = np.array([[-_INV_S3, -_INV_S3],
                        [_INV_S3, -_INV_S3],
                        [_INV_S3, _INV_S3],
                        [-_INV_S3, _INV_S3]])

    @njit_cached(fastmath=True)
    def _sd_sri(xi: float, eta: float):
        dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
        dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
        return dN_dxi, dN_deta

    @njit_cached(fastmath=True)
    def _grads_sri(xi: float, eta: float, coords: np.ndarray):
        dN_dxi, dN_deta = _sd_sri(xi, eta)
        J11 = np.dot(dN_dxi, coords[:, 0]); J12 = np.dot(dN_dxi, coords[:, 1])
        J21 = np.dot(dN_deta, coords[:, 0]); J22 = np.dot(dN_deta, coords[:, 1])
        detJ = J11 * J22 - J12 * J21
        if abs(detJ) < 1e-30:
            detJ = 1e-30
        i11 = J22 / detJ; i12 = -J12 / detJ
        i21 = -J21 / detJ; i22 = J11 / detJ
        gX = i11 * dN_dxi + i12 * dN_deta
        gY = i21 * dN_dxi + i22 * dN_deta
        return gX, gY, detJ

    @njit_cached(fastmath=True)
    def _element_rotation(coords_ref: np.ndarray, coords_curr: np.ndarray) -> np.ndarray:
        """Element rigid rotation from the mean deformed XI-edge direction.

        Must stay bit-equivalent to `q4_corotational_jax.compute_element_rotation`,
        which every JAX co-rotational kernel in this library shares. That rule
        uses the xi-edge pair (`v12 + v43`) ALONE and takes e2 as its
        perpendicular; it does NOT average in the eta-edge.

        This function used to average the two edge rotations. That is a
        defensible frame choice in itself, but it is a DIFFERENT element: the
        co-rotational frame cancels exactly out of a full-integration
        Green-Lagrange element (E = (F_l^T F_l - I)/2 with F_l = R^T F is
        R-independent, which is finding F1's "bitwise TL" result), so the
        choice is invisible there -- but SRI samples the shear off-diagonal
        of dU/dX in fixed Cartesian components, which is NOT frame-invariant
        (see q4_sri_jax._F_sri's warning), so the frame choice changes the
        answer. Measured on contract C10's tilted+distorted probe:
        theta_jax = -0.014852 deg vs theta_numba = -0.681885 deg, giving a
        1.54e-03 relative internal-force disagreement between the two
        lowerings. Everything else in the two kernels -- quadrature, the SRI
        gradient, B_L, and the J2 return map -- agrees to 2e-16; feeding the
        two-edge frame's local displacement into the JAX SRI core reproduces
        this kernel to 1.09e-15, i.e. the frame was the whole of it.

        The removed comment claimed this form matched `q4_sri_numba.py` /
        `q4_sri_hybrid_numba.py`. It matched them on the TRANSPOSE convention
        (R[0,1] = -s) only; both siblings use the xi-edge alone, as here now.
        """
        e1r = coords_ref[1] - coords_ref[0] + coords_ref[2] - coords_ref[3]
        e1c = coords_curr[1] - coords_curr[0] + coords_curr[2] - coords_curr[3]
        th = np.arctan2(e1c[1], e1c[0]) - np.arctan2(e1r[1], e1r[0])
        R = np.empty((2, 2), dtype=np.float64)
        c = np.cos(th); s = np.sin(th)
        # NOTE: was transposed (R[0,1]=s, R[1,0]=-s) -- fable review 2026-09-08
        # found this makes coords_curr @ R equal R(+theta) applied AGAIN
        # (row-vector convention: v @ M applies M^T) instead of the pull-back
        # R(-theta), so a PURE RIGID ROTATION produced spurious local strain
        # growing linearly in theta (max|u_l| ~ 2x element size at 90deg
        # instead of exactly 0), feeding nonsense strain into the J2 return
        # map -> the 1e25 residual blowup on the first real (rotated) step.
        # Matches the already-correct sibling kernels q4_sri_numba.py /
        # q4_sri_hybrid_numba.py, which both use this exact form.
        R[0, 0] = c;  R[0, 1] = -s
        R[1, 0] = s;  R[1, 1] = c
        return R

    @njit_cached(fastmath=True)
    def compute_corotational_sri_j2_single_numba(
        coords: np.ndarray, u_elem: np.ndarray, state_elem: np.ndarray,
        lam: float, mu: float, sigma_y0: float, H: float, thickness: float,
    ):
        """Returns (f_global(8,), K_global(8,8), state_new(4, n_state))."""
        coords_curr = coords + u_elem.reshape((4, 2))
        R = _element_rotation(coords, coords_curr)

        T8 = np.zeros((8, 8), dtype=np.float64)
        for i in range(4):
            T8[2 * i, 2 * i] = R[0, 0];     T8[2 * i, 2 * i + 1] = R[0, 1]
            T8[2 * i + 1, 2 * i] = R[1, 0]; T8[2 * i + 1, 2 * i + 1] = R[1, 1]

        u_l = (coords_curr @ R - coords).reshape(8)

        gX0, gY0, _ = _grads_sri(0.0, 0.0, coords)

        f_l = np.zeros(8, dtype=np.float64)
        K_l = np.zeros((8, 8), dtype=np.float64)
        state_new = np.zeros_like(state_elem)

        ux = u_l[0::2].copy()
        uy = u_l[1::2].copy()

        for k in range(4):
            gX, gY, detJ = _grads_sri(_SRI_GP[k, 0], _SRI_GP[k, 1], coords)

            # SRI deformation gradient: normal terms at the Gauss point,
            # shear terms at the centroid
            F = np.empty((2, 2), dtype=np.float64)
            F[0, 0] = 1.0 + np.dot(ux, gX)
            F[0, 1] = np.dot(ux, gY0)
            F[1, 0] = np.dot(uy, gX0)
            F[1, 1] = 1.0 + np.dot(uy, gY)

            S_v, C_v, sn = stress_and_tangent_j2_numba(
                F, state_elem[k], lam, mu, sigma_y0, H)
            for q in range(state_new.shape[1]):
                state_new[k, q] = sn[q]

            # exact dE/du for that mixed measure
            B = np.zeros((3, 8), dtype=np.float64)
            for a in range(4):
                B[0, 2 * a] = F[0, 0] * gX[a]
                B[0, 2 * a + 1] = F[1, 0] * gX0[a]
                B[1, 2 * a] = F[0, 1] * gY0[a]
                B[1, 2 * a + 1] = F[1, 1] * gY[a]
                B[2, 2 * a] = F[0, 0] * gY0[a] + F[0, 1] * gX[a]
                B[2, 2 * a + 1] = F[1, 0] * gY[a] + F[1, 1] * gX0[a]

            w = detJ * thickness
            for i in range(8):
                acc = 0.0
                for r in range(3):
                    acc += B[r, i] * S_v[r]
                f_l[i] += acc * w
            CB = C_v @ B
            K_l += (B.T @ CB) * w

        f_g = T8 @ f_l
        K_g = T8 @ K_l @ T8.T
        # Same NaN/Inf zero-guard every other kernel in this codebase has;
        # see q4_visco_eas_numba.py's identical guard for the full rationale.
        bad = False
        for i in range(8):
            if not np.isfinite(f_g[i]):
                bad = True
        for i in range(8):
            for j in range(8):
                if not np.isfinite(K_g[i, j]):
                    bad = True
        if bad:
            f_g = np.zeros(8, dtype=np.float64)
            K_g = np.zeros((8, 8), dtype=np.float64)

        return f_g, K_g, state_new

    @njit_cached(fastmath=True, parallel=True)
    def assemble_corotational_sri_j2_batch_numba(
        elem_coords: np.ndarray,   # (N, 4, 2)
        u_elems: np.ndarray,       # (N, 8)
        state_elems: np.ndarray,   # (N, 4, n_state)
        lam: float, mu: float, sigma_y0: float, H: float,
        thicknesses: np.ndarray,   # (N,)
    ):
        n = elem_coords.shape[0]
        ns = state_elems.shape[2]
        f_all = np.empty((n, 8), dtype=np.float64)
        K_all = np.empty((n, 8, 8), dtype=np.float64)
        s_all = np.empty((n, 4, ns), dtype=np.float64)
        for e in numba.prange(n):
            f_e, K_e, s_e = compute_corotational_sri_j2_single_numba(
                elem_coords[e], u_elems[e], state_elems[e],
                lam, mu, sigma_y0, H, thicknesses[e])
            f_all[e] = f_e
            K_all[e] = K_e
            s_all[e] = s_e
        return f_all, K_all, s_all

else:      # pragma: no cover
    def compute_corotational_sri_j2_single_numba(*a, **k):
        raise ImportError("Numba is not installed. Run `pip install numba`.")

    def assemble_corotational_sri_j2_batch_numba(*a, **k):
        raise ImportError("Numba is not installed. Run `pip install numba`.")
