"""Pure JAX EAS Q4 element with autodiff tangent and fixed-iteration alpha condensation.

Port of q4_eas.py to JAX. Key improvements:
1. Autodiff consistent tangent via tangent_voigt_jax (replaces FD)
2. All-JAX arrays for vmap compatibility
3. Fixed-iteration Newton for alpha condensation (vmap safe — no data-dependent exit)

Updated-Lagrangian (UL) mode
-----------------------------
When F_n_gps (shape (4,2,2)) is supplied, the element operates in UL mode:
  coords   = reference coordinates at the LAST CONVERGED step
  u_elem   = INCREMENTAL displacement from those reference coords
  F_n_gps[k] = total deformation gradient at the last converged step (GP k)

The total deformation gradient used for material evaluation is then:
  Ft_mat = Ft_inc @ F_n_gps[k]

where Ft_inc = I + grad(u_inc) (always det > 0 for small increments, preventing
element inversion). B_L and G use Ft_inc for the kinematic linearisation.

When F_n_gps = identity (or TL mode with F_n_gps=jnp.tile(eye,(4,1,1))):
  Ft_mat = Ft_inc = Ft   (standard Total Lagrangian, no change in behaviour)

References
----------
Simo, J.C. & Rifai, M.S. (1990). A class of mixed assumed strain methods.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dispsolver.material.plastic_jax import tangent_voigt_jax

# ── Gauss points & weights ──────────────────────────────────────────
_S3 = jnp.sqrt(3.0)
_GP2 = jnp.array([
    [-1.0 / _S3, -1.0 / _S3],
    [ 1.0 / _S3, -1.0 / _S3],
    [ 1.0 / _S3,  1.0 / _S3],
    [-1.0 / _S3,  1.0 / _S3],
], dtype=jnp.float64)
_W2 = jnp.ones(4, dtype=jnp.float64)

# ── Shape functions & Jacobian ──────────────────────────────────────


def _sd(xi, eta):
    dN_dxi  = 0.25 * jnp.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * jnp.array([-(1 - xi),  -(1 + xi),   (1 + xi),   (1 - xi)])
    return dN_dxi, dN_deta


def _jac(xi, eta, coords):
    dN_dxi, dN_deta = _sd(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:, 0]), jnp.dot(dN_dxi, coords[:, 1])],
        [jnp.dot(dN_deta, coords[:, 0]), jnp.dot(dN_deta, coords[:, 1])],
    ])
    return J, jnp.linalg.det(J), jnp.linalg.inv(J)


# ── EAS enhanced gradient modes ─────────────────────────────────────


def _enhanced_grad_modes(xi, eta, detJ, J0, detJ0):
    """4 enhanced deformation-gradient modes (4, 2, 2).

    Fenh_k = (detJ0/detJ) * D_k(xi,eta) @ J0^{-T}

    TRANSPOSE FIX (2026-09-08, opus review + independent numerical
    confirmation; see dev_log/plan_abaqus_element_consolidation_20260908.md):
    this used to contract `D_k` with `J0^{-1}` instead of `J0^{-T}`, which
    is an index-TYPE mismatch. `_jac` above builds `J[i][j] = dX_j/dxi_i`
    (rows natural, cols spatial), so `inv(J0)[a][b] = dxi_b/dX_a` -- rows
    SPATIAL, cols NATURAL. `D_k`'s second index is NATURAL (the modes are
    written in natural coordinates), so the pull-back
    `d(u~)_i/dX_m = D_ij * dxi_j/dX_m = D_ij * inv(J0)[m][j]`
    requires the TRANSPOSE. Written with `J0^{-1}` the contraction pairs a
    natural index against a spatial one.

    Why it went unnoticed: for an axis-aligned rectangular element `J0` is
    DIAGONAL, so `J0^{-1} == J0^{-T}` and the bug is exactly invisible --
    every patch test, the AGENTS.md 4.1 aspect-ratio sweep, and every
    existing unit test in this repo uses axis-aligned elements at their
    reference configuration.

    Why it mattered anyway: with `ul_large_rotation_mode` on (the default,
    `fold_model_config.py`), the Updated-Lagrangian reference config IS the
    last converged configuration, so `J0` carries the accumulated fold
    rotation and goes off-diagonal within a few increments. Measured
    bending-energy ratio (1.000 = exact) against reference rotation angle
    at the production free-span geometry, before vs after this fix:

        ref rotation:      0deg   10deg   30deg   45deg   60deg   90deg
        before (J0^-1):   1.000   9.384   3.377  10.279   9.804   1.000
        after  (J0^-T):   1.000   1.000   1.000   1.000   1.000   1.000

    i.e. the enhanced modes were up to ~10x too stiff in bending for any
    element whose reference frame had rotated off-axis -- which, in a
    90-degree fold, is essentially all of them. Global-axis isotropy
    likewise went from ~1e-1 to ~1e-12.
    """
    J0inv = jnp.linalg.inv(J0)
    s = detJ0 / detJ
    Dk = jnp.array([
        [[xi, 0.0], [0.0, 0.0]],
        [[0.0, eta], [0.0, 0.0]],
        [[0.0, 0.0], [xi, 0.0]],
        [[0.0, 0.0], [0.0, eta]],
    ], dtype=jnp.float64)
    # 'kij,mj->kim' == D_k @ J0^{-T}
    return jnp.einsum('kij,mj->kim', Dk, J0inv) * s  # (4, 2, 2)


# ── Strain-displacement operators ───────────────────────────────────


def _voigt_sym(P):
    """Voigt [P_11, P_22, P_12+P_21] of 2×2 P."""
    return jnp.array([P[0, 0], P[1, 1], P[0, 1] + P[1, 0]])


def _BL_columns(Ft, gX, gY):
    """Total-Lagrangian B_L (3×8):  delta E = sym(Ft^T delta F)."""
    F11, F12 = Ft[0, 0], Ft[0, 1]
    F21, F22 = Ft[1, 0], Ft[1, 1]
    B = jnp.zeros((3, 8), dtype=jnp.float64)
    for a in range(4):
        gx, gy = gX[a], gY[a]
        B = B.at[0, 2 * a].set(F11 * gx)
        B = B.at[1, 2 * a].set(F12 * gy)
        B = B.at[2, 2 * a].set(F11 * gy + F12 * gx)
        B = B.at[0, 2 * a + 1].set(F21 * gx)
        B = B.at[1, 2 * a + 1].set(F22 * gy)
        B = B.at[2, 2 * a + 1].set(F21 * gy + F22 * gx)
    return B


def _compute_G(Ft, Fenh):
    """Enhanced coupling G (3×4):  G_j = voigt_sym(Ft^T @ Fenh_j)."""
    FtFenh = jnp.einsum('ba,jbc->jac', Ft, Fenh)  # (4, 2, 2)
    return jnp.stack([_voigt_sym(FtFenh[j]) for j in range(4)], axis=1)


def _push_T(A):
    """Voigt operator of the pull-back  E_mat = A^T E_inc A.

    Acts on STRAIN-like Voigt vectors ``[e11, e22, 2*e12]``, so its transpose
    acts on STRESS-like Voigt vectors ``[S11, S22, S12]`` and realizes
    ``A S A^T``.  (Check: ``(T^T S_v)[0] = A00^2 S11 + A01^2 S22 +
    2 A00 A01 S12 = (A S A^T)[0,0]``.)
    """
    return jnp.array([
        [A[0, 0] ** 2, A[1, 0] ** 2, A[0, 0] * A[1, 0]],
        [A[0, 1] ** 2, A[1, 1] ** 2, A[0, 1] * A[1, 1]],
        [2.0 * A[0, 0] * A[0, 1], 2.0 * A[1, 0] * A[1, 1],
         A[0, 0] * A[1, 1] + A[1, 0] * A[0, 1]],
    ], dtype=jnp.float64)


def _push_forward(S_v, C_v, F_n_k):
    """Work-conjugacy push-forward of (PK2, material tangent) onto config n.

    AGENTS.md §4.14 finding F4, previously fixed in `q4_visco_simo_fs_jax.py`
    / `q4_visco_eas_jax.py` / `q4_visco_hybrid_reduced_jax.py` but NOT here.

    `S_v`/`C_v` come out of the material referred to the ORIGINAL config
    (they are evaluated at the TOTAL ``Ft_mat = Ft_inc @ F_n``), while every
    kinematic operator this element builds -- ``B_L(Ft_inc)``, ``G(Ft_inc)``,
    ``Fenh``, and the weight ``w = detJ`` on the step-n ``coords`` -- is
    conjugate to the INCREMENTAL Green-Lagrange strain on config n.
    Contracting the two directly mixes reference frames.

    Since ``E_mat = F_n^T E_inc F_n`` (exactly, because
    ``Ft_mat = Ft_inc @ F_n``) and ``dV_0 = dV_n / det(F_n)``:

        S : dE_mat dV0  =  (F_n S F_n^T / det F_n) : dE_inc dV_n

    so ``S_n = T^T S / det F_n`` and, by the same chain rule applied once
    more, ``C_n = T^T C T / det F_n``.  Both are the identity map when
    ``F_n = I``, i.e. **TL mode is bit-identical to before this fix**.

    Pushing the tangent (not just the force) is what makes ``K_aa`` the true
    Jacobian of ``f_alpha`` again: measured on the real failing element of
    the nlgeo-cantilever benchmark (39 degrees of accumulated reference
    rotation), ``||K_coded - K_finite_difference|| / ||K_fd||`` went from
    **0.976** to **4.6e-10**, the finite-difference Jacobian went from 150%
    asymmetric to symmetric at 7.8e-11, and the element-local Newton went
    from wandering (|f_a| oscillating around 1e-1 for 20+ iterations, never
    converging) to quadratic convergence in 2 iterations.
    """
    detFn = jnp.maximum(
        jnp.abs(F_n_k[0, 0] * F_n_k[1, 1] - F_n_k[0, 1] * F_n_k[1, 0]), 1e-30)
    T = _push_T(F_n_k)
    return (T.T @ S_v) / detFn, (T.T @ C_v @ T) / detFn


def _compute_Kgeo_aa(Fenh, St):
    """Geometric stiffness Kgeo_aa (4×4)."""
    K = jnp.zeros((4, 4), dtype=jnp.float64)
    for a in range(4):
        for b in range(4):
            P = 0.5 * (Fenh[a].T @ Fenh[b] + Fenh[b].T @ Fenh[a])
            K = K.at[a, b].set(
                St[0, 0] * P[0, 0] + St[1, 1] * P[1, 1]
                + 2.0 * St[0, 1] * P[0, 1]
            )
    return K


def _compute_Kgeo_ua(grad_N, St, Fenh):
    """Geometric stiffness Kgeo_ua (8×4)."""
    K = jnp.zeros((8, 4), dtype=jnp.float64)
    for a_node in range(4):
        for i in range(2):
            for kk in range(4):
                Pr = jnp.outer(grad_N[a_node], Fenh[kk][i, :])
                Ps = 0.5 * (Pr + Pr.T)
                K = K.at[2 * a_node + i, kk].set(
                    St[0, 0] * Ps[0, 0] + St[1, 1] * Ps[1, 1]
                    + 2.0 * St[0, 1] * Ps[0, 1]
                )
    return K


# ── Newton convergence parameters for alpha condensation ────────────
_ALPHA_MAX_ITER = 20
# Convergence tolerance on the LAST accepted enhanced-mode Newton correction,
# ||d(alpha)||_inf. `alpha` scales O(1) dimensionless enhanced
# deformation-gradient modes, so it is itself dimensionless and an absolute
# tolerance is meaningful independently of mesh size and units.
#
# 2026-09-10 fix: the loop's exit test used to compare `f_a_norm` (the
# ENHANCED-MODE FORCE RESIDUAL, dimensional: stress x area, so its scale
# tracks the material modulus) against a fixed absolute `_ALPHA_TOL = 1e-10`.
# `_ALPHA_CONV_TOL` existed (with this exact docstring) but was never wired
# into the actual condition -- dead code from an incomplete migration.
# Confirmed via a real Abaqus benchmark reproduction (Cook's membrane /
# nlgeocantilever, scratch/bmk_*.py, dev_log TBD): for a stiffer material
# (E=1e8, this benchmark's modulus, vs the ~1e3-1e4 range this file's other
# tests all use) the residual floor from floating-point roundoff sits
# around 1e-9-1e-10 -- ABOVE the 1e-10 threshold -- so the loop exhausted
# its iteration cap and reported non-convergence (`status=inf`) on inputs
# that a plain (line-search-free) Newton iteration converges to machine
# precision in 2 iterations. Erratic, non-monotonic in load magnitude (a
# noise-floor comparison, not a real stability boundary) -- NOT the
# Wriggers-Reese EAS material instability (K_aa stayed positive-definite,
# well-conditioned, throughout). Switching the exit test to the already-
# tracked, dimensionless `||d(alpha)||_inf` fixes this for any material
# modulus scale.
_ALPHA_CONV_TOL = 1e-8

# Backtracking line-search step lengths for the element-local alpha Newton.
# EAS elements' "lack of robustness in the Newton-Raphson scheme" is one of the
# two documented open issues of the method (Pfefferkorn, Bieber, Oesterle,
# Bischoff & Betsch, "Improving efficiency and robustness of enhanced assumed
# strain elements for nonlinear problems", IJNME 2021, DOI 10.1002/nme.6605),
# and a line search is one of the two remedies that work reports. The NumPy
# sibling `q4_eas.py` has always used a damped local line search and is the one
# EAS kernel in this repo that never needed a magnitude clamp; this brings the
# JAX kernel onto the same footing. A line search cannot move the converged
# root (f_alpha = 0 either way) -- only the path to it.
_LS_STEPS = jnp.array([1.0, 0.5, 0.25, 0.1], dtype=jnp.float64)

# REMOVED 2026-09-09: the `_ALPHA_MAX` magnitude clamp (tanh saturation of
# alpha; 0.5 as of 2026-09-08, 0.05 before that, originally 5.0).
#
# It had no basis in the EAS literature and no counterpart in any commercial
# solver's documented behaviour. The real EAS defect it was standing in for --
# the large-compression hourglass instability of Simo-Armero Q1/E4 -- is a
# RANK DEFICIENCY of the element tangent (Wriggers & Reese, "A note on
# enhanced strain methods for large deformations", CMAME 135:201-209, 1996;
# de Souza Neto et al., Comm. Num. Meth. Engng 11(11), 1995), and the
# literature cures it at the FORMULATION level -- transposed Wilson modes,
# element Q1/E4T (Glaser & Armero, Engineering Computations 14(7):759-791,
# 1997; Pfefferkorn & Betsch, IJNME 121(8):1695-1737, 2020) -- never by
# capping the magnitude of a condensed internal variable. Saturating alpha
# also violates the very stationarity condition (f_alpha = 0) that the
# condensed tangent K_uu - K_ua K_aa^-1 K_au is derived from, so the force and
# the tangent stop belonging to the same state.
#
# Replacement: the local Newton now REPORTS non-convergence through a 6th
# return value (`status`), and `DynamicSolver` abandons the increment and cuts
# back -- the same response Abaqus/Standard gives a non-converged
# element-local algorithm (its "plasticity/creep/connector friction algorithm
# did not converge at N points" path re-attempts the increment at a smaller
# size). See dev_log/eas_stabilization_modernization_20260909.md.


# ── Main entry point ────────────────────────────────────────────────


@jax.jit
def compute_eas_j2_contributions_jax_status(
    coords,          # (4, 2) reference coords (last converged in UL; original in TL)
    u_elem,          # (8,)   displacement from reference (incremental in UL; total in TL)
    alpha,           # (4,)  warm-start EAS parameters
    state_elem,      # (4, 5) per-GP J2 state or initial state
    lam, mu, sigma_y0, H,
    thickness=1.0,
    F_n_gps=None,    # (4, 2, 2) total F at last converged step; None or eye(2) → TL
):
    """Finite-strain EAS Q4 element with J2 plasticity — pure JAX.

    In UL mode (F_n_gps provided and non-identity):
      Ft_mat  = Ft_inc @ F_n_gps[k]   used for stress / tangent
      Ft_inc  = Fc_k + EAS_k           used for B_L / G (det > 0 for small steps)

    In TL mode (F_n_gps=None or identity):
      Ft_mat  = Ft_inc                 (original behaviour)

    Returns
    -------
    f_e       : (8,)   condensed internal force
    K_e       : (8, 8) condensed tangent
    alpha_new : (4,)   converged EAS parameters
    state_new : (4, 5) updated material state
    F_n_new   : (4, 2, 2) updated total F per GP (= Ft_mat at converged alpha)
    status    : ()      element-local convergence indicator -- ||d(alpha)||_inf
                        of the last accepted internal Newton correction
                        (0 = converged), or inf if the local solve hit the
                        iteration cap without satisfying ||d(alpha)||_inf <=
                        _ALPHA_CONV_TOL, or went non-finite. The solver turns
                        a value above `eas_local_tol` into an increment
                        cutback; see the `_ALPHA_MAX` removal note above.
    """
    # Default F_n_gps to identity (TL mode)
    _eye2 = jnp.eye(2, dtype=jnp.float64)
    if F_n_gps is None:
        F_n = jnp.stack([_eye2, _eye2, _eye2, _eye2])  # (4, 2, 2)
    else:
        F_n = F_n_gps

    # ── Reference geometry ──
    J0, detJ0, _ = _jac(0.0, 0.0, coords)

    gX_all = jnp.zeros((4, 4), dtype=jnp.float64)
    gY_all = jnp.zeros((4, 4), dtype=jnp.float64)
    Fenh_all = jnp.zeros((4, 4, 2, 2), dtype=jnp.float64)
    w_all = jnp.zeros(4, dtype=jnp.float64)
    Fc_all = jnp.zeros((4, 2, 2), dtype=jnp.float64)

    for k in range(4):
        xi, eta = _GP2[k]
        _, detJ, invJ = _jac(xi, eta, coords)
        dN_dxi, dN_deta = _sd(xi, eta)
        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        Fenh = _enhanced_grad_modes(xi, eta, detJ, J0, detJ0)
        w = detJ * _W2[k] * thickness

        gX_all = gX_all.at[k].set(gX)
        gY_all = gY_all.at[k].set(gY)
        Fenh_all = Fenh_all.at[k].set(Fenh)
        w_all = w_all.at[k].set(w)

        # Compatible deformation gradient  F_c = I + grad(u)
        ux = u_elem[0::2]
        uy = u_elem[1::2]
        Hc = jnp.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
        Fc_all = Fc_all.at[k].set(jnp.eye(2) + Hc)

    # ── Newton for alpha condensation with convergence check ──────────
    def _f_alpha(alpha_k):
        """Enhanced-mode residual f_alpha at `alpha_k` (line-search probe)."""
        f_a = jnp.zeros(4, dtype=jnp.float64)
        for k in range(4):
            Ft_inc = Fc_all[k] + jnp.einsum('j,jab->ab', alpha_k, Fenh_all[k])
            Ft_mat = Ft_inc @ F_n[k]
            S_v, C_v, _ = tangent_voigt_jax(
                Ft_mat[:2, :2], state_elem[k], lam, mu, sigma_y0, H,
            )
            S_v, _ = _push_forward(S_v, C_v, F_n[k])
            G = _compute_G(Ft_inc, Fenh_all[k])
            f_a = f_a + G.T @ S_v * w_all[k]
        return f_a

    def _alpha_cond(state):
        alpha_k, f_a_norm, iter_count, _da = state
        return (_da > _ALPHA_CONV_TOL) & (iter_count < _ALPHA_MAX_ITER)

    def _alpha_body(state):
        alpha_k, _, iter_count, _da_prev = state
        f_a = jnp.zeros(4, dtype=jnp.float64)
        K_aa = jnp.zeros((4, 4), dtype=jnp.float64)
        for k in range(4):
            Ft_inc = Fc_all[k] + jnp.einsum('j,jab->ab', alpha_k, Fenh_all[k])
            Ft_mat = Ft_inc @ F_n[k]
            S_v, C_v, _ = tangent_voigt_jax(
                Ft_mat[:2, :2], state_elem[k], lam, mu, sigma_y0, H,
            )
            S_v, C_v = _push_forward(S_v, C_v, F_n[k])
            G = _compute_G(Ft_inc, Fenh_all[k])
            St = jnp.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])

            f_a = f_a + G.T @ S_v * w_all[k]
            K_aa = K_aa + (G.T @ C_v @ G + _compute_Kgeo_aa(Fenh_all[k], St)) * w_all[k]

        reg_scale = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(K_aa))), 1e-30)
        K_reg = K_aa + 1e-10 * reg_scale * jnp.eye(4)
        dalpha = -jnp.linalg.solve(K_reg, f_a)

        # --- Backtracking line search on |f_alpha| over a fixed step-length
        # set (branch-free, so it stays jit/vmap safe).
        norms = jnp.stack([jnp.linalg.norm(_f_alpha(alpha_k + ls * dalpha))
                           for ls in _LS_STEPS])
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        j_ls = jnp.argmin(norms)
        step = _LS_STEPS[j_ls] * dalpha
        alpha_new = alpha_k + step

        # --- Non-finite robustness: freeze alpha, exit the loop, and LATCH
        # the failure so it reaches the caller through `status` instead of
        # being absorbed silently. (The former tanh magnitude saturation
        # lived here -- see the `_ALPHA_MAX` removal note at the top of this
        # module for why it is gone.)
        nan_detected = (~jnp.all(jnp.isfinite(alpha_new))
                        | ~jnp.all(jnp.isfinite(f_a))
                        | ~jnp.isfinite(norms[j_ls]))
        alpha_new = jnp.where(nan_detected, alpha_k, alpha_new)
        # Non-finite -> set the residual norm to 0 so the while_loop exits
        # (alpha frozen at its last good value); the inf latched into
        # `da_meas` is what actually reports the failure upward. The loop's
        # own exit test uses the residual AT the accepted iterate, so the
        # line search cannot mask a stalled solve.
        f_a_new_norm = jnp.where(nan_detected, 0.0, norms[j_ls])
        da_meas = jnp.where(nan_detected, jnp.inf, jnp.max(jnp.abs(step)))
        return (alpha_new, f_a_new_norm, iter_count + 1, da_meas)

    init_state = (alpha, jnp.array(1.0), jnp.array(0),
                  jnp.array(jnp.inf, dtype=jnp.float64))
    alpha_final, _fa_norm_end, _n_alpha_it, _da_end = jax.lax.while_loop(
        _alpha_cond, _alpha_body, init_state)
    # Exhausting the iteration cap without reaching _ALPHA_CONV_TOL is itself
    # non-convergence, even if the final residual happened to look small.
    _local_failed = ((_n_alpha_it >= _ALPHA_MAX_ITER)
                     & (_da_end > _ALPHA_CONV_TOL)) | ~jnp.isfinite(_da_end)

    # ── Final assembly at converged alpha ────────────────────────────
    K_uu = jnp.zeros((8, 8), dtype=jnp.float64)
    K_ua = jnp.zeros((8, 4), dtype=jnp.float64)
    K_aa = jnp.zeros((4, 4), dtype=jnp.float64)
    f_u = jnp.zeros(8, dtype=jnp.float64)
    f_a = jnp.zeros(4, dtype=jnp.float64)
    state_new = jnp.zeros_like(state_elem)
    F_n_new = jnp.zeros((4, 2, 2), dtype=jnp.float64)

    for k in range(4):
        Ft_inc = Fc_all[k] + jnp.einsum('j,jab->ab', alpha_final, Fenh_all[k])
        Ft_mat = Ft_inc @ F_n[k]

        S_v, C_v, sn = tangent_voigt_jax(
            Ft_mat[:2, :2], state_elem[k], lam, mu, sigma_y0, H,
        )
        state_new = state_new.at[k].set(sn)

        # Store updated total F for UL bookkeeping
        F_n_new = F_n_new.at[k].set(Ft_mat)

        # Work-conjugacy push-forward onto config n (see `_push_forward`);
        # every operator below (BL, G, Kgeo, w) lives on config n.
        S_v, C_v = _push_forward(S_v, C_v, F_n[k])

        St = jnp.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])

        # B_L uses Ft_inc (det > 0 in UL mode → no sign-flip needed;
        # the guard is kept as a safety net for unusual configurations).
        det_inc = Ft_inc[0, 0] * Ft_inc[1, 1] - Ft_inc[0, 1] * Ft_inc[1, 0]
        Ft_bl = jnp.where(det_inc < 0.0, -Ft_inc, Ft_inc)
        BL = _BL_columns(Ft_bl, gX_all[k], gY_all[k])
        G = _compute_G(Ft_inc, Fenh_all[k])
        grad_N = jnp.stack([gX_all[k], gY_all[k]], axis=1)  # (4, 2)

        # Geometric stiffness — uu block
        gamma = grad_N @ St @ grad_N.T
        Kgeo_uu = jnp.zeros((8, 8), dtype=jnp.float64)
        Kgeo_uu = Kgeo_uu.at[0::2, 0::2].set(gamma)
        Kgeo_uu = Kgeo_uu.at[1::2, 1::2].set(gamma)

        Kgeo_ua = _compute_Kgeo_ua(grad_N, St, Fenh_all[k])
        Kgeo_aa = _compute_Kgeo_aa(Fenh_all[k], St)

        f_u = f_u + BL.T @ S_v * w_all[k]
        f_a = f_a + G.T @ S_v * w_all[k]
        K_uu = K_uu + (BL.T @ C_v @ BL + Kgeo_uu) * w_all[k]
        K_ua = K_ua + (BL.T @ C_v @ G + Kgeo_ua) * w_all[k]
        K_aa = K_aa + (G.T @ C_v @ G + Kgeo_aa) * w_all[k]

    # ── Static condensation (Tikhonov-regularized so a near-inverted trial
    #    element during line search cannot make K_aa singular) ──
    reg_scale = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(K_aa))), 1e-30)
    K_aa_reg = K_aa + 1e-10 * reg_scale * jnp.eye(4)
    K_aa_inv_KuaT = jnp.linalg.solve(K_aa_reg, K_ua.T)
    K_e = K_uu - K_ua @ K_aa_inv_KuaT
    f_e = f_u - K_ua @ jnp.linalg.solve(K_aa_reg, f_a)

    # ── NaN guard: if element is unrecoverable, return zero contributions
    #    so the solver's line search / time-step cutback handles it gracefully ──
    elem_nan = ~jnp.all(jnp.isfinite(f_e)) | ~jnp.all(jnp.isfinite(K_e))
    f_e = jnp.where(elem_nan, jnp.zeros_like(f_e), f_e)
    K_e = jnp.where(elem_nan, jnp.zeros_like(K_e), K_e)
    alpha_final = jnp.where(elem_nan, alpha, alpha_final)
    status = jnp.where(elem_nan | _local_failed, jnp.inf, _da_end)

    return f_e, K_e, alpha_final, state_new, F_n_new, status


@jax.jit
def compute_eas_j2_contributions_jax(
    coords, u_elem, alpha, state_elem, lam, mu, sigma_y0, H,
    thickness=1.0, F_n_gps=None,
):
    """Back-compatible 5-tuple wrapper (drops the trailing `status`).

    Kept so existing callers/tests that unpack exactly
    (f_e, K_e, alpha_new, state_new, F_n_new) keep working; `DynamicSolver`
    calls `compute_eas_j2_contributions_jax_status` so it can act on
    element-local failure.
    """
    return compute_eas_j2_contributions_jax_status(
        coords, u_elem, alpha, state_elem, lam, mu, sigma_y0, H,
        thickness, F_n_gps)[:5]
