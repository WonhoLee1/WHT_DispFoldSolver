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

    Fenh_k = (detJ0/detJ) * D_k(xi,eta) @ J0^{-1}
    """
    J0inv = jnp.linalg.inv(J0)
    s = detJ0 / detJ
    Dk = jnp.array([
        [[xi, 0.0], [0.0, 0.0]],
        [[0.0, eta], [0.0, 0.0]],
        [[0.0, 0.0], [xi, 0.0]],
        [[0.0, 0.0], [0.0, eta]],
    ], dtype=jnp.float64)
    return jnp.einsum('kij,jm->kim', Dk, J0inv) * s  # (4, 2, 2)


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


# ── Number of Newton iterations for alpha (fixed for vmap) ─────────
_ALPHA_ITER = 12


# ── Main entry point ────────────────────────────────────────────────


@jax.jit
def compute_eas_j2_contributions_jax(
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

    # ── Fixed-iteration Newton for alpha condensation ────────────────
    def _alpha_step(alpha_k, _):
        f_a = jnp.zeros(4, dtype=jnp.float64)
        K_aa = jnp.zeros((4, 4), dtype=jnp.float64)
        for k in range(4):
            # Incremental total F (compatible + EAS enhancement)
            Ft_inc = Fc_all[k] + jnp.einsum('j,jab->ab', alpha_k, Fenh_all[k])
            # Material total F (UL: compose with F_n; TL: same as Ft_inc)
            Ft_mat = Ft_inc @ F_n[k]
            S_v, C_v, _ = tangent_voigt_jax(
                Ft_mat[:2, :2], state_elem[k], lam, mu, sigma_y0, H,
            )
            # G uses Ft_inc (incremental frame for linearisation)
            G = _compute_G(Ft_inc, Fenh_all[k])
            St = jnp.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])

            f_a = f_a + G.T @ S_v * w_all[k]
            K_aa = K_aa + (G.T @ C_v @ G + _compute_Kgeo_aa(Fenh_all[k], St)) * w_all[k]

        reg_scale = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(K_aa))), 1e-30)
        K_reg = K_aa + 1e-10 * reg_scale * jnp.eye(4)
        dalpha = -jnp.linalg.solve(K_reg, f_a)
        return alpha_k + dalpha, None

    alpha_final, _ = jax.lax.scan(_alpha_step, alpha, None, length=_ALPHA_ITER)

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

    return f_e, K_e, alpha_final, state_new, F_n_new
