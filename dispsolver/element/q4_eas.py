"""
q4_eas.py
=========
Enhanced Assumed Strain (EAS) Q4 element — Simo & Rifai (1990).

The compatible strain field of the bilinear Q4 locks in bending (parasitic
shear) and is too stiff for thin elements driven through a single edge. EAS
augments the strain with element-internal incompatible modes that are
condensed out, so a single element through the thickness captures pure bending
without spurious shear stiffness.

    eps_total(xi) = B(xi) u  +  M(xi) alpha

`alpha` (4 internal parameters for 2D) is found per element by enforcing the
enhanced equilibrium  f_alpha = integral( M^T S ) = 0, then statically
condensed from the element residual / tangent.

This module implements the *small-strain linear-elastic* core first, against
which the patch test and a bending benchmark are verified. The finite-strain
J2 path reuses the same M(xi) enhancement and condensation machinery.

References
----------
Simo, J.C. & Rifai, M.S. (1990). A class of mixed assumed strain methods and
    the method of incompatible modes. IJNME, 29(8), 1595-1638.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import q4


# Enhanced-strain natural-coordinate interpolation (3x4), Simo-Rifai EAS-4.
#   E_xi(xi, eta) = [[xi, 0,  0,  0 ],
#                    [0,  eta,0,  0 ],
#                    [0,  0,  xi, eta]]
def _M_xi(xi: float, eta: float) -> np.ndarray:
    M = np.zeros((3, 4), dtype=np.float64)
    M[0, 0] = xi
    M[1, 1] = eta
    M[2, 2] = xi
    M[2, 3] = eta
    return M


def _strain_transform(J: np.ndarray) -> np.ndarray:
    """3x3 strain transformation T such that eps_phys = T^{-1} eps_nat,
    built from a Jacobian J = [[j11, j12], [j21, j22]].

    Engineering-shear Voigt convention [eps_xx, eps_yy, gamma_xy].
    """
    j11, j12 = J[0, 0], J[0, 1]
    j21, j22 = J[1, 0], J[1, 1]
    T = np.array([
        [j11 * j11, j21 * j21, j11 * j21],
        [j12 * j12, j22 * j22, j12 * j22],
        [2.0 * j11 * j12, 2.0 * j21 * j22, j11 * j22 + j12 * j21],
    ], dtype=np.float64)
    return T


def enhancement_M(xi: float, eta: float, J: np.ndarray,
                  detJ: float, J0: np.ndarray, detJ0: float) -> np.ndarray:
    """Physical-frame enhanced-strain operator M(xi) (3x4).

    M(xi) = (detJ0 / detJ) * T0^{-1} * M_xi(xi)

    The (detJ0/detJ) scaling and the center Jacobian transform T0 guarantee
    integral(M dV) = 0, so constant-stress (patch) states are reproduced
    exactly: alpha stays zero and the element passes the patch test.
    """
    T0 = _strain_transform(J0)
    M = (detJ0 / detJ) * np.linalg.solve(T0, _M_xi(xi, eta))
    return M


def compute_eas_linear_K(coords: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Condensed EAS element stiffness (8x8) for linear elasticity.

    Used for verification (patch test, bending). Production paths call the
    block assembler below and condense alongside the material update.
    """
    _, detJ0, invJ0 = q4.jacobian(0.0, 0.0, coords)
    J0, _, _ = q4.jacobian(0.0, 0.0, coords)

    K_uu = np.zeros((8, 8), dtype=np.float64)
    K_ua = np.zeros((8, 4), dtype=np.float64)
    K_aa = np.zeros((4, 4), dtype=np.float64)

    for k in range(4):
        xi, eta = q4._GP2[k]
        J, detJ, invJ = q4.jacobian(xi, eta, coords)
        B = q4.B_matrix(xi, eta, invJ)
        M = enhancement_M(xi, eta, J, detJ, J0, detJ0)
        w = detJ * q4._W2[k]
        K_uu += B.T @ D @ B * w
        K_ua += B.T @ D @ M * w
        K_aa += M.T @ D @ M * w

    K_cond = K_uu - K_ua @ np.linalg.solve(K_aa, K_ua.T)
    return K_cond


# ==================================================================
# Finite-strain EAS for Total-Lagrangian J2 plasticity
# ==================================================================

def _enhanced_grad_modes(xi: float, eta: float, detJ: float,
                         J0: np.ndarray, detJ0: float):
    """Return the 4 enhanced deformation-gradient mode matrices (each 2x2).

    F_enh,k = (detJ0/detJ) * D_k(xi,eta) * J0^{-1}

    with the natural-frame enhanced displacement-gradient modes
        D_0 = [[xi,0],[0,0]]   D_1 = [[0,eta],[0,0]]
        D_2 = [[0,0],[xi,0]]   D_3 = [[0,0],[0,eta]]

    The (detJ0/detJ) scaling plus int(xi)=int(eta)=0 over [-1,1]^2 give
    int(F_enh dV)=0, so a homogeneous deformation keeps alpha=0 (patch test).
    """
    J0inv = np.linalg.inv(J0)
    s = detJ0 / detJ
    Dk = (
        np.array([[xi, 0.0], [0.0, 0.0]]),
        np.array([[0.0, eta], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [xi, 0.0]]),
        np.array([[0.0, 0.0], [0.0, eta]]),
    )
    return [s * (D @ J0inv) for D in Dk]


def _voigt_sym(P: np.ndarray) -> np.ndarray:
    """Voigt [.,.,engineering-shear] of sym(P) for a 2x2 P."""
    return np.array([P[0, 0], P[1, 1], P[0, 1] + P[1, 0]], dtype=np.float64)


def _BL_columns(Ft: np.ndarray, gX: np.ndarray, gY: np.ndarray) -> np.ndarray:
    """Total-Lagrangian strain-displacement operator B_L (3x8) at a GP.

    delta E = sym(Ft^T delta F);  columns ordered [ux1,uy1,ux2,uy2,...].
    """
    F11, F12 = Ft[0, 0], Ft[0, 1]
    F21, F22 = Ft[1, 0], Ft[1, 1]
    B = np.zeros((3, 8), dtype=np.float64)
    for a in range(4):
        gx, gy = gX[a], gY[a]
        # x-dof
        B[0, 2 * a] = F11 * gx
        B[1, 2 * a] = F12 * gy
        B[2, 2 * a] = F11 * gy + F12 * gx
        # y-dof
        B[0, 2 * a + 1] = F21 * gx
        B[1, 2 * a + 1] = F22 * gy
        B[2, 2 * a + 1] = F21 * gy + F22 * gx
    return B


def _stress_and_tangent(material, Ft2: np.ndarray, params: dict, state_gp, h: float = 1e-6):
    """PK2 stress S (Voigt) + a *consistent* material tangent C = dS/dE.

    The J2 material's analytic tangent is inconsistent with its own
    (logarithmic-strain) stress, which destroys Newton's quadratic rate. Here
    C is built by forward finite differences of the material stress along the
    three Green-Lagrange Voigt directions (engineering shear), via
        dF = Ft^{-T} dE_tensor   =>   sym(Ft^T dF) = dE_tensor,
    so the perturbation realises a pure unit strain increment.
    """
    # Degenerate (inverted / non-finite) guard: return a finite regularized
    # response (zero stress + stable isotropic tangent, state untouched) so the
    # global line search / cutback can pull the element back out of inversion.
    if np.linalg.det(Ft2) <= 1e-8:
        lam, mu = material.lam, material.mu
        C_iso = np.array([[lam + 2 * mu, lam, 0.0],
                          [lam, lam + 2 * mu, 0.0],
                          [0.0, 0.0, mu]], dtype=np.float64)
        return np.zeros(3), C_iso, (state_gp.copy() if state_gp is not None else None)

    S0, _, state_new = material.pk2_voigt(Ft2, params, state_gp)
    if not np.all(np.isfinite(S0)):
        lam, mu = material.lam, material.mu
        C_iso = np.array([[lam + 2 * mu, lam, 0.0],
                          [lam, lam + 2 * mu, 0.0],
                          [0.0, 0.0, mu]], dtype=np.float64)
        return np.zeros(3), C_iso, (state_gp.copy() if state_gp is not None else None)
    FinvT = np.linalg.inv(Ft2).T
    dirs = (
        np.array([[1.0, 0.0], [0.0, 0.0]]),   # dE_xx
        np.array([[0.0, 0.0], [0.0, 1.0]]),   # dE_yy
        np.array([[0.0, 0.5], [0.5, 0.0]]),   # unit engineering shear gamma_xy
    )
    C = np.zeros((3, 3), dtype=np.float64)
    for m, dE in enumerate(dirs):
        dF = FinvT @ dE
        Sp, _, _ = material.pk2_voigt(Ft2 + h * dF, params, state_gp)
        C[:, m] = (Sp - S0) / h
    C = 0.5 * (C + C.T)  # symmetrise (minor symmetry)
    return S0, C, state_new


def compute_eas_j2_contributions(
    coords: np.ndarray,       # (4,2)
    u_elem: np.ndarray,       # (8,)
    alpha: np.ndarray,        # (4,) element EAS parameters (warm start)
    state_elem: np.ndarray,   # (n_gp, n_vars) or None
    material,                 # J2Plasticity (F-based pk2_voigt)
    params: dict,
    thickness: float = 1.0,
    max_local_iter: int = 12,
    local_tol: float = 1e-11,
):
    """Finite-strain EAS Q4 element contribution with internal alpha solve.

    Returns
    -------
    f_e       : (8,)   condensed internal force
    K_e       : (8,8)  condensed tangent
    alpha_new : (4,)   converged EAS parameters
    state_new : (n_gp, n_vars) updated material state (or None)
    """
    n_gp = len(q4._GP2)
    J0, detJ0, invJ0 = q4.jacobian(0.0, 0.0, coords)

    # Per-GP reference geometry (independent of alpha)
    gp_geom = []
    for k in range(n_gp):
        xi, eta = q4._GP2[k]
        _, detJ, invJ = q4.jacobian(xi, eta, coords)
        dN_dxi, dN_deta = q4.shape_derivatives(xi, eta)
        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        Fenh = _enhanced_grad_modes(xi, eta, detJ, J0, detJ0)
        w = detJ * q4._W2[k] * thickness
        # compatible deformation gradient F_c = I + grad u
        ux = u_elem[0::2]
        uy = u_elem[1::2]
        Hc = np.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
        Fc = np.eye(2) + Hc
        gp_geom.append((xi, eta, gX, gY, Fenh, w, Fc))

    alpha = alpha.copy()
    state_new = np.zeros_like(state_elem) if state_elem is not None else None

    def _fa_Kaa(al):
        """Enhanced residual f_alpha and tangent K_aa at parameters `al`.

        Always finite: inverted GPs are regularized inside _stress_and_tangent.
        """
        K_aa = np.zeros((4, 4))
        f_a = np.zeros(4)
        for k in range(n_gp):
            _, _, _, _, Fenh, w, Fc = gp_geom[k]
            Ft = Fc + sum(al[j] * Fenh[j] for j in range(4))
            sg = state_elem[k] if state_elem is not None else None
            S_v, C_v, _ = _stress_and_tangent(material, Ft[:2, :2], params, sg)
            G = np.stack([_voigt_sym(Ft.T @ Fenh[j]) for j in range(4)], axis=1)
            f_a += G.T @ S_v * w
            St = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
            Kgeo = np.zeros((4, 4))
            for a in range(4):
                for b in range(4):
                    P = 0.5 * (Fenh[a].T @ Fenh[b] + Fenh[b].T @ Fenh[a])
                    Kgeo[a, b] = (St[0, 0] * P[0, 0] + St[1, 1] * P[1, 1]
                                  + 2.0 * St[0, 1] * P[0, 1])
            K_aa += (G.T @ C_v @ G + Kgeo) * w
        return f_a, K_aa

    # --- internal Newton (damped line search on |f_alpha|) ---
    f_a, K_aa = _fa_Kaa(alpha)
    for _ in range(max_local_iter):
        fn = np.linalg.norm(f_a)
        if fn < local_tol:
            break
        K_reg = K_aa + 1e-12 * (np.trace(K_aa) / 4.0 + 1e-30) * np.eye(4)
        dalpha = -np.linalg.solve(K_reg, f_a)
        ls = 1.0
        f_try, K_try = _fa_Kaa(alpha + ls * dalpha)
        while ls > 1e-3 and np.linalg.norm(f_try) >= fn:
            ls *= 0.5
            f_try, K_try = _fa_Kaa(alpha + ls * dalpha)
        alpha = alpha + ls * dalpha
        f_a, K_aa = f_try, K_try

    # --- final assembly of all blocks at converged alpha ---
    K_uu = np.zeros((8, 8)); K_ua = np.zeros((8, 4)); K_aa = np.zeros((4, 4))
    f_u = np.zeros(8); f_a = np.zeros(4)
    for k in range(n_gp):
        _, _, gX, gY, Fenh, w, Fc = gp_geom[k]
        Ft = Fc + sum(alpha[j] * Fenh[j] for j in range(4))
        sg = state_elem[k] if state_elem is not None else None
        S_v, C_v, sg_new = _stress_and_tangent(material, Ft[:2, :2], params, sg)
        if state_new is not None and sg_new is not None:
            state_new[k] = sg_new
        St = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        BL = _BL_columns(Ft, gX, gY)
        G = np.stack([_voigt_sym(Ft.T @ Fenh[j]) for j in range(4)], axis=1)

        f_u += BL.T @ S_v * w
        f_a += G.T @ S_v * w

        # geometric blocks
        grad_N = np.stack([gX, gY], axis=1)        # (4,2)
        gamma = grad_N @ St @ grad_N.T             # (4,4) node-node
        Kgeo_uu = np.zeros((8, 8))
        Kgeo_uu[0::2, 0::2] = gamma
        Kgeo_uu[1::2, 1::2] = gamma
        Kgeo_ua = np.zeros((8, 4))
        for a in range(4):
            for i in range(2):
                ei_row = np.zeros(2); ei_row[i] = 1.0
                for kk in range(4):
                    Pr = np.outer(grad_N[a], Fenh[kk][i, :])  # outer(g_a, row i of Fenh_k)
                    Ps = 0.5 * (Pr + Pr.T)
                    Kgeo_ua[2 * a + i, kk] = (St[0, 0] * Ps[0, 0] + St[1, 1] * Ps[1, 1]
                                              + 2.0 * St[0, 1] * Ps[0, 1])
        Kgeo_aa = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                P = 0.5 * (Fenh[a].T @ Fenh[b] + Fenh[b].T @ Fenh[a])
                Kgeo_aa[a, b] = St[0, 0] * P[0, 0] + St[1, 1] * P[1, 1] + 2.0 * St[0, 1] * P[0, 1]

        K_uu += (BL.T @ C_v @ BL + Kgeo_uu) * w
        K_ua += (BL.T @ C_v @ G + Kgeo_ua) * w
        K_aa += (G.T @ C_v @ G + Kgeo_aa) * w

    K_aa_inv_KuaT = np.linalg.solve(K_aa, K_ua.T)
    K_e = K_uu - K_ua @ K_aa_inv_KuaT
    f_e = f_u - K_ua @ np.linalg.solve(K_aa, f_a)
    return f_e, K_e, alpha, state_new
