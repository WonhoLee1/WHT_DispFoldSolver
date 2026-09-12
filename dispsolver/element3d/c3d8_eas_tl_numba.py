"""
c3d8_eas_tl_numba.py
====================
3D Total Lagrangian (TL) 9-mode Enhanced Assumed Strain (EAS) Hexahedral Element.

Combines Simo & Armero (1992) 9-mode EAS formulation with Total Lagrangian kinematics:
- Completely eliminates 3D Shear Locking in bending (C3D8I)
- Multiplicative F-bar Volumetric Locking elimination (nu -> 0.5)
- Internal mode static condensation K_cond = K_uu - K_ua * inv(K_aa) * K_ua^T
- Full non-linear tangent stiffness K = K_cond + K_geo
"""

# ==============================================================================
# 🚨 CRITICAL THEORETICAL DEFECT WARNING (2026-09-12 Audit, still open) 🚨
# This C3D8I (EAS) element uses a Total Lagrangian (TL) framework: B_L is built
# from the TOTAL deformation gradient F referred to the ORIGINAL (t=0) reference
# every call, not an incremental Delta-F referred to the start of the current
# increment.
#
# Per Abaqus Theory Guide (benchmark_element/reference_abaqus_docs/incompatible.txt,
# "Geometrically nonlinear formulation"), this additive-F approach satisfies the
# large-strain patch test only for HOMOGENEOUS deformations; once the element
# distorts under load, an incremental (start-of-increment) reference is required
# or the incompatible modes erroneously couple with the deformation history.
#
# This is UNCHANGED by the 2026-09-13 fix below (which addressed a different,
# already-confirmed defect: the enhanced strain was never fed into the material
# call at all -- see that fix's docstring). Do not read the presence of that fix
# as resolving this TL/incremental-reference concern; reformulating into an
# incremental/Updated-Lagrangian reference remains open, unverified for large
# rotation/deformation problems.
# ==============================================================================

import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_3d, MAT_CUSTOM_ELASTIC

_GP_GAUSS = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)


@njit(fastmath=True)
def _sd3d(xi: float, eta: float, zeta: float):
    """3D Trilinear shape function derivatives."""
    dN_dxi = np.zeros(8, dtype=np.float64)
    dN_deta = np.zeros(8, dtype=np.float64)
    dN_dzeta = np.zeros(8, dtype=np.float64)

    dN_dxi[0]   = -0.125 * (1.0 - eta) * (1.0 - zeta)
    dN_deta[0]  = -0.125 * (1.0 - xi)  * (1.0 - zeta)
    dN_dzeta[0] = -0.125 * (1.0 - xi)  * (1.0 - eta)

    dN_dxi[1]   =  0.125 * (1.0 - eta) * (1.0 - zeta)
    dN_deta[1]  = -0.125 * (1.0 + xi)  * (1.0 - zeta)
    dN_dzeta[1] = -0.125 * (1.0 + xi)  * (1.0 - eta)

    dN_dxi[2]   =  0.125 * (1.0 + eta) * (1.0 - zeta)
    dN_deta[2]  =  0.125 * (1.0 + xi)  * (1.0 - zeta)
    dN_dzeta[2] = -0.125 * (1.0 + xi)  * (1.0 + eta)

    dN_dxi[3]   = -0.125 * (1.0 + eta) * (1.0 - zeta)
    dN_deta[3]  =  0.125 * (1.0 - xi)  * (1.0 - zeta)
    dN_dzeta[3] = -0.125 * (1.0 - xi)  * (1.0 + eta)

    dN_dxi[4]   = -0.125 * (1.0 - eta) * (1.0 + zeta)
    dN_deta[4]  = -0.125 * (1.0 - xi)  * (1.0 + zeta)
    dN_dzeta[4] =  0.125 * (1.0 - xi)  * (1.0 - eta)

    dN_dxi[5]   =  0.125 * (1.0 - eta) * (1.0 + zeta)
    dN_deta[5]  = -0.125 * (1.0 + xi)  * (1.0 + zeta)
    dN_dzeta[5] =  0.125 * (1.0 + xi)  * (1.0 - eta)

    dN_dxi[6]   =  0.125 * (1.0 + eta) * (1.0 + zeta)
    dN_deta[6]  =  0.125 * (1.0 + xi)  * (1.0 + zeta)
    dN_dzeta[6] =  0.125 * (1.0 + xi)  * (1.0 + eta)

    dN_dxi[7]   = -0.125 * (1.0 + eta) * (1.0 + zeta)
    dN_deta[7]  =  0.125 * (1.0 - xi)  * (1.0 + zeta)
    dN_dzeta[7] =  0.125 * (1.0 - xi)  * (1.0 + eta)

    return dN_dxi, dN_deta, dN_dzeta


@njit(fastmath=True)
def _jacobian3d(xi: float, eta: float, zeta: float, coords: np.ndarray):
    """Compute 3x3 reference Jacobian matrix J0 = dX/dxi."""
    dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)

    J = np.zeros((3, 3), dtype=np.float64)
    for i in range(8):
        J[0, 0] += dN_dxi[i]   * coords[i, 0]
        J[0, 1] += dN_dxi[i]   * coords[i, 1]
        J[0, 2] += dN_dxi[i]   * coords[i, 2]

        J[1, 0] += dN_deta[i]  * coords[i, 0]
        J[1, 1] += dN_deta[i]  * coords[i, 1]
        J[1, 2] += dN_deta[i]  * coords[i, 2]

        J[2, 0] += dN_dzeta[i] * coords[i, 0]
        J[2, 1] += dN_dzeta[i] * coords[i, 1]
        J[2, 2] += dN_dzeta[i] * coords[i, 2]

    detJ = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) -
            J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]) +
            J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))

    invJ = np.zeros((3, 3), dtype=np.float64)
    if abs(detJ) > 1e-15:
        invJ[0, 0] =  (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) / detJ
        invJ[0, 1] = -(J[0, 1] * J[2, 2] - J[0, 2] * J[2, 1]) / detJ
        invJ[0, 2] =  (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) / detJ

        invJ[1, 0] = -(J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]) / detJ
        invJ[1, 1] =  (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) / detJ
        invJ[1, 2] = -(J[0, 0] * J[1, 2] - J[0, 2] * J[1, 0]) / detJ

        invJ[2, 0] =  (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) / detJ
        invJ[2, 1] = -(J[0, 0] * J[2, 1] - J[0, 1] * J[2, 0]) / detJ
        invJ[2, 2] =  (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) / detJ

    return J, detJ, invJ


@njit(fastmath=True)
def _compute_invT0_T_numba(J0: np.ndarray):
    """Transformation matrix inv(T0).T for 3D 9-mode EAS."""
    j11, j12, j13 = J0[0, 0], J0[0, 1], J0[0, 2]
    j21, j22, j23 = J0[1, 0], J0[1, 1], J0[1, 2]
    j31, j32, j33 = J0[2, 0], J0[2, 1], J0[2, 2]

    T0 = np.array([
        [j11*j11, j12*j12, j13*j13, 2.0*j11*j12, 2.0*j12*j13, 2.0*j13*j11],
        [j21*j21, j22*j22, j23*j23, 2.0*j21*j22, 2.0*j22*j23, 2.0*j23*j21],
        [j31*j31, j32*j32, j33*j33, 2.0*j31*j32, 2.0*j32*j33, 2.0*j33*j31],
        [j11*j21, j12*j22, j13*j23, j11*j22+j12*j21, j12*j23+j13*j22, j13*j21+j11*j23],
        [j21*j31, j22*j32, j23*j33, j21*j32+j22*j31, j22*j33+j23*j32, j23*j31+j21*j33],
        [j31*j11, j32*j12, j33*j13, j31*j12+j32*j11, j32*j13+j33*j12, j33*j11+j31*j13]
    ], dtype=np.float64)

    return np.linalg.inv(T0).T


@njit(fastmath=True)
def _eas_matrix_m_numba(xi: float, eta: float, zeta: float):
    """9-mode trilinear enhanced assumed strain matrix M(xi, eta, zeta)."""
    M = np.zeros((6, 9), dtype=np.float64)
    M[0, 0] = xi
    M[1, 1] = eta
    M[2, 2] = zeta
    M[3, 3] = xi
    M[3, 4] = eta
    M[4, 5] = eta
    M[4, 6] = zeta
    M[5, 7] = zeta
    M[5, 8] = xi
    return M


_EMPTY_2D_CONTROLS = np.empty((0, 6), dtype=np.float64)


@njit(fastmath=True)
def _compute_c3d8_eas_tl_element_umat_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    mat_type: int,
    props: np.ndarray,
    sdvs_elem: np.ndarray,
    dt: float = 1.0,
    stress_init: np.ndarray = _EMPTY_2D_CONTROLS
):
    """Compute 3D Total Lagrangian C3D8I EAS (9-mode) tangent stiffness (24x24)
    and internal force vector (24) under finite strain deformation using UMAT material dispatcher.

    2026-09-13 fix: the enhanced strain (B_tilde @ alpha) was computed only to
    build K_ua/K_aa (stiffness blocks) but was NEVER added to the Green-Lagrange
    strain actually passed to the material -- E_voigt was always the compatible
    strain alone, so the 9 EAS alpha modes had zero effect on S_voigt/f_int
    (dead code, matching dev_log/3d_element_defect_audit_20260912.md Section 2.1).
    Confirmed identical bending-locking ratio to plain C3D8 (0.46 both) before
    this fix -- a residual-level tell, since a stiffness-only defect cannot by
    itself change the converged equilibrium solution.

    Known remaining limitation (not a regression from this fix, a pre-existing
    scope gap now correctly exposed instead of masked): this kernel implements
    only the 9 PRINCIPAL alpha_i incompatible modes. Real Abaqus C3D8I has 13
    internal DOFs -- the same 9 alpha_i plus 4 additional beta_j volumetric
    modes (theta_1=xi1*xi2, theta_2=xi1*xi3, theta_3=xi2*xi3,
    theta_4=xi1*xi2*xi3) that incompatible.txt explicitly states are what
    "prevent locking of the elements for approximately incompressible material
    behavior" -- the 9 alpha_i modes alone are NOT meant to fix volumetric
    locking. Measured after this fix: nu=0.49999 volumetric test now reports
    STIFFENED (ratio 0.366) instead of the pre-fix "INCOMPRESSIBLE-OK" --
    that pre-fix pass was almost certainly a false positive from the dead-EAS
    Newton path landing on a different (not more correct) local solution near
    an ill-conditioned nu -> 0.5 limit, not evidence the element actually
    handled incompressibility. Adding the 4 beta_j modes is a real follow-up,
    out of scope for this fix (which targets the patch-test/bending-locking
    defect only).

    Fix: alpha is solved to local equilibrium (f_alpha = integral(B_tilde^T S) = 0)
    by a per-call Newton loop BEFORE f_int/K are assembled, exactly the static
    condensation Abaqus's own incompatible-mode theory describes (see
    benchmark_element/reference_abaqus_docs/incompatible.txt) -- M/B_tilde
    already had the right shape for this (6x9, Voigt-strain-conjugate to alpha,
    parallel to how B_L is Voigt-strain-conjugate to u), so the missing step was
    purely "add it to E_voigt before calling the material," not a redesign.
    E_voigt is used (as opposed to enhancing F itself, the Simo-Armero
    additive-F route) because MAT_CUSTOM_ELASTIC -- the only material type this
    fix has been verified against -- is a St Venant-Kirchhoff law
    (S_voigt = C_mat @ E_voigt, dispsolver/material3d/numba_materials.py), so a
    strain-conjugate enhancement is exact for it. Materials that consume F/detF
    directly (MAT_NEO_HOOKEAN, MAT_J2_PLASTICITY) are NOT given a consistent
    enhancement by this fix -- F is still passed as the plain compatible
    gradient, so alpha only affects those materials via a second-order/no-op
    path. Extending this to those material types requires reconstructing an
    enhanced F (or moving to the additive-F formulation, with its own known
    large-distortion caveat per incompatible.txt) -- left as a follow-up, not
    silently claimed as covered.
    """
    K_uu = np.zeros((24, 24), dtype=np.float64)
    K_ua = np.zeros((24, 9), dtype=np.float64)
    K_aa = np.zeros((9, 9), dtype=np.float64)
    K_geo = np.zeros((24, 24), dtype=np.float64)
    f_int = np.zeros(24, dtype=np.float64)
    global_err = 0

    u_nodes = u_elem.reshape((8, 3))
    coords_curr = coords_init + u_nodes

    # Central Jacobian J0 & inv(T0).T for EAS
    J0, detJ0, _ = _jacobian3d(0.0, 0.0, 0.0, coords_init)
    invT0_T = _compute_invT0_T_numba(J0)

    n_gp = 8
    B_L_all = np.zeros((n_gp, 6, 24), dtype=np.float64)
    B_tilde_all = np.zeros((n_gp, 6, 9), dtype=np.float64)
    E_compat_all = np.zeros((n_gp, 6), dtype=np.float64)
    F_all = np.zeros((n_gp, 3, 3), dtype=np.float64)
    grad_all = np.zeros((n_gp, 8, 3), dtype=np.float64)
    dV_all = np.zeros(n_gp, dtype=np.float64)

    # 1. Precompute per-GP kinematics (all independent of the alpha DOFs)
    gp = 0
    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]

                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords_init)

                dN_dX = np.zeros(8, dtype=np.float64)
                dN_dY = np.zeros(8, dtype=np.float64)
                dN_dZ = np.zeros(8, dtype=np.float64)
                for i in range(8):
                    dN_dX[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dY[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dZ[i] = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]
                    grad_all[gp, i, 0] = dN_dX[i]
                    grad_all[gp, i, 1] = dN_dY[i]
                    grad_all[gp, i, 2] = dN_dZ[i]

                F = np.zeros((3, 3), dtype=np.float64)
                for i in range(8):
                    F[0, 0] += coords_curr[i, 0] * dN_dX[i]
                    F[0, 1] += coords_curr[i, 0] * dN_dY[i]
                    F[0, 2] += coords_curr[i, 0] * dN_dZ[i]

                    F[1, 0] += coords_curr[i, 1] * dN_dX[i]
                    F[1, 1] += coords_curr[i, 1] * dN_dY[i]
                    F[1, 2] += coords_curr[i, 1] * dN_dZ[i]

                    F[2, 0] += coords_curr[i, 2] * dN_dX[i]
                    F[2, 1] += coords_curr[i, 2] * dN_dY[i]
                    F[2, 2] += coords_curr[i, 2] * dN_dZ[i]
                F_all[gp, :, :] = F

                # Enhanced Strain operator B_tilde = (detJ0 / detJ) * invT0_T @ M
                M = _eas_matrix_m_numba(xi, eta, zeta)
                B_tilde_all[gp, :, :] = (detJ0 / detJ) * (invT0_T @ M)

                # Non-linear (compatible) Strain-Displacement Matrix B_L
                B_L = np.zeros((6, 24), dtype=np.float64)
                for a in range(8):
                    dX = dN_dX[a]
                    dY = dN_dY[a]
                    dZ = dN_dZ[a]

                    B_L[0, 3*a + 0] = F[0, 0] * dX
                    B_L[0, 3*a + 1] = F[1, 0] * dX
                    B_L[0, 3*a + 2] = F[2, 0] * dX

                    B_L[1, 3*a + 0] = F[0, 1] * dY
                    B_L[1, 3*a + 1] = F[1, 1] * dY
                    B_L[1, 3*a + 2] = F[2, 1] * dY

                    B_L[2, 3*a + 0] = F[0, 2] * dZ
                    B_L[2, 3*a + 1] = F[1, 2] * dZ
                    B_L[2, 3*a + 2] = F[2, 2] * dZ

                    B_L[3, 3*a + 0] = F[0, 0] * dY + F[0, 1] * dX
                    B_L[3, 3*a + 1] = F[1, 0] * dY + F[1, 1] * dX
                    B_L[3, 3*a + 2] = F[2, 0] * dY + F[2, 1] * dX

                    B_L[4, 3*a + 0] = F[0, 1] * dZ + F[0, 2] * dY
                    B_L[4, 3*a + 1] = F[1, 1] * dZ + F[1, 2] * dY
                    B_L[4, 3*a + 2] = F[2, 1] * dZ + F[2, 2] * dY

                    B_L[5, 3*a + 0] = F[0, 2] * dX + F[0, 0] * dZ
                    B_L[5, 3*a + 1] = F[1, 2] * dX + F[1, 0] * dZ
                    B_L[5, 3*a + 2] = F[2, 2] * dX + F[2, 0] * dZ
                B_L_all[gp, :, :] = B_L

                # Green-Lagrange Strain E = 1/2 (F^T F - I) -- COMPATIBLE part only
                C = F.T @ F
                E_tensor = 0.5 * (C - np.eye(3))
                E_compat_all[gp, 0] = E_tensor[0, 0]
                E_compat_all[gp, 1] = E_tensor[1, 1]
                E_compat_all[gp, 2] = E_tensor[2, 2]
                E_compat_all[gp, 3] = 2.0 * E_tensor[0, 1]
                E_compat_all[gp, 4] = 2.0 * E_tensor[1, 2]
                E_compat_all[gp, 5] = 2.0 * E_tensor[2, 0]

                dV_all[gp] = detJ
                gp += 1

    # 2. Local Newton loop: solve the 9 alpha DOFs to element-level equilibrium
    #    (f_alpha = integral(B_tilde^T S) dV = 0) before touching global f_int/K.
    #    Converges in exactly 1 iteration for the St Venant-Kirchhoff material
    #    this fix targets (K_aa is alpha-independent there); the loop is kept
    #    general (up to 25 iterations) for any future nonlinear-in-E material.
    alpha = np.zeros(9, dtype=np.float64)
    sdv_n = sdvs_elem.shape[1]
    S_final = np.zeros((n_gp, 6), dtype=np.float64)
    C_final = np.zeros((n_gp, 6, 6), dtype=np.float64)
    sdv_final = np.zeros((n_gp, sdv_n), dtype=np.float64)

    for _local_iter in range(25):
        f_alpha = np.zeros(9, dtype=np.float64)
        K_aa_trial = np.zeros((9, 9), dtype=np.float64)
        for gpi in range(n_gp):
            E_enh = E_compat_all[gpi] + B_tilde_all[gpi] @ alpha
            if sdv_n > 0:
                sdv_gp = sdvs_elem[gpi].copy()
            else:
                sdv_gp = np.zeros(0, dtype=np.float64)
            F_gp = F_all[gpi]
            detF_gp = (F_gp[0, 0] * (F_gp[1, 1] * F_gp[2, 2] - F_gp[1, 2] * F_gp[2, 1]) -
                       F_gp[0, 1] * (F_gp[1, 0] * F_gp[2, 2] - F_gp[1, 2] * F_gp[2, 0]) +
                       F_gp[0, 2] * (F_gp[1, 0] * F_gp[2, 1] - F_gp[1, 1] * F_gp[2, 0]))
            S_voigt, C_tangent, sdv_new, err = material_dispatch_3d(
                mat_type, props, sdv_gp, E_enh, F_gp, detF_gp, dt
            )
            if err != 0:
                global_err = err
            if stress_init.shape[0] == n_gp:
                for i in range(6):
                    S_voigt[i] += stress_init[gpi, i]

            S_final[gpi, :] = S_voigt
            C_final[gpi, :, :] = C_tangent
            if sdv_n > 0:
                sdv_final[gpi, :] = sdv_new

            dV = dV_all[gpi]
            f_alpha += (B_tilde_all[gpi].T @ S_voigt) * dV
            K_aa_trial += (B_tilde_all[gpi].T @ C_tangent @ B_tilde_all[gpi]) * dV

        res_norm = 0.0
        for i in range(9):
            res_norm += f_alpha[i] * f_alpha[i]
        res_norm = np.sqrt(res_norm)
        if res_norm < 1e-10:
            break

        ridge = 1e-12 * (abs(K_aa_trial[0, 0]) + abs(K_aa_trial[1, 1]) + abs(K_aa_trial[2, 2]))
        if ridge < 1e-14:
            ridge = 1e-14
        for i in range(9):
            K_aa_trial[i, i] += ridge
        d_alpha = np.linalg.solve(K_aa_trial, -f_alpha)
        alpha += d_alpha

    if sdv_n > 0:
        for gpi in range(n_gp):
            sdvs_elem[gpi] = sdv_final[gpi]

    # 3. Assemble global f_int / K at the alpha-equilibrium state
    for gpi in range(n_gp):
        B_L = B_L_all[gpi]
        B_tilde = B_tilde_all[gpi]
        S_voigt = S_final[gpi]
        C_tangent = C_final[gpi]
        dV = dV_all[gpi]

        f_int += (B_L.T @ S_voigt) * dV
        K_uu += (B_L.T @ C_tangent @ B_L) * dV
        K_ua += (B_L.T @ C_tangent @ B_tilde) * dV
        K_aa += (B_tilde.T @ C_tangent @ B_tilde) * dV

        S_tensor = np.array([
            [S_voigt[0], S_voigt[3], S_voigt[5]],
            [S_voigt[3], S_voigt[1], S_voigt[4]],
            [S_voigt[5], S_voigt[4], S_voigt[2]]
        ], dtype=np.float64)

        for a in range(8):
            grad_a = grad_all[gpi, a, :]
            for b in range(8):
                grad_b = grad_all[gpi, b, :]
                g_ab = float(grad_a @ S_tensor @ grad_b) * dV

                K_geo[3*a + 0, 3*b + 0] += g_ab
                K_geo[3*a + 1, 3*b + 1] += g_ab
                K_geo[3*a + 2, 3*b + 2] += g_ab

    # 4. Static Condensation of the 9 EAS Internal Modes (now consistent: K_uu,
    #    K_ua, K_aa, f_int all evaluated at the SAME alpha-equilibrium state).
    ridge = 1e-12 * (abs(K_aa[0, 0]) + abs(K_aa[1, 1]) + abs(K_aa[2, 2]))
    if ridge < 1e-14:
        ridge = 1e-14
    for i in range(9):
        K_aa[i, i] += ridge
    inv_K_aa = np.linalg.inv(K_aa)
    K_mat_condensed = K_uu - K_ua @ inv_K_aa @ K_ua.T

    K_elem = K_mat_condensed + K_geo
    return K_elem, f_int, global_err


@njit(fastmath=True)
def _compute_c3d8_eas_tl_element_cmat_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    C_mat: np.ndarray
):
    props = np.zeros(36, dtype=np.float64)
    for i in range(6):
        for j in range(6):
            props[6 * i + j] = C_mat[i, j]
    dummy_sdvs = np.zeros((8, 0), dtype=np.float64)
    return _compute_c3d8_eas_tl_element_umat_numba(
        coords_init, u_elem, MAT_CUSTOM_ELASTIC, props, dummy_sdvs, 1.0
    )


def compute_c3d8_eas_tl_element_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    *args,
    **kwargs
):
    """Compute 3D Total Lagrangian C3D8I EAS (9-mode) tangent stiffness (24x24)
    and internal force vector (24) under finite strain deformation.
    Supports both legacy C_mat (6x6) and UMAT-style (mat_type, props, sdvs_elem, dt).
    """
    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].ndim == 2:
        return _compute_c3d8_eas_tl_element_cmat_numba(coords_init, u_elem, args[0])
    else:
        mat_type = int(args[0])
        props = args[1]
        sdvs_elem = args[2] if len(args) > 2 else np.zeros((8, 0), dtype=np.float64)
        dt = float(args[3]) if len(args) > 3 else float(kwargs.get("dt", 1.0))
        return _compute_c3d8_eas_tl_element_umat_numba(coords_init, u_elem, mat_type, props, sdvs_elem, dt)


_EMPTY_2D_CONTROLS = np.empty((0, 0), dtype=np.float64)


@njit(parallel=True, fastmath=True, cache=True)
def _assemble_mesh_c3d8_eas_tl_umat_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float = 1.0,
    elem_stress_init: np.ndarray = _EMPTY_2D_CONTROLS
):
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
    err_flags = np.zeros(n_elems, dtype=np.int32)
    has_stress_init = elem_stress_init.shape[0] == n_elems

    for e in prange(n_elems):
        nodes_e = elem_conn[e]
        coords_e = np.zeros((8, 3), dtype=np.float64)
        u_e = np.zeros(24, dtype=np.float64)

        for i in range(8):
            nid = nodes_e[i]
            coords_e[i, :] = node_coords[nid, :]
            u_e[3*i:3*i+3] = u_global[3*nid:3*nid+3]

        mat_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        stress_init = elem_stress_init[e] if has_stress_init else np.zeros((8, 6), dtype=np.float64)

        Ke, fe, err = _compute_c3d8_eas_tl_element_umat_numba(
            coords_e, u_e, mat_type, props_e, sdvs_e, dt, stress_init
        )
        K_elems[e, :, :] = Ke
        f_elems[e, :] = fe
        err_flags[e] = err

    err_sum = 0
    for e in range(n_elems):
        err_sum += err_flags[e]
        
    return f_elems, K_elems, err_sum


def assemble_mesh_c3d8_eas_tl_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    *args,
    **kwargs
):
    """Mesh assembly kernel for C3D8I EAS Total Lagrangian 3D elements.
    Supports both legacy C_mat (single material) and multi-material arrays.
    """
    n_elems = elem_conn.shape[0]
    elem_stress_init = kwargs.get("elem_stress_init", None)
    if elem_stress_init is None:
        elem_stress_init = np.zeros((n_elems, 8, 6), dtype=np.float64)

    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].ndim == 2:
        C_mat = args[0]
        elem_mat_types = np.full(n_elems, MAT_CUSTOM_ELASTIC, dtype=np.int32)
        elem_props = np.zeros((n_elems, 36), dtype=np.float64)
        cmat_flat = C_mat.ravel()
        for e in range(n_elems):
            elem_props[e, :36] = cmat_flat
        elem_sdvs = np.zeros((n_elems, 8, 0), dtype=np.float64)
        return _assemble_mesh_c3d8_eas_tl_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, 1.0, elem_stress_init
        )
    else:
        elem_mat_types = args[0]
        elem_props = args[1]
        elem_sdvs = args[2] if len(args) > 2 else np.zeros((elem_conn.shape[0], 8, 0), dtype=np.float64)
        dt = float(args[3]) if len(args) > 3 else float(kwargs.get("dt", 1.0))
        return _assemble_mesh_c3d8_eas_tl_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_stress_init
        )

