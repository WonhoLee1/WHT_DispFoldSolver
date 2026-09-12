"""
c3d8_fbar_tl_numba.py
=====================
3D Multiplicative F-bar Total Lagrangian (TL) Finite Strain Hexahedral 8-node Element.

Completely eliminates Volumetric Locking (nu -> 0.5) under finite strain deformation:
- Centroidal volume ratio J0 = det(F(0,0,0))
- Gauss point volume ratio J = det(F(xi,eta,zeta))
- F-bar deformation gradient F_bar = (J0 / J)^(1/3) * F
- Green-Lagrange strain E_bar = 1/2 (F_bar^T F_bar - I)
- Second Piola-Kirchhoff Stress S = C_mat : E_bar
- Tangent Stiffness K_elem = K_mat + K_geo
"""

import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_3d, MAT_CUSTOM_ELASTIC

_GP_GAUSS = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)


@njit(fastmath=True)
def _sd3d(xi: float, eta: float, zeta: float):
    """3D Trilinear shape function derivatives w.r.t isoparametric coordinates."""
    dN_dxi = np.zeros(8, dtype=np.float64)
    dN_deta = np.zeros(8, dtype=np.float64)
    dN_dzeta = np.zeros(8, dtype=np.float64)

    # Node 1: (-1,-1,-1)
    dN_dxi[0]   = -0.125 * (1.0 - eta) * (1.0 - zeta)
    dN_deta[0]  = -0.125 * (1.0 - xi)  * (1.0 - zeta)
    dN_dzeta[0] = -0.125 * (1.0 - xi)  * (1.0 - eta)

    # Node 2: (+1,-1,-1)
    dN_dxi[1]   =  0.125 * (1.0 - eta) * (1.0 - zeta)
    dN_deta[1]  = -0.125 * (1.0 + xi)  * (1.0 - zeta)
    dN_dzeta[1] = -0.125 * (1.0 + xi)  * (1.0 - eta)

    # Node 3: (+1,+1,-1)
    dN_dxi[2]   =  0.125 * (1.0 + eta) * (1.0 - zeta)
    dN_deta[2]  =  0.125 * (1.0 + xi)  * (1.0 - zeta)
    dN_dzeta[2] = -0.125 * (1.0 + xi)  * (1.0 + eta)

    # Node 4: (-1,+1,-1)
    dN_dxi[3]   = -0.125 * (1.0 + eta) * (1.0 - zeta)
    dN_deta[3]  =  0.125 * (1.0 - xi)  * (1.0 - zeta)
    dN_dzeta[3] = -0.125 * (1.0 - xi)  * (1.0 + eta)

    # Node 5: (-1,-1,+1)
    dN_dxi[4]   = -0.125 * (1.0 - eta) * (1.0 + zeta)
    dN_deta[4]  = -0.125 * (1.0 - xi)  * (1.0 + zeta)
    dN_dzeta[4] =  0.125 * (1.0 - xi)  * (1.0 - eta)

    # Node 6: (+1,-1,+1)
    dN_dxi[5]   =  0.125 * (1.0 - eta) * (1.0 + zeta)
    dN_deta[5]  = -0.125 * (1.0 + xi)  * (1.0 + zeta)
    dN_dzeta[5] =  0.125 * (1.0 + xi)  * (1.0 - eta)

    # Node 7: (+1,+1,+1)
    dN_dxi[6]   =  0.125 * (1.0 + eta) * (1.0 + zeta)
    dN_deta[6]  =  0.125 * (1.0 + xi)  * (1.0 + zeta)
    dN_dzeta[6] =  0.125 * (1.0 + xi)  * (1.0 + eta)

    # Node 8: (-1,+1,+1)
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


_EMPTY_2D_CONTROLS = np.empty((0, 6), dtype=np.float64)


@njit(fastmath=True)
def _c3d8_fbar_tl_residual_only(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    mat_type: int,
    props: np.ndarray,
    sdvs_elem: np.ndarray,
    dt: float,
    stress_init: np.ndarray,
    commit_sdv: bool,
):
    """Internal force only (no tangent) -- the ONE place F-bar's residual is
    assembled, reused both by the real element call and by the FD tangent
    below so the two can never drift apart. `commit_sdv` controls whether
    state variables are written back to `sdvs_elem` (True only for the
    single real call the Newton solver actually keeps; False for the
    perturbed evaluations the FD tangent uses so a probe does not corrupt
    the converged element's history).
    """
    f_int = np.zeros(24, dtype=np.float64)
    global_err = 0

    u_nodes = u_elem.reshape((8, 3))
    coords_curr = coords_init + u_nodes

    # 1. Compute Centroidal Deformation Gradient F0
    dN0_dxi, dN0_deta, dN0_dzeta = _sd3d(0.0, 0.0, 0.0)
    _, _, invJ0 = _jacobian3d(0.0, 0.0, 0.0, coords_init)

    dN0_dX = np.zeros(8, dtype=np.float64)
    dN0_dY = np.zeros(8, dtype=np.float64)
    dN0_dZ = np.zeros(8, dtype=np.float64)
    for i in range(8):
        dN0_dX[i] = invJ0[0, 0] * dN0_dxi[i] + invJ0[0, 1] * dN0_deta[i] + invJ0[0, 2] * dN0_dzeta[i]
        dN0_dY[i] = invJ0[1, 0] * dN0_dxi[i] + invJ0[1, 1] * dN0_deta[i] + invJ0[1, 2] * dN0_dzeta[i]
        dN0_dZ[i] = invJ0[2, 0] * dN0_dxi[i] + invJ0[2, 1] * dN0_deta[i] + invJ0[2, 2] * dN0_dzeta[i]

    F0 = np.zeros((3, 3), dtype=np.float64)
    for i in range(8):
        F0[0, 0] += coords_curr[i, 0] * dN0_dX[i]
        F0[0, 1] += coords_curr[i, 0] * dN0_dY[i]
        F0[0, 2] += coords_curr[i, 0] * dN0_dZ[i]

        F0[1, 0] += coords_curr[i, 1] * dN0_dX[i]
        F0[1, 1] += coords_curr[i, 1] * dN0_dY[i]
        F0[1, 2] += coords_curr[i, 1] * dN0_dZ[i]

        F0[2, 0] += coords_curr[i, 2] * dN0_dX[i]
        F0[2, 1] += coords_curr[i, 2] * dN0_dY[i]
        F0[2, 2] += coords_curr[i, 2] * dN0_dZ[i]

    detF0 = (F0[0, 0] * (F0[1, 1] * F0[2, 2] - F0[1, 2] * F0[2, 1]) -
             F0[0, 1] * (F0[1, 0] * F0[2, 2] - F0[1, 2] * F0[2, 0]) +
             F0[0, 2] * (F0[1, 0] * F0[2, 1] - F0[1, 1] * F0[2, 0]))
    if detF0 < 1e-12:
        detF0 = 1e-12

    # 2. Integration Loop over 8 Gauss Points
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

                detF = (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]) -
                        F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0]) +
                        F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))
                if detF < 1e-12:
                    detF = 1e-12

                # F-bar Volumetric Split: F_bar = (detF0 / detF)^(1/3) * F
                scale_vol = (detF0 / detF) ** (1.0 / 3.0)
                F_bar = scale_vol * F

                # Green-Lagrange strain E_bar = 1/2 (F_bar^T F_bar - I)
                C_bar = F_bar.T @ F_bar
                E_tensor = 0.5 * (C_bar - np.eye(3))

                E_voigt = np.array([
                    E_tensor[0, 0],
                    E_tensor[1, 1],
                    E_tensor[2, 2],
                    2.0 * E_tensor[0, 1],
                    2.0 * E_tensor[1, 2],
                    2.0 * E_tensor[2, 0]
                ], dtype=np.float64)

                gp = 4 * gi + 2 * gj + gk
                if sdvs_elem.shape[1] > 0:
                    sdv_gp = sdvs_elem[gp]
                else:
                    sdv_gp = np.zeros(0, dtype=np.float64)

                # Call UMAT dispatcher to decouple kinematics from constitutive response
                S_voigt, C_tangent, sdv_gp_new, err = material_dispatch_3d(
                    mat_type, props, sdv_gp, E_voigt, F_bar, detF0, dt
                )
                if commit_sdv and sdvs_elem.shape[1] > 0:
                    sdvs_elem[gp] = sdv_gp_new
                if err != 0:
                    global_err = err

                if stress_init.shape[0] == 8:
                    for i in range(6):
                        S_voigt[i] += stress_init[gp, i]

                # Non-linear Strain-Displacement Matrix B_L (6 x 24).
                # Built from the REAL deformation gradient F, not F_bar --
                # F-bar modifies the deformation gradient fed to the
                # CONSTITUTIVE call only; the virtual strain used to build
                # the residual must stay that of the real kinematics
                # (de Souza Neto et al. 1996). This makes f_int the correct
                # F-bar residual (root-verified: reduces exactly to a plain
                # compatible C3D8 residual when F_bar=F, i.e. at any
                # homogeneous/affine state -- so the exact patch-test
                # solution is an exact root of this residual). The TANGENT
                # consistent with THIS f_int is NOT B_L^T @ C_tangent @ B_L
                # (that ignores both K_geo and the fact that F_bar depends
                # on ALL 8 nodes through detF0/detF, not just the local GP)
                # -- see _compute_c3d8_fbar_tl_element_umat_numba below,
                # which builds the tangent as a finite difference of THIS
                # SAME residual instead of an incomplete analytical form
                # (measured 20% wrong against its own FD Jacobian before
                # this fix, dev_log/solve_step_false_convergence_20260913.md).
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

                dV = detJ

                f_int += (B_L.T @ S_voigt) * dV

    return f_int, global_err


@njit(fastmath=True)
def _compute_c3d8_fbar_tl_element_umat_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    mat_type: int,
    props: np.ndarray,
    sdvs_elem: np.ndarray,
    dt: float = 1.0,
    stress_init: np.ndarray = _EMPTY_2D_CONTROLS
):
    """Compute Multiplicative F-bar Total Lagrangian 3D C3D8 element tangent
    stiffness (24x24) and internal force vector (24) under finite
    deformation using UMAT material dispatcher.

    The tangent is a central finite difference of the element's own
    residual (`_c3d8_fbar_tl_residual_only`), not a hand-derived analytical
    form. This is a deliberate choice, not a stopgap: F_bar depends on
    every node's displacement through the element-averaged detF0/detF
    ratio (not just the local Gauss point's), so the exact analytical
    tangent has a nonlocal cross-node term that a naive
    B_L^T @ C_tangent @ B_L (plus the usual local geometric-stiffness term)
    omits entirely -- measured 20% error against this same FD check before
    this fix (dev_log/solve_step_false_convergence_20260913.md), matching
    dev_log/3d_element_defect_audit_20260912.md 2.5's original "F-bar
    omits its volumetric-consistency tangent term" finding. An FD tangent
    is exact-by-construction against whatever residual is coded (it cannot
    reintroduce this class of defect), at the cost of 48 extra element
    residual evaluations per Newton iteration -- acceptable here since
    C3D8_FBAR is used specifically for expensive near-incompressible
    problems, not a hot path.

    Per-column adaptive step h_j = sqrt(eps)*max(|u_j|, L_elem) (Dennis &
    Schnabel 1983 Sec 5.4), same convention this project's B4 fix already
    established for other Numba kernels (AGENTS.md 4.15) -- a fixed
    absolute step is a latent mesh-refinement bug (truncation error scales
    as h/L_elem, doubling on every uniform refinement).
    """
    f_int, global_err = _c3d8_fbar_tl_residual_only(
        coords_init, u_elem, mat_type, props, sdvs_elem, dt, stress_init, True
    )

    # Characteristic element length, for the adaptive FD step -- mean
    # distance from the element centroid to its 8 corners.
    centroid = np.zeros(3, dtype=np.float64)
    for i in range(8):
        centroid += coords_init[i]
    centroid /= 8.0
    L_elem = 0.0
    for i in range(8):
        d = coords_init[i] - centroid
        L_elem += np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    L_elem /= 8.0

    sqrt_eps = 1.4901161193847656e-08  # sqrt(machine epsilon), float64
    K_elem = np.zeros((24, 24), dtype=np.float64)
    for j in range(24):
        h = sqrt_eps * max(abs(u_elem[j]), L_elem)
        u_p = u_elem.copy()
        u_p[j] += h
        u_m = u_elem.copy()
        u_m[j] -= h
        f_p, _ = _c3d8_fbar_tl_residual_only(
            coords_init, u_p, mat_type, props, sdvs_elem, dt, stress_init, False
        )
        f_m, _ = _c3d8_fbar_tl_residual_only(
            coords_init, u_m, mat_type, props, sdvs_elem, dt, stress_init, False
        )
        K_elem[:, j] = (f_p - f_m) / (2.0 * h)

    return K_elem, f_int, global_err


@njit(fastmath=True)
def _compute_c3d8_fbar_tl_element_cmat_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    C_mat: np.ndarray
):
    props = np.zeros(36, dtype=np.float64)
    for i in range(6):
        for j in range(6):
            props[6 * i + j] = C_mat[i, j]
    dummy_sdvs = np.zeros((8, 0), dtype=np.float64)
    return _compute_c3d8_fbar_tl_element_umat_numba(
        coords_init, u_elem, MAT_CUSTOM_ELASTIC, props, dummy_sdvs, 1.0
    )


def compute_c3d8_fbar_tl_element_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    *args,
    **kwargs
):
    """Compute Multiplicative F-bar Total Lagrangian 3D C3D8 element tangent stiffness (24x24)
    and internal force vector (24) under finite deformation.
    Supports both legacy C_mat (6x6) and UMAT-style (mat_type, props, sdvs_elem, dt).
    """
    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].ndim == 2:
        return _compute_c3d8_fbar_tl_element_cmat_numba(coords_init, u_elem, args[0])
    else:
        mat_type = int(args[0])
        props = args[1]
        sdvs_elem = args[2] if len(args) > 2 else np.zeros((8, 0), dtype=np.float64)
        dt = float(args[3]) if len(args) > 3 else float(kwargs.get("dt", 1.0))
        return _compute_c3d8_fbar_tl_element_umat_numba(coords_init, u_elem, mat_type, props, sdvs_elem, dt)


_EMPTY_2D_CONTROLS = np.empty((0, 0), dtype=np.float64)

@njit(parallel=True, fastmath=True, cache=True)
def _assemble_mesh_c3d8_fbar_tl_umat_numba(
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

        Ke, fe, err = _compute_c3d8_fbar_tl_element_umat_numba(
            coords_e, u_e, mat_type, props_e, sdvs_e, dt, stress_init
        )
        K_elems[e, :, :] = Ke
        f_elems[e, :] = fe
        err_flags[e] = err

    err_sum = 0
    for e in range(n_elems):
        err_sum += err_flags[e]
        
    return f_elems, K_elems, err_sum


def assemble_mesh_c3d8_fbar_tl_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    *args,
    **kwargs
):
    """Mesh assembly kernel for Multiplicative F-bar Total Lagrangian 3D elements.
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
        return _assemble_mesh_c3d8_fbar_tl_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, 1.0, elem_stress_init
        )
    else:
        elem_mat_types = args[0]
        elem_props = args[1]
        elem_sdvs = args[2] if len(args) > 2 else np.zeros((elem_conn.shape[0], 8, 0), dtype=np.float64)
        dt = float(args[3]) if len(args) > 3 else float(kwargs.get("dt", 1.0))
        return _assemble_mesh_c3d8_fbar_tl_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_stress_init
        )

