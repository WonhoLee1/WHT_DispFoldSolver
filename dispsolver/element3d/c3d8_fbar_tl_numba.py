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


@njit(fastmath=True)
def _compute_c3d8_fbar_tl_element_umat_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    mat_type: int,
    props: np.ndarray,
    sdvs_elem: np.ndarray,
    dt: float = 1.0
):
    """Compute Multiplicative F-bar Total Lagrangian 3D C3D8 element tangent stiffness (24x24)
    and internal force vector (24) under finite deformation using UMAT material dispatcher.
    """
    K_mat = np.zeros((24, 24), dtype=np.float64)
    K_geo = np.zeros((24, 24), dtype=np.float64)
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
                if sdvs_elem.shape[1] > 0:
                    sdvs_elem[gp] = sdv_gp_new

                S_tensor = np.array([
                    [S_voigt[0], S_voigt[3], S_voigt[5]],
                    [S_voigt[3], S_voigt[1], S_voigt[4]],
                    [S_voigt[5], S_voigt[4], S_voigt[2]]
                ], dtype=np.float64)

                # Non-linear Strain-Displacement Matrix B_L (6 x 24)
                B_L = np.zeros((6, 24), dtype=np.float64)
                for a in range(8):
                    dX = dN_dX[a]
                    dY = dN_dY[a]
                    dZ = dN_dZ[a]

                    B_L[0, 3*a + 0] = F_bar[0, 0] * dX
                    B_L[0, 3*a + 1] = F_bar[1, 0] * dX
                    B_L[0, 3*a + 2] = F_bar[2, 0] * dX

                    B_L[1, 3*a + 0] = F_bar[0, 1] * dY
                    B_L[1, 3*a + 1] = F_bar[1, 1] * dY
                    B_L[1, 3*a + 2] = F_bar[2, 1] * dY

                    B_L[2, 3*a + 0] = F_bar[0, 2] * dZ
                    B_L[2, 3*a + 1] = F_bar[1, 2] * dZ
                    B_L[2, 3*a + 2] = F_bar[2, 2] * dZ

                    B_L[3, 3*a + 0] = F_bar[0, 0] * dY + F_bar[0, 1] * dX
                    B_L[3, 3*a + 1] = F_bar[1, 0] * dY + F_bar[1, 1] * dX
                    B_L[3, 3*a + 2] = F_bar[2, 0] * dY + F_bar[2, 1] * dX

                    B_L[4, 3*a + 0] = F_bar[0, 1] * dZ + F_bar[0, 2] * dY
                    B_L[4, 3*a + 1] = F_bar[1, 1] * dZ + F_bar[1, 2] * dY
                    B_L[4, 3*a + 2] = F_bar[2, 1] * dZ + F_bar[2, 2] * dY

                    B_L[5, 3*a + 0] = F_bar[0, 2] * dX + F_bar[0, 0] * dZ
                    B_L[5, 3*a + 1] = F_bar[1, 2] * dX + F_bar[1, 0] * dZ
                    B_L[5, 3*a + 2] = F_bar[2, 2] * dX + F_bar[2, 0] * dZ

                dV = detJ

                f_int += (B_L.T @ S_voigt) * dV
                K_mat += (B_L.T @ C_tangent @ B_L) * dV

                for a in range(8):
                    grad_a = np.array([dN_dX[a], dN_dY[a], dN_dZ[a]], dtype=np.float64)
                    for b in range(8):
                        grad_b = np.array([dN_dX[b], dN_dY[b], dN_dZ[b]], dtype=np.float64)
                        g_ab = float(grad_a @ S_tensor @ grad_b) * dV

                        K_geo[3*a + 0, 3*b + 0] += g_ab
                        K_geo[3*a + 1, 3*b + 1] += g_ab
                        K_geo[3*a + 2, 3*b + 2] += g_ab

    K_elem = K_mat + K_geo
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


@njit(fastmath=True)
def _assemble_mesh_c3d8_fbar_tl_umat_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float = 1.0
):
    n_elems = elem_conn.shape[0]
    has_error = 0
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)

    for e in range(n_elems):
        nodes_e = elem_conn[e]
        coords_e = np.zeros((8, 3), dtype=np.float64)
        u_e = np.zeros(24, dtype=np.float64)

        for i in range(8):
            nid = nodes_e[i]
            coords_e[i, 0] = node_coords[nid, 0]
            coords_e[i, 1] = node_coords[nid, 1]
            coords_e[i, 2] = node_coords[nid, 2]

            u_e[3*i + 0] = u_global[3*nid + 0]
            u_e[3*i + 1] = u_global[3*nid + 1]
            u_e[3*i + 2] = u_global[3*nid + 2]

        mat_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]

        Ke, fe, err = _compute_c3d8_fbar_tl_element_umat_numba(
            coords_e, u_e, mat_type, props_e, sdvs_e, dt
        )
        K_elems[e] = Ke
        f_elems[e] = fe
        if err > 0:
            has_error = err

    return f_elems, K_elems, has_error


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
    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].ndim == 2:
        C_mat = args[0]
        n_elems = elem_conn.shape[0]
    has_error = 0
        elem_mat_types = np.full(n_elems, MAT_CUSTOM_ELASTIC, dtype=np.int32)
        elem_props = np.zeros((n_elems, 36), dtype=np.float64)
        cmat_flat = C_mat.ravel()
        for e in range(n_elems):
            elem_props[e, :36] = cmat_flat
        elem_sdvs = np.zeros((n_elems, 8, 0), dtype=np.float64)
        return _assemble_mesh_c3d8_fbar_tl_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, 1.0
        )
    else:
        elem_mat_types = args[0]
        elem_props = args[1]
        elem_sdvs = args[2] if len(args) > 2 else np.zeros((elem_conn.shape[0], 8, 0), dtype=np.float64)
        dt = float(args[3]) if len(args) > 3 else float(kwargs.get("dt", 1.0))
        return _assemble_mesh_c3d8_fbar_tl_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt
        )

