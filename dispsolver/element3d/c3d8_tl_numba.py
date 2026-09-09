"""
c3d8_tl_numba.py
================
3D Total Lagrangian (TL) Finite Strain Hexahedral 8-node Element.

Exact Geometrically Non-linear (NLGEOM) Formulation:
- Deformation Gradient F = I + grad_X(u)
- Green-Lagrange Strain E = 1/2 (F^T F - I)
- Second Piola-Kirchhoff Stress S = C_mat : E
- Non-linear Strain-Displacement Matrix B_L(F)
- Full Tangent Stiffness K_elem = K_mat + K_geo
"""

import numpy as np
from numba import njit, prange

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
def compute_c3d8_tl_element_numba(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    C_mat: np.ndarray
):
    """Compute Total Lagrangian 3D C3D8 element tangent stiffness K_elem (24x24)
    and internal force vector f_int (24) under finite deformation.

    Parameters
    ----------
    coords_init : (8, 3) initial reference nodal coordinates X
    u_elem      : (24,) element displacement vector u
    C_mat       : (6, 6) material elasticity matrix

    Returns
    -------
    K_elem : (24, 24) total tangent stiffness matrix K = K_mat + K_geo
    f_int  : (24,) internal force vector
    """
    K_mat = np.zeros((24, 24), dtype=np.float64)
    K_geo = np.zeros((24, 24), dtype=np.float64)
    f_int = np.zeros(24, dtype=np.float64)

    # Reshape u into (8, 3) for node displacements
    u_nodes = u_elem.reshape((8, 3))
    coords_curr = coords_init + u_nodes

    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]

                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords_init)

                # Reference shape function derivatives dN/dX
                dN_dX = np.zeros(8, dtype=np.float64)
                dN_dY = np.zeros(8, dtype=np.float64)
                dN_dZ = np.zeros(8, dtype=np.float64)

                for i in range(8):
                    dN_dX[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dY[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dZ[i] = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]

                # Deformation Gradient F = d(x)/d(X)
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

                # Green-Lagrange Strain E = 1/2 (F^T F - I)
                C = F.T @ F
                E_tensor = 0.5 * (C - np.eye(3))

                E_voigt = np.array([
                    E_tensor[0, 0],
                    E_tensor[1, 1],
                    E_tensor[2, 2],
                    2.0 * E_tensor[0, 1],
                    2.0 * E_tensor[1, 2],
                    2.0 * E_tensor[2, 0]
                ], dtype=np.float64)

                # PK2 Stress S = C_mat : E
                S_voigt = C_mat @ E_voigt
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

                    # row 0: E_xx
                    B_L[0, 3*a + 0] = F[0, 0] * dX
                    B_L[0, 3*a + 1] = F[1, 0] * dX
                    B_L[0, 3*a + 2] = F[2, 0] * dX

                    # row 1: E_yy
                    B_L[1, 3*a + 0] = F[0, 1] * dY
                    B_L[1, 3*a + 1] = F[1, 1] * dY
                    B_L[1, 3*a + 2] = F[2, 1] * dY

                    # row 2: E_zz
                    B_L[2, 3*a + 0] = F[0, 2] * dZ
                    B_L[2, 3*a + 1] = F[1, 2] * dZ
                    B_L[2, 3*a + 2] = F[2, 2] * dZ

                    # row 3: 2 E_xy
                    B_L[3, 3*a + 0] = F[0, 0] * dY + F[0, 1] * dX
                    B_L[3, 3*a + 1] = F[1, 0] * dY + F[1, 1] * dX
                    B_L[3, 3*a + 2] = F[2, 0] * dY + F[2, 1] * dX

                    # row 4: 2 E_yz
                    B_L[4, 3*a + 0] = F[0, 1] * dZ + F[0, 2] * dY
                    B_L[4, 3*a + 1] = F[1, 1] * dZ + F[1, 2] * dY
                    B_L[4, 3*a + 2] = F[2, 1] * dZ + F[2, 2] * dY

                    # row 5: 2 E_zx
                    B_L[5, 3*a + 0] = F[0, 2] * dX + F[0, 0] * dZ
                    B_L[5, 3*a + 1] = F[1, 2] * dX + F[1, 0] * dZ
                    B_L[5, 3*a + 2] = F[2, 2] * dX + F[2, 0] * dZ

                dV = detJ  # Gauss weight = 1.0

                f_int += (B_L.T @ S_voigt) * dV
                K_mat += (B_L.T @ C_mat @ B_L) * dV

                # Geometric Stiffness K_geo
                for a in range(8):
                    grad_a = np.array([dN_dX[a], dN_dY[a], dN_dZ[a]], dtype=np.float64)
                    for b in range(8):
                        grad_b = np.array([dN_dX[b], dN_dY[b], dN_dZ[b]], dtype=np.float64)
                        g_ab = float(grad_a @ S_tensor @ grad_b) * dV
                        
                        K_geo[3*a + 0, 3*b + 0] += g_ab
                        K_geo[3*a + 1, 3*b + 1] += g_ab
                        K_geo[3*a + 2, 3*b + 2] += g_ab

    K_elem = K_mat + K_geo
    return K_elem, f_int


@njit(parallel=True, fastmath=True, nogil=True)
def assemble_mesh_c3d8_tl_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    C_mat: np.ndarray
):
    """Parallel OpenMP mesh assembly kernel for Total Lagrangian 3D C3D8 elements.

    Parameters
    ----------
    node_coords : (n_nodes, 3)
    elem_conn   : (n_elems, 8)
    u_global    : (3 * n_nodes,)
    C_mat       : (6, 6)

    Returns
    -------
    f_elems : (n_elems, 24)
    K_elems : (n_elems, 24, 24)
    """
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)

    for e in prange(n_elems):
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

        Ke, fe = compute_c3d8_tl_element_numba(coords_e, u_e, C_mat)
        K_elems[e] = Ke
        f_elems[e] = fe

    return f_elems, K_elems
