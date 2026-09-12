"""
c3d4_numba.py
=============
Numba Kernel for Standard 3D 4-Node Linear Tetrahedron (C3D4) Elements.

Note: Bonet & Burton (1998) Average Nodal Pressure (ANP) requires a 2-pass
global mesh-wide nodal volume averaging scheme (Va = sum 1/4 V^(e), theta_a = va / Va).
On an isolated single element, theta_a == J and ANP collapses identically to standard C3D4.
This kernel is an honest 1-point linear tetrahedron element (C3D4).
"""

from __future__ import annotations
import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(fastmath=True)
    def compute_c3d4_element_numba(coords: np.ndarray, u_elem: np.ndarray, C_mat: np.ndarray):
        """Compute 3D 4-node Tet4 element stiffness (12x12) and internal force vector (12)."""
        # Natural derivatives dN / d(xi, eta, zeta) for Tet4 (3, 4)
        dN_dxi = np.array([
            [-1.0,  1.0,  0.0,  0.0],
            [-1.0,  0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0,  1.0]
        ], dtype=np.float64)

        # 3x3 Jacobian matrix
        J = dN_dxi @ coords
        detJ = (
            J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) -
            J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]) +
            J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
        )

        invJ = np.zeros((3, 3), dtype=np.float64)
        invJ[0, 0] = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) / detJ
        invJ[0, 1] = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) / detJ
        invJ[0, 2] = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) / detJ

        invJ[1, 0] = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) / detJ
        invJ[1, 1] = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) / detJ
        invJ[1, 2] = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) / detJ

        invJ[2, 0] = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) / detJ
        invJ[2, 1] = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) / detJ
        invJ[2, 2] = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) / detJ

        # Physical derivatives dN / dX = invJ @ dN_dxi (3, 4)
        dN_dX = invJ @ dN_dxi

        # Construct 6 x 12 Strain-Displacement B matrix
        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            B[0, 3*i + 0] = dN_dX[0, i]
            B[1, 3*i + 1] = dN_dX[1, i]
            B[2, 3*i + 2] = dN_dX[2, i]

            B[3, 3*i + 0] = dN_dX[1, i]
            B[3, 3*i + 1] = dN_dX[0, i]

            B[4, 3*i + 1] = dN_dX[2, i]
            B[4, 3*i + 2] = dN_dX[1, i]

            B[5, 3*i + 0] = dN_dX[2, i]
            B[5, 3*i + 2] = dN_dX[0, i]

        strain = B @ u_elem
        stress = C_mat @ strain

        # Tet4 Volume V = detJ / 6.0
        Ve = abs(detJ) / 6.0

        f_int = (B.T @ stress) * Ve
        K_elem = (B.T @ C_mat @ B) * Ve

        return K_elem, f_int

    @njit(parallel=True, fastmath=True, nogil=True)
    def assemble_mesh_c3d4_numba(
        node_coords: np.ndarray,
        elem_conn: np.ndarray,
        u_global: np.ndarray,
        C_mat: np.ndarray
    ):
        """Parallel OpenMP mesh assembly kernel for C3D4 Tet4 elements."""
        n_elems = elem_conn.shape[0]
        f_elems = np.zeros((n_elems, 12), dtype=np.float64)
        K_elems = np.zeros((n_elems, 12, 12), dtype=np.float64)

        for e in prange(n_elems):
            nodes_e = elem_conn[e]
            coords_e = np.zeros((4, 3), dtype=np.float64)
            u_e = np.zeros(12, dtype=np.float64)

            for i in range(4):
                nid = nodes_e[i]
                coords_e[i, 0] = node_coords[nid, 0]
                coords_e[i, 1] = node_coords[nid, 1]
                coords_e[i, 2] = node_coords[nid, 2]

                u_e[3*i + 0] = u_global[3*nid + 0]
                u_e[3*i + 1] = u_global[3*nid + 1]
                u_e[3*i + 2] = u_global[3*nid + 2]

            Ke, fe = compute_c3d4_element_numba(coords_e, u_e, C_mat)
            K_elems[e] = Ke
            f_elems[e] = fe

        return f_elems, K_elems

    # Legacy aliases
    compute_c3d4_anp_element_numba = compute_c3d4_element_numba
    assemble_mesh_c3d4_anp_numba = assemble_mesh_c3d4_numba
