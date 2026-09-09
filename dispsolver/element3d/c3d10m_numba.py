"""
c3d10m_numba.py
===============
Numba OpenMP C-Extension Kernel for 3D 10-Node Quadratic Tetrahedron (C3D10M) Elements.

Provides high-order quadratic shape function interpolation for complex CAD geometries and contact surfaces,
executed in a fast parallel JIT kernel.
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
    # 4-point Gauss Quadrature natural coordinates & weights for 3D Tet10
    _A_TET10 = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0  # ~ 0.58541020
    _B_TET10 = (5.0 - np.sqrt(5.0)) / 20.0        # ~ 0.13819660

    _GAUSS_POINTS_NATURAL_TET10 = np.array([
        [_A_TET10, _B_TET10, _B_TET10],
        [_B_TET10, _A_TET10, _B_TET10],
        [_B_TET10, _B_TET10, _A_TET10],
        [_B_TET10, _B_TET10, _B_TET10]
    ], dtype=np.float64)

    _GAUSS_WEIGHTS_TET10 = np.array([1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0], dtype=np.float64)

    @njit(fastmath=True)
    def _sd3d_tet10(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Evaluate 10-node natural shape derivatives dN_i / d(xi, eta, zeta) (3, 10)."""
        x, y, z = xi, eta, zeta
        w = 1.0 - x - y - z

        dN = np.zeros((3, 10), dtype=np.float64)

        # dN / dxi
        dN[0, 0] = 1.0 - 4.0 * w
        dN[0, 1] = 4.0 * x - 1.0
        dN[0, 2] = 0.0
        dN[0, 3] = 0.0
        dN[0, 4] = 4.0 * (w - x)
        dN[0, 5] = 4.0 * y
        dN[0, 6] = -4.0 * y
        dN[0, 7] = -4.0 * z
        dN[0, 8] = 4.0 * z
        dN[0, 9] = 0.0

        # dN / deta
        dN[1, 0] = 1.0 - 4.0 * w
        dN[1, 1] = 0.0
        dN[1, 2] = 4.0 * y - 1.0
        dN[1, 3] = 0.0
        dN[1, 4] = -4.0 * x
        dN[1, 5] = 4.0 * x
        dN[1, 6] = 4.0 * (w - y)
        dN[1, 7] = -4.0 * z
        dN[1, 8] = 0.0
        dN[1, 9] = 4.0 * z

        # dN / dzeta
        dN[2, 0] = 1.0 - 4.0 * w
        dN[2, 1] = 0.0
        dN[2, 2] = 0.0
        dN[2, 3] = 4.0 * z - 1.0
        dN[2, 4] = -4.0 * x
        dN[2, 5] = 0.0
        dN[2, 6] = -4.0 * y
        dN[2, 7] = 4.0 * (w - z)
        dN[2, 8] = 4.0 * x
        dN[2, 9] = 4.0 * y

        return dN

    @njit(fastmath=True)
    def compute_c3d10m_element_numba(coords: np.ndarray, u_elem: np.ndarray, C_mat: np.ndarray):
        """Compute 3D 10-node Tet10 element stiffness (30x30) and internal force vector (30)."""
        K_elem = np.zeros((30, 30), dtype=np.float64)
        f_int = np.zeros(30, dtype=np.float64)

        for pt_idx in range(4):
            xi = _GAUSS_POINTS_NATURAL_TET10[pt_idx, 0]
            eta = _GAUSS_POINTS_NATURAL_TET10[pt_idx, 1]
            zeta = _GAUSS_POINTS_NATURAL_TET10[pt_idx, 2]
            weight = _GAUSS_WEIGHTS_TET10[pt_idx]

            dN_dxi = _sd3d_tet10(xi, eta, zeta)

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

            # Physical derivatives dN / dX = invJ @ dN_dxi (3, 10)
            dN_dX = invJ @ dN_dxi

            # Construct 6 x 30 Strain-Displacement B matrix
            B = np.zeros((6, 30), dtype=np.float64)
            for i in range(10):
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

            dV = detJ * weight
            K_elem += (B.T @ C_mat @ B) * dV
            f_int += (B.T @ stress) * dV

        return K_elem, f_int

    @njit(parallel=True, fastmath=True, nogil=True)
    def assemble_mesh_c3d10m_numba(
        node_coords: np.ndarray,
        elem_conn: np.ndarray,
        u_global: np.ndarray,
        C_mat: np.ndarray
    ):
        """Parallel OpenMP mesh assembly kernel for C3D10M Tet10 elements."""
        n_elems = elem_conn.shape[0]
        f_elems = np.zeros((n_elems, 30), dtype=np.float64)
        K_elems = np.zeros((n_elems, 30, 30), dtype=np.float64)

        for e in prange(n_elems):
            nodes_e = elem_conn[e]
            coords_e = np.zeros((10, 3), dtype=np.float64)
            u_e = np.zeros(30, dtype=np.float64)

            for i in range(10):
                nid = nodes_e[i]
                coords_e[i, 0] = node_coords[nid, 0]
                coords_e[i, 1] = node_coords[nid, 1]
                coords_e[i, 2] = node_coords[nid, 2]

                u_e[3*i + 0] = u_global[3*nid + 0]
                u_e[3*i + 1] = u_global[3*nid + 1]
                u_e[3*i + 2] = u_global[3*nid + 2]

            Ke, fe = compute_c3d10m_element_numba(coords_e, u_e, C_mat)
            K_elems[e] = Ke
            f_elems[e] = fe

        return f_elems, K_elems
