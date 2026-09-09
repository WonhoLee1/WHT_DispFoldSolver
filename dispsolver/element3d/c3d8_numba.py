"""
c3d8_numba.py
=============
Ultra-Fast Numba OpenMP C-Extension Assembly Kernel for 3D C3D8 Hexahedral Elements.

Enables multi-core parallel assembly of 3D stiffness matrices (24x24 per element)
and internal force vectors without Python GIL overhead, delivering C/Fortran speed.
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
    # 2-point Gauss Quadrature rules in 1D (-1/sqrt(3), +1/sqrt(3))
    _GP_GAUSS = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)

    @njit(fastmath=True)
    def _sd3d(xi: float, eta: float, zeta: float):
        """Compute derivatives of 3D 8-node Hexahedral shape functions w.r.t xi, eta, zeta."""
        dN_dxi = np.array([
            -0.125 * (1.0 - eta) * (1.0 - zeta),
             0.125 * (1.0 - eta) * (1.0 - zeta),
             0.125 * (1.0 + eta) * (1.0 - zeta),
            -0.125 * (1.0 + eta) * (1.0 - zeta),
            -0.125 * (1.0 - eta) * (1.0 + zeta),
             0.125 * (1.0 - eta) * (1.0 + zeta),
             0.125 * (1.0 + eta) * (1.0 + zeta),
            -0.125 * (1.0 + eta) * (1.0 + zeta),
        ], dtype=np.float64)

        dN_deta = np.array([
            -0.125 * (1.0 - xi) * (1.0 - zeta),
            -0.125 * (1.0 + xi) * (1.0 - zeta),
             0.125 * (1.0 + xi) * (1.0 - zeta),
             0.125 * (1.0 - xi) * (1.0 - zeta),
            -0.125 * (1.0 - xi) * (1.0 + zeta),
            -0.125 * (1.0 + xi) * (1.0 + zeta),
             0.125 * (1.0 + xi) * (1.0 + zeta),
             0.125 * (1.0 - xi) * (1.0 + zeta),
        ], dtype=np.float64)

        dN_dzeta = np.array([
            -0.125 * (1.0 - xi) * (1.0 - eta),
            -0.125 * (1.0 + xi) * (1.0 - eta),
            -0.125 * (1.0 + xi) * (1.0 + eta),
            -0.125 * (1.0 - xi) * (1.0 + eta),
             0.125 * (1.0 - xi) * (1.0 - eta),
             0.125 * (1.0 + xi) * (1.0 - eta),
             0.125 * (1.0 + xi) * (1.0 + eta),
             0.125 * (1.0 - xi) * (1.0 + eta),
        ], dtype=np.float64)

        return dN_dxi, dN_deta, dN_dzeta

    @njit(fastmath=True)
    def _jacobian3d(xi: float, eta: float, zeta: float, coords: np.ndarray):
        """Compute 3D 3x3 Jacobian matrix, determinant det(J), and inverse inv(J)."""
        dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
        J = np.zeros((3, 3), dtype=np.float64)

        for i in range(8):
            J[0, 0] += dN_dxi[i] * coords[i, 0]
            J[0, 1] += dN_dxi[i] * coords[i, 1]
            J[0, 2] += dN_dxi[i] * coords[i, 2]

            J[1, 0] += dN_deta[i] * coords[i, 0]
            J[1, 1] += dN_deta[i] * coords[i, 1]
            J[1, 2] += dN_deta[i] * coords[i, 2]

            J[2, 0] += dN_dzeta[i] * coords[i, 0]
            J[2, 1] += dN_dzeta[i] * coords[i, 1]
            J[2, 2] += dN_dzeta[i] * coords[i, 2]

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

        return J, detJ, invJ

    @njit(fastmath=True)
    def compute_c3d8_element_numba(coords: np.ndarray, u_elem: np.ndarray, C_mat: np.ndarray):
        """Compute 3D 8-node Hexahedral element stiffness matrix (24x24) and internal force vector (24)."""
        K_elem = np.zeros((24, 24), dtype=np.float64)
        f_int = np.zeros(24, dtype=np.float64)

        for gi in range(2):
            xi = _GP_GAUSS[gi]
            for gj in range(2):
                eta = _GP_GAUSS[gj]
                for gk in range(2):
                    zeta = _GP_GAUSS[gk]

                    dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                    _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)

                    # Physical shape function derivatives dN/dX, dN/dY, dN/dZ
                    dN_dX = np.zeros(8, dtype=np.float64)
                    dN_dY = np.zeros(8, dtype=np.float64)
                    dN_dZ = np.zeros(8, dtype=np.float64)

                    for i in range(8):
                        dN_dX[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                        dN_dY[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                        dN_dZ[i] = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]

                    # Construct 6 x 24 Strain-Displacement B matrix
                    B = np.zeros((6, 24), dtype=np.float64)
                    for i in range(8):
                        B[0, 3*i + 0] = dN_dX[i]
                        B[1, 3*i + 1] = dN_dY[i]
                        B[2, 3*i + 2] = dN_dZ[i]

                        B[3, 3*i + 0] = dN_dY[i]
                        B[3, 3*i + 1] = dN_dX[i]

                        B[4, 3*i + 1] = dN_dZ[i]
                        B[4, 3*i + 2] = dN_dY[i]

                        B[5, 3*i + 0] = dN_dZ[i]
                        B[5, 3*i + 2] = dN_dX[i]

                    # Linear strain & stress
                    strain = B @ u_elem
                    stress = C_mat @ strain

                    dV = detJ  # Gauss weight = 1.0 * 1.0 * 1.0 = 1.0
                    f_int += (B.T @ stress) * dV
                    K_elem += (B.T @ C_mat @ B) * dV

        return K_elem, f_int

    @njit(parallel=True, fastmath=True, nogil=True)
    def assemble_mesh_c3d8_numba(
        node_coords: np.ndarray,
        elem_conn: np.ndarray,
        u_global: np.ndarray,
        C_mat: np.ndarray
    ):
        """Parallel OpenMP mesh assembly kernel for C3D8 elements.

        Parameters
        ----------
        node_coords : (n_nodes, 3)
        elem_conn : (n_elems, 8)
        u_global : (3 * n_nodes,)
        C_mat : (6, 6)

        Returns
        -------
        f_int_global : (3 * n_nodes,)
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

            Ke, fe = compute_c3d8_element_numba(coords_e, u_e, C_mat)
            K_elems[e] = Ke
            f_elems[e] = fe

        return f_elems, K_elems
