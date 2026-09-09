"""
c3d8_eas_numba.py
=================
Numba OpenMP C-Extension Kernel for 3D 8-Node EAS (Enhanced Assumed Strain - C3D8I) Elements.

Static condensation of 9 internal EAS modes performed in a high-speed parallel JIT kernel
without Python GIL overhead.
"""

from __future__ import annotations
import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from .c3d8_numba import _sd3d, _jacobian3d, _GP_GAUSS


if HAS_NUMBA:
    @njit(fastmath=True)
    def _compute_invT0_T_numba(J0: np.ndarray) -> np.ndarray:
        """Compute transposed inverse 6x6 Voigt Transformation Matrix inv(T0).T (Simo-Armero 1992)."""
        J11, J12, J13 = J0[0, 0], J0[0, 1], J0[0, 2]
        J21, J22, J23 = J0[1, 0], J0[1, 1], J0[1, 2]
        J31, J32, J33 = J0[2, 0], J0[2, 1], J0[2, 2]

        T0 = np.array([
            [J11**2, J21**2, J31**2, 2.0*J11*J21, 2.0*J21*J31, 2.0*J31*J11],
            [J12**2, J22**2, J32**2, 2.0*J12*J22, 2.0*J22*J32, 2.0*J32*J12],
            [J13**2, J23**2, J33**2, 2.0*J13*J23, 2.0*J23*J33, 2.0*J33*J13],
            [J11*J12, J21*J22, J31*J32, J11*J22+J12*J21, J21*J32+J22*J31, J31*J12+J32*J11],
            [J12*J13, J22*J23, J32*J33, J12*J23+J13*J22, J22*J33+J23*J32, J32*J13+J33*J12],
            [J13*J11, J23*J21, J33*J31, J13*J21+J11*J23, J23*J31+J21*J33, J33*J11+J31*J13]
        ], dtype=np.float64)

        invT0 = np.linalg.inv(T0)
        return invT0.T

    @njit(fastmath=True)
    def _enhanced_strain_matrix_m_numba(xi: float, eta: float, zeta: float) -> np.ndarray:
        """9-mode natural strain interpolation matrix M(xi, eta, zeta) (6, 9)."""
        M = np.zeros((6, 9), dtype=np.float64)
        # Normal modes
        M[0, 0] = xi
        M[1, 1] = eta
        M[2, 2] = zeta
        # Shear modes
        M[3, 3] = xi
        M[3, 4] = eta
        M[4, 5] = eta
        M[4, 6] = zeta
        M[5, 7] = zeta
        M[5, 8] = xi
        return M

    @njit(fastmath=True)
    def compute_c3d8_eas_element_numba(coords: np.ndarray, u_elem: np.ndarray, C_mat: np.ndarray):
        """Compute 3D C3D8I EAS element stiffness (24x24) and internal force vector (24)."""
        # Central Jacobian J0 and Transformation Matrix inv(T0).T
        J0, detJ0, _ = _jacobian3d(0.0, 0.0, 0.0, coords)
        invT0_T = _compute_invT0_T_numba(J0)

        K_uu = np.zeros((24, 24), dtype=np.float64)
        K_ua = np.zeros((24, 9), dtype=np.float64)
        K_aa = np.zeros((9, 9), dtype=np.float64)
        f_int = np.zeros(24, dtype=np.float64)

        for gi in range(2):
            xi = _GP_GAUSS[gi]
            for gj in range(2):
                eta = _GP_GAUSS[gj]
                for gk in range(2):
                    zeta = _GP_GAUSS[gk]

                    dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                    _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)

                    # Physical shape function derivatives
                    dN_dX = np.zeros(8, dtype=np.float64)
                    dN_dY = np.zeros(8, dtype=np.float64)
                    dN_dZ = np.zeros(8, dtype=np.float64)

                    for i in range(8):
                        dN_dX[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                        dN_dY[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                        dN_dZ[i] = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]

                    # Standard Strain-Displacement B matrix (6 x 24)
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

                    # Enhanced B-matrix B_tilde = (detJ0 / detJ) * invT0_T @ M
                    M = _enhanced_strain_matrix_m_numba(xi, eta, zeta)
                    B_tilde = (detJ0 / detJ) * (invT0_T @ M)

                    # Linear strain & stress
                    strain = B @ u_elem
                    stress = C_mat @ strain

                    dV = detJ  # Gauss weight = 1.0

                    K_uu += (B.T @ C_mat @ B) * dV
                    K_ua += (B.T @ C_mat @ B_tilde) * dV
                    K_aa += (B_tilde.T @ C_mat @ B_tilde) * dV
                    f_int += (B.T @ stress) * dV

        # Static Condensation: K_cond = K_uu - K_ua * K_aa^-1 * K_ua^T
        inv_K_aa = np.linalg.inv(K_aa)
        K_condensed = K_uu - K_ua @ inv_K_aa @ K_ua.T
        alpha_opt = -inv_K_aa @ (K_ua.T @ u_elem)

        return K_condensed, f_int, alpha_opt

    @njit(parallel=True, fastmath=True, nogil=True)
    def assemble_mesh_c3d8_eas_numba(
        node_coords: np.ndarray,
        elem_conn: np.ndarray,
        u_global: np.ndarray,
        C_mat: np.ndarray
    ):
        """Parallel OpenMP mesh assembly kernel for C3D8I EAS elements.

        Parameters
        ----------
        node_coords : (n_nodes, 3)
        elem_conn : (n_elems, 8)
        u_global : (3 * n_nodes,)
        C_mat : (6, 6)

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

            Ke, fe, _ = compute_c3d8_eas_element_numba(coords_e, u_e, C_mat)
            K_elems[e] = Ke
            f_elems[e] = fe

        return f_elems, K_elems
