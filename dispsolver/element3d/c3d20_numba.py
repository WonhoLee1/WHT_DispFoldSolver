"""
c3d20_numba.py
==============
Ultra-Fast Numba OpenMP C-Extension Assembly Kernel for 3D C3D20 / C3D20R
20-Node Quadratic Serendipity Hexahedral Elements.

Enables multi-core parallel assembly of 3D stiffness matrices (60x60 per element)
and internal force vectors without Python GIL overhead, delivering C/Fortran speed.
Supports full 27-point (3x3x3) Gauss integration and 8-point (2x2x2) reduced integration.
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
    # 3-point 1D Gauss quadrature: (-sqrt(0.6), 0, sqrt(0.6)), weights: (5/9, 8/9, 5/9)
    _G3_PTS = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)], dtype=np.float64)
    _G3_WTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)

    # 2-point 1D Gauss quadrature: (-1/sqrt(3), 1/sqrt(3)), weights: (1.0, 1.0)
    _G2_PTS = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
    _G2_WTS = np.array([1.0, 1.0], dtype=np.float64)

    # Pre-computed corner and midside sign arrays for fast SIMD in Numba
    _XI_I = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float64)
    _ETA_I = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)
    _ZETA_I = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    @njit(fastmath=True)
    def _sd3d_c3d20(xi: float, eta: float, zeta: float):
        """Evaluate 20-node natural shape derivatives dN / d(xi, eta, zeta) of shape (3, 20)."""
        dN = np.zeros((3, 20), dtype=np.float64)

        # 1. Corner nodes (0..7)
        for i in range(8):
            xi0 = xi * _XI_I[i]
            eta0 = eta * _ETA_I[i]
            zeta0 = zeta * _ZETA_I[i]

            # dN / dxi
            dN[0, i] = 0.125 * _XI_I[i] * (1.0 + eta0) * (1.0 + zeta0) * (2.0 * xi0 + eta0 + zeta0 - 1.0)
            # dN / deta
            dN[1, i] = 0.125 * _ETA_I[i] * (1.0 + xi0) * (1.0 + zeta0) * (xi0 + 2.0 * eta0 + zeta0 - 1.0)
            # dN / dzeta
            dN[2, i] = 0.125 * _ZETA_I[i] * (1.0 + xi0) * (1.0 + eta0) * (xi0 + eta0 + 2.0 * zeta0 - 1.0)

        # 2. Midside nodes parallel to xi (8, 10, 12, 14): (0, -1, -1), (0, 1, -1), (0, -1, 1), (0, 1, 1)
        # Node 8: (0, -1, -1)
        dN[0, 8] = -0.5 * xi * (1.0 - eta) * (1.0 - zeta)
        dN[1, 8] = -0.25 * (1.0 - xi * xi) * (1.0 - zeta)
        dN[2, 8] = -0.25 * (1.0 - xi * xi) * (1.0 - eta)

        # Node 10: (0, 1, -1)
        dN[0, 10] = -0.5 * xi * (1.0 + eta) * (1.0 - zeta)
        dN[1, 10] = 0.25 * (1.0 - xi * xi) * (1.0 - zeta)
        dN[2, 10] = -0.25 * (1.0 - xi * xi) * (1.0 + eta)

        # Node 12: (0, -1, 1)
        dN[0, 12] = -0.5 * xi * (1.0 - eta) * (1.0 + zeta)
        dN[1, 12] = -0.25 * (1.0 - xi * xi) * (1.0 + zeta)
        dN[2, 12] = 0.25 * (1.0 - xi * xi) * (1.0 - eta)

        # Node 14: (0, 1, 1)
        dN[0, 14] = -0.5 * xi * (1.0 + eta) * (1.0 + zeta)
        dN[1, 14] = 0.25 * (1.0 - xi * xi) * (1.0 + zeta)
        dN[2, 14] = 0.25 * (1.0 - xi * xi) * (1.0 + eta)

        # 3. Midside nodes parallel to eta (9, 11, 13, 15): (1, 0, -1), (-1, 0, -1), (1, 0, 1), (-1, 0, 1)
        # Node 9: (1, 0, -1)
        dN[0, 9] = 0.25 * (1.0 - eta * eta) * (1.0 - zeta)
        dN[1, 9] = -0.5 * eta * (1.0 + xi) * (1.0 - zeta)
        dN[2, 9] = -0.25 * (1.0 + xi) * (1.0 - eta * eta)

        # Node 11: (-1, 0, -1)
        dN[0, 11] = -0.25 * (1.0 - eta * eta) * (1.0 - zeta)
        dN[1, 11] = -0.5 * eta * (1.0 - xi) * (1.0 - zeta)
        dN[2, 11] = -0.25 * (1.0 - xi) * (1.0 - eta * eta)

        # Node 13: (1, 0, 1)
        dN[0, 13] = 0.25 * (1.0 - eta * eta) * (1.0 + zeta)
        dN[1, 13] = -0.5 * eta * (1.0 + xi) * (1.0 + zeta)
        dN[2, 13] = 0.25 * (1.0 + xi) * (1.0 - eta * eta)

        # Node 15: (-1, 0, 1)
        dN[0, 15] = -0.25 * (1.0 - eta * eta) * (1.0 + zeta)
        dN[1, 15] = -0.5 * eta * (1.0 - xi) * (1.0 + zeta)
        dN[2, 15] = 0.25 * (1.0 - xi) * (1.0 - eta * eta)

        # 4. Midside nodes parallel to zeta (16, 17, 18, 19): (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)
        # Node 16: (-1, -1, 0)
        dN[0, 16] = -0.25 * (1.0 - eta) * (1.0 - zeta * zeta)
        dN[1, 16] = -0.25 * (1.0 - xi) * (1.0 - zeta * zeta)
        dN[2, 16] = -0.5 * zeta * (1.0 - xi) * (1.0 - eta)

        # Node 17: (1, -1, 0)
        dN[0, 17] = 0.25 * (1.0 - eta) * (1.0 - zeta * zeta)
        dN[1, 17] = -0.25 * (1.0 + xi) * (1.0 - zeta * zeta)
        dN[2, 17] = -0.5 * zeta * (1.0 + xi) * (1.0 - eta)

        # Node 18: (1, 1, 0)
        dN[0, 18] = 0.25 * (1.0 + eta) * (1.0 - zeta * zeta)
        dN[1, 18] = 0.25 * (1.0 + xi) * (1.0 - zeta * zeta)
        dN[2, 18] = -0.5 * zeta * (1.0 + xi) * (1.0 + eta)

        # Node 19: (-1, 1, 0)
        dN[0, 19] = -0.25 * (1.0 + eta) * (1.0 - zeta * zeta)
        dN[1, 19] = 0.25 * (1.0 - xi) * (1.0 - zeta * zeta)
        dN[2, 19] = -0.5 * zeta * (1.0 - xi) * (1.0 + eta)

        return dN

    @njit(fastmath=True)
    def _jacobian3d_c3d20(dN: np.ndarray, coords: np.ndarray):
        """Compute 3D 3x3 Jacobian matrix, determinant det(J), and inverse inv(J) for 20 nodes."""
        J = np.zeros((3, 3), dtype=np.float64)

        for i in range(20):
            J[0, 0] += dN[0, i] * coords[i, 0]
            J[0, 1] += dN[0, i] * coords[i, 1]
            J[0, 2] += dN[0, i] * coords[i, 2]

            J[1, 0] += dN[1, i] * coords[i, 0]
            J[1, 1] += dN[1, i] * coords[i, 1]
            J[1, 2] += dN[1, i] * coords[i, 2]

            J[2, 0] += dN[2, i] * coords[i, 0]
            J[2, 1] += dN[2, i] * coords[i, 1]
            J[2, 2] += dN[2, i] * coords[i, 2]

        detJ = (
            J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) -
            J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]) +
            J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
        )

        invJ = np.zeros((3, 3), dtype=np.float64)
        if abs(detJ) > 1e-15:
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
    def _sf3d_c3d20(xi: float, eta: float, zeta: float):
        """Evaluate 20-node serendipity shape functions N(xi, eta, zeta) of shape (20,)."""
        N = np.zeros(20, dtype=np.float64)
        for i in range(8):
            xi0 = xi * _XI_I[i]
            eta0 = eta * _ETA_I[i]
            zeta0 = zeta * _ZETA_I[i]
            N[i] = 0.125 * (1.0 + xi0) * (1.0 + eta0) * (1.0 + zeta0) * (xi0 + eta0 + zeta0 - 2.0)

        # xi parallel (8, 10, 12, 14)
        N[8] = 0.25 * (1.0 - xi * xi) * (1.0 - eta) * (1.0 - zeta)
        N[10] = 0.25 * (1.0 - xi * xi) * (1.0 + eta) * (1.0 - zeta)
        N[12] = 0.25 * (1.0 - xi * xi) * (1.0 - eta) * (1.0 + zeta)
        N[14] = 0.25 * (1.0 - xi * xi) * (1.0 + eta) * (1.0 + zeta)

        # eta parallel (9, 11, 13, 15)
        N[9] = 0.25 * (1.0 + xi) * (1.0 - eta * eta) * (1.0 - zeta)
        N[11] = 0.25 * (1.0 - xi) * (1.0 - eta * eta) * (1.0 - zeta)
        N[13] = 0.25 * (1.0 + xi) * (1.0 - eta * eta) * (1.0 + zeta)
        N[15] = 0.25 * (1.0 - xi) * (1.0 - eta * eta) * (1.0 + zeta)

        # zeta parallel (16, 17, 18, 19)
        N[16] = 0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta * zeta)
        N[17] = 0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta * zeta)
        N[18] = 0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta * zeta)
        N[19] = 0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta * zeta)
        return N

    @njit(fastmath=True)
    def compute_c3d20_element_numba(
        coords: np.ndarray,
        u_elem: np.ndarray,
        C_mat_iso: np.ndarray,
        reduced_integration: bool = False,
        C_mat_loc: np.ndarray = None,
        is_cylindrical: bool = False
    ):
        """Compute C3D20/C3D20R element stiffness matrix (60x60) and internal force vector (60)."""
        K_elem = np.zeros((60, 60), dtype=np.float64)
        f_int = np.zeros(60, dtype=np.float64)

        pts = _G2_PTS if reduced_integration else _G3_PTS
        wts = _G2_WTS if reduced_integration else _G3_WTS
        n_gp = len(pts)

        for gi in range(n_gp):
            xi = pts[gi]
            wi = wts[gi]
            for gj in range(n_gp):
                eta = pts[gj]
                wj = wts[gj]
                for gk in range(n_gp):
                    zeta = pts[gk]
                    wk = wts[gk]

                    dN = _sd3d_c3d20(xi, eta, zeta)
                    _, detJ, invJ = _jacobian3d_c3d20(dN, coords)

                    if detJ <= 0.0:
                        continue

                    # Physical shape function derivatives dN/dX, dN/dY, dN/dZ
                    dN_dX = np.zeros(20, dtype=np.float64)
                    dN_dY = np.zeros(20, dtype=np.float64)
                    dN_dZ = np.zeros(20, dtype=np.float64)

                    for i in range(20):
                        dN_dX[i] = invJ[0, 0] * dN[0, i] + invJ[0, 1] * dN[1, i] + invJ[0, 2] * dN[2, i]
                        dN_dY[i] = invJ[1, 0] * dN[0, i] + invJ[1, 1] * dN[1, i] + invJ[1, 2] * dN[2, i]
                        dN_dZ[i] = invJ[2, 0] * dN[0, i] + invJ[2, 1] * dN[1, i] + invJ[2, 2] * dN[2, i]

                    # Construct 6 x 60 Strain-Displacement B matrix
                    B = np.zeros((6, 60), dtype=np.float64)
                    for i in range(20):
                        B[0, 3*i + 0] = dN_dX[i]
                        B[1, 3*i + 1] = dN_dY[i]
                        B[2, 3*i + 2] = dN_dZ[i]

                        B[3, 3*i + 0] = dN_dY[i]
                        B[3, 3*i + 1] = dN_dX[i]

                        B[4, 3*i + 1] = dN_dZ[i]
                        B[4, 3*i + 2] = dN_dY[i]

                        B[5, 3*i + 0] = dN_dZ[i]
                        B[5, 3*i + 2] = dN_dX[i]

                    if is_cylindrical and C_mat_loc is not None:
                        # Evaluate GP physical coordinates using shape functions
                        N_gp = _sf3d_c3d20(xi, eta, zeta)
                        xgp = 0.0
                        ygp = 0.0
                        for i in range(20):
                            xgp += N_gp[i] * coords[i, 0]
                            ygp += N_gp[i] * coords[i, 1]
                        rgp = np.sqrt(xgp * xgp + ygp * ygp)
                        if rgp > 1e-12:
                            c = xgp / rgp
                            s = ygp / rgp
                        else:
                            c = 1.0
                            s = 0.0

                        T = np.zeros((6, 6), dtype=np.float64)
                        T[0, 0] = c * c;        T[0, 1] = s * s;        T[0, 3] = c * s
                        T[1, 0] = s * s;        T[1, 1] = c * c;        T[1, 3] = -c * s
                        T[2, 2] = 1.0
                        T[3, 0] = -2.0 * c * s; T[3, 1] = 2.0 * c * s;  T[3, 3] = c * c - s * s
                        T[4, 4] = c;            T[4, 5] = -s
                        T[5, 4] = s;            T[5, 5] = c

                        C_mat = T.T @ C_mat_loc @ T
                    else:
                        C_mat = C_mat_iso

                    strain = B @ u_elem
                    stress = C_mat @ strain

                    dV = detJ * wi * wj * wk
                    f_int += (B.T @ stress) * dV
                    K_elem += (B.T @ C_mat @ B) * dV

        return K_elem, f_int

    @njit(parallel=True, fastmath=True, nogil=True)
    def assemble_mesh_c3d20_numba(
        node_coords: np.ndarray,
        elem_conn: np.ndarray,
        u_global: np.ndarray,
        elem_mat_types: np.ndarray,
        elem_props: np.ndarray,
        elem_sdvs: np.ndarray,
        dt: float,
        reduced_integration: bool = False
    ):
        """Parallel OpenMP mesh assembly kernel for C3D20 elements.

        Parameters
        ----------
        node_coords : (n_nodes, 3)
        elem_conn : (n_elems, 20)
        u_global : (3 * n_nodes,)
        elem_mat_types : (n_elems,)
        elem_props : (n_elems, n_props)
        elem_sdvs : (n_elems, 8, max_sdvs)
        dt : float
        reduced_integration : bool

        Returns
        -------
        f_elems : (n_elems, 60)
        K_elems : (n_elems, 60, 60)
        err : int
        """
        n_elems = elem_conn.shape[0]
        f_elems = np.zeros((n_elems, 60), dtype=np.float64)
        K_elems = np.zeros((n_elems, 60, 60), dtype=np.float64)

        for e in prange(n_elems):
            coords = np.zeros((20, 3), dtype=np.float64)
            u_elem = np.zeros(60, dtype=np.float64)

            for i in range(20):
                nid = elem_conn[e, i]
                coords[i, 0] = node_coords[nid, 0]
                coords[i, 1] = node_coords[nid, 1]
                coords[i, 2] = node_coords[nid, 2]

                u_elem[3*i + 0] = u_global[3*nid + 0]
                u_elem[3*i + 1] = u_global[3*nid + 1]
                u_elem[3*i + 2] = u_global[3*nid + 2]

            mtype = elem_mat_types[e]
            if mtype == 17:  # MAT_ORTHOTROPIC_CYLINDRICAL
                E1 = elem_props[e, 0]
                E2 = elem_props[e, 1]
                E3 = elem_props[e, 2]
                nu12 = elem_props[e, 3]
                nu13 = elem_props[e, 4]
                nu23 = elem_props[e, 5]
                G12 = elem_props[e, 6]
                G13 = elem_props[e, 7]
                G23 = elem_props[e, 8]

                S = np.zeros((6, 6), dtype=np.float64)
                S[0, 0] = 1.0 / E1;      S[0, 1] = -nu12 / E1;  S[0, 2] = -nu13 / E1
                S[1, 0] = -nu12 / E1;    S[1, 1] = 1.0 / E2;    S[1, 2] = -nu23 / E2
                S[2, 0] = -nu13 / E1;    S[2, 1] = -nu23 / E2;  S[2, 2] = 1.0 / E3
                S[3, 3] = 1.0 / G12
                S[4, 4] = 1.0 / G23
                S[5, 5] = 1.0 / G13

                C_loc = np.linalg.inv(S)
                C_dummy = np.zeros((6, 6), dtype=np.float64)
                K_e, f_e = compute_c3d20_element_numba(coords, u_elem, C_dummy, reduced_integration, C_loc, True)
            else:
                # Linear isotropic elasticity tensor C_mat (6x6)
                E_val = elem_props[e, 0] if elem_props.shape[1] > 0 else 200000.0
                nu_val = elem_props[e, 1] if elem_props.shape[1] > 1 else 0.3

                fac = E_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
                c11 = fac * (1.0 - nu_val)
                c12 = fac * nu_val
                c44 = fac * 0.5 * (1.0 - 2.0 * nu_val)

                C_mat = np.zeros((6, 6), dtype=np.float64)
                C_mat[0, 0] = c11; C_mat[0, 1] = c12; C_mat[0, 2] = c12
                C_mat[1, 0] = c12; C_mat[1, 1] = c11; C_mat[1, 2] = c12
                C_mat[2, 0] = c12; C_mat[2, 1] = c12; C_mat[2, 2] = c11
                C_mat[3, 3] = c44
                C_mat[4, 4] = c44
                C_mat[5, 5] = c44

                K_e, f_e = compute_c3d20_element_numba(coords, u_elem, C_mat, reduced_integration)

            K_elems[e] = K_e
            f_elems[e] = f_e

        return f_elems, K_elems, 0

