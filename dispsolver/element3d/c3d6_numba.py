"""
c3d6_numba.py
=============
Numba Kernel for 3D 6-Node Linear Triangular Wedge/Prism (C3D6) Elements.

Features:
- Full 6-point numerical integration (3 triangle x 2 axial points) guaranteeing full rank 12
  (18 DOFs - 6 rigid body modes).
- UMAT-style material decoupling via material_dispatch_3d (supports elastic, J2, viscoelastic).
- OpenMP multi-threaded parallel mesh assembly (assemble_mesh_c3d6_numba).
"""

from __future__ import annotations
import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from dispsolver.material3d.numba_materials import material_dispatch_3d, MAT_CUSTOM_ELASTIC


if HAS_NUMBA:
    _SQ3_INV = 1.0 / np.sqrt(3.0)  # ~ 0.577350269

    _GAUSS_POINTS_NATURAL_WEDGE6 = np.array([
        [1.0 / 6.0, 1.0 / 6.0, -_SQ3_INV],
        [2.0 / 3.0, 1.0 / 6.0, -_SQ3_INV],
        [1.0 / 6.0, 2.0 / 3.0, -_SQ3_INV],
        [1.0 / 6.0, 1.0 / 6.0,  _SQ3_INV],
        [2.0 / 3.0, 1.0 / 6.0,  _SQ3_INV],
        [1.0 / 6.0, 2.0 / 3.0,  _SQ3_INV],
    ], dtype=np.float64)

    _GAUSS_WEIGHTS_WEDGE6 = np.full(6, 1.0 / 6.0, dtype=np.float64)

    _EMPTY_1D_FLOAT = np.empty(0, dtype=np.float64)
    _EMPTY_2D_FLOAT = np.empty((0, 0), dtype=np.float64)

    @njit(fastmath=True)
    def _sd3d_wedge6(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Evaluate 6-node wedge shape derivatives dN_i / d(xi, eta, zeta) (3, 6)."""
        x, y, z = xi, eta, zeta
        lam1 = 1.0 - x - y
        lam2 = x
        lam3 = y

        half_m = 0.5 * (1.0 - z)
        half_p = 0.5 * (1.0 + z)

        dN = np.zeros((3, 6), dtype=np.float64)

        # dN / dxi
        dN[0, 0] = -half_m
        dN[0, 1] =  half_m
        dN[0, 2] =  0.0
        dN[0, 3] = -half_p
        dN[0, 4] =  half_p
        dN[0, 5] =  0.0

        # dN / deta
        dN[1, 0] = -half_m
        dN[1, 1] =  0.0
        dN[1, 2] =  half_m
        dN[1, 3] = -half_p
        dN[1, 4] =  0.0
        dN[1, 5] =  half_p

        # dN / dzeta
        dN[2, 0] = -0.5 * lam1
        dN[2, 1] = -0.5 * lam2
        dN[2, 2] = -0.5 * lam3
        dN[2, 3] =  0.5 * lam1
        dN[2, 4] =  0.5 * lam2
        dN[2, 5] =  0.5 * lam3

        return dN

    @njit(fastmath=True)
    def compute_c3d6_element_numba(
        coords: np.ndarray,
        u_elem: np.ndarray,
        mat_type: int = 0,
        props: np.ndarray = _EMPTY_1D_FLOAT,
        sdvs: np.ndarray = _EMPTY_2D_FLOAT,
        dt: float = 1.0,
        controls: np.ndarray = _EMPTY_1D_FLOAT,
        stress_init: np.ndarray = _EMPTY_2D_FLOAT
    ):
        """Compute C3D6 element stiffness (18x18) and internal force vector (18)."""
        K_elem = np.zeros((18, 18), dtype=np.float64)
        f_int = np.zeros(18, dtype=np.float64)
        error_flag = 0
        I_3x3 = np.eye(3, dtype=np.float64)

        for pt_idx in range(6):
            xi = _GAUSS_POINTS_NATURAL_WEDGE6[pt_idx, 0]
            eta = _GAUSS_POINTS_NATURAL_WEDGE6[pt_idx, 1]
            zeta = _GAUSS_POINTS_NATURAL_WEDGE6[pt_idx, 2]
            weight = _GAUSS_WEIGHTS_WEDGE6[pt_idx]

            dN_dxi = _sd3d_wedge6(xi, eta, zeta)
            J = dN_dxi @ coords

            detJ = (
                J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) -
                J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]) +
                J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
            )
            if detJ <= 0.0:
                return np.zeros((18, 18), dtype=np.float64), np.zeros(18, dtype=np.float64), 1

            dV = detJ * weight

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

            dN_dX = invJ @ dN_dxi

            B = np.zeros((6, 18), dtype=np.float64)
            for i in range(6):
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
            vol_ratio = 1.0 + float(strain[0] + strain[1] + strain[2])

            sdv_gp = sdvs[pt_idx] if sdvs.shape[0] > pt_idx else np.zeros(0, dtype=np.float64)

            stress, C_tangent, sdv_new, mat_err = material_dispatch_3d(
                mat_type, props, sdv_gp, strain, I_3x3, vol_ratio, dt
            )
            if mat_err != 0:
                error_flag = mat_err

            if sdvs.shape[0] > pt_idx:
                sdvs[pt_idx] = sdv_new

            if stress_init.shape[0] > pt_idx:
                for i in range(6):
                    stress[i] += stress_init[pt_idx, i]

            f_int += (B.T @ stress) * dV
            K_elem += (B.T @ C_tangent @ B) * dV

        return K_elem, f_int, error_flag

    @njit(parallel=True, fastmath=True, nogil=True)
    def assemble_mesh_c3d6_numba(
        node_coords: np.ndarray,
        elem_conn: np.ndarray,
        u_global: np.ndarray,
        elem_mat_types: np.ndarray,
        elem_props: np.ndarray,
        elem_sdvs: np.ndarray,
        dt: float = 1.0,
        elem_controls: np.ndarray = _EMPTY_2D_FLOAT,
        elem_stress_init: np.ndarray = _EMPTY_2D_FLOAT
    ):
        """Parallel OpenMP mesh assembly kernel for C3D6 6-node linear wedge elements."""
        n_elems = elem_conn.shape[0]
        f_elems = np.zeros((n_elems, 18), dtype=np.float64)
        K_elems = np.zeros((n_elems, 18, 18), dtype=np.float64)
        has_error = False

        use_controls = (elem_controls.shape[0] == n_elems)
        has_stress_init = (elem_stress_init.shape[0] == n_elems)

        for e in prange(n_elems):
            nodes_e = elem_conn[e]
            coords_e = np.zeros((6, 3), dtype=np.float64)
            u_e = np.zeros(18, dtype=np.float64)

            for i in range(6):
                nid = nodes_e[i]
                coords_e[i, 0] = node_coords[nid, 0]
                coords_e[i, 1] = node_coords[nid, 1]
                coords_e[i, 2] = node_coords[nid, 2]

                u_e[3*i + 0] = u_global[3*nid + 0]
                u_e[3*i + 1] = u_global[3*nid + 1]
                u_e[3*i + 2] = u_global[3*nid + 2]

            mtype = elem_mat_types[e]
            props_e = elem_props[e]
            sdvs_e = elem_sdvs[e]
            ctrl_e = elem_controls[e] if use_controls else _EMPTY_1D_FLOAT
            sinit_e = elem_stress_init[e] if has_stress_init else _EMPTY_2D_FLOAT

            Ke, fe, err = compute_c3d6_element_numba(
                coords_e, u_e, mtype, props_e, sdvs_e, dt, ctrl_e, sinit_e
            )
            if err != 0:
                has_error = True

            K_elems[e] = Ke
            f_elems[e] = fe

        return f_elems, K_elems, has_error
