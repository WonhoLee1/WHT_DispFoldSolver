"""
c3d8r_numba.py
==============
High-Performance Numba Kernel for 3D 8-Node Hexahedral Elements
with 1-Point Reduced Numerical Integration and Flanagan-Belytschko Hourglass Control (C3D8R).

Features:
- Single-point numerical integration at centroid (xi=0, eta=0, zeta=0).
- Flanagan & Belytschko (1981) orthogonal hourglass vectors gamma_alpha (alpha=1..4).
- Rank 18 (24 DOFs - 6 rigid body modes) with 0 spurious zero-energy modes.
- Fully decoupled UMAT constitutive integration via material_dispatch_3d.
- OpenMP multi-threaded parallel mesh assembly (assemble_mesh_c3d8r_numba).
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
    # Flanagan & Belytschko (1981) base hourglass vectors (4, 8)
    _H_BASE = np.array([
        [ 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],  # h1
        [ 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],  # h2
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],  # h3
        [-1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0],  # h4
    ], dtype=np.float64)

    # Node signs in natural coordinates (8, 3)
    _XI_N = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0]
    ], dtype=np.float64)

    _EMPTY_1D_FLOAT = np.empty(0, dtype=np.float64)
    _EMPTY_2D_FLOAT = np.empty((0, 0), dtype=np.float64)

    @njit(fastmath=True)
    def _inv3x3_det(A: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute inverse and determinant of 3x3 matrix."""
        a00 = A[0, 0]; a01 = A[0, 1]; a02 = A[0, 2]
        a10 = A[1, 0]; a11 = A[1, 1]; a12 = A[1, 2]
        a20 = A[2, 0]; a21 = A[2, 1]; a22 = A[2, 2]

        c00 =  (a11 * a22 - a12 * a21)
        c01 = -(a10 * a22 - a12 * a20)
        c02 =  (a10 * a21 - a11 * a20)

        det = a00 * c00 + a01 * c01 + a02 * c02

        if abs(det) < 1e-15:
            inv = np.zeros((3, 3), dtype=np.float64)
            return inv, 0.0

        inv_det = 1.0 / det
        inv = np.zeros((3, 3), dtype=np.float64)
        inv[0, 0] = c00 * inv_det
        inv[0, 1] = -(a01 * a22 - a02 * a21) * inv_det
        inv[0, 2] =  (a01 * a12 - a02 * a11) * inv_det

        inv[1, 0] = c01 * inv_det
        inv[1, 1] =  (a00 * a22 - a02 * a20) * inv_det
        inv[1, 2] = -(a00 * a12 - a02 * a10) * inv_det

        inv[2, 0] = c02 * inv_det
        inv[2, 1] = -(a00 * a21 - a01 * a20) * inv_det
        inv[2, 2] =  (a00 * a11 - a01 * a10) * inv_det

        return inv, det

    @njit(fastmath=True)
    def compute_c3d8r_element_numba(
        node_coords: np.ndarray,      # (8, 3)
        u_elem: np.ndarray,           # (24,)
        mat_type: int = 0,
        props: np.ndarray = _EMPTY_1D_FLOAT,
        sdvs: np.ndarray = _EMPTY_2D_FLOAT,
        dt: float = 1.0,
        elem_controls: np.ndarray = _EMPTY_1D_FLOAT
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compute single C3D8R element tangent stiffness (24, 24) and internal force (24,)."""
        K_elem = np.zeros((24, 24), dtype=np.float64)
        f_int = np.zeros(24, dtype=np.float64)

        # 1. Shape derivatives at centroid (0, 0, 0)
        # dN_dxi (3, 8): dN_i / dxi_j = 0.125 * xi_n[i, j]
        dN_dxi = np.zeros((3, 8), dtype=np.float64)
        for a in range(8):
            dN_dxi[0, a] = 0.125 * _XI_N[a, 0]
            dN_dxi[1, a] = 0.125 * _XI_N[a, 1]
            dN_dxi[2, a] = 0.125 * _XI_N[a, 2]

        # Centroidal Jacobian J0 (3, 3) = dN_dxi @ node_coords
        J0 = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                val = 0.0
                for a in range(8):
                    val += dN_dxi[i, a] * node_coords[a, j]
                J0[i, j] = val

        invJ0, detJ0 = _inv3x3_det(J0)
        if detJ0 <= 1e-15:
            return K_elem, f_int, 1

        V0 = 8.0 * detJ0

        # Physical derivatives at centroid: dN_dx (3, 8) = invJ0 @ dN_dxi
        dN_dx = np.zeros((3, 8), dtype=np.float64)
        for i in range(3):
            for a in range(8):
                val = 0.0
                for j in range(3):
                    val += invJ0[i, j] * dN_dxi[j, a]
                dN_dx[i, a] = val

        # 2. Flanagan & Belytschko orthogonalized hourglass vectors gamma (4, 8)
        # gamma_alpha = h_alpha - sum_{i=0..2} (h_alpha^T x_i) * dN_dx[i, :]
        gamma = np.zeros((4, 8), dtype=np.float64)
        for alpha in range(4):
            for a in range(8):
                gamma[alpha, a] = _H_BASE[alpha, a]

            for i in range(3):
                h_dot_x = 0.0
                for a in range(8):
                    h_dot_x += _H_BASE[alpha, a] * node_coords[a, i]
                for a in range(8):
                    gamma[alpha, a] -= h_dot_x * dN_dx[i, a]

        # 3. B-matrix at centroid (6, 24)
        B0 = np.zeros((6, 24), dtype=np.float64)
        for a in range(8):
            dNx = dN_dx[0, a]
            dNy = dN_dx[1, a]
            dNz = dN_dx[2, a]
            col = 3 * a

            B0[0, col]     = dNx
            B0[1, col + 1] = dNy
            B0[2, col + 2] = dNz

            B0[3, col + 1] = dNz
            B0[3, col + 2] = dNy

            B0[4, col]     = dNz
            B0[4, col + 2] = dNx

            B0[5, col]     = dNy
            B0[5, col + 1] = dNx

        # 4. Deformation gradient F at centroid (3, 3)
        # F_ij = delta_ij + sum_a u_{a, i} * dN_dx[j, a]
        F0 = np.eye(3, dtype=np.float64)
        for a in range(8):
            ua_x = u_elem[3 * a]
            ua_y = u_elem[3 * a + 1]
            ua_z = u_elem[3 * a + 2]

            dNx = dN_dx[0, a]
            dNy = dN_dx[1, a]
            dNz = dN_dx[2, a]

            F0[0, 0] += ua_x * dNx
            F0[0, 1] += ua_x * dNy
            F0[0, 2] += ua_x * dNz

            F0[1, 0] += ua_y * dNx
            F0[1, 1] += ua_y * dNy
            F0[1, 2] += ua_y * dNz

            F0[2, 0] += ua_z * dNx
            F0[2, 1] += ua_z * dNy
            F0[2, 2] += ua_z * dNz

        # Linear strain vector (6,)
        eps0 = np.zeros(6, dtype=np.float64)
        for i in range(6):
            val = 0.0
            for j in range(24):
                val += B0[i, j] * u_elem[j]
            eps0[i] = val

        # 5. UMAT Material evaluation at centroid
        _, detF0 = _inv3x3_det(F0)
        sdv_pt = sdvs[0] if sdvs.shape[0] > 0 else _EMPTY_1D_FLOAT
        stress0, C_mat, sdv_new, mat_err = material_dispatch_3d(mat_type, props, sdv_pt, eps0, F0, detF0, dt)
        if mat_err != 0:
            return K_elem, f_int, 1
        if sdvs.shape[0] > 0:
            sdvs[0, :] = sdv_new

        # Centroid internal force f_int_0 = V0 * B0^T @ stress0
        for j in range(24):
            val = 0.0
            for i in range(6):
                val += B0[i, j] * stress0[i]
            f_int[j] = V0 * val

        # Centroid material stiffness K_0 = V0 * B0^T @ C_mat @ B0
        CB = np.zeros((6, 24), dtype=np.float64)
        for i in range(6):
            for j in range(24):
                val = 0.0
                for k in range(6):
                    val += C_mat[i, k] * B0[k, j]
                CB[i, j] = val

        for i in range(24):
            for j in range(24):
                val = 0.0
                for k in range(6):
                    val += B0[k, i] * CB[k, j]
                K_elem[i, j] = V0 * val

        # 6. Flanagan & Belytschko Hourglass Control
        # Extract shear modulus G from props or C_mat
        if mat_type == MAT_CUSTOM_ELASTIC and props.shape[0] >= 36:
            G = props[3 * 6 + 3]
        elif props.shape[0] >= 2:
            E_val = props[0]
            nu_val = props[1]
            G = E_val / (2.0 * (1.0 + nu_val))
        else:
            G = C_mat[3, 3]

        alpha_hg = 0.05
        if elem_controls.shape[0] > 5 and elem_controls[5] > 0.0:
            alpha_hg = elem_controls[5]

        # kappa_hg = 0.5 * alpha_hg * G * V0^(1/3)
        Le = V0 ** (1.0 / 3.0)
        kappa_hg = 0.5 * alpha_hg * G * Le

        # Hourglass modal displacements q (4, 3) across the 3 directions
        for alpha in range(4):
            g = gamma[alpha]  # (8,)
            qx = 0.0
            qy = 0.0
            qz = 0.0
            for a in range(8):
                qx += g[a] * u_elem[3 * a]
                qy += g[a] * u_elem[3 * a + 1]
                qz += g[a] * u_elem[3 * a + 2]

            # Add to internal force: f_hg = kappa_hg * q_i * g_a
            for a in range(8):
                ga = g[a]
                f_int[3 * a]     += kappa_hg * qx * ga
                f_int[3 * a + 1] += kappa_hg * qy * ga
                f_int[3 * a + 2] += kappa_hg * qz * ga

                # Add to tangent stiffness: K_hg = kappa_hg * (g_a * g_b) * delta_ij
                for b in range(8):
                    k_val = kappa_hg * ga * g[b]
                    K_elem[3 * a,     3 * b]     += k_val
                    K_elem[3 * a + 1, 3 * b + 1] += k_val
                    K_elem[3 * a + 2, 3 * b + 2] += k_val

        return K_elem, f_int, 0

    @njit(parallel=True)
    def assemble_mesh_c3d8r_numba(
        node_coords: np.ndarray,      # (n_nodes, 3)
        elem_conn: np.ndarray,        # (n_elems, 8)
        u_global: np.ndarray,         # (n_dofs,)
        elem_mat_types: np.ndarray,   # (n_elems,)
        elem_props: np.ndarray,       # (n_elems, max_props)
        elem_sdvs: np.ndarray,        # (n_elems, 8, max_sdvs)
        dt: float = 1.0,
        elem_controls: np.ndarray = _EMPTY_2D_FLOAT,
        elem_stress_init: np.ndarray = _EMPTY_2D_FLOAT
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Assemble all C3D8R elements in parallel with OpenMP prange."""
        n_elems = elem_conn.shape[0]
        f_elems = np.zeros((n_elems, 24), dtype=np.float64)
        K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
        has_error = 0

        for e in prange(n_elems):
            coords_e = np.zeros((8, 3), dtype=np.float64)
            u_e = np.zeros(24, dtype=np.float64)

            for a in range(8):
                nid = elem_conn[e, a]
                coords_e[a, 0] = node_coords[nid, 0]
                coords_e[a, 1] = node_coords[nid, 1]
                coords_e[a, 2] = node_coords[nid, 2]

                u_e[3 * a]     = u_global[3 * nid]
                u_e[3 * a + 1] = u_global[3 * nid + 1]
                u_e[3 * a + 2] = u_global[3 * nid + 2]

            ctrl_e = elem_controls[e] if elem_controls.shape[0] > 0 else _EMPTY_1D_FLOAT

            K_e, f_e, err = compute_c3d8r_element_numba(
                coords_e,
                u_e,
                mat_type=elem_mat_types[e],
                props=elem_props[e],
                sdvs=elem_sdvs[e],
                dt=dt,
                elem_controls=ctrl_e
            )

            if err != 0:
                has_error = 1

            for i in range(24):
                f_elems[e, i] = f_e[i]
                for j in range(24):
                    K_elems[e, i, j] = K_e[i, j]

        return f_elems, K_elems, has_error
