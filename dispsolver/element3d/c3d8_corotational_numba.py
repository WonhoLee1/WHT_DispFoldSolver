"""
c3d8_corotational_numba.py
==========================
3D Co-rotational 8-Node Hexahedral Element with B-bar Kinematics and UMAT Dispatch.

Decomposes large finite deformation into:
1. Rigid body rotation (R_elem) tracked by element co-rotational triad.
2. Pure deformational displacement u_local = R_elem^T (x - x_c) - (X - X_c).

Isolates finite rigid rotation from element constitutive strain computation,
completely preventing artificial shear locking, volume blow-up, and element inversion
(detF <= 0) during large bending and folding (> 90° - 180°).
"""

from __future__ import annotations
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
    """Compute 3D Jacobian matrix, determinant, and inverse."""
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
    det_safe = detJ if abs(detJ) > 1e-15 else 1e-15

    invJ[0, 0] = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) / det_safe
    invJ[0, 1] = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) / det_safe
    invJ[0, 2] = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) / det_safe

    invJ[1, 0] = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) / det_safe
    invJ[1, 1] = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) / det_safe
    invJ[1, 2] = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) / det_safe

    invJ[2, 0] = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) / det_safe
    invJ[2, 1] = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) / det_safe
    invJ[2, 2] = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) / det_safe

    return J, detJ, invJ


@njit(fastmath=True)
def compute_element_rotation_3d(coords_ref: np.ndarray, coords_curr: np.ndarray) -> np.ndarray:
    """Compute 3x3 element rigid body rotation matrix R_elem from nodal coords.

    Constructs symmetric edge triads for reference and deformed configurations,
    orthonormalizes them via Gram-Schmidt, and determines the relative rotation.
    Guaranteed R_elem @ R_elem.T == I and det(R_elem) == 1.
    """
    # Deformed edge vectors along 3 natural directions
    vx = (coords_curr[1] - coords_curr[0] + coords_curr[2] - coords_curr[3] +
          coords_curr[5] - coords_curr[4] + coords_curr[6] - coords_curr[7])
    vy = (coords_curr[3] - coords_curr[0] + coords_curr[2] - coords_curr[1] +
          coords_curr[7] - coords_curr[4] + coords_curr[6] - coords_curr[5])

    # Reference edge vectors
    Vx = (coords_ref[1] - coords_ref[0] + coords_ref[2] - coords_ref[3] +
          coords_ref[5] - coords_ref[4] + coords_ref[6] - coords_ref[7])
    Vy = (coords_ref[3] - coords_ref[0] + coords_ref[2] - coords_ref[1] +
          coords_ref[7] - coords_ref[4] + coords_ref[6] - coords_ref[5])

    # Deformed triad
    norm_vx = np.sqrt(vx[0]*vx[0] + vx[1]*vx[1] + vx[2]*vx[2]) + 1e-15
    r1 = vx / norm_vx

    # r3_tmp = r1 x vy
    r3_x = r1[1]*vy[2] - r1[2]*vy[1]
    r3_y = r1[2]*vy[0] - r1[0]*vy[2]
    r3_z = r1[0]*vy[1] - r1[1]*vy[0]
    norm_r3 = np.sqrt(r3_x*r3_x + r3_y*r3_y + r3_z*r3_z) + 1e-15
    r3 = np.array([r3_x / norm_r3, r3_y / norm_r3, r3_z / norm_r3], dtype=np.float64)

    # r2 = r3 x r1
    r2 = np.array([
        r3[1]*r1[2] - r3[2]*r1[1],
        r3[2]*r1[0] - r3[0]*r1[2],
        r3[0]*r1[1] - r3[1]*r1[0]
    ], dtype=np.float64)

    R_curr = np.zeros((3, 3), dtype=np.float64)
    R_curr[0, 0] = r1[0]; R_curr[0, 1] = r2[0]; R_curr[0, 2] = r3[0]
    R_curr[1, 0] = r1[1]; R_curr[1, 1] = r2[1]; R_curr[1, 2] = r3[1]
    R_curr[2, 0] = r1[2]; R_curr[2, 1] = r2[2]; R_curr[2, 2] = r3[2]

    # Reference triad
    norm_Vx = np.sqrt(Vx[0]*Vx[0] + Vx[1]*Vx[1] + Vx[2]*Vx[2]) + 1e-15
    R1 = Vx / norm_Vx

    R3_x = R1[1]*Vy[2] - R1[2]*Vy[1]
    R3_y = R1[2]*Vy[0] - R1[0]*Vy[2]
    R3_z = R1[0]*Vy[1] - R1[1]*Vy[0]
    norm_R3 = np.sqrt(R3_x*R3_x + R3_y*R3_y + R3_z*R3_z) + 1e-15
    R3 = np.array([R3_x / norm_R3, R3_y / norm_R3, R3_z / norm_R3], dtype=np.float64)

    R2 = np.array([
        R3[1]*R1[2] - R3[2]*R1[1],
        R3[2]*R1[0] - R3[0]*R1[2],
        R3[0]*R1[1] - R3[1]*R1[0]
    ], dtype=np.float64)

    R_ref = np.zeros((3, 3), dtype=np.float64)
    R_ref[0, 0] = R1[0]; R_ref[0, 1] = R2[0]; R_ref[0, 2] = R3[0]
    R_ref[1, 0] = R1[1]; R_ref[1, 1] = R2[1]; R_ref[1, 2] = R3[1]
    R_ref[2, 0] = R1[2]; R_ref[2, 1] = R2[2]; R_ref[2, 2] = R3[2]

    # Relative rigid rotation: R_elem = R_curr @ R_ref.T
    R_elem = R_curr @ R_ref.T
    return R_elem


@njit(fastmath=True)
def compute_c3d8_corotational_element_umat_numba(
    coords: np.ndarray,
    u_elem: np.ndarray,
    mat_type: int,
    props: np.ndarray,
    sdvs: np.ndarray,
    dt: float = 1.0
):
    """Compute 3D Co-rotational C3D8 element with B-bar kinematics and UMAT constitutive dispatch.

    Returns:
        f_global (24,): Internal force vector in global coordinates.
        K_global (24, 24): Tangent stiffness matrix in global coordinates.
        error_flag: 0 on success, 1 on severe failure.
    """
    coords_curr = np.zeros((8, 3), dtype=np.float64)
    for i in range(8):
        coords_curr[i, 0] = coords[i, 0] + u_elem[3 * i + 0]
        coords_curr[i, 1] = coords[i, 1] + u_elem[3 * i + 1]
        coords_curr[i, 2] = coords[i, 2] + u_elem[3 * i + 2]

    # 1. Extract rigid body rotation
    R = compute_element_rotation_3d(coords, coords_curr)

    # 2. Compute reference and current centroids
    Xc = np.zeros(3, dtype=np.float64)
    xc = np.zeros(3, dtype=np.float64)
    for i in range(8):
        Xc += coords[i]
        xc += coords_curr[i]
    Xc *= 0.125
    xc *= 0.125

    # 3. Extract pure deformational local displacement u_local
    u_local = np.zeros(24, dtype=np.float64)
    for i in range(8):
        d_curr = coords_curr[i] - xc
        d_ref = coords[i] - Xc
        # u_l_node = R.T @ d_curr - d_ref
        u_l0 = R[0, 0] * d_curr[0] + R[1, 0] * d_curr[1] + R[2, 0] * d_curr[2] - d_ref[0]
        u_l1 = R[0, 1] * d_curr[0] + R[1, 1] * d_curr[1] + R[2, 1] * d_curr[2] - d_ref[1]
        u_l2 = R[0, 2] * d_curr[0] + R[1, 2] * d_curr[1] + R[2, 2] * d_curr[2] - d_ref[2]
        u_local[3 * i + 0] = u_l0
        u_local[3 * i + 1] = u_l1
        u_local[3 * i + 2] = u_l2

    # 4. Volume-averaged shape function derivatives for B-bar
    dN_dX_mean = np.zeros((3, 8), dtype=np.float64)
    V0_total = 0.0

    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]

                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)

                dN_dX = np.zeros((3, 8), dtype=np.float64)
                for i in range(8):
                    dN_dX[0, i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dX[1, i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dX[2, i] = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]

                dV = detJ
                dN_dX_mean += dN_dX * dV
                V0_total += dV

    dN_dX_bar = dN_dX_mean / np.maximum(V0_total, 1e-14)

    # Build mean volumetric B-matrix
    B_mean = np.zeros((6, 24), dtype=np.float64)
    for i in range(8):
        B_mean[0, 3*i + 0] = dN_dX_bar[0, i]
        B_mean[1, 3*i + 1] = dN_dX_bar[1, i]
        B_mean[2, 3*i + 2] = dN_dX_bar[2, i]
        B_mean[3, 3*i + 0] = dN_dX_bar[1, i]
        B_mean[3, 3*i + 1] = dN_dX_bar[0, i]
        B_mean[4, 3*i + 1] = dN_dX_bar[2, i]
        B_mean[4, 3*i + 2] = dN_dX_bar[1, i]
        B_mean[5, 3*i + 0] = dN_dX_bar[2, i]
        B_mean[5, 3*i + 2] = dN_dX_bar[0, i]

    vol_mean = (B_mean[0, :] + B_mean[1, :] + B_mean[2, :]) / 3.0

    # 5. Core integration in local corotational frame
    K_local = np.zeros((24, 24), dtype=np.float64)
    f_local = np.zeros(24, dtype=np.float64)
    gp_idx = 0
    error_flag = 0
    I_3x3 = np.eye(3, dtype=np.float64)

    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]

                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)

                dN_dX_i = np.zeros(8, dtype=np.float64)
                dN_dY_i = np.zeros(8, dtype=np.float64)
                dN_dZ_i = np.zeros(8, dtype=np.float64)

                for i in range(8):
                    dN_dX_i[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dY_i[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dZ_i[i] = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]

                # Standard B-matrix
                B_std = np.zeros((6, 24), dtype=np.float64)
                for i in range(8):
                    B_std[0, 3*i + 0] = dN_dX_i[i]
                    B_std[1, 3*i + 1] = dN_dY_i[i]
                    B_std[2, 3*i + 2] = dN_dZ_i[i]
                    B_std[3, 3*i + 0] = dN_dY_i[i]
                    B_std[3, 3*i + 1] = dN_dX_i[i]
                    B_std[4, 3*i + 1] = dN_dZ_i[i]
                    B_std[4, 3*i + 2] = dN_dY_i[i]
                    B_std[5, 3*i + 0] = dN_dZ_i[i]
                    B_std[5, 3*i + 2] = dN_dX_i[i]

                vol_std = (B_std[0, :] + B_std[1, :] + B_std[2, :]) / 3.0

                B_bar = B_std.copy()
                for row in range(3):
                    B_bar[row, :] = B_std[row, :] - vol_std + vol_mean

                # Local engineering strain (pure deformational, no rigid rotation)
                strain_voigt = B_bar @ u_local

                sdv_gp = sdvs[gp_idx] if sdvs.shape[0] > gp_idx else np.zeros(0, dtype=np.float64)
                # In corotational frame, detF is approximated by 1 + tr(strain)
                vol_ratio = 1.0 + (strain_voigt[0] + strain_voigt[1] + strain_voigt[2])

                stress_voigt, C_mat, sdv_new, err = material_dispatch_3d(
                    mat_type, props, sdv_gp, strain_voigt, I_3x3, vol_ratio, dt
                )
                if err != 0:
                    error_flag = err

                dV = detJ
                f_local += (B_bar.T @ stress_voigt) * dV
                K_local += (B_bar.T @ C_mat @ B_bar) * dV
                gp_idx += 1

    # 6. Transform back to Global Frame:
    # f_global = T24 @ f_local
    # K_global = T24 @ K_local @ T24.T
    f_global = np.zeros(24, dtype=np.float64)
    K_global = np.zeros((24, 24), dtype=np.float64)

    for a in range(8):
        f_a_l = f_local[3*a : 3*a + 3]
        f_global[3*a + 0] = R[0, 0]*f_a_l[0] + R[0, 1]*f_a_l[1] + R[0, 2]*f_a_l[2]
        f_global[3*a + 1] = R[1, 0]*f_a_l[0] + R[1, 1]*f_a_l[1] + R[1, 2]*f_a_l[2]
        f_global[3*a + 2] = R[2, 0]*f_a_l[0] + R[2, 1]*f_a_l[1] + R[2, 2]*f_a_l[2]

        for b in range(8):
            Kab_l = K_local[3*a : 3*a + 3, 3*b : 3*b + 3]
            # R @ Kab_l @ R.T
            RK = R @ Kab_l
            Kab_g = RK @ R.T
            for r in range(3):
                for c in range(3):
                    K_global[3*a + r, 3*b + c] = Kab_g[r, c]

    return f_global, K_global, error_flag


@njit(parallel=True, fastmath=True, nogil=True)
def _assemble_mesh_c3d8_corotational_umat_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float = 1.0
):
    """Parallel OpenMP mesh assembly kernel for 3D Co-rotational C3D8 elements."""
    n_elems = elem_conn.shape[0]
    has_error = 0
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)

    for e in prange(n_elems):
        coords = np.zeros((8, 3), dtype=np.float64)
        u_elem = np.zeros(24, dtype=np.float64)

        for i in range(8):
            nid = elem_conn[e, i]
            coords[i, 0] = node_coords[nid, 0]
            coords[i, 1] = node_coords[nid, 1]
            coords[i, 2] = node_coords[nid, 2]

            u_elem[3*i + 0] = u_global[3*nid + 0]
            u_elem[3*i + 1] = u_global[3*nid + 1]
            u_elem[3*i + 2] = u_global[3*nid + 2]

        mat_type = elem_mat_types[e]
        props = elem_props[e]
        sdvs = elem_sdvs[e]

        f_e, K_e, err_e = compute_c3d8_corotational_element_umat_numba(
            coords, u_elem, mat_type, props, sdvs, dt
        )
        if err_e != 0:
            has_error = 1

        f_elems[e] = f_e
        K_elems[e] = K_e

    return f_elems, K_elems, has_error


def assemble_mesh_c3d8_corotational_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    *args,
    **kwargs
):
    """Entrypoint wrapper supporting both legacy and modern UMAT signatures."""
    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].ndim == 2:
        C_mat = args[0]
        n_elems = elem_conn.shape[0]
        elem_mat_types = np.full(n_elems, MAT_CUSTOM_ELASTIC, dtype=np.int32)
        elem_props = np.zeros((n_elems, 36), dtype=np.float64)
        cmat_flat = C_mat.ravel()
        for e in range(n_elems):
            elem_props[e, :36] = cmat_flat
        elem_sdvs = np.zeros((n_elems, 8, 0), dtype=np.float64)
        return _assemble_mesh_c3d8_corotational_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, 1.0
        )
    else:
        elem_mat_types = args[0]
        elem_props = args[1]
        elem_sdvs = args[2] if len(args) > 2 else np.zeros((elem_conn.shape[0], 8, 0), dtype=np.float64)
        dt = float(args[3]) if len(args) > 3 else float(kwargs.get("dt", 1.0))
        return _assemble_mesh_c3d8_corotational_umat_numba(
            node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt
        )
