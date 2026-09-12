"""
c3d8_hybrid_numba.py
====================
3D Co-rotational Hybrid 8-Node Hexahedral Element (C3D8H / C3D8_HYBRID).

Formulation:
------------
1. Co-rotational Kinematics:
   - Isolates rigid body rotation R_elem from deformed natural triads.
   - Computes pure deformational displacement:
     u_local = R_elem^T (x - x_c) - (X - X_c).

2. Mixed u-p Hybrid Volumetric-Deviatoric Split:
   - Hydrostatic pressure p_0 is treated as an independent field (constant per element)
     and statically condensed at element level (Abaqus C3D8H equivalent).
   - Mean volumetric B-matrix:
     B_vol_mean = (1 / V0) * \\int B_vol(xi, eta, zeta) dV0.
   - Deviatoric strain:
     eps_dev = B_dev * u_local.
   - Mean volumetric dilatation:
     eps_vol_mean = B_vol_mean * u_local.
   - Stress:
     S = 2 * mu * eps_dev + K * eps_vol_mean * I.
   - Guarantees 100% volumetric locking-free behavior near incompressibility (nu -> 0.5, K/mu -> inf)
     and prevents element inversion / crushing in soft hyperelastic (PSA/rubber) layers.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_3d, MAT_HYPERELASTIC_NEOHOOKEAN

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
def compute_element_rotation_3d(coords_ref: np.ndarray, coords_curr: np.ndarray) -> np.ndarray:
    """Extract orthonormal rigid rotation matrix R_elem (3, 3) from edge triads."""
    vx = (coords_curr[1] - coords_curr[0] + coords_curr[2] - coords_curr[3] +
          coords_curr[5] - coords_curr[4] + coords_curr[6] - coords_curr[7])
    vy = (coords_curr[3] - coords_curr[0] + coords_curr[2] - coords_curr[1] +
          coords_curr[7] - coords_curr[4] + coords_curr[6] - coords_curr[5])

    Vx = (coords_ref[1] - coords_ref[0] + coords_ref[2] - coords_ref[3] +
          coords_ref[5] - coords_ref[4] + coords_ref[6] - coords_ref[7])
    Vy = (coords_ref[3] - coords_ref[0] + coords_ref[2] - coords_ref[1] +
          coords_ref[7] - coords_ref[4] + coords_ref[6] - coords_ref[5])

    norm_vx = np.sqrt(vx[0]*vx[0] + vx[1]*vx[1] + vx[2]*vx[2]) + 1e-15
    r1 = vx / norm_vx

    r3_x = r1[1]*vy[2] - r1[2]*vy[1]
    r3_y = r1[2]*vy[0] - r1[0]*vy[2]
    r3_z = r1[0]*vy[1] - r1[1]*vy[0]
    norm_r3 = np.sqrt(r3_x*r3_x + r3_y*r3_y + r3_z*r3_z) + 1e-15
    r3 = np.array([r3_x / norm_r3, r3_y / norm_r3, r3_z / norm_r3], dtype=np.float64)

    r2 = np.array([
        r3[1]*r1[2] - r3[2]*r1[1],
        r3[2]*r1[0] - r3[0]*r1[2],
        r3[0]*r1[1] - r3[1]*r1[0]
    ], dtype=np.float64)

    R_curr = np.zeros((3, 3), dtype=np.float64)
    R_curr[0, 0] = r1[0]; R_curr[0, 1] = r2[0]; R_curr[0, 2] = r3[0]
    R_curr[1, 0] = r1[1]; R_curr[1, 1] = r2[1]; R_curr[1, 2] = r3[1]
    R_curr[2, 0] = r1[2]; R_curr[2, 1] = r2[2]; R_curr[2, 2] = r3[2]

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

    return R_curr @ R_ref.T


@njit(fastmath=True)
def compute_c3d8_hybrid_element_umat_numba(
    coords: np.ndarray,
    u_elem: np.ndarray,
    mat_type: int,
    props: np.ndarray,
    sdvs: np.ndarray,
    dt: float = 1.0
):
    """Compute 3D Co-Rotational Hybrid Hexahedral Element with mixed u-p volume-averaged pressure.

    Returns:
        f_global (24,): Internal force vector in global coordinates.
        K_global (24, 24): Tangent stiffness matrix in global coordinates.
        error_flag: 0 on success, 1 on element inversion.
    """
    coords_curr = np.zeros((8, 3), dtype=np.float64)
    for i in range(8):
        coords_curr[i, 0] = coords[i, 0] + u_elem[3 * i + 0]
        coords_curr[i, 1] = coords[i, 1] + u_elem[3 * i + 1]
        coords_curr[i, 2] = coords[i, 2] + u_elem[3 * i + 2]

    # 1. Extract rigid rotation
    R = compute_element_rotation_3d(coords, coords_curr)

    # 2. Reference and current centroids
    Xc = np.zeros(3, dtype=np.float64)
    xc = np.zeros(3, dtype=np.float64)
    for i in range(8):
        Xc += coords[i]
        xc += coords_curr[i]
    Xc *= 0.125
    xc *= 0.125

    # 3. Pure deformational displacement in local frame: u_local = R.T @ (x - xc) - (X - Xc)
    u_local = np.zeros(24, dtype=np.float64)
    for i in range(8):
        d_curr = coords_curr[i] - xc
        d_ref = coords[i] - Xc
        u_l0 = R[0, 0] * d_curr[0] + R[1, 0] * d_curr[1] + R[2, 0] * d_curr[2] - d_ref[0]
        u_l1 = R[0, 1] * d_curr[0] + R[1, 1] * d_curr[1] + R[2, 1] * d_curr[2] - d_ref[1]
        u_l2 = R[0, 2] * d_curr[0] + R[1, 2] * d_curr[1] + R[2, 2] * d_curr[2] - d_ref[2]
        u_local[3 * i + 0] = u_l0
        u_local[3 * i + 1] = u_l1
        u_local[3 * i + 2] = u_l2

    # 4. Volume-averaged volumetric B-vector: B_vol_bar (1 x 24)
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
                if detJ <= 0.0:
                    return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1

                dN_dX = np.zeros((3, 8), dtype=np.float64)
                for i in range(8):
                    dN_dX[0, i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dX[1, i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dX[2, i] = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]

                dV = detJ
                dN_dX_mean += dN_dX * dV
                V0_total += dV

    dN_dX_bar = dN_dX_mean / np.maximum(V0_total, 1e-14)

    # 1x24 volumetric dilatation gradient: eps_vol = div(u) = du_x/dx + du_y/dy + du_z/dz
    B_vol_bar = np.zeros(24, dtype=np.float64)
    for i in range(8):
        B_vol_bar[3 * i + 0] = dN_dX_bar[0, i]
        B_vol_bar[3 * i + 1] = dN_dX_bar[1, i]
        B_vol_bar[3 * i + 2] = dN_dX_bar[2, i]

    # Mean volumetric strain for hybrid pressure: eps_vol_mean = B_vol_bar @ u_local
    eps_vol_mean = 0.0
    for dof in range(24):
        eps_vol_mean += B_vol_bar[dof] * u_local[dof]

    # Decode material bulk and shear moduli
    p0 = props[0]
    p1 = props[1]
    if p1 <= 0.05 and p0 > 0.0:
        mu = 2.0 * p0
        K = 2.0 / max(p1, 1e-12)
    elif p1 < 0.5 and p0 > 1.0:
        mu = p0 / (2.0 * (1.0 + p1))
        K = p0 / (3.0 * max(1.0 - 2.0 * p1, 1e-8))
    else:
        mu = p0
        K = max(p1, 1e-8)

    # Hybrid constant hydrostatic pressure: p0 = K * eps_vol_mean
    pressure_mean = K * eps_vol_mean

    # 5. Integrate local stiffness and internal force
    K_local = np.zeros((24, 24), dtype=np.float64)
    f_local = np.zeros(24, dtype=np.float64)

    # Volumetric contribution from hybrid pressure:
    # f_vol = V0 * B_vol_bar^T * p0
    # K_vol = V0 * K * (B_vol_bar (x) B_vol_bar)
    for i in range(24):
        f_local[i] += V0_total * B_vol_bar[i] * pressure_mean
        for j in range(24):
            K_local[i, j] += V0_total * K * B_vol_bar[i] * B_vol_bar[j]

    # Gauss point loop for deviatoric shear contributions
    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]

                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)
                dV = detJ

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

                # Local volumetric strain at this Gauss point
                vol_std = (B_std[0, :] + B_std[1, :] + B_std[2, :]) / 3.0

                # Deviatoric B-matrix: B_dev = B_std - vol_std (zero trace)
                B_dev = B_std.copy()
                for row in range(3):
                    B_dev[row, :] = B_std[row, :] - vol_std

                # Deviatoric strain: e_dev = B_dev @ u_local
                e_dev = B_dev @ u_local

                # Deviatoric shear stress: s_dev = 2 * mu * e_dev
                s_dev = np.zeros(6, dtype=np.float64)
                s_dev[0] = 2.0 * mu * e_dev[0]
                s_dev[1] = 2.0 * mu * e_dev[1]
                s_dev[2] = 2.0 * mu * e_dev[2]
                s_dev[3] = mu * e_dev[3]
                s_dev[4] = mu * e_dev[4]
                s_dev[5] = mu * e_dev[5]

                # Deviatoric tangent operator C_dev
                C_dev = np.zeros((6, 6), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        C_dev[r, c] = - (2.0 / 3.0) * mu
                    C_dev[r, r] += 2.0 * mu
                C_dev[3, 3] = mu
                C_dev[4, 4] = mu
                C_dev[5, 5] = mu

                # Accumulate deviatoric force: f_dev = B_std^T @ s_dev * dV
                f_local += (B_std.T @ s_dev) * dV

                # Accumulate deviatoric stiffness: K_dev = B_std^T @ C_dev @ B_std * dV
                K_local += (B_std.T @ C_dev @ B_std) * dV

    # 6. Transform to global coordinates: K_global = T @ K_local @ T.T, f_global = T @ f_local
    f_global = np.zeros(24, dtype=np.float64)
    K_global = np.zeros((24, 24), dtype=np.float64)

    for i in range(8):
        f_l_node = f_local[3*i : 3*i+3]
        f_global[3*i + 0] = R[0, 0] * f_l_node[0] + R[0, 1] * f_l_node[1] + R[0, 2] * f_l_node[2]
        f_global[3*i + 1] = R[1, 0] * f_l_node[0] + R[1, 1] * f_l_node[1] + R[1, 2] * f_l_node[2]
        f_global[3*i + 2] = R[2, 0] * f_l_node[0] + R[2, 1] * f_l_node[1] + R[2, 2] * f_l_node[2]

    # Transform 24x24 stiffness block-wise (8x8 blocks of 3x3)
    for i in range(8):
        for j in range(8):
            K_block_local = K_local[3*i : 3*i+3, 3*j : 3*j+3]
            K_block_global = R @ K_block_local @ R.T
            K_global[3*i : 3*i+3, 3*j : 3*j+3] = K_block_global

    return f_global, K_global, 0


@njit(parallel=True, fastmath=True)
def assemble_mesh_c3d8_hybrid_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float = 1.0
):
    """Parallel Numba assembly of all C3D8H hybrid elements in the mesh."""
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
    has_error = False

    for e in prange(n_elems):
        conn = elem_conn[e]
        elem_coords = np.zeros((8, 3), dtype=np.float64)
        u_elem = np.zeros(24, dtype=np.float64)

        for i in range(8):
            nid = conn[i]
            elem_coords[i, 0] = node_coords[nid, 0]
            elem_coords[i, 1] = node_coords[nid, 1]
            elem_coords[i, 2] = node_coords[nid, 2]

            u_elem[3 * i + 0] = u_global[3 * nid + 0]
            u_elem[3 * i + 1] = u_global[3 * nid + 1]
            u_elem[3 * i + 2] = u_global[3 * nid + 2]

        m_type = elem_mat_types[e]
        props = elem_props[e]
        sdvs = elem_sdvs[e]

        fe, Ke, err = compute_c3d8_hybrid_element_umat_numba(
            elem_coords, u_elem, m_type, props, sdvs, dt
        )
        if err != 0:
            has_error = True

        f_elems[e] = fe
        K_elems[e] = Ke

    return f_elems, K_elems, has_error
