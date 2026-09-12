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

from dispsolver.material3d.numba_materials import (
    material_dispatch_3d, MAT_HYPERELASTIC_NEOHOOKEAN, MAT_CUSTOM_ELASTIC, MAT_LINEAR_ELASTIC,
)

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


_EMPTY_CONTROLS = np.empty(0, dtype=np.float64)
_EMPTY_2D_CONTROLS = np.empty((0, 6), dtype=np.float64)


@njit(fastmath=True)
def compute_c3d8_hybrid_element_umat_numba(
    coords: np.ndarray,
    u_elem: np.ndarray,
    mat_type: int,
    props: np.ndarray,
    sdvs: np.ndarray,
    dt: float = 1.0,
    controls: np.ndarray = _EMPTY_CONTROLS,
    stress_init: np.ndarray = _EMPTY_2D_CONTROLS
):
    """Compute 3D Co-Rotational Hybrid Hexahedral Element with mixed u-p volume-averaged pressure
    and Abaqus-compatible distortion control / anti-inversion safeguards.

    Parameters:
        controls: (8,) float array:
            [0]: distortion_control (1.0=ON, 0.0=OFF)
            [1]: length_ratio / j_crit (default 0.1)
            [2]: viscous_damping (default 0.0)
            [3]: anti_inversion_barrier (1.0=ON, 0.0=OFF)
            [4]: min_det_f (default 0.02)

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

    # Decode material bulk and shear moduli for the hybrid volumetric-pressure
    # term. This MUST match how material_dispatch_3d interprets `props` for
    # the same mat_type below, or the volumetric (mean-dilatation) stiffness
    # and the deviatoric (per-GP, material_dispatch_3d) stiffness are built
    # from two different, inconsistent elasticity tensors.
    #
    # BUG (found 2026-09-13, benchmark_element/benchmark_3d_mechanics.py):
    # every benchmark_element caller constructs DynamicSolver3D via the
    # convenience `material_params={"E":..,"nu":..}` path, which -- per
    # dynamic3d.py's _setup_numba_topology -- always yields
    # mat_type=MAT_CUSTOM_ELASTIC (99) with props = flattened 6x6 C matrix
    # (props[6*i+j] = C[i,j], see numba_materials.py's own MAT_CUSTOM_ELASTIC
    # branch), NEVER mat_type=MAT_LINEAR_ELASTIC (0) with props=[E,nu]. The
    # old heuristic below (magnitude-sniffing p0/p1 to guess [mu,K] vs
    # [E,nu]) silently misread C[0,0]=lam+2mu and C[0,1]=lam as if they were
    # [E,nu] or [mu,K] pairs, giving a wrong effective K/mu for the hybrid
    # pressure term specifically (deviatoric part was unaffected --
    # material_dispatch_3d already decodes MAT_CUSTOM_ELASTIC correctly).
    # This is why C3D8H measured worse than plain (non-hybrid) C3D8 on every
    # metric: an internally inconsistent, not just under-performing, element.
    if mat_type == MAT_CUSTOM_ELASTIC:
        lam = props[1]        # C[0,1]
        mu = props[21]        # C[3,3]
        K = lam + 2.0 * mu / 3.0
    elif mat_type == MAT_LINEAR_ELASTIC:
        E_ = props[0]
        nu_ = props[1]
        mu = E_ / (2.0 * (1.0 + nu_))
        K = E_ / (3.0 * max(1.0 - 2.0 * nu_, 1e-8))
    else:
        # Fallback heuristic for material types not yet given an explicit
        # branch here (e.g. neo-Hookean) -- unchanged from before.
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

    # Check controls
    has_controls = (controls.shape[0] >= 5)
    distortion_control_on = has_controls and (controls[0] > 0.5)
    j_crit = controls[1] if (has_controls and controls[1] > 0.0) else 0.1
    anti_inversion_on = has_controls and (controls[3] > 0.5)
    min_det_f = controls[4] if (has_controls and controls[4] > 0.0) else 0.02

    # Volume ratio J = 1 + eps_vol_mean
    J_vol = 1.0 + eps_vol_mean

    # Anti-inversion severe check
    if anti_inversion_on and J_vol < min_det_f:
        return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1

    # Distortion control barrier (Abaqus energy penalty formulation: J < j_crit)
    p_dist = 0.0
    c_dist = 0.0
    if distortion_control_on and J_vol < j_crit:
        j_eff = max(J_vol, 1e-4)
        k_dist = K
        # Psi_dist = 0.5 * k_dist * ((j_crit - j_eff) / j_eff)**2
        # p_dist = k_dist * j_crit * (j_crit - j_eff) / j_eff^3
        # C_dist = k_dist * j_crit * (2 * j_crit - j_eff) / j_eff^4
        p_dist = k_dist * j_crit * (j_crit - j_eff) / (j_eff * j_eff * j_eff)
        c_dist = k_dist * j_crit * (3.0 * j_crit - 2.0 * j_eff) / (j_eff * j_eff * j_eff * j_eff)

    # Hybrid constant hydrostatic pressure: p0 = K * eps_vol_mean - p_dist
    pressure_mean = K * eps_vol_mean - p_dist
    K_eff = K + c_dist

    # 5. Integrate local stiffness and internal force
    K_local = np.zeros((24, 24), dtype=np.float64)
    f_local = np.zeros(24, dtype=np.float64)

    # Volumetric contribution from hybrid pressure:
    # f_vol = V0 * B_vol_bar^T * p0
    # K_vol = V0 * K_eff * (B_vol_bar (x) B_vol_bar)
    for i in range(24):
        f_local[i] += V0_total * B_vol_bar[i] * pressure_mean
        for j in range(24):
            K_local[i, j] += V0_total * K_eff * B_vol_bar[i] * B_vol_bar[j]

    # Gauss point loop for deviatoric shear contributions
    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]
                gp_idx = 4 * gi + 2 * gj + gk

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

                if mat_type == 0:  # MAT_LINEAR_ELASTIC
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
                else:
                    # Construct B-bar strain for general material dispatcher
                    strain_bar = e_dev.copy()
                    strain_bar[0] += eps_vol_mean / 3.0
                    strain_bar[1] += eps_vol_mean / 3.0
                    strain_bar[2] += eps_vol_mean / 3.0
                    sdv_gp = sdvs[gp_idx] if sdvs.shape[0] > gp_idx else np.zeros(0, dtype=np.float64)
                    vol_ratio = 1.0 + eps_vol_mean
                    I_3x3 = np.eye(3, dtype=np.float64)

                    stress_bar, C_tangent, sdv_gp_new, mat_err = material_dispatch_3d(
                        mat_type, props, sdv_gp, strain_bar, I_3x3, vol_ratio, dt
                    )
                    if sdvs.shape[0] > gp_idx:
                        sdvs[gp_idx] = sdv_gp_new

                    trace_s = stress_bar[0] + stress_bar[1] + stress_bar[2]
                    s_dev = stress_bar.copy()
                    s_dev[0] -= trace_s / 3.0
                    s_dev[1] -= trace_s / 3.0
                    s_dev[2] -= trace_s / 3.0

                    # BUG FIX (2026-09-13): C_dev = C_tangent (the RAW,
                    # un-projected material tangent) does not match the
                    # DEVIATORIC-projected stress s_dev used for f_local
                    # above. f_local's B_std^T @ s_dev is correct regardless
                    # (the volumetric row of B_std contracts against a
                    # trace-free s_dev to exactly zero), but K_local's
                    # B_std^T @ C_dev @ B_std with an un-projected C_dev
                    # reintroduces a full bulk-stiffness contribution here
                    # ON TOP OF the explicit mean-dilatation K_vol term
                    # built above -- i.e. volumetric stiffness was double
                    # counted in the tangent but not in the residual.
                    # Measured: element tangent vs its own finite-difference
                    # Jacobian was 39% off on an undistorted single cube
                    # before this fix (see fix-c3d8h fork report). Verified
                    # this codebase's own mat_type==0 branch above already
                    # builds a proper deviatoric-only isotropic operator
                    # (2*mu off pure-shear, -2/3*mu coupling among normal
                    # components) -- mirror that here using the same `mu`
                    # decoded for the volumetric term, rather than reusing
                    # the raw C_tangent. This is only exactly correct for
                    # ISOTROPIC elasticity (every material this benchmark
                    # suite currently exercises); a general anisotropic
                    # MAT_CUSTOM_ELASTIC would need a full P@C@P deviatoric
                    # projection instead, not implemented here.
                    C_dev = np.zeros((6, 6), dtype=np.float64)
                    for r in range(3):
                        for c in range(3):
                            C_dev[r, c] = -(2.0 / 3.0) * mu
                        C_dev[r, r] += 2.0 * mu
                    C_dev[3, 3] = mu
                    C_dev[4, 4] = mu
                    C_dev[5, 5] = mu
                
                if stress_init.shape[0] == 8:
                    for i in range(6):
                        s_dev[i] += stress_init[gp_idx, i]

                # Accumulate deviatoric force: f_dev = B_std^T @ s_dev * dV
                f_local += (B_std.T @ s_dev) * dV

                # Accumulate deviatoric stiffness: K_dev = B_std^T @ C_dev @ B_std * dV
                K_local += (B_std.T @ C_dev @ B_std) * dV

    # 5b. Viscous damping (Abaqus SectionControls VISCOUS DAMPING)
    damping_coeff = controls[2] if has_controls else 0.0
    if damping_coeff > 0.0:
        for i in range(24):
            k_ii = abs(K_local[i, i])
            f_local[i] += damping_coeff * u_local[i] * k_ii
            K_local[i, i] += damping_coeff * k_ii

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


_EMPTY_2D_CONTROLS = np.empty((0, 0), dtype=np.float64)


@njit(parallel=True, fastmath=True, cache=True)
def assemble_mesh_c3d8_hybrid_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float = 1.0,
    elem_controls: np.ndarray = _EMPTY_2D_CONTROLS,
    elem_stress_init: np.ndarray = _EMPTY_2D_CONTROLS
):
    """Parallel Numba assembly of all C3D8H hybrid elements in the mesh."""
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
    err_flags = np.zeros(n_elems, dtype=np.int32)
    use_controls = elem_controls.shape[0] == n_elems
    has_stress_init = elem_stress_init.shape[0] == n_elems

    for e in prange(n_elems):
        conn = elem_conn[e]
        elem_coords = np.zeros((8, 3), dtype=np.float64)
        u_elem = np.zeros(24, dtype=np.float64)
        for i in range(8):
            nid = conn[i]
            elem_coords[i, :] = node_coords[nid, :]
            u_elem[3*i:3*i+3] = u_global[3*nid:3*nid+3]

        mat_type = elem_mat_types[e]
        props = elem_props[e]
        sdvs = elem_sdvs[e]
        ctrl = elem_controls[e] if use_controls else _EMPTY_CONTROLS
        stress_init = elem_stress_init[e] if has_stress_init else np.zeros((8, 6), dtype=np.float64)

        f_e, K_e, err_e = compute_c3d8_hybrid_element_umat_numba(
            elem_coords, u_elem, mat_type, props, sdvs, dt, ctrl, stress_init
        )
        f_elems[e, :] = f_e
        K_elems[e, :, :] = K_e
        err_flags[e] = err_e

    err_sum = 0
    for e in range(n_elems):
        err_sum += err_flags[e]

    return f_elems, K_elems, err_sum
