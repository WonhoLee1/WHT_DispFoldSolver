"""
cpe4r_numba.py
==============
High-Performance Numba JIT accelerated 4-node Reduced Integration Quadrilateral (CPE4R)
with Abaqus-grade ENHANCED / STIFFNESS Hourglass Control and Universal Distortion Control.

Formulation & Mechanics:
- Single centroid Gauss point (xi=0, eta=0) uniform strain Total Lagrangian.
- ENHANCED mode: Exact physical hourglass stiffness from Q1E4 (CPE4I) incompatible modes
  via static condensation: K_E = K_1pt + Gamma^T (H/16) Gamma (Puso 2000 / Belytschko & Bindeman 1993).
  Objective under large global rotation via polar decomposition pull-back:
  q_ref = R(theta)^T q. Eliminates aspect ratio sensitivity and shear locking in bending.
- STIFFNESS mode: Flanagan & Belytschko (1981) orthogonal hourglass control with Abaqus
  0.005 * G * V * sum(b^2) total-stiffness scaling.
- Universal Distortion Control: Energetic C^1 barrier preventing element inversion
  when local volume ratio J = det(F) < length_ratio (Abaqus *SECTION CONTROLS, DISTORTION CONTROL=YES).
- OpenMP multi-threaded parallel mesh assembly (assemble_mesh_cpe4r_numba).
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_2d

# Flanagan-Belytschko base hourglass mode at 4 corners
_HG = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)

# Abaqus standard hourglass factor (x G)
_ABAQUS_HG_FACTOR = 0.005


@njit(fastmath=True)
def _jac_2d(xi: float, eta: float, coords: np.ndarray):
    """Shape function derivatives and Jacobian at natural coordinate (xi, eta)."""
    dN_dxi = 0.25 * np.array([-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)], dtype=np.float64)
    dN_deta = 0.25 * np.array([-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)], dtype=np.float64)
    J00 = np.sum(dN_dxi * coords[:, 0])
    J01 = np.sum(dN_dxi * coords[:, 1])
    J10 = np.sum(dN_deta * coords[:, 0])
    J11 = np.sum(dN_deta * coords[:, 1])
    detJ = J00 * J11 - J01 * J10
    if detJ <= 1e-15:
        detJ = 1e-15
    invJ00 =  J11 / detJ
    invJ01 = -J01 / detJ
    invJ10 = -J10 / detJ
    invJ11 =  J00 / detJ
    return detJ, invJ00, invJ01, invJ10, invJ11, dN_dxi, dN_deta


@njit(fastmath=True)
def q1e4_elastic_stiffness_numba(coords: np.ndarray, lam: float, mu: float):
    """Small-strain linear-elastic Q1E4 (CPE4I) stiffness matrix (8x8) for unit thickness."""
    D11 = lam + 2.0 * mu
    D12 = lam
    D33 = mu

    detJ0, invJ00, invJ01, invJ10, invJ11, _, _ = _jac_2d(0.0, 0.0, coords)

    K_uu = np.zeros((8, 8), dtype=np.float64)
    K_ua = np.zeros((8, 4), dtype=np.float64)
    K_aa = np.zeros((4, 4), dtype=np.float64)

    gp_val = 1.0 / np.sqrt(3.0)
    gp_pts = [(-gp_val, -gp_val),
              ( gp_val, -gp_val),
              ( gp_val,  gp_val),
              (-gp_val,  gp_val)]

    E0 = np.array([[ 1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    E1 = np.array([[ 0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    E2 = np.array([[ 0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    E3 = np.array([[ 0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    for xi, eta in gp_pts:
        detJ, inv00, inv01, inv10, inv11, dN_dxi, dN_deta = _jac_2d(xi, eta, coords)
        gX = inv00 * dN_dxi + inv01 * dN_deta
        gY = inv10 * dN_dxi + inv11 * dN_deta

        B = np.zeros((3, 8), dtype=np.float64)
        for a in range(4):
            B[0, 2 * a]     = gX[a]
            B[1, 2 * a + 1] = gY[a]
            B[2, 2 * a]     = gY[a]
            B[2, 2 * a + 1] = gX[a]

        fac = detJ0 / detJ
        G = np.zeros((3, 4), dtype=np.float64)
        modes = [xi * E0, eta * E1, eta * E2, xi * E3]

        for m in range(4):
            Em = modes[m]
            F00 = fac * (invJ00 * Em[0, 0] + invJ10 * Em[1, 0])
            F01 = fac * (invJ00 * Em[0, 1] + invJ10 * Em[1, 1])
            F10 = fac * (invJ01 * Em[0, 0] + invJ11 * Em[1, 0])
            F11 = fac * (invJ01 * Em[0, 1] + invJ11 * Em[1, 1])
            G[0, m] = F00
            G[1, m] = F11
            G[2, m] = F01 + F10

        DB = np.zeros((3, 8), dtype=np.float64)
        DG = np.zeros((3, 4), dtype=np.float64)
        for j in range(8):
            DB[0, j] = D11 * B[0, j] + D12 * B[1, j]
            DB[1, j] = D12 * B[0, j] + D11 * B[1, j]
            DB[2, j] = D33 * B[2, j]
        for j in range(4):
            DG[0, j] = D11 * G[0, j] + D12 * G[1, j]
            DG[1, j] = D12 * G[0, j] + D11 * G[1, j]
            DG[2, j] = D33 * G[2, j]

        for i in range(8):
            for j in range(8):
                K_uu[i, j] += (B[0, i] * DB[0, j] + B[1, i] * DB[1, j] + B[2, i] * DB[2, j]) * detJ
            for j in range(4):
                K_ua[i, j] += (B[0, i] * DG[0, j] + B[1, i] * DG[1, j] + B[2, i] * DG[2, j]) * detJ
        for i in range(4):
            for j in range(4):
                K_aa[i, j] += (G[0, i] * DG[0, j] + G[1, i] * DG[1, j] + G[2, i] * DG[2, j]) * detJ

    inv_Kaa = np.linalg.inv(K_aa)
    K_E = K_uu - K_ua @ inv_Kaa @ K_ua.T
    return K_E


@njit(fastmath=True)
def enhanced_hourglass_matrix_numba(coords: np.ndarray, lam: float, mu: float):
    """H/16 matrix (2x2) of Puso (2000) EAS hourglass formulation."""
    K_E = q1e4_elastic_stiffness_numba(coords, lam, mu)
    hx = np.zeros(8, dtype=np.float64)
    hy = np.zeros(8, dtype=np.float64)
    for a in range(4):
        hx[2 * a] = _HG[a]
        hy[2 * a + 1] = _HG[a]
    Hv = np.zeros((2, 8), dtype=np.float64)
    Hv[0] = hx
    Hv[1] = hy
    return (Hv @ K_E @ Hv.T) / 16.0


@njit(fastmath=True)
def compute_cpe4r_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (1, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes 8x8 Ke and 8-element fe for 1 CPE4R element with Enhanced/Stiffness
    Hourglass Control and Distortion Control.
    """
    # State variables of the (single) integration point; the hourglass and distortion constants come from the
    # material's INITIAL elastic tangent (Abaqus takes them from the elastic moduli, so they must not soften
    # with the plastic tangent), and the stress below from the current state.
    sdv_pt = sdvs[0] if (sdvs is not None and len(sdvs) > 0) else np.zeros(8, dtype=np.float64)
    _, C0, _, _ = material_dispatch_2d(mat_type, props, np.zeros(len(sdv_pt), dtype=np.float64),
                                       np.zeros(3, dtype=np.float64), 0.0)
    lam = C0[0, 1]
    mu = C0[2, 2]
    kappa = lam + mu  # 2D plane strain bulk modulus

    # 1. Centroid derivatives and geometry
    detJ0, invJ00, invJ01, invJ10, invJ11, dN_dxi, dN_deta = _jac_2d(0.0, 0.0, coords)
    area = 4.0 * detJ0
    has_error = 0
    if detJ0 <= 1e-15:
        has_error = 1

    bx = invJ00 * dN_dxi + invJ01 * dN_deta
    by = invJ10 * dN_dxi + invJ11 * dN_deta

    # 2. Centroid deformation gradient F0 (2, 2)
    ux = np.array([u_elem[0], u_elem[2], u_elem[4], u_elem[6]])
    uy = np.array([u_elem[1], u_elem[3], u_elem[5], u_elem[7]])
    F00 = 1.0 + np.sum(ux * bx)
    F01 = np.sum(ux * by)
    F10 = np.sum(uy * bx)
    F11 = 1.0 + np.sum(uy * by)
    detF0 = F00 * F11 - F01 * F10

    # 3. Total Lagrangian BL matrix (3, 8): delta E = sym(F^T delta F)
    BL = np.zeros((3, 8), dtype=np.float64)
    for a in range(4):
        BL[0, 2 * a]     = F00 * bx[a]
        BL[1, 2 * a]     = F01 * by[a]
        BL[2, 2 * a]     = F00 * by[a] + F01 * bx[a]
        BL[0, 2 * a + 1] = F10 * bx[a]
        BL[1, 2 * a + 1] = F11 * by[a]
        BL[2, 2 * a + 1] = F10 * by[a] + F11 * bx[a]

    # Green-Lagrange strain: E = 0.5 * (F^T F - I)
    C00 = F00 * F00 + F10 * F10
    C01 = F00 * F01 + F10 * F11
    C11 = F01 * F01 + F11 * F11
    E00 = 0.5 * (C00 - 1.0)
    E11 = 0.5 * (C11 - 1.0)
    E01 = C01  # engineering shear = 2 * E01

    # PK2 stress and material tangent from the material model at the Green-Lagrange strain (E01 is the
    # engineering shear); for a linear elastic material this is the St. Venant-Kirchhoff law as before.
    S_gl, Dm, sdv_new, mat_err = material_dispatch_2d(
        mat_type, props, sdv_pt, np.array([E00, E11, E01], dtype=np.float64), dt)
    if mat_err != 0:
        has_error = 1
    if sdvs is not None and len(sdvs) > 0:
        for s_i in range(min(len(sdv_new), len(sdvs[0]))):
            sdvs[0, s_i] = sdv_new[s_i]
    S0 = S_gl[0]
    S1 = S_gl[1]
    S2 = S_gl[2]
    S_v = np.array([S0, S1, S2], dtype=np.float64)

    # Internal force from 1-point material: fe = area * BL^T @ S_v
    fe = np.zeros(8, dtype=np.float64)
    for i in range(8):
        fe[i] = (BL[0, i] * S0 + BL[1, i] * S1 + BL[2, i] * S2) * area

    # 1-point material stiffness Ke_mat = area * BL^T @ D @ BL
    DBL = Dm @ BL

    Ke = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            Ke[i, j] = (BL[0, i] * DBL[0, j] + BL[1, i] * DBL[1, j] + BL[2, i] * DBL[2, j]) * area

    # Initial-stress (geometric) stiffness: grad_N @ S @ grad_N^T
    k_sig = np.zeros((4, 4), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            k_sig[a, b] = (bx[a] * S0 * bx[b] + bx[a] * S2 * by[b] + by[a] * S2 * bx[b] + by[a] * S1 * by[b]) * area
    for a in range(4):
        for b in range(4):
            Ke[2 * a,     2 * b]     += k_sig[a, b]
            Ke[2 * a + 1, 2 * b + 1] += k_sig[a, b]

    # 4. Universal Distortion Control (*SECTION CONTROLS, DISTORTION CONTROL=YES)
    use_distortion = False
    r_c = 0.10
    if elem_controls is not None and len(elem_controls) > 1:
        use_distortion = (elem_controls[0] > 0.5)
        if elem_controls[1] > 0.0:
            r_c = elem_controls[1]

    if use_distortion and detF0 < r_c:
        J_safe = max(detF0, 1e-12)
        # Potential: Psi_dist = 0.5 * kappa * ((r_c - J) / J)^2
        # Restoring pressure: p_dist = - dPsi/dJ = kappa * r_c * (r_c - J) / J^3
        p_dist = kappa * r_c * (r_c - J_safe) / (J_safe ** 3)
        invC00 =  C11 / (J_safe * J_safe)
        invC01 = -C01 / (J_safe * J_safe)
        invC11 =  C00 / (J_safe * J_safe)
        S_dist0 = -p_dist * J_safe * invC00
        S_dist1 = -p_dist * J_safe * invC11
        S_dist2 = -p_dist * J_safe * invC01
        for i in range(8):
            fe[i] += (BL[0, i] * S_dist0 + BL[1, i] * S_dist1 + BL[2, i] * S_dist2) * area

        c_p = kappa * r_c * (2.0 * r_c - J_safe) / (J_safe * J_safe)
        # Volumetric tangent injection
        for i in range(8):
            for j in range(8):
                tr_i = BL[0, i] * invC00 + BL[1, i] * invC11 + 2.0 * BL[2, i] * invC01
                tr_j = BL[0, j] * invC00 + BL[1, j] * invC11 + 2.0 * BL[2, j] * invC01
                Ke[i, j] += c_p * tr_i * tr_j * area

    # 5. Hourglass Control: Flanagan-Belytschko gamma vector
    hx = np.sum(_HG * coords[:, 0])
    hy = np.sum(_HG * coords[:, 1])
    gamma = _HG - hx * bx - hy * by

    hg_mode = 0.0  # default 0: ENHANCED, 1: STIFFNESS
    hg_scale = 1.0
    if elem_controls is not None and len(elem_controls) > 5:
        hg_mode = elem_controls[5]
    if elem_controls is not None and len(elem_controls) > 6 and elem_controls[6] > 0.0:
        hg_scale = elem_controls[6]

    if hg_mode == 1.0:
        # STIFFNESS Mode: Flanagan-Belytschko (1981) with Abaqus factor
        k_fb = hg_scale * _ABAQUS_HG_FACTOR * mu * area * (np.sum(bx * bx) + np.sum(by * by))
        qx = np.sum(gamma * ux)
        qy = np.sum(gamma * uy)
        for a in range(4):
            fe[2 * a + 0] += k_fb * qx * gamma[a]
            fe[2 * a + 1] += k_fb * qy * gamma[a]
            for b in range(4):
                Ke[2 * a + 0, 2 * b + 0] += k_fb * gamma[a] * gamma[b]
                Ke[2 * a + 1, 2 * b + 1] += k_fb * gamma[a] * gamma[b]
    else:
        # ENHANCED Mode: Puso (2000) Q1E4 EAS static condensation + polar rotation pull-back
        Hq = enhanced_hourglass_matrix_numba(coords, lam, mu)
        a_rot = 2.0 + np.sum(ux * bx) + np.sum(uy * by)
        b_rot = np.sum(uy * bx) - np.sum(ux * by)
        n_rot = np.sqrt(a_rot * a_rot + b_rot * b_rot)
        c_rot = a_rot / n_rot
        s_rot = b_rot / n_rot

        qx = np.sum(gamma * ux)
        qy = np.sum(gamma * uy)
        q_ref_0 = c_rot * qx + s_rot * qy
        q_ref_1 = -s_rot * qx + c_rot * qy

        t0 = Hq[0, 0] * q_ref_0 + Hq[0, 1] * q_ref_1
        t1 = Hq[1, 0] * q_ref_0 + Hq[1, 1] * q_ref_1
        fq0 = c_rot * t0 - s_rot * t1
        fq1 = s_rot * t0 + c_rot * t1
        tau_spin = t0 * q_ref_1 - t1 * q_ref_0

        vx = (-c_rot * by - s_rot * bx) / n_rot
        vy = ( c_rot * bx - s_rot * by) / n_rot

        for i in range(4):
            fe[2 * i + 0] += fq0 * gamma[i] + tau_spin * vx[i]
            fe[2 * i + 1] += fq1 * gamma[i] + tau_spin * vy[i]

        # Tangent stiffness K_hg (consistent polar variation)
        # Material rotated part: K_mat = R @ Hq @ R^T
        # R = [[c, -s], [s, c]], R^T = [[c, s], [-s, c]]
        # M = Hq @ R^T:
        # M00 = H00 * c - H01 * s, M01 = H00 * s + H01 * c
        # M10 = H10 * c - H11 * s, M11 = H10 * s + H11 * c
        # Kmat = R @ M:
        M00 = Hq[0, 0] * c_rot - Hq[0, 1] * s_rot
        M01 = Hq[0, 0] * s_rot + Hq[0, 1] * c_rot
        M10 = Hq[1, 0] * c_rot - Hq[1, 1] * s_rot
        M11 = Hq[1, 0] * s_rot + Hq[1, 1] * c_rot

        Kmat00 = c_rot * M00 - s_rot * M10
        Kmat01 = c_rot * M01 - s_rot * M11
        Kmat10 = s_rot * M00 + c_rot * M10
        Kmat11 = s_rot * M01 + c_rot * M11

        for a in range(4):
            for b in range(4):
                Ke[2 * a + 0, 2 * b + 0] += Kmat00 * gamma[a] * gamma[b]
                Ke[2 * a + 0, 2 * b + 1] += Kmat01 * gamma[a] * gamma[b]
                Ke[2 * a + 1, 2 * b + 0] += Kmat10 * gamma[a] * gamma[b]
                Ke[2 * a + 1, 2 * b + 1] += Kmat11 * gamma[a] * gamma[b]

    # Enforce exact symmetry of tangent stiffness
    Ke = 0.5 * (Ke + Ke.T)
    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4r_numba(
    node_coords_all: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float,
    elem_controls: np.ndarray = None,
    elem_stress_init: np.ndarray = None
):
    """Assembles all CPE4R elements in parallel with OpenMP prange."""
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 8), dtype=np.float64)
    k_elems = np.zeros((n_elems, 8, 8), dtype=np.float64)
    errors = np.zeros(n_elems, dtype=np.int32)

    for e in prange(n_elems):
        conn_e = elem_conn[e]
        coords_e = np.zeros((4, 2), dtype=np.float64)
        u_e = np.zeros(8, dtype=np.float64)
        for i in range(4):
            nid = conn_e[i]
            coords_e[i, 0] = node_coords_all[nid, 0]
            coords_e[i, 1] = node_coords_all[nid, 1]
            u_e[2 * i + 0] = u_global[2 * nid + 0]
            u_e[2 * i + 1] = u_global[2 * nid + 1]

        m_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        ctrl_e = elem_controls[e] if elem_controls is not None else None

        fe, Ke, err = compute_cpe4r_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
