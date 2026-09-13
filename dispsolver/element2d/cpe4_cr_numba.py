"""
cpe4_cr_numba.py
================
Numba JIT accelerated 4-node Co-rotational Quadrilateral (CPE4_CR).
Felippa & Haugen (2005) unified small-strain corotational frame + Hughes B-bar.
SectionControls distortion control and anti-inversion support.
Equivalents: Abaqus CPE4 with *SECTION CONTROLS.
"""

import numpy as np
from numba import njit, prange

_GP1D = 1.0 / np.sqrt(3.0)
_GPS = np.array([
    [-_GP1D, -_GP1D],
    [ _GP1D, -_GP1D],
    [ _GP1D,  _GP1D],
    [-_GP1D,  _GP1D],
], dtype=np.float64)
_WTS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@njit(fastmath=True)
def _shape_derivs_cpe4_cr(xi: float, eta: float) -> np.ndarray:
    dN = np.zeros((2, 4), dtype=np.float64)
    dN[0, 0] = -0.25 * (1.0 - eta)
    dN[0, 1] =  0.25 * (1.0 - eta)
    dN[0, 2] =  0.25 * (1.0 + eta)
    dN[0, 3] = -0.25 * (1.0 + eta)

    dN[1, 0] = -0.25 * (1.0 - xi)
    dN[1, 1] = -0.25 * (1.0 + xi)
    dN[1, 2] =  0.25 * (1.0 + xi)
    dN[1, 3] =  0.25 * (1.0 - xi)
    return dN


@njit(fastmath=True)
def compute_cpe4_cr_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (4, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes 8x8 Ke and 8-element fe for 1 CPE4_CR element with corotational kinematics.
    """
    E = props[0]
    nu = props[1]
    if nu > 0.499999:
        nu = 0.499999

    c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C11 = c_fac * (1.0 - nu)
    C12 = c_fac * nu
    C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

    # Deformed coordinates: x = coords + u
    x_curr = np.zeros((4, 2), dtype=np.float64)
    for a in range(4):
        x_curr[a, 0] = coords[a, 0] + u_elem[2 * a + 0]
        x_curr[a, 1] = coords[a, 1] + u_elem[2 * a + 1]

    # Deformed side vectors to extract rigid rotation
    # Vector along xi-axis: from edge 4-1 to edge 2-3
    v1_ref = 0.5 * ((coords[1] + coords[2]) - (coords[0] + coords[3]))
    v1_cur = 0.5 * ((x_curr[1] + x_curr[2]) - (x_curr[0] + x_curr[3]))

    len_cur = np.sqrt(v1_cur[0] * v1_cur[0] + v1_cur[1] * v1_cur[1])
    len_ref = np.sqrt(v1_ref[0] * v1_ref[0] + v1_ref[1] * v1_ref[1])

    if len_cur < 1e-12:
        len_cur = 1e-12
    if len_ref < 1e-12:
        len_ref = 1e-12

    e1_ref = v1_ref / len_ref
    e1_cur = v1_cur / len_cur

    # Relative rotation angle theta
    cos_th = e1_cur[0] * e1_ref[0] + e1_cur[1] * e1_ref[1]
    sin_th = e1_cur[1] * e1_ref[0] - e1_cur[0] * e1_ref[1]

    # Rotation matrix R (cur -> ref/local)
    R = np.array([
        [ cos_th, sin_th],
        [-sin_th, cos_th]
    ], dtype=np.float64)

    # Transform current positions to local frame: x_local = x_curr @ R.T
    # Local displacement: u_local = x_local - coords
    u_loc = np.zeros(8, dtype=np.float64)
    for a in range(4):
        xl_a = R[0, 0] * x_curr[a, 0] + R[0, 1] * x_curr[a, 1]
        yl_a = R[1, 0] * x_curr[a, 0] + R[1, 1] * x_curr[a, 1]
        u_loc[2 * a + 0] = xl_a - coords[a, 0]
        u_loc[2 * a + 1] = yl_a - coords[a, 1]

    # Standard B-bar on local frame
    dN0 = _shape_derivs_cpe4_cr(0.0, 0.0)
    J0 = dN0 @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    has_error = 0
    if detJ0 <= 1e-14:
        has_error = 1
        detJ0 = 1e-14

    invJ00 =  J0[1, 1] / detJ0
    invJ01 = -J0[0, 1] / detJ0
    invJ10 = -J0[1, 0] / detJ0
    invJ11 =  J0[0, 0] / detJ0

    B_vol0 = np.zeros(8, dtype=np.float64)
    for a in range(4):
        bx0 = invJ00 * dN0[0, a] + invJ01 * dN0[1, a]
        by0 = invJ10 * dN0[0, a] + invJ11 * dN0[1, a]
        B_vol0[2 * a + 0] = bx0
        B_vol0[2 * a + 1] = by0

    K_loc = np.zeros((8, 8), dtype=np.float64)
    f_loc = np.zeros(8, dtype=np.float64)

    B = np.zeros((3, 8), dtype=np.float64)
    B_bar = np.zeros((3, 8), dtype=np.float64)

    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        wt = _WTS[q]

        dN_dxi = _shape_derivs_cpe4_cr(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
            has_error = 1
            detJ = 1e-14

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        B.fill(0.0)
        for a in range(4):
            bx = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            by = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]
            B[0, 2 * a + 0] = bx
            B[1, 2 * a + 1] = by
            B[2, 2 * a + 0] = by
            B[2, 2 * a + 1] = bx

        for j in range(8):
            trB = (B[0, j] + B[1, j]) * 0.5
            trB0 = B_vol0[j] * 0.5
            B_bar[0, j] = B[0, j] - trB + trB0
            B_bar[1, j] = B[1, j] - trB + trB0
            B_bar[2, j] = B[2, j]

        eps = B_bar @ u_loc
        s_xx = C11 * eps[0] + C12 * eps[1]
        s_yy = C12 * eps[0] + C11 * eps[1]
        s_xy = C33 * eps[2]
        sigma = np.array([s_xx, s_yy, s_xy], dtype=np.float64)

        dV = detJ * wt

        for i in range(8):
            f_loc[i] += (B_bar[0, i] * sigma[0] + B_bar[1, i] * sigma[1] + B_bar[2, i] * sigma[2]) * dV

        CB = np.zeros((3, 8), dtype=np.float64)
        for j in range(8):
            CB[0, j] = C11 * B_bar[0, j] + C12 * B_bar[1, j]
            CB[1, j] = C12 * B_bar[0, j] + C11 * B_bar[1, j]
            CB[2, j] = C33 * B_bar[2, j]

        for i in range(8):
            for j in range(8):
                K_loc[i, j] += (B_bar[0, i] * CB[0, j] + B_bar[1, i] * CB[1, j] + B_bar[2, i] * CB[2, j]) * dV

    # Global transformation: T_block (8x8) consists of R.T blocks
    # f_global = T^T * f_loc, Ke_global = T^T * K_loc * T
    # Note: R transforms (x_curr -> x_local), so R.T transforms (local -> global).
    RT = R.T
    fe = np.zeros(8, dtype=np.float64)
    for a in range(4):
        fl_x = f_loc[2 * a + 0]
        fl_y = f_loc[2 * a + 1]
        fe[2 * a + 0] = RT[0, 0] * fl_x + RT[0, 1] * fl_y
        fe[2 * a + 1] = RT[1, 0] * fl_x + RT[1, 1] * fl_y

    Ke = np.zeros((8, 8), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            # K_ab block (2x2) = RT * K_loc_ab * R
            Klab = K_loc[2 * a: 2 * a + 2, 2 * b: 2 * b + 2]
            Kab = RT @ Klab @ R
            Ke[2 * a: 2 * a + 2, 2 * b: 2 * b + 2] = Kab

    # SectionControls: distortion control barrier
    if elem_controls is not None and len(elem_controls) > 3 and elem_controls[3] > 0.0:
        # Anti-inversion penalty if current area ratio < min_det
        # detJ_cur
        J_cur = dN0 @ x_curr
        detJ_cur = J_cur[0, 0] * J_cur[1, 1] - J_cur[0, 1] * J_cur[1, 0]
        area_ratio = detJ_cur / detJ0
        min_det = elem_controls[4] if len(elem_controls) > 4 else 0.02
        if area_ratio < min_det:
            has_error = 1

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4_cr_numba(
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

        fe, Ke, err = compute_cpe4_cr_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
