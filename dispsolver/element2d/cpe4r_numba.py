"""
cpe4r_numba.py
==============
Numba JIT accelerated 4-node Reduced Integration Quadrilateral (CPE4R).
Single centroid Gauss point + Flanagan & Belytschko (1981) orthogonal hourglass control.
Equivalents: Abaqus CPE4R / LS-DYNA ELFORM 1.
"""

import numpy as np
from numba import njit, prange


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
    Computes 8x8 Ke and 8-element fe for 1 CPE4R element with orthogonal hourglass control.
    """
    E = props[0]
    nu = props[1]
    G = E / (2.0 * (1.0 + nu))

    # Plane strain C matrix (3x3)
    c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C11 = c_fac * (1.0 - nu)
    C12 = c_fac * nu
    C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

    # Centroid derivatives (xi=0, eta=0)
    # dN/dxi = [-0.25, 0.25, 0.25, -0.25]
    # dN/deta = [-0.25, -0.25, 0.25, 0.25]
    dN_dxi = np.array([
        [-0.25,  0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25,  0.25]
    ], dtype=np.float64)

    J0 = dN_dxi @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]

    has_error = 0
    if detJ0 <= 1e-14:
        has_error = 1
        detJ0 = 1e-14

    area = 4.0 * detJ0

    invJ00 =  J0[1, 1] / detJ0
    invJ01 = -J0[0, 1] / detJ0
    invJ10 = -J0[1, 0] / detJ0
    invJ11 =  J0[0, 0] / detJ0

    bx = np.zeros(4, dtype=np.float64)
    by = np.zeros(4, dtype=np.float64)
    for a in range(4):
        bx[a] = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
        by[a] = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]

    # B-matrix at centroid (3, 8)
    B = np.zeros((3, 8), dtype=np.float64)
    for a in range(4):
        B[0, 2 * a + 0] = bx[a]
        B[1, 2 * a + 1] = by[a]
        B[2, 2 * a + 0] = by[a]
        B[2, 2 * a + 1] = bx[a]

    # Centroid strain and stress
    eps = B @ u_elem
    s_xx = C11 * eps[0] + C12 * eps[1]
    s_yy = C12 * eps[0] + C11 * eps[1]
    s_xy = C33 * eps[2]
    sigma = np.array([s_xx, s_yy, s_xy], dtype=np.float64)

    # Internal force from 1-point integration: fe = area * B^T * sigma
    fe = np.zeros(8, dtype=np.float64)
    for i in range(8):
        fe[i] = (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * area

    # 1-point stiffness: Ke_0 = area * B^T * C * B
    CB = np.zeros((3, 8), dtype=np.float64)
    for j in range(8):
        CB[0, j] = C11 * B[0, j] + C12 * B[1, j]
        CB[1, j] = C12 * B[0, j] + C11 * B[1, j]
        CB[2, j] = C33 * B[2, j]

    Ke = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            Ke[i, j] = (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * area

    # -------------------------------------------------------------
    # Flanagan & Belytschko (1981) Orthogonal Hourglass Stabilization
    # -------------------------------------------------------------
    h = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    hx = 0.0
    hy = 0.0
    for a in range(4):
        hx += h[a] * coords[a, 0]
        hy += h[a] * coords[a, 1]

    # gamma = h - hx * bx - hy * by
    gamma = np.zeros(4, dtype=np.float64)
    for a in range(4):
        gamma[a] = h[a] - hx * bx[a] - hy * by[a]

    # Hourglass stiffness scaling (SectionControls or default 0.05)
    hg_factor = 0.05
    if elem_controls is not None and len(elem_controls) > 0:
        hg_factor = elem_controls[0] if elem_controls[0] > 0.0 else 0.05

    # Stabilization modulus Q_hg
    gamma_sq = np.sum(gamma * gamma)
    if gamma_sq < 1e-12:
        gamma_sq = 1e-12
    Q_hg = hg_factor * G * area * (1.0 / gamma_sq)

    # Displacements in x and y
    ux = np.array([u_elem[0], u_elem[2], u_elem[4], u_elem[6]], dtype=np.float64)
    uy = np.array([u_elem[1], u_elem[3], u_elem[5], u_elem[7]], dtype=np.float64)

    qh_x = np.sum(gamma * ux)
    qh_y = np.sum(gamma * uy)

    # Hourglass forces: f_hg = Q_hg * qh * gamma
    for a in range(4):
        fe[2 * a + 0] += Q_hg * qh_x * gamma[a]
        fe[2 * a + 1] += Q_hg * qh_y * gamma[a]

    # Hourglass stiffness: Ke_hg
    for a in range(4):
        for b in range(4):
            Ke[2 * a + 0, 2 * b + 0] += Q_hg * gamma[a] * gamma[b]
            Ke[2 * a + 1, 2 * b + 1] += Q_hg * gamma[a] * gamma[b]

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
