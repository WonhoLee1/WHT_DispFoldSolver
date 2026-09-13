"""
cpe3_numba.py
=============
Numba JIT accelerated 3-node Linear Constant Strain Triangle Element (CPE3).
Single integration point (centroid).
Equivalents: Abaqus CPE3 / Ansys PLANE182 (Triangle).
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def compute_cpe3_element_numba(
    coords: np.ndarray,      # (3, 2)
    u_elem: np.ndarray,      # (6,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (1, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes 6x6 tangent stiffness Ke and 6-element internal force vector fe for 1 CPE3 element.
    """
    E = props[0]
    nu = props[1]

    # Plane strain elasticity matrix C (3x3)
    c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C11 = c_fac * (1.0 - nu)
    C12 = c_fac * nu
    C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

    Ke = np.zeros((6, 6), dtype=np.float64)
    fe = np.zeros(6, dtype=np.float64)

    # Coords of 3 vertices
    x1, y1 = coords[0, 0], coords[0, 1]
    x2, y2 = coords[1, 0], coords[1, 1]
    x3, y3 = coords[2, 0], coords[2, 1]

    # Determinant of Jacobian: 2 * Area
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    has_error = 0
    if detJ <= 1e-14:
        has_error = 1
        detJ = 1e-14

    area = 0.5 * detJ

    # Derivatives of shape functions w.r.t physical x, y:
    # N1 = ((x2*y3 - x3*y2) + (y2 - y3)*x + (x3 - x2)*y) / (2*Area)
    # N2 = ((x3*y1 - x1*y3) + (y3 - y1)*x + (x1 - x3)*y) / (2*Area)
    # N3 = ((x1*y2 - x2*y1) + (y1 - y2)*x + (x2 - x1)*y) / (2*Area)
    b1 = (y2 - y3) / detJ
    b2 = (y3 - y1) / detJ
    b3 = (y1 - y2) / detJ

    c1 = (x3 - x2) / detJ
    c2 = (x1 - x3) / detJ
    c3 = (x2 - x1) / detJ

    # B-matrix: (3, 6)
    B = np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3]
    ], dtype=np.float64)

    # In-plane strain: eps = B * u_elem (3,)
    eps = B @ u_elem

    # Stress: sigma = C * eps
    s_xx = C11 * eps[0] + C12 * eps[1]
    s_yy = C12 * eps[0] + C11 * eps[1]
    s_xy = C33 * eps[2]
    sigma = np.array([s_xx, s_yy, s_xy], dtype=np.float64)

    # fe = B^T * sigma * area
    for i in range(6):
        fe[i] = (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * area

    # CB = C * B
    CB = np.zeros((3, 6), dtype=np.float64)
    for j in range(6):
        CB[0, j] = C11 * B[0, j] + C12 * B[1, j]
        CB[1, j] = C12 * B[0, j] + C11 * B[1, j]
        CB[2, j] = C33 * B[2, j]

    for i in range(6):
        for j in range(6):
            Ke[i, j] = (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * area

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe3_numba(
    node_coords_all: np.ndarray,  # (N_nodes, 2)
    elem_conn: np.ndarray,        # (N_elems, 3)
    u_global: np.ndarray,         # (N_nodes * 2,)
    elem_mat_types: np.ndarray,   # (N_elems,)
    elem_props: np.ndarray,       # (N_elems, 36)
    elem_sdvs: np.ndarray,        # (N_elems, 1, 7)
    dt: float,
    elem_controls: np.ndarray = None,
    elem_stress_init: np.ndarray = None
):
    """Parallel assembly of all CPE3 elements in the mesh."""
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 6), dtype=np.float64)
    k_elems = np.zeros((n_elems, 6, 6), dtype=np.float64)
    errors = np.zeros(n_elems, dtype=np.int32)

    for e in prange(n_elems):
        conn_e = elem_conn[e]
        coords_e = np.zeros((3, 2), dtype=np.float64)
        u_e = np.zeros(6, dtype=np.float64)
        for i in range(3):
            nid = conn_e[i]
            coords_e[i, 0] = node_coords_all[nid, 0]
            coords_e[i, 1] = node_coords_all[nid, 1]
            u_e[2 * i + 0] = u_global[2 * nid + 0]
            u_e[2 * i + 1] = u_global[2 * nid + 1]

        m_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        ctrl_e = elem_controls[e] if elem_controls is not None else None

        fe, Ke, err = compute_cpe3_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
