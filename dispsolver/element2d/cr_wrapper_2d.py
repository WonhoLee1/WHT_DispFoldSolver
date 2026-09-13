"""
cr_wrapper_2d.py
================
Universal Co-Rotational (CR) Kinematic Wrapper for 2D Solid Elements.
Enables large-rotation, geometrically non-linear (NLGEOM) capability
for arbitrary 2D core element formulations:
- CPE4I_CR: 4-node Incompatible EAS Quad + Corotational
- CPE4H_CR: 4-node Mixed u-P Hybrid Quad + Corotational
- CPE4R_CR: 4-node Reduced-Integration Hourglass Quad + Corotational
- CPE6M_CR: 6-node Modified Quadratic Triangle + Corotational

Theory:
- Felippa & Haugen (2005) Unified Corotational Framework
- Extracts pure rigid body rotation R at the element level.
- Computes deformational local displacements: u_loc = R @ x_curr - X_ref.
- Evaluates core small-strain element tangent K_loc and internal force f_loc.
- Performs global pull-back transformation: f_glob = T^T f_loc, K_glob = T^T K_loc T.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange

from dispsolver.element2d.cpe4i_numba import compute_cpe4i_element_numba
from dispsolver.element2d.cpe4h_numba import compute_cpe4h_element_numba
from dispsolver.element2d.cpe4r_numba import compute_cpe4r_element_numba
from dispsolver.element2d.cpe6m_numba import compute_cpe6m_element_numba


# ==============================================================================
# 2D 4-node Quad Corotational Kinematics (CPE4I, CPE4H, CPE4R)
# ==============================================================================

@njit(fastmath=True)
def _extract_corotational_frame_quad2d(coords: np.ndarray, u_elem: np.ndarray):
    """Extract 2x2 rotation matrix R and 8-element deformational displacement u_loc for Quad."""
    x_curr = np.zeros((4, 2), dtype=np.float64)
    for a in range(4):
        x_curr[a, 0] = coords[a, 0] + u_elem[2 * a + 0]
        x_curr[a, 1] = coords[a, 1] + u_elem[2 * a + 1]

    # Reference and current side vectors (xi axis: midpoint of (1,2) minus midpoint of (0,3))
    v1_ref = 0.5 * ((coords[1] + coords[2]) - (coords[0] + coords[3]))
    v1_cur = 0.5 * ((x_curr[1] + x_curr[2]) - (x_curr[0] + x_curr[3]))

    len_cur = np.sqrt(v1_cur[0] * v1_cur[0] + v1_cur[1] * v1_cur[1])
    len_ref = np.sqrt(v1_ref[0] * v1_ref[0] + v1_ref[1] * v1_ref[1])
    if len_cur < 1e-14:
        len_cur = 1e-14
    if len_ref < 1e-14:
        len_ref = 1e-14

    e1_ref = v1_ref / len_ref
    e1_cur = v1_cur / len_cur

    cos_th = e1_cur[0] * e1_ref[0] + e1_cur[1] * e1_ref[1]
    sin_th = e1_cur[1] * e1_ref[0] - e1_cur[0] * e1_ref[1]

    # R transforms current coords to local orientation (cur -> local)
    R = np.array([
        [ cos_th, sin_th],
        [-sin_th, cos_th]
    ], dtype=np.float64)

    # Local deformational displacement: u_loc = R @ x_curr - coords
    u_loc = np.zeros(8, dtype=np.float64)
    for a in range(4):
        xl_a = R[0, 0] * x_curr[a, 0] + R[0, 1] * x_curr[a, 1]
        yl_a = R[1, 0] * x_curr[a, 0] + R[1, 1] * x_curr[a, 1]
        u_loc[2 * a + 0] = xl_a - coords[a, 0]
        u_loc[2 * a + 1] = yl_a - coords[a, 1]

    return R, u_loc, x_curr


@njit(fastmath=True)
def _transform_corotational_system_quad2d(R: np.ndarray, f_loc: np.ndarray, K_loc: np.ndarray):
    """Transform 8-element f_loc and 8x8 K_loc from local corotational frame to global frame."""
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
            Klab = K_loc[2 * a: 2 * a + 2, 2 * b: 2 * b + 2]
            Kab = RT @ Klab @ R
            Ke[2 * a: 2 * a + 2, 2 * b: 2 * b + 2] = Kab

    return fe, Ke


# ==============================================================================
# 2D 6-node Triangle Corotational Kinematics (CPE6M)
# ==============================================================================

@njit(fastmath=True)
def _extract_corotational_frame_tri6_2d(coords: np.ndarray, u_elem: np.ndarray):
    """Extract 2x2 rotation matrix R and 12-element deformational displacement u_loc for Tri6."""
    x_curr = np.zeros((6, 2), dtype=np.float64)
    for a in range(6):
        x_curr[a, 0] = coords[a, 0] + u_elem[2 * a + 0]
        x_curr[a, 1] = coords[a, 1] + u_elem[2 * a + 1]

    # Base side vector: from corner node 0 to corner node 1
    v1_ref = coords[1] - coords[0]
    v1_cur = x_curr[1] - x_curr[0]

    len_cur = np.sqrt(v1_cur[0] * v1_cur[0] + v1_cur[1] * v1_cur[1])
    len_ref = np.sqrt(v1_ref[0] * v1_ref[0] + v1_ref[1] * v1_ref[1])
    if len_cur < 1e-14:
        len_cur = 1e-14
    if len_ref < 1e-14:
        len_ref = 1e-14

    e1_ref = v1_ref / len_ref
    e1_cur = v1_cur / len_cur

    cos_th = e1_cur[0] * e1_ref[0] + e1_cur[1] * e1_ref[1]
    sin_th = e1_cur[1] * e1_ref[0] - e1_cur[0] * e1_ref[1]

    R = np.array([
        [ cos_th, sin_th],
        [-sin_th, cos_th]
    ], dtype=np.float64)

    u_loc = np.zeros(12, dtype=np.float64)
    for a in range(6):
        xl_a = R[0, 0] * x_curr[a, 0] + R[0, 1] * x_curr[a, 1]
        yl_a = R[1, 0] * x_curr[a, 0] + R[1, 1] * x_curr[a, 1]
        u_loc[2 * a + 0] = xl_a - coords[a, 0]
        u_loc[2 * a + 1] = yl_a - coords[a, 1]

    return R, u_loc, x_curr


@njit(fastmath=True)
def _transform_corotational_system_tri6_2d(R: np.ndarray, f_loc: np.ndarray, K_loc: np.ndarray):
    """Transform 12-element f_loc and 12x12 K_loc from local corotational frame to global frame."""
    RT = R.T
    fe = np.zeros(12, dtype=np.float64)
    for a in range(6):
        fl_x = f_loc[2 * a + 0]
        fl_y = f_loc[2 * a + 1]
        fe[2 * a + 0] = RT[0, 0] * fl_x + RT[0, 1] * fl_y
        fe[2 * a + 1] = RT[1, 0] * fl_x + RT[1, 1] * fl_y

    Ke = np.zeros((12, 12), dtype=np.float64)
    for a in range(6):
        for b in range(6):
            Klab = K_loc[2 * a: 2 * a + 2, 2 * b: 2 * b + 2]
            Kab = RT @ Klab @ R
            Ke[2 * a: 2 * a + 2, 2 * b: 2 * b + 2] = Kab

    return fe, Ke


# ==============================================================================
# Specific Element Compute Kernels (CPE4I_CR, CPE4H_CR, CPE4R_CR, CPE6M_CR)
# ==============================================================================

@njit(fastmath=True)
def compute_cpe4i_cr_element_numba(coords, u_elem, mat_type, props, sdvs, dt, elem_controls=None):
    R, u_loc, _ = _extract_corotational_frame_quad2d(coords, u_elem)
    f_loc, K_loc, err = compute_cpe4i_element_numba(coords, u_loc, mat_type, props, sdvs, dt, elem_controls)
    fe, Ke = _transform_corotational_system_quad2d(R, f_loc, K_loc)
    return fe, Ke, err


@njit(fastmath=True)
def compute_cpe4h_cr_element_numba(coords, u_elem, mat_type, props, sdvs, dt, elem_controls=None):
    R, u_loc, _ = _extract_corotational_frame_quad2d(coords, u_elem)
    f_loc, K_loc, err = compute_cpe4h_element_numba(coords, u_loc, mat_type, props, sdvs, dt, elem_controls)
    fe, Ke = _transform_corotational_system_quad2d(R, f_loc, K_loc)
    return fe, Ke, err


@njit(fastmath=True)
def compute_cpe4r_cr_element_numba(coords, u_elem, mat_type, props, sdvs, dt, elem_controls=None):
    R, u_loc, _ = _extract_corotational_frame_quad2d(coords, u_elem)
    f_loc, K_loc, err = compute_cpe4r_element_numba(coords, u_loc, mat_type, props, sdvs, dt, elem_controls)
    fe, Ke = _transform_corotational_system_quad2d(R, f_loc, K_loc)
    return fe, Ke, err


@njit(fastmath=True)
def compute_cpe6m_cr_element_numba(coords, u_elem, mat_type, props, sdvs, dt, elem_controls=None):
    R, u_loc, _ = _extract_corotational_frame_tri6_2d(coords, u_elem)
    f_loc, K_loc, err = compute_cpe6m_element_numba(coords, u_loc, mat_type, props, sdvs, dt, elem_controls)
    fe, Ke = _transform_corotational_system_tri6_2d(R, f_loc, K_loc)
    return fe, Ke, err


# ==============================================================================
# Parallel Mesh Assembly Dispatchers
# ==============================================================================

@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4i_cr_numba(node_coords_all, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls=None, elem_stress_init=None):
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
        fe, Ke, err = compute_cpe4i_cr_element_numba(coords_e, u_e, elem_mat_types[e], elem_props[e], elem_sdvs[e], dt, elem_controls[e] if elem_controls is not None else None)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err
    return f_elems, k_elems, int(np.sum(errors))


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4h_cr_numba(node_coords_all, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls=None, elem_stress_init=None):
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
        fe, Ke, err = compute_cpe4h_cr_element_numba(coords_e, u_e, elem_mat_types[e], elem_props[e], elem_sdvs[e], dt, elem_controls[e] if elem_controls is not None else None)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err
    return f_elems, k_elems, int(np.sum(errors))


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4r_cr_numba(node_coords_all, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls=None, elem_stress_init=None):
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
        fe, Ke, err = compute_cpe4r_cr_element_numba(coords_e, u_e, elem_mat_types[e], elem_props[e], elem_sdvs[e], dt, elem_controls[e] if elem_controls is not None else None)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err
    return f_elems, k_elems, int(np.sum(errors))


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe6m_cr_numba(node_coords_all, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls=None, elem_stress_init=None):
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 12), dtype=np.float64)
    k_elems = np.zeros((n_elems, 12, 12), dtype=np.float64)
    errors = np.zeros(n_elems, dtype=np.int32)
    for e in prange(n_elems):
        conn_e = elem_conn[e]
        coords_e = np.zeros((6, 2), dtype=np.float64)
        u_e = np.zeros(12, dtype=np.float64)
        for i in range(6):
            nid = conn_e[i]
            coords_e[i, 0] = node_coords_all[nid, 0]
            coords_e[i, 1] = node_coords_all[nid, 1]
            u_e[2 * i + 0] = u_global[2 * nid + 0]
            u_e[2 * i + 1] = u_global[2 * nid + 1]
        fe, Ke, err = compute_cpe6m_cr_element_numba(coords_e, u_e, elem_mat_types[e], elem_props[e], elem_sdvs[e], dt, elem_controls[e] if elem_controls is not None else None)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err
    return f_elems, k_elems, int(np.sum(errors))
