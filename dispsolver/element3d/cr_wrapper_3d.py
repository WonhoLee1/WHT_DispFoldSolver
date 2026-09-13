"""
cr_wrapper_3d.py
================
Universal Co-Rotational (CR) Kinematic Wrapper for 3D Solid Elements.
Enables large-rotation, geometrically non-linear (NLGEOM) capability
for arbitrary 3D core element formulations:
- C3D8I_CR: 8-node Incompatible EAS Hexahedron + Corotational
- C3D8H_CR: 8-node Mixed u-P Hybrid Hexahedron + Corotational
- C3D8R_CR: 8-node Reduced-Integration Hourglass Hexahedron + Corotational

Theory:
- Felippa & Haugen (2005) & Belytschko & Hsieh (1973)
- Extracts rigid body rotation R from deformed natural triads.
- Computes deformational local displacements: u_local = R^T (x - x_c) - (X - X_c).
- Evaluates core small-strain element tangent K_loc and internal force f_loc.
- Performs global pull-back transformation: f_glob = T f_loc, K_glob = T K_loc T^T.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange

from dispsolver.element3d.c3d8_corotational_numba import compute_element_rotation_3d
from dispsolver.element3d.c3d8_eas_numba import compute_c3d8_eas_element_numba
from dispsolver.element3d.c3d8_hybrid_numba import compute_c3d8_hybrid_element_umat_numba, assemble_mesh_c3d8_hybrid_numba
from dispsolver.element3d.c3d8r_numba import compute_c3d8r_element_numba


# ==============================================================================
# 3D 8-node Hex Corotational Kinematics
# ==============================================================================

@njit(fastmath=True)
def _extract_corotational_frame_hex3d(coords: np.ndarray, u_elem: np.ndarray):
    """Extract 3x3 rotation matrix R and 24-element deformational displacement u_local for Hex8."""
    coords_curr = np.zeros((8, 3), dtype=np.float64)
    for i in range(8):
        coords_curr[i, 0] = coords[i, 0] + u_elem[3 * i + 0]
        coords_curr[i, 1] = coords[i, 1] + u_elem[3 * i + 1]
        coords_curr[i, 2] = coords[i, 2] + u_elem[3 * i + 2]

    # R is orthogonal rotation matrix (ref triad -> cur triad)
    R = compute_element_rotation_3d(coords, coords_curr)

    # Reference and current centroids
    Xc = np.zeros(3, dtype=np.float64)
    xc = np.zeros(3, dtype=np.float64)
    for i in range(8):
        Xc += coords[i]
        xc += coords_curr[i]
    Xc *= 0.125
    xc *= 0.125

    # Pure deformational displacement in local element frame
    u_local = np.zeros(24, dtype=np.float64)
    for i in range(8):
        d_curr = coords_curr[i] - xc
        d_ref = coords[i] - Xc
        # u_local_node = R.T @ d_curr - d_ref
        u_local[3 * i + 0] = R[0, 0] * d_curr[0] + R[1, 0] * d_curr[1] + R[2, 0] * d_curr[2] - d_ref[0]
        u_local[3 * i + 1] = R[0, 1] * d_curr[0] + R[1, 1] * d_curr[1] + R[2, 1] * d_curr[2] - d_ref[1]
        u_local[3 * i + 2] = R[0, 2] * d_curr[0] + R[1, 2] * d_curr[1] + R[2, 2] * d_curr[2] - d_ref[2]

    return R, u_local, coords_curr


@njit(fastmath=True)
def _transform_corotational_system_hex3d(R: np.ndarray, f_local: np.ndarray, K_local: np.ndarray):
    """Transform 24-element f_local and 24x24 K_local from local corotational frame to global frame."""
    f_global = np.zeros(24, dtype=np.float64)
    K_global = np.zeros((24, 24), dtype=np.float64)

    for a in range(8):
        f_a_l = f_local[3 * a : 3 * a + 3]
        f_global[3 * a + 0] = R[0, 0] * f_a_l[0] + R[0, 1] * f_a_l[1] + R[0, 2] * f_a_l[2]
        f_global[3 * a + 1] = R[1, 0] * f_a_l[0] + R[1, 1] * f_a_l[1] + R[1, 2] * f_a_l[2]
        f_global[3 * a + 2] = R[2, 0] * f_a_l[0] + R[2, 1] * f_a_l[1] + R[2, 2] * f_a_l[2]

        for b in range(8):
            K_lab = K_local[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
            K_ab = R @ K_lab @ R.T
            K_global[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = K_ab

    return f_global, K_global


# ==============================================================================
# Specific Element Compute Kernels (C3D8I_CR, C3D8H_CR, C3D8R_CR)
# ==============================================================================

@njit(fastmath=True)
def compute_c3d8i_cr_element_numba(coords, u_elem, mat_type, props, sdvs, dt, elem_controls=None, elem_stress_init=None):
    """Compute 3D C3D8I EAS element with Co-rotational kinematics."""
    R, u_loc, _ = _extract_corotational_frame_hex3d(coords, u_elem)
    
    # Constitutive matrix C_mat (isotropic 3D)
    E = props[0]
    nu = props[1]
    if nu > 0.499999:
        nu = 0.499999
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C_mat = np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],
        [lam,        lam + 2*mu, lam,        0,  0,  0],
        [lam,        lam,        lam + 2*mu, 0,  0,  0],
        [0,          0,          0,          mu, 0,  0],
        [0,          0,          0,          0,  mu, 0],
        [0,          0,          0,          0,  0,  mu]
    ], dtype=np.float64)

    K_loc, f_loc, _ = compute_c3d8_eas_element_numba(coords, u_loc, C_mat)
    f_glob, K_glob = _transform_corotational_system_hex3d(R, f_loc, K_loc)
    return f_glob, K_glob, 0


@njit(fastmath=True)
def compute_c3d8h_cr_element_numba(coords, u_elem, mat_type, props, sdvs, dt, elem_controls=None, elem_stress_init=None):
    """C3D8H already has corotational kinematics embedded inside compute_c3d8_hybrid_element_umat_numba."""
    ctrl = elem_controls if elem_controls is not None else np.empty(0, dtype=np.float64)
    stress_init = elem_stress_init if elem_stress_init is not None else np.empty((0, 6), dtype=np.float64)
    return compute_c3d8_hybrid_element_umat_numba(coords, u_elem, mat_type, props, sdvs, dt, ctrl, stress_init)


@njit(fastmath=True)
def compute_c3d8r_cr_element_numba(coords, u_elem, mat_type, props, sdvs, dt, elem_controls=None, elem_stress_init=None):
    """Compute 3D C3D8R Reduced Integration element with Co-rotational kinematics."""
    R, u_loc, _ = _extract_corotational_frame_hex3d(coords, u_elem)
    ctrl = elem_controls if elem_controls is not None else np.empty(0, dtype=np.float64)
    K_loc, f_loc, err = compute_c3d8r_element_numba(coords, u_loc, mat_type, props, sdvs, dt, ctrl)
    f_glob, K_glob = _transform_corotational_system_hex3d(R, f_loc, K_loc)
    return f_glob, K_glob, err


# ==============================================================================
# Parallel Mesh Assembly Dispatchers
# ==============================================================================

@njit(parallel=True, fastmath=True)
def assemble_mesh_c3d8i_cr_numba(node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls=None, elem_stress_init=None):
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
    err_flags = np.zeros(n_elems, dtype=np.int32)
    for e in prange(n_elems):
        conn = elem_conn[e]
        elem_nodes = np.zeros((8, 3), dtype=np.float64)
        u_elem = np.zeros(24, dtype=np.float64)
        for i in range(8):
            nid = conn[i]
            elem_nodes[i, :] = node_coords[nid, :]
            u_elem[3*i:3*i+3] = u_global[3*nid:3*nid+3]
        f_e, K_e, err = compute_c3d8i_cr_element_numba(elem_nodes, u_elem, elem_mat_types[e], elem_props[e], elem_sdvs[e], dt)
        f_elems[e] = f_e
        K_elems[e] = K_e
        err_flags[e] = err
    return f_elems, K_elems, int(np.sum(err_flags))


@njit(parallel=True, fastmath=True)
def assemble_mesh_c3d8h_cr_numba(node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls=None, elem_stress_init=None):
    return assemble_mesh_c3d8_hybrid_numba(node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls, elem_stress_init)


@njit(parallel=True, fastmath=True)
def assemble_mesh_c3d8r_cr_numba(node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs, dt, elem_controls=None, elem_stress_init=None):
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
    err_flags = np.zeros(n_elems, dtype=np.int32)
    for e in prange(n_elems):
        conn = elem_conn[e]
        elem_nodes = np.zeros((8, 3), dtype=np.float64)
        u_elem = np.zeros(24, dtype=np.float64)
        for i in range(8):
            nid = conn[i]
            elem_nodes[i, :] = node_coords[nid, :]
            u_elem[3*i:3*i+3] = u_global[3*nid:3*nid+3]
        ctrl = elem_controls[e] if elem_controls is not None else None
        f_e, K_e, err = compute_c3d8r_cr_element_numba(elem_nodes, u_elem, elem_mat_types[e], elem_props[e], elem_sdvs[e], dt, ctrl)
        f_elems[e] = f_e
        K_elems[e] = K_e
        err_flags[e] = err
    return f_elems, K_elems, int(np.sum(err_flags))
