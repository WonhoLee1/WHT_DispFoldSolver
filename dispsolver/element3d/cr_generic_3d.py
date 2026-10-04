"""
cr_generic_3d.py
================
Topology-independent co-rotational (CR) wrapper for 3D small-strain element
kernels, used when NLGEOM is on.

Why: the tetrahedral, wedge and 20-node kernels (C3D4, C3D10, C3D10M, C3D6,
C3D20) are small-strain formulations. Under NLGEOM=YES the solver used to run
them unchanged, so a 30 deg rigid rotation of one element produced internal
forces of the same order as a 1 % stretch (element NLGEOM probe, 2026-10-04):
large-deformation analyses with these elements were not objective.

How (same scheme as cr_wrapper_3d.py's hex wrappers, Belytschko & Hsieh 1973,
Felippa & Haugen 2005, generalised to any node count):
  1. Element rigid rotation R = the rotation that best maps the reference node
     cloud onto the current one about their centroids (Kabsch / orthogonal
     Procrustes: R = polar factor of H = sum (x_a - x_c)(X_a - X_c)^T). It is
     exact for a rigid motion and does not depend on node numbering.
  2. Local deformational displacement u_l,a = R^T (x_a - x_c) - (X_a - X_c).
  3. The unchanged small-strain kernel is evaluated on (X, u_l): to reuse the
     mesh-level kernels without touching them, each element gets its own copy
     of its nodes ("expanded mesh"), so node sharing cannot mix local frames.
  4. f = T f_l, K = T K_l T^T with T = blockdiag(R).

K is the material part only (the dT/du geometric term is dropped), i.e. the
same modified-Newton tangent the existing CR wrappers use; the residual is
exact. Strains stay small in the co-rotated frame, so this is a large-rotation
/ small-strain formulation, like kernel 1 (C3D8_CR).

Not usable for kernels that couple elements through shared nodes (C3D4 ANP's
nodal pressure averaging): the expanded mesh would silently turn the
averaging into a per-element one.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from dispsolver.element3d.c3d8_eas_numba import compute_c3d8_eas_element_numba


@njit(cache=True)
def _kabsch_rotation(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Proper rotation R minimising sum |R (X_a - X_c) - (x_a - x_c)|^2."""
    n = X.shape[0]
    Xc = np.zeros(3)
    xc = np.zeros(3)
    for a in range(n):
        Xc += X[a]
        xc += x[a]
    Xc /= n
    xc /= n
    H = np.zeros((3, 3))
    for a in range(n):
        dX = X[a] - Xc
        dx = x[a] - xc
        for i in range(3):
            for j in range(3):
                H[i, j] += dx[i] * dX[j]
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    D = np.eye(3)
    if d < 0.0:
        D[2, 2] = -1.0
    return U @ D @ Vt


@njit(parallel=True, cache=True)
def corotational_local_mesh(node_coords: np.ndarray, conn: np.ndarray, u_global: np.ndarray):
    """Expanded per-element mesh in the co-rotated frames.

    Returns (X_exp (ne*nn, 3), conn_exp (ne, nn), u_exp (3*ne*nn,), R (ne, 3, 3)).
    """
    ne, nn = conn.shape
    X_exp = np.zeros((ne * nn, 3))
    u_exp = np.zeros(3 * ne * nn)
    conn_exp = np.zeros((ne, nn), dtype=np.int64)
    R_all = np.zeros((ne, 3, 3))
    for e in prange(ne):
        X = np.zeros((nn, 3))
        x = np.zeros((nn, 3))
        for a in range(nn):
            g = conn[e, a]
            for i in range(3):
                X[a, i] = node_coords[g, i]
                x[a, i] = node_coords[g, i] + u_global[3 * g + i]
        R = _kabsch_rotation(X, x)
        R_all[e] = R
        Xc = np.zeros(3)
        xc = np.zeros(3)
        for a in range(nn):
            Xc += X[a]
            xc += x[a]
        Xc /= nn
        xc /= nn
        for a in range(nn):
            row = e * nn + a
            conn_exp[e, a] = row
            dx = x[a] - xc
            for i in range(3):
                X_exp[row, i] = X[a, i]
                # (R^T dx)_i - (X_a - X_c)_i
                v = R[0, i] * dx[0] + R[1, i] * dx[1] + R[2, i] * dx[2]
                u_exp[3 * row + i] = v - (X[a, i] - Xc[i])
    return X_exp, conn_exp, u_exp, R_all


@njit(parallel=True, cache=True)
def corotational_to_global(f_elems: np.ndarray, K_elems: np.ndarray, R_all: np.ndarray) -> None:
    """In place: f <- T f, K <- T K T^T, T = blockdiag(R) per element."""
    ne, nd = f_elems.shape
    nn = nd // 3
    for e in prange(ne):
        R = R_all[e]
        f_new = np.zeros(nd)
        for a in range(nn):
            for i in range(3):
                s = 0.0
                for j in range(3):
                    s += R[i, j] * f_elems[e, 3 * a + j]
                f_new[3 * a + i] = s
        f_elems[e, :] = f_new
        # K <- T K T^T, blockwise: K_ab <- R K_ab R^T
        Kt = np.zeros((nd, nd))
        for a in range(nn):
            for b in range(nn):
                for i in range(3):
                    for j in range(3):
                        s = 0.0
                        for p in range(3):
                            for q in range(3):
                                s += R[i, p] * K_elems[e, 3 * a + p, 3 * b + q] * R[j, q]
                        Kt[3 * a + i, 3 * b + j] = s
        K_elems[e, :, :] = Kt


@njit(parallel=True, cache=True)
def assemble_mesh_c3d8i_small_strain_numba(node_coords: np.ndarray, conn: np.ndarray, u_global: np.ndarray,
                                           C_all: np.ndarray):
    """Small-strain C3D8I (incompatible modes / EAS) for NLGEOM off: the core
    the C3D8I_CR wrapper evaluates in its co-rotated frame, called directly
    (no rotation). C_all (ne, 6, 6): each element's linear elastic matrix."""
    ne = conn.shape[0]
    f_elems = np.zeros((ne, 24))
    K_elems = np.zeros((ne, 24, 24))
    for e in prange(ne):
        X = np.zeros((8, 3))
        ue = np.zeros(24)
        for a in range(8):
            g = conn[e, a]
            for i in range(3):
                X[a, i] = node_coords[g, i]
                ue[3 * a + i] = u_global[3 * g + i]
        K, f, _ = compute_c3d8_eas_element_numba(X, ue, C_all[e])
        f_elems[e] = f
        K_elems[e] = K
    return f_elems, K_elems
