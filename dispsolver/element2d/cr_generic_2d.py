"""
cr_generic_2d.py
================
Topology-independent co-rotational wrapper for 2D plane small-strain kernels
under NLGEOM -- the 2D counterpart of element3d/cr_generic_3d.py (see that
module for the scheme and its limits).

The in-plane rotation of an element is the polar rotation of its least-squares
deformation gradient F = (sum dx dX^T)(sum dX dX^T)^-1 about the centroids:
theta = atan2(F10 - F01, F00 + F11). Exact for any homogeneous motion x = R U X
whatever the node layout, independent of node numbering. A two-node truss has
a singular node matrix and uses the best-fit axis angle atan2(sum dX x dx,
sum dX . dx) (2D orthogonal Procrustes), which is also what this module used
for every element before 2026-10-18; it equals the polar rotation only for an
isotropic node set and was off by O(strain) on triangles.

Plane elements only. An axisymmetric element cannot be wrapped: a rotation in
the r-z plane changes the radius of its nodes, so it is not a rigid motion
of the ring and carries real hoop strain.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def corotational_local_mesh_2d(node_coords: np.ndarray, conn: np.ndarray, u_global: np.ndarray):
    """Expanded per-element mesh in co-rotated frames.

    Returns (X_exp (ne*nn, 2), conn_exp (ne, nn), u_exp (2*ne*nn,), cs (ne, 2) = (cos, sin)).
    """
    ne, nn = conn.shape
    X_exp = np.zeros((ne * nn, 2))
    u_exp = np.zeros(2 * ne * nn)
    conn_exp = np.zeros((ne, nn), dtype=np.int64)
    cs = np.zeros((ne, 2))
    for e in prange(ne):
        Xc0 = 0.0
        Xc1 = 0.0
        xc0 = 0.0
        xc1 = 0.0
        for a in range(nn):
            g = conn[e, a]
            Xc0 += node_coords[g, 0]
            Xc1 += node_coords[g, 1]
            xc0 += node_coords[g, 0] + u_global[2 * g]
            xc1 += node_coords[g, 1] + u_global[2 * g + 1]
        Xc0 /= nn
        Xc1 /= nn
        xc0 /= nn
        xc1 /= nn
        sd = 0.0
        sc = 0.0
        m00 = 0.0
        m01 = 0.0
        m11 = 0.0
        p00 = 0.0
        p01 = 0.0
        p10 = 0.0
        p11 = 0.0
        for a in range(nn):
            g = conn[e, a]
            dX0 = node_coords[g, 0] - Xc0
            dX1 = node_coords[g, 1] - Xc1
            dx0 = node_coords[g, 0] + u_global[2 * g] - xc0
            dx1 = node_coords[g, 1] + u_global[2 * g + 1] - xc1
            sd += dX0 * dx0 + dX1 * dx1
            sc += dX0 * dx1 - dX1 * dx0
            m00 += dX0 * dX0
            m01 += dX0 * dX1
            m11 += dX1 * dX1
            p00 += dx0 * dX0
            p01 += dx0 * dX1
            p10 += dx1 * dX0
            p11 += dx1 * dX1
        # Frame = rotation of the least-squares deformation gradient F = P M^-1 (P = sum dx dX^T,
        # M = sum dX dX^T), i.e. the polar rotation of the element's mean motion. Exact for any homogeneous
        # x = R U X whatever the node layout. The best-fit angle of the node cloud (atan2(sc, sd)) equals it
        # only for an isotropic node set (a square); on a triangle it is off by O(strain). A two-node truss
        # has a singular M and keeps the axis-rotation formula.
        det_m = m00 * m11 - m01 * m01
        tr_m = m00 + m11
        if det_m > 1e-12 * tr_m * tr_m:
            i00 = m11 / det_m
            i01 = -m01 / det_m
            i11 = m00 / det_m
            f00 = p00 * i00 + p01 * i01
            f01 = p00 * i01 + p01 * i11
            f10 = p10 * i00 + p11 * i01
            f11 = p10 * i01 + p11 * i11
            th = np.arctan2(f10 - f01, f00 + f11)
        else:
            th = np.arctan2(sc, sd)
        c = np.cos(th)
        s = np.sin(th)
        cs[e, 0] = c
        cs[e, 1] = s
        for a in range(nn):
            g = conn[e, a]
            row = e * nn + a
            conn_exp[e, a] = row
            X_exp[row, 0] = node_coords[g, 0]
            X_exp[row, 1] = node_coords[g, 1]
            dx0 = node_coords[g, 0] + u_global[2 * g] - xc0
            dx1 = node_coords[g, 1] + u_global[2 * g + 1] - xc1
            # R^T dx - dX
            u_exp[2 * row] = (c * dx0 + s * dx1) - (node_coords[g, 0] - Xc0)
            u_exp[2 * row + 1] = (-s * dx0 + c * dx1) - (node_coords[g, 1] - Xc1)
    return X_exp, conn_exp, u_exp, cs


@njit(parallel=True, cache=True)
def corotational_to_global_2d(f_elems: np.ndarray, K_elems: np.ndarray, cs: np.ndarray) -> None:
    """In place: f <- T f, K <- T K T^T, T = blockdiag(R), R = [[c, -s], [s, c]]."""
    ne, nd = f_elems.shape
    nn = nd // 2
    for e in prange(ne):
        c = cs[e, 0]
        s = cs[e, 1]
        R = np.array([[c, -s], [s, c]])
        f_new = np.zeros(nd)
        for a in range(nn):
            f_new[2 * a] = c * f_elems[e, 2 * a] - s * f_elems[e, 2 * a + 1]
            f_new[2 * a + 1] = s * f_elems[e, 2 * a] + c * f_elems[e, 2 * a + 1]
        f_elems[e, :] = f_new
        Kt = np.zeros((nd, nd))
        for a in range(nn):
            for b in range(nn):
                for i in range(2):
                    for j in range(2):
                        v = 0.0
                        for p in range(2):
                            for q in range(2):
                                v += R[i, p] * K_elems[e, 2 * a + p, 2 * b + q] * R[j, q]
                        Kt[2 * a + i, 2 * b + j] = v
        K_elems[e, :, :] = Kt
