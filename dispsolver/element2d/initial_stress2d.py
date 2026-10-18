"""Internal force of an initial stress field, f_i = int (dN_i/dx) . s0 dA, for the 2D plane-strain kernels
(element_catalog.KernelInfo.initial_stress == "added"; unit thickness).

No 2D kernel takes the field itself, so the assembly adds this term. The tangent is unchanged: the
initial-stress stiffness is not formed, as for the 3D "added" kernels. The stress is the element value in
Abaqus plane-strain order xx yy zz xy; only xx, yy, xy enter the in-plane force (zz is out of plane).
Coordinates are the reference ones the kernel sees; for a co-rotated kernel the stress is carried with
the element.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

_S3 = 1.0 / np.sqrt(3.0)
_G2 = np.array([-_S3, _S3])
_G3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_W3 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
_Q8 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.float64)
_T6_PTS = np.array([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])


@njit(cache=True)
def _dn_quad4(xi, eta):
    dn = np.zeros((4, 2))
    for a in range(4):
        sx = _Q8[a, 0]
        sy = _Q8[a, 1]
        dn[a, 0] = 0.25 * sx * (1 + sy * eta)
        dn[a, 1] = 0.25 * sy * (1 + sx * xi)
    return dn


@njit(cache=True)
def _dn_quad8(xi, eta):
    dn = np.zeros((8, 2))
    for a in range(8):
        sx = _Q8[a, 0]
        sy = _Q8[a, 1]
        if sx != 0 and sy != 0:                       # corner
            fx = 1 + sx * xi
            fy = 1 + sy * eta
            g = sx * xi + sy * eta - 1.0
            dn[a, 0] = 0.25 * sx * fy * (g + fx)
            dn[a, 1] = 0.25 * sy * fx * (g + fy)
        elif sx == 0:                                 # mid-edge along xi
            fy = 1 + sy * eta
            dn[a, 0] = -xi * fy
            dn[a, 1] = 0.5 * (1 - xi * xi) * sy
        else:                                         # mid-edge along eta
            fx = 1 + sx * xi
            dn[a, 0] = 0.5 * (1 - eta * eta) * sx
            dn[a, 1] = -eta * fx
    return dn


@njit(cache=True)
def _dn_tri3():
    dn = np.zeros((3, 2))
    dn[0, :] = -1.0
    dn[1, 0] = 1.0
    dn[2, 1] = 1.0
    return dn


@njit(cache=True)
def _dn_tri6(r, s):
    L = np.array([1.0 - r - s, r, s])
    dL = np.zeros((3, 2))
    dL[0, :] = -1.0
    dL[1, 0] = 1.0
    dL[2, 1] = 1.0
    dn = np.zeros((6, 2))
    for a in range(3):
        for k in range(2):
            dn[a, k] = (4.0 * L[a] - 1.0) * dL[a, k]
    pa = (0, 1, 2)
    pb = (1, 2, 0)                                    # mid nodes 4:1-2, 5:2-3, 6:3-1
    for m in range(3):
        i = pa[m]
        j = pb[m]
        for k in range(2):
            dn[3 + m, k] = 4.0 * (dL[i, k] * L[j] + L[i] * dL[j, k])
    return dn


@njit(cache=True)
def _accumulate(f, coords, dn, w, sig):
    nn = coords.shape[0]
    J = np.zeros((2, 2))
    for a in range(nn):
        for i in range(2):
            for k in range(2):
                J[i, k] += coords[a, i] * dn[a, k]
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    Ji = np.linalg.inv(J)
    for a in range(nn):
        g = np.zeros(2)
        for i in range(2):
            for k in range(2):
                g[i] += Ji[k, i] * dn[a, k]
        for i in range(2):
            acc = 0.0
            for k in range(2):
                acc += sig[i, k] * g[k]
            f[2 * a + i] += w * detJ * acc


@njit(parallel=True, cache=True)
def initial_stress_force_2d(node_coords, elem_conn, stress_init):
    """Per-element internal force (n_elems, 2*nodes); the element type follows from the connectivity width:
    3 linear triangle, 4 bilinear quad, 6 quadratic triangle, 8 quadratic quad. stress_init: (n_elems, 4)."""
    ne, nn = elem_conn.shape
    f_out = np.zeros((ne, 2 * nn))
    for e in prange(ne):
        sig = np.array([[stress_init[e, 0], stress_init[e, 3]], [stress_init[e, 3], stress_init[e, 1]]])
        coords = np.zeros((nn, 2))
        for a in range(nn):
            for i in range(2):
                coords[a, i] = node_coords[elem_conn[e, a], i]
        f = np.zeros(2 * nn)
        if nn == 4:
            for i in range(2):
                for j in range(2):
                    _accumulate(f, coords, _dn_quad4(_G2[i], _G2[j]), 1.0, sig)
        elif nn == 8:
            for i in range(3):
                for j in range(3):
                    _accumulate(f, coords, _dn_quad8(_G3[i], _G3[j]), _W3[i] * _W3[j], sig)
        elif nn == 3:
            _accumulate(f, coords, _dn_tri3(), 0.5, sig)
        elif nn == 6:
            for p in range(3):
                _accumulate(f, coords, _dn_tri6(_T6_PTS[p, 0], _T6_PTS[p, 1]), 1.0 / 6.0, sig)
        f_out[e, :] = f
    return f_out
