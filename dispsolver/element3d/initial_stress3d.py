"""Internal force of an initial stress field, f_i = int (dN_i/dx) . s0 dV, for the 3D solid kernels
that do not take the field themselves (element_catalog.KernelInfo.initial_stress == "added").

The kernels with a native initial-stress input add s0 to the stress at their own Gauss points. The others
(elastic-only EAS, reduced integration, the quadratic hexahedron and the plain tetrahedra) assemble only
the displacement-dependent part, so the assembly adds this term. The tangent is unchanged: the initial
stress stiffness is not formed, exactly as for the native kernels.

The stress is the element mean of the Gauss-point rows (the model writes one value per element), in the
kernel Voigt order xx yy zz xy yz xz. Coordinates are the reference ones the kernel sees: for a
co-rotated kernel those are the reference coordinates and the stress is carried with the element.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

_S3 = 1.0 / np.sqrt(3.0)
_G2 = np.array([-_S3, _S3])
_G3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_W3 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

# natural coordinates of the 20 hexahedron nodes in Abaqus order
_H20 = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
    [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float64)

_TET4_PT = np.array([[0.25, 0.25, 0.25]])
_A = 0.5854101966249685
_B = 0.1381966011250105
_TET10_PTS = np.array([[_B, _B, _B], [_A, _B, _B], [_B, _A, _B], [_B, _B, _A]])


@njit(cache=True)
def _dn_hex8(xi, eta, zeta):
    dn = np.zeros((8, 3))
    for a in range(8):
        sx = _H20[a, 0]; sy = _H20[a, 1]; sz = _H20[a, 2]
        dn[a, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
        dn[a, 1] = 0.125 * sy * (1 + sx * xi) * (1 + sz * zeta)
        dn[a, 2] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
    return dn


@njit(cache=True)
def _dn_hex20(xi, eta, zeta):
    dn = np.zeros((20, 3))
    for a in range(20):
        sx = _H20[a, 0]; sy = _H20[a, 1]; sz = _H20[a, 2]
        if sx != 0 and sy != 0 and sz != 0:           # corner
            fx = 1 + sx * xi; fy = 1 + sy * eta; fz = 1 + sz * zeta
            g = sx * xi + sy * eta + sz * zeta - 2.0
            dn[a, 0] = 0.125 * sx * fy * fz * (g + fx)
            dn[a, 1] = 0.125 * sy * fx * fz * (g + fy)
            dn[a, 2] = 0.125 * sz * fx * fy * (g + fz)
        elif sx == 0:                                  # mid-edge along xi
            fy = 1 + sy * eta; fz = 1 + sz * zeta
            dn[a, 0] = -0.5 * xi * fy * fz
            dn[a, 1] = 0.25 * (1 - xi * xi) * sy * fz
            dn[a, 2] = 0.25 * (1 - xi * xi) * fy * sz
        elif sy == 0:                                  # mid-edge along eta
            fx = 1 + sx * xi; fz = 1 + sz * zeta
            dn[a, 0] = 0.25 * (1 - eta * eta) * sx * fz
            dn[a, 1] = -0.5 * eta * fx * fz
            dn[a, 2] = 0.25 * (1 - eta * eta) * fx * sz
        else:                                          # mid-edge along zeta
            fx = 1 + sx * xi; fy = 1 + sy * eta
            dn[a, 0] = 0.25 * (1 - zeta * zeta) * sx * fy
            dn[a, 1] = 0.25 * (1 - zeta * zeta) * fx * sy
            dn[a, 2] = -0.5 * zeta * fx * fy
    return dn


@njit(cache=True)
def _dn_tet4():
    dn = np.zeros((4, 3))
    dn[0, :] = -1.0
    dn[1, 0] = 1.0
    dn[2, 1] = 1.0
    dn[3, 2] = 1.0
    return dn


@njit(cache=True)
def _dn_tet10(r, s, t):
    # barycentric L = (1-r-s-t, r, s, t); corner N = L(2L-1), mid N = 4 La Lb
    L = np.array([1.0 - r - s - t, r, s, t])
    dL = np.zeros((4, 3))
    dL[0, :] = -1.0
    dL[1, 0] = 1.0
    dL[2, 1] = 1.0
    dL[3, 2] = 1.0
    dn = np.zeros((10, 3))
    for a in range(4):
        for k in range(3):
            dn[a, k] = (4.0 * L[a] - 1.0) * dL[a, k]
    # Abaqus mid-node edges: 5:1-2, 6:2-3, 7:3-1, 8:1-4, 9:2-4, 10:3-4
    pa = (0, 1, 2, 0, 1, 2)
    pb = (1, 2, 0, 3, 3, 3)
    for m in range(6):
        i = pa[m]; j = pb[m]
        for k in range(3):
            dn[4 + m, k] = 4.0 * (dL[i, k] * L[j] + L[i] * dL[j, k])
    return dn


@njit(cache=True)
def _accumulate(f, coords, dn, w, sig):
    nn = coords.shape[0]
    J = np.zeros((3, 3))
    for a in range(nn):
        for i in range(3):
            for k in range(3):
                J[i, k] += coords[a, i] * dn[a, k]
    detJ = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
            - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
            + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
    Ji = np.linalg.inv(J)
    for a in range(nn):
        g = np.zeros(3)
        for i in range(3):
            for k in range(3):
                g[i] += Ji[k, i] * dn[a, k]
        for i in range(3):
            acc = 0.0
            for k in range(3):
                acc += sig[i, k] * g[k]
            f[3 * a + i] += w * detJ * acc


@njit(parallel=True, cache=True)
def initial_stress_force(node_coords, elem_conn, stress_init):
    """Per-element internal force (n_elems, 3*nodes) of the initial stress; the element type follows
    from the connectivity width: 4 linear tet, 8 hex, 10 quadratic tet, 20 quadratic hex."""
    ne, nn = elem_conn.shape
    f_out = np.zeros((ne, 3 * nn))
    for e in prange(ne):
        s = np.zeros(6)
        ng = stress_init.shape[1]
        for g in range(ng):
            for c in range(6):
                s[c] += stress_init[e, g, c] / ng
        sig = np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])
        coords = np.zeros((nn, 3))
        for a in range(nn):
            for i in range(3):
                coords[a, i] = node_coords[elem_conn[e, a], i]
        f = np.zeros(3 * nn)
        if nn == 8:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        _accumulate(f, coords, _dn_hex8(_G2[i], _G2[j], _G2[k]), 1.0, sig)
        elif nn == 20:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        _accumulate(f, coords, _dn_hex20(_G3[i], _G3[j], _G3[k]), _W3[i] * _W3[j] * _W3[k], sig)
        elif nn == 4:
            _accumulate(f, coords, _dn_tet4(), 1.0 / 6.0, sig)
        elif nn == 10:
            for p in range(4):
                _accumulate(f, coords, _dn_tet10(_TET10_PTS[p, 0], _TET10_PTS[p, 1], _TET10_PTS[p, 2]), 1.0 / 24.0, sig)
        f_out[e, :] = f
    return f_out
