"""
dispsolver/loads/integration.py
================================
Pure NumPy integration routines for consistent body force weights and
face pressure loads across all 1D/2D/3D element types runnable in the solver.

Semantics & Theoretical References:
- Body force weights:
    W_a = int_{Omega_e} N_a(x) dOmega
    Exact nodal partition of a uniform body load b (f_a = W_a * b).
    Sum of weights equals the geometric measure of the reference configuration:
      - 3D elements: volume V = int dV
      - 2D plane strain / plane stress elements: area A = int dA (unit thickness t = 1.0)
      - Axisymmetric elements (CAX4, CAX8): full ring volume V = 2*pi * int r dA
        matching the internal force and stiffness integration in
        dispsolver/element2d/cpe4_numba.py and cpe8_numba.py (Abaqus Theory Manual §3.2.8).
      - Truss (T2D2): length L = ||x_1 - x_0||.
    Quadratic elements (C3D20, C3D10, CPE8, CPE6, CAX8) use consistent Gauss
    quadrature on their own shape functions; their weights are not an equal split
    (corner terms are negative or zero, midsides carry the majority).

- Face pressure forces:
    f_a = int_{Gamma_e} N_a(x) (-p n_{out}) dGamma
    Abaqus sign convention:
      Positive pressure p > 0 pushes INTO the element (compressive traction t = -p n_{out}).
      n_{out} is the outward unit normal on the face in the reference configuration.
    Sum of nodal forces satisfies equilibrium:
      sum f_a = -p * A_{face} * n_{out} (or 2*pi int r (-p n) ds for axisymmetric).
    Face numbering matches Abaqus element tables:
      - C3D8 / C3D20: S1..S6
      - C3D4 / C3D10: S1..S4
      - C3D6: S1..S5
      - CPE4 / CPE8 / CAX4 / CAX8: S1..S4
      - CPE3 / CPE6: S1..S3
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np


# -----------------------------------------------------------------------------
# Gauss Quadrature Constants
# -----------------------------------------------------------------------------
_G1 = 1.0 / math.sqrt(3.0)
_GP_1D_2PT = np.array([-_G1, _G1], dtype=np.float64)
_WT_1D_2PT = np.array([1.0, 1.0], dtype=np.float64)

_G3_SQ = math.sqrt(3.0 / 5.0)
_GP_1D_3PT = np.array([-_G3_SQ, 0.0, _G3_SQ], dtype=np.float64)
_WT_1D_3PT = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)

# 6-point triangle quadrature (Cowper 1973 / Dunavant degree 4, exact for quadratics)
_TRI6_ALPHAS = np.array([
    [0.091576213509771, 0.091576213509771, 0.816847572980459],
    [0.091576213509771, 0.816847572980459, 0.091576213509771],
    [0.816847572980459, 0.091576213509771, 0.091576213509771],
    [0.445948490915965, 0.445948490915965, 0.108103018168070],
    [0.445948490915965, 0.108103018168070, 0.445948490915965],
    [0.108103018168070, 0.445948490915965, 0.445948490915965],
], dtype=np.float64)
_TRI6_WEIGHTS = np.array([
    0.109951743655322, 0.109951743655322, 0.109951743655322,
    0.223381589678011, 0.223381589678011, 0.223381589678011,
], dtype=np.float64)  # sum = 1.0; normalized over reference triangle of area 1/2

# 4-point tetrahedron quadrature (Keast, degree 2 exact)
_TET4_A = (5.0 - math.sqrt(5.0)) / 20.0
_TET4_B = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
_TET4_L = np.array([
    [_TET4_B, _TET4_A, _TET4_A, _TET4_A],
    [_TET4_A, _TET4_B, _TET4_A, _TET4_A],
    [_TET4_A, _TET4_A, _TET4_B, _TET4_A],
    [_TET4_A, _TET4_A, _TET4_A, _TET4_B],
], dtype=np.float64)
_TET4_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)  # normalized over ref tet of volume 1/6

# 3-point triangle quadrature (degree 2 exact)
_TRI3_L = np.array([
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
], dtype=np.float64)
_TRI3_WEIGHTS = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)


# -----------------------------------------------------------------------------
# Element Classification Helper
# -----------------------------------------------------------------------------
def _classify_element(elem_type: str, n_nodes: int) -> str:
    """Return normalized category for an element type."""
    t = elem_type.strip().upper()
    if t in ("T2D2", "TRUSS2D", "LINK") or n_nodes == 2:
        return "T2D2"
    if t.startswith("CAX") or "AXISYMMETRIC" in t:
        if n_nodes == 8 or t in ("CAX8", "CAX8R", "CAX8H", "CAX8RH"):
            return "CAX8"
        return "CAX4"
    if t.startswith("C3D20") or n_nodes == 20:
        return "C3D20"
    if t.startswith("C3D10") or n_nodes == 10:
        return "C3D10"
    if t in ("C3D4", "C3D4H") or (t.startswith("C3D") and n_nodes == 4):
        return "C3D4"
    if t in ("C3D6", "C3D6_WEDGE", "WEDGE6") or (t.startswith("C3D") and n_nodes == 6):
        return "C3D6"
    if t.startswith("C3D8") or t.startswith("SOLID_SHELL") or (t.startswith("C3D") and n_nodes == 8):
        return "C3D8"
    if t.startswith("CPE8") or t.startswith("CPS8") or t in ("Q8", "Q8R") or n_nodes == 8:
        return "CPE8"
    if t.startswith("CPE6") or t.startswith("CPS6") or t in ("T6", "CPE6M", "CPS6M", "CPE6MH", "CPS6MH", "CPE6M_CR") or n_nodes == 6:
        return "CPE6"
    if t.startswith("CPE3") or t.startswith("CPS3") or t in ("T3", "CST", "TRIA3", "CPE3H", "CPS3H") or n_nodes == 3:
        return "CPE3"
    if t.startswith("CPE4") or t.startswith("CPS4") or t.startswith("Q4") or n_nodes == 4:
        return "CPE4"
    raise NotImplementedError(f"Unsupported element type {elem_type} (nodes={n_nodes})")


# -----------------------------------------------------------------------------
# Shape Functions & Derivatives
# -----------------------------------------------------------------------------
def _shape_c3d8(xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
    """C3D8 shape functions and natural derivatives (8, 3)."""
    sgn = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float64)
    p = np.array([xi, eta, zeta], dtype=np.float64)
    N = 0.125 * np.prod(1.0 + sgn * p, axis=1)
    dN = np.empty((8, 3), dtype=np.float64)
    dN[:, 0] = 0.125 * sgn[:, 0] * (1.0 + sgn[:, 1] * eta) * (1.0 + sgn[:, 2] * zeta)
    dN[:, 1] = 0.125 * sgn[:, 1] * (1.0 + sgn[:, 0] * xi) * (1.0 + sgn[:, 2] * zeta)
    dN[:, 2] = 0.125 * sgn[:, 2] * (1.0 + sgn[:, 0] * xi) * (1.0 + sgn[:, 1] * eta)
    return N, dN


def _shape_c3d20(xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
    """C3D20 20-node serendipity shape functions and natural derivatives (20, 3)."""
    N = np.zeros(20, dtype=np.float64)
    dN = np.zeros((20, 3), dtype=np.float64)
    # Corners
    sgn = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float64)
    for a in range(8):
        ca, cb, cc = sgn[a]
        term = ca * xi + cb * eta + cc * zeta - 2.0
        lin = (1.0 + ca * xi) * (1.0 + cb * eta) * (1.0 + cc * zeta)
        N[a] = 0.125 * lin * term
        dN[a, 0] = 0.125 * ca * (1.0 + cb * eta) * (1.0 + cc * zeta) * term + 0.125 * lin * ca
        dN[a, 1] = 0.125 * cb * (1.0 + ca * xi) * (1.0 + cc * zeta) * term + 0.125 * lin * cb
        dN[a, 2] = 0.125 * cc * (1.0 + ca * xi) * (1.0 + cb * eta) * term + 0.125 * lin * cc

    # Midside edges (8..19)
    # 8: (0, -1, -1), 9: (1, 0, -1), 10: (0, 1, -1), 11: (-1, 0, -1)
    # 12: (0, -1, 1), 13: (1, 0, 1), 14: (0, 1, 1), 15: (-1, 0, 1)
    # 16: (-1, -1, 0), 17: (1, -1, 0), 18: (1, 1, 0), 19: (-1, 1, 0)
    mids = [
        (8,  0, -1, -1), (9,  1,  0, -1), (10, 0,  1, -1), (11, -1,  0, -1),
        (12, 0, -1,  1), (13, 1,  0,  1), (14, 0,  1,  1), (15, -1,  0,  1),
        (16, -1, -1, 0), (17, 1, -1,  0), (18, 1,  1,  0), (19, -1,  1,  0)
    ]
    for idx, cx, cy, cz in mids:
        if cx == 0:
            N[idx] = 0.25 * (1.0 - xi * xi) * (1.0 + cy * eta) * (1.0 + cz * zeta)
            dN[idx, 0] = -0.5 * xi * (1.0 + cy * eta) * (1.0 + cz * zeta)
            dN[idx, 1] = 0.25 * cy * (1.0 - xi * xi) * (1.0 + cz * zeta)
            dN[idx, 2] = 0.25 * cz * (1.0 - xi * xi) * (1.0 + cy * eta)
        elif cy == 0:
            N[idx] = 0.25 * (1.0 + cx * xi) * (1.0 - eta * eta) * (1.0 + cz * zeta)
            dN[idx, 0] = 0.25 * cx * (1.0 - eta * eta) * (1.0 + cz * zeta)
            dN[idx, 1] = -0.5 * eta * (1.0 + cx * xi) * (1.0 + cz * zeta)
            dN[idx, 2] = 0.25 * cz * (1.0 + cx * xi) * (1.0 - eta * eta)
        else:  # cz == 0
            N[idx] = 0.25 * (1.0 + cx * xi) * (1.0 + cy * eta) * (1.0 - zeta * zeta)
            dN[idx, 0] = 0.25 * cx * (1.0 + cy * eta) * (1.0 - zeta * zeta)
            dN[idx, 1] = 0.25 * cy * (1.0 + cx * xi) * (1.0 - zeta * zeta)
            dN[idx, 2] = -0.5 * zeta * (1.0 + cx * xi) * (1.0 + cy * eta)
    return N, dN


def _shape_c3d10(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """C3D10 10-node quadratic tet shape functions and derivatives w.r.t [L1, L2, L3, L4]."""
    l1, l2, l3, l4 = L
    N = np.array([
        (2.0 * l1 - 1.0) * l1,
        (2.0 * l2 - 1.0) * l2,
        (2.0 * l3 - 1.0) * l3,
        (2.0 * l4 - 1.0) * l4,
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
        4.0 * l1 * l4,
        4.0 * l2 * l4,
        4.0 * l3 * l4,
    ], dtype=np.float64)
    # dN / d(r, s, t) where L1 = 1 - r - s - t, L2 = r, L3 = s, L4 = t
    dL = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)  # (4, 3)
    dNdL = np.zeros((10, 4), dtype=np.float64)
    dNdL[0, 0] = 4.0 * l1 - 1.0
    dNdL[1, 1] = 4.0 * l2 - 1.0
    dNdL[2, 2] = 4.0 * l3 - 1.0
    dNdL[3, 3] = 4.0 * l4 - 1.0
    dNdL[4, 0] = 4.0 * l2; dNdL[4, 1] = 4.0 * l1
    dNdL[5, 1] = 4.0 * l3; dNdL[5, 2] = 4.0 * l2
    dNdL[6, 2] = 4.0 * l1; dNdL[6, 0] = 4.0 * l3
    dNdL[7, 0] = 4.0 * l4; dNdL[7, 3] = 4.0 * l1
    dNdL[8, 1] = 4.0 * l4; dNdL[8, 3] = 4.0 * l2
    dNdL[9, 2] = 4.0 * l4; dNdL[9, 3] = 4.0 * l3
    dNdRST = dNdL @ dL  # (10, 3)
    return N, dNdRST


def _shape_c3d6(r: float, s: float, z: float) -> Tuple[np.ndarray, np.ndarray]:
    """C3D6 6-node wedge shape functions and derivatives (6, 3)."""
    l1 = 1.0 - r - s
    l2 = r
    l3 = s
    N = np.array([
        0.5 * l1 * (1.0 - z),
        0.5 * l2 * (1.0 - z),
        0.5 * l3 * (1.0 - z),
        0.5 * l1 * (1.0 + z),
        0.5 * l2 * (1.0 + z),
        0.5 * l3 * (1.0 + z),
    ], dtype=np.float64)
    dN = np.zeros((6, 3), dtype=np.float64)
    dN[0] = [-0.5 * (1.0 - z), -0.5 * (1.0 - z), -0.5 * l1]
    dN[1] = [ 0.5 * (1.0 - z),  0.0,           -0.5 * l2]
    dN[2] = [ 0.0,            0.5 * (1.0 - z), -0.5 * l3]
    dN[3] = [-0.5 * (1.0 + z), -0.5 * (1.0 + z),  0.5 * l1]
    dN[4] = [ 0.5 * (1.0 + z),  0.0,            0.5 * l2]
    dN[5] = [ 0.0,            0.5 * (1.0 + z),  0.5 * l3]
    return N, dN


def _shape_cpe4(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """CPE4 / CAX4 shape functions and derivatives (4, 2)."""
    N = 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ], dtype=np.float64)
    dN = 0.25 * np.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)],
    ], dtype=np.float64).T  # (4, 2)
    return N, dN


def _shape_cpe8(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """CPE8 / CAX8 8-node serendipity quad shape functions and derivatives (8, 2)."""
    N = np.zeros(8, dtype=np.float64)
    dN = np.zeros((8, 2), dtype=np.float64)
    xi_i = np.array([-1.0, 1.0, 1.0, -1.0])
    eta_i = np.array([-1.0, -1.0, 1.0, 1.0])
    for i in range(4):
        xi0 = xi * xi_i[i]
        eta0 = eta * eta_i[i]
        N[i] = 0.25 * (1.0 + xi0) * (1.0 + eta0) * (xi0 + eta0 - 1.0)
        dN[i, 0] = 0.25 * xi_i[i] * (1.0 + eta0) * (2.0 * xi0 + eta0)
        dN[i, 1] = 0.25 * eta_i[i] * (1.0 + xi0) * (xi0 + 2.0 * eta0)

    # Midside nodes: 4 (0, -1), 5 (1, 0), 6 (0, 1), 7 (-1, 0)
    N[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    dN[4, 0] = -xi * (1.0 - eta)
    dN[4, 1] = -0.5 * (1.0 - xi * xi)

    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    dN[5, 0] = 0.5 * (1.0 - eta * eta)
    dN[5, 1] = -(1.0 + xi) * eta

    N[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    dN[6, 0] = -xi * (1.0 + eta)
    dN[6, 1] = 0.5 * (1.0 - xi * xi)

    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta)
    dN[7, 0] = -0.5 * (1.0 - eta * eta)
    dN[7, 1] = -(1.0 - xi) * eta
    return N, dN


def _shape_cpe6(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CPE6 6-node triangle shape functions and derivatives (6, 2) w.r.t (r, s)."""
    l1, l2, l3 = L
    N = np.array([
        (2.0 * l1 - 1.0) * l1,
        (2.0 * l2 - 1.0) * l2,
        (2.0 * l3 - 1.0) * l3,
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
    ], dtype=np.float64)
    # L1 = 1 - r - s, L2 = r, L3 = s
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)  # (3, 2)
    dNdL = np.zeros((6, 3), dtype=np.float64)
    dNdL[0, 0] = 4.0 * l1 - 1.0
    dNdL[1, 1] = 4.0 * l2 - 1.0
    dNdL[2, 2] = 4.0 * l3 - 1.0
    dNdL[3, 0] = 4.0 * l2; dNdL[3, 1] = 4.0 * l1
    dNdL[4, 1] = 4.0 * l3; dNdL[4, 2] = 4.0 * l2
    dNdL[5, 2] = 4.0 * l1; dNdL[5, 0] = 4.0 * l3
    dN = dNdL @ dL  # (6, 2)
    return N, dN


# -----------------------------------------------------------------------------
# Body Force Weights
# -----------------------------------------------------------------------------
def body_force_weights(X: np.ndarray, elem_type: str) -> np.ndarray:
    """Consistent body force weights int_{Omega_e} N_a dOmega for each node a.

    For uniform body force b, the consistent nodal load is f_a = W_a * b.
    Sums to reference volume V (3D), area A * 1.0 (2D plane), or full ring
    volume 2*pi int r dA (axisymmetric CAX*).

    Parameters
    ----------
    X : np.ndarray
        Nodal coordinates of the element in reference configuration (n_nodes, dim).
    elem_type : str
        Element type name (e.g. 'C3D8', 'C3D20', 'CPE4', 'CAX4', 'CPE8', etc.).

    Returns
    -------
    weights : np.ndarray
        Vector of length n_nodes containing the integrated weight for each node.
    """
    X = np.asarray(X, dtype=np.float64)
    n_nodes, dim = X.shape
    cat = _classify_element(elem_type, n_nodes)

    if cat == "T2D2":
        L = float(np.linalg.norm(X[1] - X[0]))
        return np.full(2, 0.5 * L, dtype=np.float64)

    if cat == "C3D8":
        w = np.zeros(8, dtype=np.float64)
        for xi in _GP_1D_2PT:
            for eta in _GP_1D_2PT:
                for zeta in _GP_1D_2PT:
                    N, dN = _shape_c3d8(xi, eta, zeta)
                    J = X.T @ dN  # (3, 3)
                    detJ = abs(float(np.linalg.det(J)))
                    w += N * detJ
        return w

    if cat == "C3D20":
        w = np.zeros(20, dtype=np.float64)
        for i, xi in enumerate(_GP_1D_3PT):
            wi = _WT_1D_3PT[i]
            for j, eta in enumerate(_GP_1D_3PT):
                wj = _WT_1D_3PT[j]
                for k, zeta in enumerate(_GP_1D_3PT):
                    wk = _WT_1D_3PT[k]
                    wt = wi * wj * wk
                    N, dN = _shape_c3d20(xi, eta, zeta)
                    J = X.T @ dN
                    detJ = abs(float(np.linalg.det(J)))
                    w += N * (detJ * wt)
        return w

    if cat == "C3D4":
        V = abs(float(np.linalg.det(X[1:] - X[0]))) / 6.0
        return np.full(4, 0.25 * V, dtype=np.float64)

    if cat == "C3D10":
        w = np.zeros(10, dtype=np.float64)
        # 4-point Keast quadrature on reference tetrahedron
        # Ref tet vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1), volume = 1/6
        for q in range(4):
            L = _TET4_L[q]
            wt = _TET4_WEIGHTS[q]
            N, dNdRST = _shape_c3d10(L)
            # Jacobian of mapping from ref tet (r, s, t) to physical space
            # X(r, s, t) = sum N_a(r, s, t) X_a
            J = X.T @ dNdRST  # (3, 3)
            detJ = abs(float(np.linalg.det(J)))
            w += N * (detJ * wt / 6.0)
        return w

    if cat == "C3D6":
        w = np.zeros(6, dtype=np.float64)
        for q in range(3):
            r, s_ = _TRI3_L[q, 1], _TRI3_L[q, 2]
            wt_tri = _TRI3_WEIGHTS[q] * 0.5  # area of ref triangle is 1/2
            for z in _GP_1D_2PT:
                N, dN = _shape_c3d6(r, s_, z)
                J = X.T @ dN
                detJ = abs(float(np.linalg.det(J)))
                w += N * (detJ * wt_tri)
        return w

    if cat == "CPE4":
        w = np.zeros(4, dtype=np.float64)
        for xi in _GP_1D_2PT:
            for eta in _GP_1D_2PT:
                N, dN = _shape_cpe4(xi, eta)
                J = X.T @ dN  # (2, 2)
                detJ = abs(float(np.linalg.det(J)))
                w += N * detJ
        return w

    if cat == "CPE8":
        w = np.zeros(8, dtype=np.float64)
        for i, xi in enumerate(_GP_1D_3PT):
            wi = _WT_1D_3PT[i]
            for j, eta in enumerate(_GP_1D_3PT):
                wj = _WT_1D_3PT[j]
                wt = wi * wj
                N, dN = _shape_cpe8(xi, eta)
                J = X.T @ dN
                detJ = abs(float(np.linalg.det(J)))
                w += N * (detJ * wt)
        return w

    if cat == "CPE3":
        # 2D 3-node triangle area
        v1 = X[1] - X[0]
        v2 = X[2] - X[0]
        A = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
        return np.full(3, A / 3.0, dtype=np.float64)

    if cat == "CPE6":
        w = np.zeros(6, dtype=np.float64)
        for q in range(6):
            L = _TRI6_ALPHAS[q]
            wt = _TRI6_WEIGHTS[q] * 0.5  # normalized by ref tri area = 1/2
            N, dNdRS = _shape_cpe6(L)
            J = X.T @ dNdRS
            detJ = abs(float(np.linalg.det(J)))
            w += N * (detJ * wt)
        return w

    if cat == "CAX4":
        # Axisymmetric 4-node ring: dV = 2*pi * r * det(J) dxi deta
        w = np.zeros(4, dtype=np.float64)
        for xi in _GP_1D_2PT:
            for eta in _GP_1D_2PT:
                N, dN = _shape_cpe4(xi, eta)
                r_gp = float(np.dot(N, X[:, 0]))
                J = X.T @ dN
                detJ = abs(float(np.linalg.det(J)))
                dV = 2.0 * math.pi * r_gp * detJ
                w += N * dV
        return w

    if cat == "CAX8":
        # Axisymmetric 8-node ring: dV = 2*pi * r * det(J) dxi deta
        w = np.zeros(8, dtype=np.float64)
        for i, xi in enumerate(_GP_1D_3PT):
            wi = _WT_1D_3PT[i]
            for j, eta in enumerate(_GP_1D_3PT):
                wj = _WT_1D_3PT[j]
                wt = wi * wj
                N, dN = _shape_cpe8(xi, eta)
                r_gp = float(np.dot(N, X[:, 0]))
                J = X.T @ dN
                detJ = abs(float(np.linalg.det(J)))
                dV = 2.0 * math.pi * r_gp * detJ * wt
                w += N * dV
        return w

    raise NotImplementedError(f"body_force_weights is not implemented for {elem_type}")


def _quadratic_points(X: np.ndarray, cat: str):
    """(N, dV) at quadrature points exact for N_a*N_b on the reference element, for the
    quadratic categories whose row-sum mass is not positive."""
    if cat in ("C3D20", "CPE8", "CAX8"):
        n_dim = 3 if cat == "C3D20" else 2
        shape = _shape_c3d20 if cat == "C3D20" else _shape_cpe8
        import itertools
        for idx in itertools.product(range(3), repeat=n_dim):
            pt = [_GP_1D_3PT[i] for i in idx]
            wt = float(np.prod([_WT_1D_3PT[i] for i in idx]))
            N, dN = shape(*pt)
            dV = abs(float(np.linalg.det(X.T @ dN))) * wt
            if cat == "CAX8":
                dV *= 2.0 * math.pi * float(np.dot(N, X[:, 0]))
            yield N, dV
        return
    if cat == "CPE6":
        for q in range(6):
            N, dN = _shape_cpe6(_TRI6_ALPHAS[q])
            yield N, abs(float(np.linalg.det(X.T @ dN))) * _TRI6_WEIGHTS[q] * 0.5
        return
    if cat == "C3D10":
        # collapsed-cube (Duffy) rule, 4 Gauss points per direction: exact to degree 7
        # in each cube coordinate, enough for N_a*N_b (degree 4) times the collapse Jacobian
        g, w = np.polynomial.legendre.leggauss(4)
        g, w = 0.5 * (g + 1.0), 0.5 * w
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a, b, c = g[i], g[j], g[k]
                    r = a
                    s = b * (1.0 - a)
                    t = c * (1.0 - a) * (1.0 - b)
                    jac = (1.0 - a) ** 2 * (1.0 - b)
                    N, dN = _shape_c3d10(np.array([1.0 - r - s - t, r, s, t]))
                    yield N, abs(float(np.linalg.det(X.T @ dN))) * jac * w[i] * w[j] * w[k]
        return
    raise NotImplementedError(f"no lumped mass rule for element category {cat}")


def lumped_mass_weights(X: np.ndarray, elem_type: str) -> np.ndarray:
    """Diagonal (lumped) mass per unit density, m_a / rho, for each node of one element.

    Row-sum lumping, m_a = rho * int N_a dV (= body_force_weights), wherever every row sum
    is positive -- all first-order elements, where it is the usual equal-share lumping.
    For the quadratic elements whose row sums are zero or negative at the corners (C3D10,
    C3D20, CPE6, CPE8, CAX8) row-sum lumping is not a mass matrix, so the diagonal of the
    consistent mass is scaled to the element mass instead (Hinton, Rock & Zienkiewicz 1976):
    m_a = rho * V * int N_a^2 dV / sum_b int N_b^2 dV. Either way sum_a m_a = rho * V, the
    element's volume (area x unit thickness for plane elements, ring volume for CAX*) --
    the same measure the 2D/3D kernels integrate the internal force over."""
    X = np.asarray(X, dtype=np.float64)
    w = body_force_weights(X, elem_type)
    if np.all(w > 1e-8 * abs(w.sum())):           # CPE6 corners integrate to round-off zero
        return w
    cat = _classify_element(elem_type, X.shape[0])
    diag = np.zeros(X.shape[0], dtype=np.float64)
    vol = 0.0
    for N, dV in _quadratic_points(X, cat):
        diag += N * N * dV
        vol += dV
    return diag * (vol / diag.sum())


# -----------------------------------------------------------------------------
# Face Definitions & Pressure Integration
# -----------------------------------------------------------------------------
# Abaqus Face Table: 1-based face numbering
_C3D8_FACES: dict[int, list[int]] = {
    1: [0, 1, 2, 3],  # S1 (r = -1)
    2: [4, 7, 6, 5],  # S2 (r = +1, outward CCW from outside)
    3: [0, 4, 5, 1],  # S3 (h = -1)
    4: [1, 5, 6, 2],  # S4 (g = +1)
    5: [2, 6, 7, 3],  # S5 (h = +1)
    6: [3, 7, 4, 0],  # S6 (g = -1)
}

_C3D20_FACES: dict[int, Tuple[list[int], list[int]]] = {
    1: ([0, 1, 2, 3], [8, 9, 10, 11]),
    2: ([4, 7, 6, 5], [15, 14, 13, 12]),
    3: ([0, 4, 5, 1], [16, 12, 17, 8]),
    4: ([1, 5, 6, 2], [17, 13, 18, 9]),
    5: ([2, 6, 7, 3], [18, 14, 19, 10]),
    6: ([3, 7, 4, 0], [19, 15, 16, 11]),
}

_C3D4_FACES: dict[int, list[int]] = {
    1: [0, 1, 2],  # S1 (r = 0 face)
    2: [0, 3, 1],  # S2 (h = 0 face)
    3: [1, 3, 2],  # S3 (g + h + r = 1 inclined face)
    4: [2, 3, 0],  # S4 (g = 0 face)
}

_C3D10_FACES: dict[int, Tuple[list[int], list[int]]] = {
    1: ([0, 1, 2], [4, 5, 6]),
    2: ([0, 3, 1], [7, 8, 4]),
    3: ([1, 3, 2], [8, 9, 5]),
    4: ([2, 3, 0], [9, 7, 6]),
}

_C3D6_FACES: dict[int, list[int]] = {
    1: [0, 1, 2],        # S1 (tri, r = -1)
    2: [3, 5, 4],        # S2 (tri, r = +1, outward CCW)
    3: [0, 1, 4, 3],     # S3 (quad, h = 0)
    4: [1, 2, 5, 4],     # S4 (quad, g + h = 1)
    5: [2, 0, 3, 5],     # S5 (quad, g = 0)
}

_2D_QUAD_EDGES: dict[int, list[int]] = {
    1: [0, 1],
    2: [1, 2],
    3: [2, 3],
    4: [3, 0],
}

_2D_QUAD8_EDGES: dict[int, list[int]] = {
    1: [0, 1, 4],
    2: [1, 2, 5],
    3: [2, 3, 6],
    4: [3, 0, 7],
}

_2D_TRI_EDGES: dict[int, list[int]] = {
    1: [0, 1],
    2: [1, 2],
    3: [2, 0],
}

_2D_TRI6_EDGES: dict[int, list[int]] = {
    1: [0, 1, 3],
    2: [1, 2, 4],
    3: [2, 0, 5],
}


def face_area_normal(X: np.ndarray, elem_type: str, face: int) -> Tuple[float, np.ndarray]:
    """Compute the surface area and outward unit normal of element face `face`.

    Parameters
    ----------
    X : np.ndarray
        Element reference coordinates (n_nodes, dim).
    elem_type : str
        Element type name.
    face : int
        1-based Abaqus face index.

    Returns
    -------
    area : float
        Surface area of the face (3D), edge length * 1.0 (2D plane),
        or axisymmetric surface area 2*pi int r ds (CAX*).
    normal : np.ndarray
        Outward-pointing unit normal vector (dim,).
    """
    X = np.asarray(X, dtype=np.float64)
    n_nodes, dim = X.shape
    cat = _classify_element(elem_type, n_nodes)
    elem_center = np.mean(X, axis=0)

    if dim == 3:
        if cat in ("C3D8", "C3D20"):
            if cat == "C3D8":
                f_idx = _C3D8_FACES[face]
                coords_f = X[f_idx]
                f_center = np.mean(coords_f, axis=0)
                v_out = f_center - elem_center
                area = 0.0
                n_acc = np.zeros(3, dtype=np.float64)
                for xi in _GP_1D_2PT:
                    for eta in _GP_1D_2PT:
                        dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
                        dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
                        t1 = dNx @ coords_f
                        t2 = dNe @ coords_f
                        dA = np.cross(t1, t2)
                        if np.dot(dA, v_out) < 0.0:
                            dA = -dA
                        area += float(np.linalg.norm(dA))
                        n_acc += dA
                n_unit = n_acc / max(float(np.linalg.norm(n_acc)), 1e-15)
                return area, n_unit

            else:  # C3D20
                corners, midsides = _C3D20_FACES[face]
                f_idx = corners + midsides
                coords_f = X[f_idx]
                f_center = np.mean(coords_f[:4], axis=0)
                v_out = f_center - elem_center
                area = 0.0
                n_acc = np.zeros(3, dtype=np.float64)
                for xi in _GP_1D_2PT:
                    for eta in _GP_1D_2PT:
                        _, dN = _shape_cpe8(xi, eta)
                        t1 = dN[:, 0] @ coords_f
                        t2 = dN[:, 1] @ coords_f
                        dA = np.cross(t1, t2)
                        if np.dot(dA, v_out) < 0.0:
                            dA = -dA
                        area += float(np.linalg.norm(dA))
                        n_acc += dA
                n_unit = n_acc / max(float(np.linalg.norm(n_acc)), 1e-15)
                return area, n_unit

        elif cat in ("C3D4", "C3D10"):
            if cat == "C3D4":
                f_idx = _C3D4_FACES[face]
                coords_f = X[f_idx]
                f_center = np.mean(coords_f, axis=0)
                v_out = f_center - elem_center
                t1 = coords_f[1] - coords_f[0]
                t2 = coords_f[2] - coords_f[0]
                dA = 0.5 * np.cross(t1, t2)
                if np.dot(dA, v_out) < 0.0:
                    dA = -dA
                area = float(np.linalg.norm(dA))
                n_unit = dA / max(area, 1e-15)
                return area, n_unit
            else:  # C3D10
                corners, midsides = _C3D10_FACES[face]
                f_idx = corners + midsides
                coords_f = X[f_idx]
                f_center = np.mean(coords_f[:3], axis=0)
                v_out = f_center - elem_center
                area = 0.0
                n_acc = np.zeros(3, dtype=np.float64)
                for q in range(6):
                    L = _TRI6_ALPHAS[q]
                    wt = _TRI6_WEIGHTS[q] * 0.5
                    _, dNdRS = _shape_cpe6(L)
                    t1 = dNdRS[:, 0] @ coords_f
                    t2 = dNdRS[:, 1] @ coords_f
                    dA = np.cross(t1, t2)
                    if np.dot(dA, v_out) < 0.0:
                        dA = -dA
                    area += float(np.linalg.norm(dA)) * wt
                    n_acc += dA * wt
                n_unit = n_acc / max(float(np.linalg.norm(n_acc)), 1e-15)
                return area, n_unit

        elif cat == "C3D6":
            f_idx = _C3D6_FACES[face]
            coords_f = X[f_idx]
            f_center = np.mean(coords_f, axis=0)
            v_out = f_center - elem_center
            if len(f_idx) == 3:  # triangular face S1 or S2
                t1 = coords_f[1] - coords_f[0]
                t2 = coords_f[2] - coords_f[0]
                dA = 0.5 * np.cross(t1, t2)
                if np.dot(dA, v_out) < 0.0:
                    dA = -dA
                area = float(np.linalg.norm(dA))
                n_unit = dA / max(area, 1e-15)
                return area, n_unit
            else:  # quadrilateral face S3, S4, S5
                area = 0.0
                n_acc = np.zeros(3, dtype=np.float64)
                for xi in _GP_1D_2PT:
                    for eta in _GP_1D_2PT:
                        dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
                        dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
                        t1 = dNx @ coords_f
                        t2 = dNe @ coords_f
                        dA = np.cross(t1, t2)
                        if np.dot(dA, v_out) < 0.0:
                            dA = -dA
                        area += float(np.linalg.norm(dA))
                        n_acc += dA
                n_unit = n_acc / max(float(np.linalg.norm(n_acc)), 1e-15)
                return area, n_unit

    else:  # 2D
        if cat in ("CPE4", "CAX4"):
            f_idx = _2D_QUAD_EDGES[face]
            p0, p1 = X[f_idx[0]], X[f_idx[1]]
            edge_center = 0.5 * (p0 + p1)
            v_out = edge_center - elem_center
            dx = p1 - p0
            L = float(np.linalg.norm(dx))
            # Normal perpendicular to dx: [dy, -dx] or [-dy, dx]
            n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
            if np.dot(n_raw, v_out) < 0.0:
                n_raw = -n_raw
            n_unit = n_raw / max(float(np.linalg.norm(n_raw)), 1e-15)
            if cat == "CAX4":
                # Axisymmetric area: 2*pi * r_mean * L
                r_mean = edge_center[0]
                return 2.0 * math.pi * r_mean * L, n_unit
            return L, n_unit

        elif cat in ("CPE8", "CAX8"):
            f_idx = _2D_QUAD8_EDGES[face]
            p0, p1, p2 = X[f_idx[0]], X[f_idx[1]], X[f_idx[2]]
            f_center = p2  # midside
            v_out = f_center - elem_center
            # Integrate length along quadratic curve
            L = 0.0
            r_int = 0.0
            coords_e = np.array([p0, p1, p2], dtype=np.float64)
            n_acc = np.zeros(2, dtype=np.float64)
            for i, s in enumerate(_GP_1D_3PT):
                wt = _WT_1D_3PT[i]
                # 3-node line shape functions: N0 = -0.5*(1-s)*s, N1 = 0.5*(1+s)*s, N2 = 1-s^2
                # dN/ds:
                dN0 = s - 0.5
                dN1 = s + 0.5
                dN2 = -2.0 * s
                dN = np.array([dN0, dN1, dN2])
                dx = dN @ coords_e
                ds_len = float(np.linalg.norm(dx))
                n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
                if np.dot(n_raw, v_out) < 0.0:
                    n_raw = -n_raw
                n_unit_pt = n_raw / max(ds_len, 1e-15)
                n_acc += n_unit_pt * wt * ds_len
                L += ds_len * wt
                r_pt = (-0.5 * (1 - s) * s) * p0[0] + (0.5 * (1 + s) * s) * p1[0] + (1.0 - s * s) * p2[0]
                r_int += r_pt * ds_len * wt
            n_unit = n_acc / max(float(np.linalg.norm(n_acc)), 1e-15)
            if cat == "CAX8":
                return 2.0 * math.pi * r_int, n_unit
            return L, n_unit

        elif cat == "CPE3":
            f_idx = _2D_TRI_EDGES[face]
            p0, p1 = X[f_idx[0]], X[f_idx[1]]
            edge_center = 0.5 * (p0 + p1)
            v_out = edge_center - elem_center
            dx = p1 - p0
            L = float(np.linalg.norm(dx))
            n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
            if np.dot(n_raw, v_out) < 0.0:
                n_raw = -n_raw
            n_unit = n_raw / max(float(np.linalg.norm(n_raw)), 1e-15)
            return L, n_unit

        elif cat == "CPE6":
            f_idx = _2D_TRI6_EDGES[face]
            p0, p1, p2 = X[f_idx[0]], X[f_idx[1]], X[f_idx[2]]
            f_center = p2
            v_out = f_center - elem_center
            L = 0.0
            coords_e = np.array([p0, p1, p2], dtype=np.float64)
            n_acc = np.zeros(2, dtype=np.float64)
            for i, s in enumerate(_GP_1D_3PT):
                wt = _WT_1D_3PT[i]
                dN = np.array([s - 0.5, s + 0.5, -2.0 * s])
                dx = dN @ coords_e
                ds_len = float(np.linalg.norm(dx))
                n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
                if np.dot(n_raw, v_out) < 0.0:
                    n_raw = -n_raw
                n_unit_pt = n_raw / max(ds_len, 1e-15)
                n_acc += n_unit_pt * wt * ds_len
                L += ds_len * wt
            n_unit = n_acc / max(float(np.linalg.norm(n_acc)), 1e-15)
            return L, n_unit

    raise NotImplementedError(f"face_area_normal is not implemented for {elem_type} face {face}")


def face_pressure_forces(
    X: np.ndarray,
    elem_type: str,
    face: int,
    p: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Consistent nodal forces of uniform pressure p on face `face` (1-based Abaqus index).

    Parameters
    ----------
    X : np.ndarray
        Element reference coordinates (n_nodes, dim).
    elem_type : str
        Element type name.
    face : int
        1-based Abaqus face index (S1..S6).
    p : float
        Pressure magnitude. Positive p pushes INTO the element (compressive traction).

    Returns
    -------
    local_nodes : np.ndarray
        Local node indices of the face in the element (k,).
    forces : np.ndarray
        Consistent nodal force vectors on the face (k, dim).
    """
    X = np.asarray(X, dtype=np.float64)
    n_nodes, dim = X.shape
    cat = _classify_element(elem_type, n_nodes)
    elem_center = np.mean(X, axis=0)

    if dim == 3:
        if cat in ("C3D8", "C3D20"):
            if cat == "C3D8":
                f_idx = np.array(_C3D8_FACES[face], dtype=np.int32)
                coords_f = X[f_idx]
                f_center = np.mean(coords_f, axis=0)
                v_in = elem_center - f_center
                forces = np.zeros((4, 3), dtype=np.float64)
                for xi in _GP_1D_2PT:
                    for eta in _GP_1D_2PT:
                        N = 0.25 * np.array([
                            (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)
                        ])
                        dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
                        dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
                        t1 = dNx @ coords_f
                        t2 = dNe @ coords_f
                        dA = np.cross(t1, t2)
                        # v_in points inward; positive p pushes inward
                        if np.dot(dA, v_in) < 0.0:
                            dA = -dA
                        forces += np.outer(N, p * dA)
                return f_idx, forces

            else:  # C3D20
                corners, midsides = _C3D20_FACES[face]
                f_idx = np.array(corners + midsides, dtype=np.int32)
                coords_f = X[f_idx]
                f_center = np.mean(coords_f[:4], axis=0)
                v_in = elem_center - f_center
                forces = np.zeros((8, 3), dtype=np.float64)
                for xi in _GP_1D_2PT:
                    for eta in _GP_1D_2PT:
                        N, dN = _shape_cpe8(xi, eta)
                        t1 = dN[:, 0] @ coords_f
                        t2 = dN[:, 1] @ coords_f
                        dA = np.cross(t1, t2)
                        if np.dot(dA, v_in) < 0.0:
                            dA = -dA
                        forces += np.outer(N, p * dA)
                return f_idx, forces

        elif cat in ("C3D4", "C3D10"):
            if cat == "C3D4":
                f_idx = np.array(_C3D4_FACES[face], dtype=np.int32)
                coords_f = X[f_idx]
                f_center = np.mean(coords_f, axis=0)
                v_in = elem_center - f_center
                t1 = coords_f[1] - coords_f[0]
                t2 = coords_f[2] - coords_f[0]
                dA = 0.5 * np.cross(t1, t2)
                if np.dot(dA, v_in) < 0.0:
                    dA = -dA
                # Linear triangle: 1/3 load to each corner
                forces = np.tile(dA * (p / 3.0), (3, 1))
                return f_idx, forces

            else:  # C3D10
                corners, midsides = _C3D10_FACES[face]
                f_idx = np.array(corners + midsides, dtype=np.int32)
                coords_f = X[f_idx]
                f_center = np.mean(coords_f[:3], axis=0)
                v_in = elem_center - f_center
                forces = np.zeros((6, 3), dtype=np.float64)
                for q in range(6):
                    L = _TRI6_ALPHAS[q]
                    wt = _TRI6_WEIGHTS[q] * 0.5
                    N, dNdRS = _shape_cpe6(L)
                    t1 = dNdRS[:, 0] @ coords_f
                    t2 = dNdRS[:, 1] @ coords_f
                    dA = np.cross(t1, t2)
                    if np.dot(dA, v_in) < 0.0:
                        dA = -dA
                    forces += np.outer(N, p * dA * wt)
                return f_idx, forces

        elif cat == "C3D6":
            f_idx = np.array(_C3D6_FACES[face], dtype=np.int32)
            coords_f = X[f_idx]
            f_center = np.mean(coords_f, axis=0)
            v_in = elem_center - f_center
            if len(f_idx) == 3:  # triangle
                t1 = coords_f[1] - coords_f[0]
                t2 = coords_f[2] - coords_f[0]
                dA = 0.5 * np.cross(t1, t2)
                if np.dot(dA, v_in) < 0.0:
                    dA = -dA
                forces = np.tile(dA * (p / 3.0), (3, 1))
                return f_idx, forces
            else:  # quad
                forces = np.zeros((4, 3), dtype=np.float64)
                for xi in _GP_1D_2PT:
                    for eta in _GP_1D_2PT:
                        N = 0.25 * np.array([
                            (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)
                        ])
                        dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
                        dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
                        t1 = dNx @ coords_f
                        t2 = dNe @ coords_f
                        dA = np.cross(t1, t2)
                        if np.dot(dA, v_in) < 0.0:
                            dA = -dA
                        forces += np.outer(N, p * dA)
                return f_idx, forces

    else:  # 2D
        if cat in ("CPE4", "CAX4"):
            f_idx = np.array(_2D_QUAD_EDGES[face], dtype=np.int32)
            p0, p1 = X[f_idx[0]], X[f_idx[1]]
            edge_center = 0.5 * (p0 + p1)
            v_in = elem_center - edge_center
            dx = p1 - p0
            # Inward normal
            n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
            if np.dot(n_raw, v_in) < 0.0:
                n_raw = -n_raw
            L = float(np.linalg.norm(dx))
            n_in = n_raw / max(float(np.linalg.norm(n_raw)), 1e-15)

            if cat == "CAX4":
                # Axisymmetric: int 2*pi * r * N_a * p * n_in ds
                forces = np.zeros((2, 2), dtype=np.float64)
                for s in _GP_1D_2PT:
                    N = 0.5 * np.array([1.0 - s, 1.0 + s])
                    r_s = N[0] * p0[0] + N[1] * p1[0]
                    ds = 0.5 * L
                    forces += np.outer(N, p * n_in * (2.0 * math.pi * r_s * ds))
                return f_idx, forces

            # 2D plane: half load to each node
            f_node = 0.5 * p * L * n_in
            return f_idx, np.array([f_node, f_node], dtype=np.float64)

        elif cat in ("CPE8", "CAX8"):
            f_idx = np.array(_2D_QUAD8_EDGES[face], dtype=np.int32)
            p0, p1, p2 = X[f_idx[0]], X[f_idx[1]], X[f_idx[2]]
            f_center = p2
            v_in = elem_center - f_center
            forces = np.zeros((3, 2), dtype=np.float64)
            coords_e = np.array([p0, p1, p2], dtype=np.float64)

            for i, s in enumerate(_GP_1D_3PT):
                wt = _WT_1D_3PT[i]
                N = np.array([-0.5 * (1.0 - s) * s, 0.5 * (1.0 + s) * s, 1.0 - s * s])
                dN = np.array([s - 0.5, s + 0.5, -2.0 * s])
                dx = dN @ coords_e
                ds_len = float(np.linalg.norm(dx))
                n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
                if np.dot(n_raw, v_in) < 0.0:
                    n_raw = -n_raw
                n_in = n_raw / max(float(np.linalg.norm(n_raw)), 1e-15)

                if cat == "CAX8":
                    r_pt = float(np.dot(N, coords_e[:, 0]))
                    w_circ = 2.0 * math.pi * r_pt
                else:
                    w_circ = 1.0

                forces += np.outer(N, p * n_in * (ds_len * wt * w_circ))
            return f_idx, forces

        elif cat == "CPE3":
            f_idx = np.array(_2D_TRI_EDGES[face], dtype=np.int32)
            p0, p1 = X[f_idx[0]], X[f_idx[1]]
            edge_center = 0.5 * (p0 + p1)
            v_in = elem_center - edge_center
            dx = p1 - p0
            L = float(np.linalg.norm(dx))
            n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
            if np.dot(n_raw, v_in) < 0.0:
                n_raw = -n_raw
            n_in = n_raw / max(float(np.linalg.norm(n_raw)), 1e-15)
            f_node = 0.5 * p * L * n_in
            return f_idx, np.array([f_node, f_node], dtype=np.float64)

        elif cat == "CPE6":
            f_idx = np.array(_2D_TRI6_EDGES[face], dtype=np.int32)
            p0, p1, p2 = X[f_idx[0]], X[f_idx[1]], X[f_idx[2]]
            f_center = p2
            v_in = elem_center - f_center
            forces = np.zeros((3, 2), dtype=np.float64)
            coords_e = np.array([p0, p1, p2], dtype=np.float64)
            for i, s in enumerate(_GP_1D_3PT):
                wt = _WT_1D_3PT[i]
                N = np.array([-0.5 * (1.0 - s) * s, 0.5 * (1.0 + s) * s, 1.0 - s * s])
                dN = np.array([s - 0.5, s + 0.5, -2.0 * s])
                dx = dN @ coords_e
                ds_len = float(np.linalg.norm(dx))
                n_raw = np.array([dx[1], -dx[0]], dtype=np.float64)
                if np.dot(n_raw, v_in) < 0.0:
                    n_raw = -n_raw
                n_in = n_raw / max(float(np.linalg.norm(n_raw)), 1e-15)
                forces += np.outer(N, p * n_in * (ds_len * wt))
            return f_idx, forces

    raise NotImplementedError(f"face_pressure_forces is not implemented for {elem_type} face {face}")
