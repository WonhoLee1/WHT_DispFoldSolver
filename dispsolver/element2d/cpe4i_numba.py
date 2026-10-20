"""
cpe4i_numba.py
==============
Numba JIT accelerated 4-node Incompatible Modes / Enhanced Assumed Strain Quadrilateral (CPE4I).
Simo & Rifai (1990) 4-mode EAS formulation plus ONE volumetric mode, with element-level static condensation.
Eliminates shear locking in bending at ANY aspect ratio and arbitrary orientation.
Equivalents: Abaqus CPE4I / Ansys SOLID182 (Enhanced Strain) / Wilson Incompatible Quad.

Internal degrees of freedom: 5 (Abaqus Theory Guide 3.2.5: "4 for CPS4I, 5 for CPE4I, CAX4I, CPEG4I,
13 for C3D8I"). The Simo-Rifai deviatoric/shear modes are 4; the Abaqus non-hybrid incompatible-mode
elements add "bilinear volumetric terms" against volumetric locking -- one in 2D (13 = 9 + 4 in 3D). The
fifth mode here is the isotropic strain  (detJ0/detJ) * xi*eta * [1, 1, 0]^T : orthogonal to every constant
strain (sum over the Gauss points of xi*eta is zero), so the Irons patch test still passes. The hybrid
element CPE4IH does NOT take it (its linear pressure does that job); cpe4ih_numba imports the 4-mode shape
helpers only.
"""

import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_2d

N_ENH = 5            # internal (incompatible-mode) degrees of freedom of CPE4I, see the module docstring
_GP1D = 1.0 / np.sqrt(3.0)
_GPS = np.array([
    [-_GP1D, -_GP1D],
    [ _GP1D, -_GP1D],
    [ _GP1D,  _GP1D],
    [-_GP1D,  _GP1D],
], dtype=np.float64)
_WTS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@njit(fastmath=True)
def solve_small(A: np.ndarray, B: np.ndarray):
    """Solve A X = B (A n x n, B n x m) by Gaussian elimination with partial pivoting.

    Returns (X, ok). ok is False when a pivot vanishes relative to the matrix scale, which is what a
    condensation block does at a plastic collapse load. `np.linalg.solve/inv` raise an exception there, and
    an exception inside the parallel assembly kills the whole run (SystemError) instead of letting the
    solver cut the increment back; this routine never raises."""
    n = A.shape[0]
    m = B.shape[1]
    M = A.copy()
    X = B.copy()
    scale = 0.0
    for i in range(n):
        for j in range(n):
            scale = max(scale, abs(M[i, j]))
    tiny = 1e-14 * max(scale, 1e-300)
    for c in range(n):
        piv = c
        best = abs(M[c, c])
        for r in range(c + 1, n):
            if abs(M[r, c]) > best:
                best = abs(M[r, c])
                piv = r
        if not (best > tiny):
            return np.zeros((n, m), dtype=np.float64), False
        if piv != c:
            for j in range(n):
                t = M[c, j]; M[c, j] = M[piv, j]; M[piv, j] = t
            for j in range(m):
                t = X[c, j]; X[c, j] = X[piv, j]; X[piv, j] = t
        d = M[c, c]
        for r in range(c + 1, n):
            f = M[r, c] / d
            if f != 0.0:
                for j in range(c, n):
                    M[r, j] -= f * M[c, j]
                for j in range(m):
                    X[r, j] -= f * X[c, j]
    for c in range(n - 1, -1, -1):
        for j in range(m):
            acc = X[c, j]
            for k in range(c + 1, n):
                acc -= M[c, k] * X[k, j]
            X[c, j] = acc / M[c, c]
    return X, True


@njit(fastmath=True)
def _shape_derivs_cpe4i(xi: float, eta: float) -> np.ndarray:
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
def _voigt_transform_2d(J0: np.ndarray) -> np.ndarray:
    """Computes T0_inv_T for transformation of enhanced strain tensor: shape (3, 3)."""
    # NOTE (2026-10-20): `J0 = dN_dxi @ coords` is J0[a, b] = dx_b/dxi_a, the TRANSPOSE of the textbook Jacobian,
    # and this function is written for that convention. An "exact inverse of Simo-Rifai T0" built from the same
    # J0 with the textbook reading (j12 = J0[0, 1] as dx/deta) is not frame invariant: K(rotated element) differs
    # from R K R^T by ~20 % (tests/test_cpe4i_ih_repair.py::test_element_stiffness_rotates_with_the_element).
    # Inverse of centroid Jacobian J0
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    invJ00 =  J0[1, 1] / detJ0
    invJ01 = -J0[0, 1] / detJ0
    invJ10 = -J0[1, 0] / detJ0
    invJ11 =  J0[0, 0] / detJ0

    # T0_inv_T maps natural Voigt strain [eps_xi, eps_eta, gamma_xi_eta]^T to physical [eps_x, eps_y, gamma_xy]^T
    # via eps_X = J0^{-1} E_xi J0^{-T}
    T = np.zeros((3, 3), dtype=np.float64)
    T[0, 0] = invJ00 * invJ00
    T[0, 1] = invJ01 * invJ01
    T[0, 2] = invJ00 * invJ01

    T[1, 0] = invJ10 * invJ10
    T[1, 1] = invJ11 * invJ11
    T[1, 2] = invJ10 * invJ11

    T[2, 0] = 2.0 * invJ00 * invJ10
    T[2, 1] = 2.0 * invJ01 * invJ11
    T[2, 2] = invJ00 * invJ11 + invJ01 * invJ10
    return T


@njit(fastmath=True)
def compute_cpe4i_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # material properties (layout depends on mat_type)
    sdvs: np.ndarray,        # (4, n_sdv), updated in place with the converged state
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Statically condensed 8x8 tangent Ke and 8-element fe of one CPE4I element.

    The enhanced strain parameters alpha are the solution of  fa(alpha) = sum M^T sigma(B u + M alpha) dV = 0
    for the CURRENT displacements. For a linear material one Newton step from alpha = 0 is exact
    (alpha = -Kaa^-1 fa0, fe = fu0 - Kua Kaa^-1 fa0); for a nonlinear material (plasticity, hyperelasticity)
    alpha is iterated to equilibrium, otherwise the condensed force is not the derivative of the element energy and
    the condensed tangent is not its Jacobian (measured: 1.9e-3 off, and the Newton loop of plastic bending
    stalled). The material is evaluated from the state variables at the START of the increment on every iteration;
    only the converged evaluation is written back.
    """
    # Centroid Jacobian J0
    dN0 = _shape_derivs_cpe4i(0.0, 0.0)
    J0 = dN0 @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    if detJ0 <= 1e-14:
        detJ0 = 1e-14

    T0_inv_T = _voigt_transform_2d(J0)
    has_error = 0

    # geometry of the four Gauss points: B (3x8), M (3x4), volume weight
    Bq = np.zeros((4, 3, 8), dtype=np.float64)
    Mq = np.zeros((4, 3, N_ENH), dtype=np.float64)
    dVq = np.zeros(4, dtype=np.float64)
    M0 = np.zeros((3, 4), dtype=np.float64)
    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        dN_dxi = _shape_derivs_cpe4i(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
            has_error = 1
            detJ = 1e-14
        invJ00 = J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 = J[0, 0] / detJ
        for a in range(4):
            dx = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dy = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]
            Bq[q, 0, 2 * a + 0] = dx
            Bq[q, 1, 2 * a + 1] = dy
            Bq[q, 2, 2 * a + 0] = dy
            Bq[q, 2, 2 * a + 1] = dx
        # Natural EAS modes M0 = [[xi, 0, 0, 0], [0, eta, 0, 0], [0, 0, xi, eta]]; physical M = (detJ0/detJ) T M0
        M0.fill(0.0)
        M0[0, 0] = xi
        M0[1, 1] = eta
        M0[2, 2] = xi
        M0[2, 3] = eta
        Mq[q, :, :4] = (detJ0 / detJ) * (T0_inv_T @ M0)
        # fifth mode: isotropic bilinear volumetric strain (no frame transformation needed: [1,1,0] is invariant)
        Mq[q, 0, 4] = (detJ0 / detJ) * xi * eta
        Mq[q, 1, 4] = (detJ0 / detJ) * xi * eta
        dVq[q] = detJ * _WTS[q]

    # state variables at the start of the increment (the material is always asked from these)
    n_sdv = sdvs.shape[1] if (sdvs is not None and sdvs.ndim == 2) else 0
    sdv0 = np.zeros((4, max(n_sdv, 1)), dtype=np.float64)
    for q in range(4):
        for s_i in range(n_sdv):
            sdv0[q, s_i] = sdvs[q, s_i]
    sdv_last = np.zeros((4, max(n_sdv, 1)), dtype=np.float64)

    alpha = np.zeros(N_ENH, dtype=np.float64)
    Kuu = np.zeros((8, 8), dtype=np.float64)
    Kua = np.zeros((8, N_ENH), dtype=np.float64)
    Kaa = np.zeros((N_ENH, N_ENH), dtype=np.float64)
    fu = np.zeros(8, dtype=np.float64)
    fa = np.zeros(N_ENH, dtype=np.float64)

    for it in range(30):
        Kuu.fill(0.0)
        Kua.fill(0.0)
        Kaa.fill(0.0)
        fu.fill(0.0)
        fa.fill(0.0)
        for q in range(4):
            B = Bq[q]
            M = Mq[q]
            dV = dVq[q]
            eps = B @ u_elem + M @ alpha
            sdv_pt = sdv0[q, :n_sdv].copy() if n_sdv > 0 else np.zeros(7, dtype=np.float64)
            sigma, C_mat_2d, sdv_new, mat_err = material_dispatch_2d(mat_type, props, sdv_pt, eps, dt)
            if mat_err != 0:
                has_error = 1
            for s_i in range(min(len(sdv_new), n_sdv)):
                sdv_last[q, s_i] = sdv_new[s_i]

            CB = C_mat_2d @ B
            CM = C_mat_2d @ M
            for i in range(8):
                for j in range(8):
                    Kuu[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV
                for j in range(N_ENH):
                    Kua[i, j] += (B[0, i] * CM[0, j] + B[1, i] * CM[1, j] + B[2, i] * CM[2, j]) * dV
                fu[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * dV
            for i in range(N_ENH):
                for j in range(N_ENH):
                    Kaa[i, j] += (M[0, i] * CM[0, j] + M[1, i] * CM[1, j] + M[2, i] * CM[2, j]) * dV
                fa[i] += (M[0, i] * sigma[0] + M[1, i] * sigma[1] + M[2, i] * sigma[2]) * dV

        fa_norm = np.sqrt(np.sum(fa * fa))
        fu_norm = np.sqrt(np.sum(fu * fu))
        if fa_norm <= 1e-12 * max(fu_norm, 1.0e-3):
            break
        d_alpha, solved = solve_small(Kaa, fa.reshape(N_ENH, 1))
        if not solved:
            has_error = 1               # singular enhanced block (collapse): the solver cuts the increment back
            break
        alpha = alpha - d_alpha[:, 0]
    else:
        fa_norm = np.sqrt(np.sum(fa * fa))
        if fa_norm > 1e-6 * max(np.sqrt(np.sum(fu * fu)), 1.0e-3):
            has_error = 1               # alpha did not reach equilibrium: let the increment be cut back

    # write back the converged material state
    for q in range(4):
        for s_i in range(n_sdv):
            sdvs[q, s_i] = sdv_last[q, s_i]

    # Static condensation at the converged alpha:  Ke = Kuu - Kua Kaa^-1 Kua^T ,  fe = fu - Kua Kaa^-1 fa (fa ~ 0)
    invKaa, inverted = solve_small(Kaa, np.eye(N_ENH))
    if not inverted:
        return fu, Kuu, 1               # finite placeholders; has_error makes the solver retry with a smaller step
    Ke = Kuu - Kua @ invKaa @ Kua.T
    fe = fu - Kua @ (invKaa @ fa)

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4i_numba(
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

        fe, Ke, err = compute_cpe4i_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
