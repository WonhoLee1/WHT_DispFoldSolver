"""
t3.py
=====
T3 Plane Strain Element — F-bar formulation.

Formulation
-----------
- Linear triangle (3 nodes, 2 DOF/node → 6 DOF/element)
- Constant strain (linear displacement field)
- F-bar method: deformation gradient is modified so that
  volumetric part uses element-averaged Jacobian
- Plane strain assumption (ε_zz = 0)

For small-strain (linear) analysis, T3 has constant B matrix.
The F-bar correction applies only in large-deformation (hyperelastic)
analysis; for linear elastic, standard B-matrix is sufficient.

References
----------
- de Souza Neto, E.A. et al. "Computational Methods for Plasticity" (2008)
- Bonet, J. & Wood, R.D. "Nonlinear Continuum Mechanics for FEM" (2008)
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

# T3: 1-point quadrature at element centroid (area = 0.5 for unit triangle)
_GP_T3 = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)  # barycentric coords
_W_T3 = np.array([1.0], dtype=np.float64)  # integration weight = area in nat. coords


def shape_functions(xi: float, eta: float) -> np.ndarray:
    """Linear shape functions for T3 in natural coordinates.

    For the standard triangle (ξ≥0, η≥0, ξ+η≤1):
        N1 = 1 - ξ - η
        N2 = ξ
        N3 = η

    Returns
    -------
    N : (3,) array
    """
    return np.array([1.0 - xi - eta, xi, eta])


def shape_derivatives() -> Tuple[np.ndarray, np.ndarray]:
    """Shape function derivatives for T3 (constant).

    Returns
    -------
    dN_dxi : (3,) array
    dN_deta : (3,) array
    """
    dN_dxi = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
    dN_deta = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    return dN_dxi, dN_deta


def jacobian(coords: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Jacobian for T3 (constant over element).

    Since shape function derivatives are constant, the Jacobian is
    constant for the linear triangle.

    Parameters
    ----------
    coords : (3, 2) ndarray — node coordinates in CCW order

    Returns
    -------
    J : (2, 2) ndarray
    detJ : float — = 2 × element area
    invJ : (2, 2) ndarray
    """
    dN_dxi, dN_deta = shape_derivatives()
    J = np.zeros((2, 2), dtype=np.float64)
    for i in range(3):
        J[0, 0] += dN_dxi[i] * coords[i, 0]
        J[0, 1] += dN_dxi[i] * coords[i, 1]
        J[1, 0] += dN_deta[i] * coords[i, 0]
        J[1, 1] += dN_deta[i] * coords[i, 1]
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    return J, detJ, invJ


def B_matrix(invJ: np.ndarray) -> np.ndarray:
    """Constant strain-displacement B matrix (3×6) for T3.

    Parameters
    ----------
    invJ : (2, 2) ndarray — inverse Jacobian (constant)

    Returns
    -------
    B : (3, 6) ndarray
    """
    dN_dxi, dN_deta = shape_derivatives()
    B = np.zeros((3, 6), dtype=np.float64)
    for i in range(3):
        dN_dx = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
        dN_dy = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]
        col = 2 * i
        B[0, col] = dN_dx
        B[1, col + 1] = dN_dy
        B[2, col] = dN_dy
        B[2, col + 1] = dN_dx
    return B


def plane_strain_D(E: float, nu: float) -> np.ndarray:
    """Plane strain constitutive matrix D (3×3).

    Same as Q4.
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ], dtype=np.float64)
    return D


def N_matrix(xi: float, eta: float) -> np.ndarray:
    """Shape function matrix N (2×6) for T3.

    Returns
    -------
    N : (2, 6) ndarray
    """
    N = np.zeros((2, 6), dtype=np.float64)
    Ns = shape_functions(xi, eta)
    for i in range(3):
        N[0, 2 * i] = Ns[i]
        N[1, 2 * i + 1] = Ns[i]
    return N


# ------------------------------------------------------------------
# Element stiffness (linear elastic)
# ------------------------------------------------------------------

def compute_K_elem(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Compute T3 element stiffness matrix.

    For T3, B is constant → K = B^T D B × (element area)

    Parameters
    ----------
    coords : (3, 2) ndarray — node coordinates in CCW order
    E : float — Young's modulus
    nu : float — Poisson's ratio

    Returns
    -------
    K : (6, 6) ndarray — element stiffness matrix
    """
    D_mat = plane_strain_D(E, nu)
    _, detJ, invJ = jacobian(coords)
    B = B_matrix(invJ)
    # Area of element in physical space: A = detJ/2 (detJ = 2*area for triangle mapping)
    area = detJ / 2.0
    return B.T @ D_mat @ B * area


# ------------------------------------------------------------------
# Consistent mass matrix
# ------------------------------------------------------------------

def compute_M_elem(coords: np.ndarray, rho: float) -> np.ndarray:
    """Compute T3 consistent mass matrix (6×6).

    M_ij = ρ ∫ N_i N_j dV  (unit thickness)

    For T3 with constant Jacobian, the integral simplifies to:
        M = ρ * A * (1/12) * [[2, 0, 1, 0, 1, 0], ...] pattern

    Parameters
    ----------
    coords : (3, 2) ndarray
    rho : float — density

    Returns
    -------
    M : (6, 6) ndarray
    """
    _, detJ, invJ = jacobian(coords)
    area = detJ / 2.0

    # Consistent mass for T3 (constant N):
    # ∫ N_i N_j dA = A/12 for i≠j, A/6 for i=j
    M = np.zeros((6, 6), dtype=np.float64)
    factor = rho * area / 12.0
    for i in range(3):
        for j in range(3):
            val = 2.0 * factor if i == j else factor
            M[2 * i, 2 * j] = val          # ux rows × ux cols
            M[2 * i + 1, 2 * j + 1] = val  # uy rows × uy cols
    return M


def compute_M_lumped(coords: np.ndarray, rho: float) -> np.ndarray:
    """Compute T3 lumped mass matrix by row-summing.

    Simple row-sum: equal distribution (1/3 of total mass per node).
    """
    _, detJ, _ = jacobian(coords)
    area = detJ / 2.0
    total_mass = rho * area
    diag_vals = np.full(6, total_mass / 3.0)
    return np.diag(diag_vals)


# ------------------------------------------------------------------
# Element strain recovery
# ------------------------------------------------------------------

def compute_strains(coords: np.ndarray, u_elem: np.ndarray) -> np.ndarray:
    """Compute constant strains for T3.

    Parameters
    ----------
    coords : (3, 2) ndarray
    u_elem : (6,) ndarray — nodal displacements [ux1, uy1, ux2, uy2, ux3, uy3]

    Returns
    -------
    epsilon : (3,) ndarray — [ε_xx, ε_yy, γ_xy]
    """
    _, _, invJ = jacobian(coords)
    B = B_matrix(invJ)
    return B @ u_elem


def compute_stress(coords: np.ndarray, u_elem: np.ndarray,
                   E: float, nu: float) -> np.ndarray:
    """Compute constant stresses for T3.

    Returns
    -------
    sigma : (3,) ndarray — [σ_xx, σ_yy, τ_xy]
    """
    D_mat = plane_strain_D(E, nu)
    eps = compute_strains(coords, u_elem)
    return D_mat @ eps
