"""
q4.py
=====
Q4 Plane Strain Element — Selective Reduced Integration (B-bar).

The bilinear Q4 locks in bending (parasitic shear) and in
near-incompressibility (volumetric locking). The B-bar method (mean
dilatation) cures the latter: the volumetric part of the B-matrix is
evaluated at the element centroid (1-point reduced integration) while the
deviatoric part retains the full 2×2 quadrature.  Written in operator form:

    B̄(ξ,η) = B_dev(ξ,η) + B̄_vol

    B̄_vol = (1/Ωₑ) ∫ B_vol(ξ,η) dΩ   (mean dilatation operator)

    ε(u) = B̄(ξ,η)·u                  (B-bar strain-displacement relation)

This preserves the patch test (B̄ reproduces constant strain exactly) and
eliminates volumetric locking while keeping 2×2 deviatoric accuracy.

Element type hierarchy
----------------------
This codebase provides three Q4 variants for the multi-material fold mesh:

    Q4_BBAR (this file)  —  baseline B-bar element
        * Used for: general elastic continuum (when bending locking is
          not the primary concern)
        * Limitation: still locks in bending for thin (0.017mm) PET
          sub-layers because it has no incompatible bending modes

    Q4_EAS  (q4_eas.py)  —  Enhanced Assumed Strain (Simo & Rifai 1990)
        * Adds 4 internal parameters (EAS-4 modes) that permit the element
          to represent pure bending without spurious shear stress
        * Used for: PET layers in the folding problem (thin bending-dominated)

    Q4_UP   (q4_visco_hybrid_fs_jax.py)  —  Q1P0 mean-dilatation hybrid
        * Replaces the B-bar with a true mixed formulation: displacement
          field + pressure field (P0, constant per element)
        * Used for: PSA layers (ν=0.49, near-incompressible)

Kinematics
----------
Total Lagrangian: all quantities (deformation gradient F, B-matrix,
stiffness) are computed with respect to the reference configuration.
The Jacobian determinant J0 is cached at element construction.

Mass matrix: consistent mass via 2×2 Gaussian quadrature.

References
----------
- Hughes, T.J.R. (1987) "The Finite Element Method", Prentice-Hall.
  Ch. 4: B-bar method, selective reduced integration.
- Bathe, K.J. (2006) "Finite Element Procedures", 2nd ed., Klaus-Jurgen Bathe.
  Ch. 5: isoparametric Q4 formulation.
- Simo, J.C. & Rifai, M.S. (1990). A class of mixed assumed strain methods
  and the method of incompatible modes. IJNME, 29(8), 1595-1638.
  — EAS extension of the B-bar concept.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

# Gauss-Legendre quadrature points and weights for 2×2
_GP2 = np.array([
    [-1.0 / np.sqrt(3), -1.0 / np.sqrt(3)],
    [ 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)],
    [ 1.0 / np.sqrt(3),  1.0 / np.sqrt(3)],
    [-1.0 / np.sqrt(3),  1.0 / np.sqrt(3)],
], dtype=np.float64)
_W2 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

# Center point (xi=0, eta=0) for B-bar volumetric part
_GP0 = np.array([[0.0, 0.0]], dtype=np.float64)
_W0 = np.array([4.0], dtype=np.float64)  # integral of 1 over -1..1 = 4


def shape_functions(xi: float, eta: float) -> np.ndarray:
    """Bilinear shape functions at (ξ, η) for Q4.

    Returns
    -------
    N : (4,) array — [N1, N2, N3, N4]
    """
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


def shape_derivatives(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Shape function derivatives in natural coordinates.

    Returns
    -------
    dN_dxi : (4,) array — ∂N_i/∂ξ
    dN_deta : (4,) array — ∂N_i/∂η
    """
    dN_dxi = 0.25 * np.array([
        -(1.0 - eta),
         (1.0 - eta),
         (1.0 + eta),
        -(1.0 + eta),
    ])
    dN_deta = 0.25 * np.array([
        -(1.0 - xi),
        -(1.0 + xi),
         (1.0 + xi),
         (1.0 - xi),
    ])
    return dN_dxi, dN_deta


def jacobian(xi: float, eta: float, coords: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute Jacobian and its determinant at (ξ, η).

    Parameters
    ----------
    xi, eta : float
        Natural coordinates.
    coords : (4, 2) ndarray
        Element node coordinates [x, y] in CCW order.

    Returns
    -------
    J : (2, 2) ndarray — Jacobian matrix
    detJ : float — Jacobian determinant
    invJ : (2, 2) ndarray — Inverse Jacobian
    """
    dN_dxi, dN_deta = shape_derivatives(xi, eta)
    J = np.zeros((2, 2), dtype=np.float64)
    for i in range(4):
        J[0, 0] += dN_dxi[i] * coords[i, 0]
        J[0, 1] += dN_dxi[i] * coords[i, 1]
        J[1, 0] += dN_deta[i] * coords[i, 0]
        J[1, 1] += dN_deta[i] * coords[i, 1]
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    return J, detJ, invJ


def B_matrix(xi: float, eta: float, invJ: np.ndarray) -> np.ndarray:
    """Standard strain-displacement B matrix (3×8).

    B maps nodal displacements to strains at (ξ, η):
        ε = B · u

    For plane strain: ε = [ε_xx, ε_yy, γ_xy]^T

    Parameters
    ----------
    xi, eta : float — natural coordinates (unused directly, but shape derivatives need them)
    invJ : (2, 2) ndarray — inverse Jacobian

    Returns
    -------
    B : (3, 8) ndarray
    """
    dN_dxi, dN_deta = shape_derivatives(xi, eta)
    B = np.zeros((3, 8), dtype=np.float64)
    for i in range(4):
        # ∂N_i/∂x and ∂N_i/∂y via chain rule
        dN_dx = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
        dN_dy = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]

        col = 2 * i
        B[0, col] = dN_dx          # ε_xx
        B[1, col + 1] = dN_dy      # ε_yy
        B[2, col] = dN_dy          # γ_xy
        B[2, col + 1] = dN_dx      # γ_xy (symmetric)
    return B


def B_bar_matrix(xi: float, eta: float, invJ: np.ndarray,
                 B0: np.ndarray, invJ0: np.ndarray) -> np.ndarray:
    """B-bar strain-displacement matrix for SRI.

    Volumetric part (dilatational) is computed from B0 at element center (1-pt),
    deviatoric part from B at the current Gauss point.

    B_bar = B_dev + B0_vol

    where:
        B_vol = (1/3) * [1, 1, 0; 1, 1, 0; 0, 0, 0] · B      (plane strain)
        B_dev = B - B_vol

    Parameters
    ----------
    xi, eta : float — current Gauss point coordinates
    invJ : (2, 2) — inverse Jacobian at current point
    B0 : (3, 8) — standard B matrix at element center (ξ=0, η=0)
    invJ0 : (2, 2) — inverse Jacobian at element center

    Returns
    -------
    B_bar : (3, 8) ndarray — modified B matrix
    """
    # Standard B at current point
    B_std = B_matrix(xi, eta, invJ)

    # Volumetric projector for plane strain (3×3)
    # ε_vol = (1/3) * (ε_xx + ε_yy) * [1, 1, 0]^T
    P_vol = (1.0 / 2.0) * np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=np.float64)

    # Deviatoric: B_dev = B_std - P_vol @ B_std
    B_dev = B_std - P_vol @ B_std

    # Volumetric part from center: B0_vol = P_vol @ B0
    B0_vol = P_vol @ B0

    return B_dev + B0_vol


def plane_strain_D(E: float, nu: float) -> np.ndarray:
    """Plane strain constitutive matrix D (3×3).

    D = E/((1+ν)(1-2ν)) * [[1-ν, ν,   0  ],
                           [ν,   1-ν,  0  ],
                           [0,   0,   (1-2ν)/2]]

    Parameters
    ----------
    E : float — Young's modulus
    nu : float — Poisson's ratio

    Returns
    -------
    D : (3, 3) ndarray
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ], dtype=np.float64)
    return D


def N_matrix(xi: float, eta: float) -> np.ndarray:
    """Shape function matrix N (2×8) for the element.

    Maps nodal displacements to displacement at (ξ, η):
        u(x) = N(ξ,η) · u_elem

    Returns
    -------
    N : (2, 8) ndarray
    """
    N = np.zeros((2, 8), dtype=np.float64)
    Ns = shape_functions(xi, eta)
    for i in range(4):
        N[0, 2 * i] = Ns[i]
        N[1, 2 * i + 1] = Ns[i]
    return N


# ------------------------------------------------------------------
# Element stiffness (linear elastic)
# ------------------------------------------------------------------

def compute_K_elem(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Compute Q4 element stiffness matrix using SRI B-bar.

    Parameters
    ----------
    coords : (4, 2) ndarray — node coordinates in CCW order
    E : float — Young's modulus
    nu : float — Poisson's ratio

    Returns
    -------
    K : (8, 8) ndarray — element stiffness matrix
    """
    D_mat = plane_strain_D(E, nu)

    # Precompute B0 at element center for volumetric part
    _, _, invJ0 = jacobian(0.0, 0.0, coords)
    B0 = B_matrix(0.0, 0.0, invJ0)

    K = np.zeros((8, 8), dtype=np.float64)
    for k in range(4):
        xi, eta = _GP2[k]
        _, detJ, invJ = jacobian(xi, eta, coords)
        Bb = B_bar_matrix(xi, eta, invJ, B0, invJ0)
        K += Bb.T @ D_mat @ Bb * detJ * _W2[k]
    return K


# ------------------------------------------------------------------
# Consistent mass matrix
# ------------------------------------------------------------------

def compute_M_elem(coords: np.ndarray, rho: float) -> np.ndarray:
    """Compute Q4 consistent mass matrix.

    M = ρ ∫ N^T N dV   (unit thickness)

    Parameters
    ----------
    coords : (4, 2) ndarray — node coordinates
    rho : float — density (mass per unit volume)

    Returns
    -------
    M : (8, 8) ndarray — element consistent mass matrix
    """
    M = np.zeros((8, 8), dtype=np.float64)
    for k in range(4):
        xi, eta = _GP2[k]
        _, detJ, _ = jacobian(xi, eta, coords)
        N = N_matrix(xi, eta)
        M += rho * N.T @ N * detJ * _W2[k]
    return M


# ------------------------------------------------------------------
# Lumped mass matrix
# ------------------------------------------------------------------

def compute_M_lumped(coords: np.ndarray, rho: float) -> np.ndarray:
    """Compute Q4 lumped (diagonal) mass matrix by row-summing.

    Parameters
    ----------
    coords : (4, 4) ndarray — node coordinates
    rho : float — density

    Returns
    -------
    M_lumped : (8, 8) ndarray — diagonal mass matrix
    """
    M_consistent = compute_M_elem(coords, rho)
    diag = np.sum(M_consistent, axis=1)  # row sum
    return np.diag(diag)


# ------------------------------------------------------------------
# Element strain recovery
# ------------------------------------------------------------------

def compute_strains(coords: np.ndarray, u_elem: np.ndarray,
                    xi: float = 0.0, eta: float = 0.0) -> np.ndarray:
    """Compute strains at a given natural coordinate.

    Parameters
    ----------
    coords : (4, 2) ndarray — node coordinates
    u_elem : (8,) ndarray — element nodal displacements [ux1, uy1, ...]
    xi, eta : float — natural coordinate

    Returns
    -------
    epsilon : (3,) ndarray — [ε_xx, ε_yy, γ_xy]
    """
    _, _, invJ = jacobian(xi, eta, coords)
    _, _, invJ0 = jacobian(0.0, 0.0, coords)
    B0 = B_matrix(0.0, 0.0, invJ0)
    Bb = B_bar_matrix(xi, eta, invJ, B0, invJ0)
    return Bb @ u_elem


def compute_stress(coords: np.ndarray, u_elem: np.ndarray,
                   E: float, nu: float,
                   xi: float = 0.0, eta: float = 0.0) -> np.ndarray:
    """Compute stresses at a given natural coordinate.

    Parameters
    ----------
    coords : (4, 2) ndarray
    u_elem : (8,) ndarray
    E : float
    nu : float
    xi, eta : float

    Returns
    -------
    sigma : (3,) ndarray — [σ_xx, σ_yy, τ_xy]
    """
    D_mat = plane_strain_D(E, nu)
    eps = compute_strains(coords, u_elem, xi, eta)
    return D_mat @ eps
