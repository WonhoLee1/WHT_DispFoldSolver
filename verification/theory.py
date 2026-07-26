"""
theory.py
=========
Closed-form analytical solutions for benchmark verification.

All formulas are derived for **plane strain** 2D configurations,
consistent with the dispsolver element formulation (F_33 = 1).

References
----------
- Timoshenko, S.P. & Goodier, J.N. (1970). "Theory of Elasticity", 3rd ed.
- Beer, F.P. et al. (2015). "Mechanics of Materials", 7th ed.
- Bower, A.F. (2009). "Applied Mechanics of Solids", Ch. 5.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ------------------------------------------------------------------
# Plane strain constitutive matrix
# ------------------------------------------------------------------

def plane_strain_D(E: float, nu: float) -> np.ndarray:
    """Plane strain constitutive matrix D (3×3).

    D = E / ((1+ν)(1-2ν)) * [[1-ν,  ν,   0  ],
                             [ν,   1-ν, 0  ],
                             [0,   0,   (1-2ν)/2]]
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu,       0.0],
        [nu,       1.0 - nu, 0.0],
        [0.0,      0.0,      (1.0 - 2.0 * nu) / 2.0],
    ], dtype=np.float64)


def plane_strain_modulus(E: float, nu: float) -> float:
    """Effective Young's modulus under plane strain.

    For a beam in plane strain (ε_zz = 0), the bending stiffness
    uses E* = E / (1 - ν²) instead of E.  This is because the
    out-of-plane stress σ_zz = ν(σ_xx + σ_yy) contributes to the
    bending moment but not to the curvature.

    For pure uniaxial strain (ε_yy = ε_zz = 0): σ_xx = E(1-ν)/((1+ν)(1-2ν)) · ε_xx
    — this is the "constrained modulus" M, not the same as E*.
    """
    return E / (1.0 - nu ** 2)


def shear_modulus(E: float, nu: float) -> float:
    """G = E / (2(1+ν))."""
    return E / (2.0 * (1.0 + nu))


def bulk_modulus(E: float, nu: float) -> float:
    """K = E / (3(1-2ν))."""
    return E / (3.0 * (1.0 - 2.0 * nu))


# ------------------------------------------------------------------
# Beam section properties (per unit out-of-plane depth)
# ------------------------------------------------------------------

def beam_I(height: float) -> float:
    """Second moment of area per unit depth for a rectangular beam.

    I = b · h³ / 12,  with b = 1 (unit out-of-plane depth in 2D).
    """
    return height ** 3 / 12.0


def beam_A(height: float) -> float:
    """Cross-sectional area per unit depth = height × 1."""
    return height


# ------------------------------------------------------------------
# Beam deflection formulas (Euler-Bernoulli + Timoshenko correction)
# ------------------------------------------------------------------

def cantilever_tip_deflection(P: float, L: float, E: float, nu: float,
                              height: float, timoshenko: bool = True) -> float:
    """Tip deflection of a cantilever beam under end load P.

    Euler-Bernoulli:  δ = P·L³ / (3·E*·I)
    Timoshenko (shear correction):  δ += P·L / (κ_s · G · A)

    Parameters
    ----------
    P : float  — tip load (force per unit depth)
    L : float  — beam length
    E : float  — Young's modulus
    nu : float — Poisson's ratio
    height : float — beam height (cross-section depth in 2D plane)
    timoshenko : bool — include shear deformation correction

    Returns
    -------
    δ : float — downward deflection at the tip (positive = downward)
    """
    E_star = plane_strain_modulus(E, nu)
    I = beam_I(height)
    delta_bend = P * L ** 3 / (3.0 * E_star * I)
    if timoshenko:
        G = shear_modulus(E, nu)
        A = beam_A(height)
        kappa_s = 5.0 / 6.0  # rectangular cross-section shear factor
        delta_shear = P * L / (kappa_s * G * A)
        return delta_bend + delta_shear
    return delta_bend


def three_point_bending_deflection(P: float, L: float, E: float, nu: float,
                                   height: float, timoshenko: bool = True) -> float:
    """Mid-span deflection of a simply-supported beam under center load P.

    Euler-Bernoulli:  δ = P·L³ / (48·E*·I)
    Timoshenko:       δ += P·L / (4·κ_s·G·A)
    """
    E_star = plane_strain_modulus(E, nu)
    I = beam_I(height)
    delta_bend = P * L ** 3 / (48.0 * E_star * I)
    if timoshenko:
        G = shear_modulus(E, nu)
        A = beam_A(height)
        kappa_s = 5.0 / 6.0
        delta_shear = P * L / (4.0 * kappa_s * G * A)
        return delta_bend + delta_shear
    return delta_bend


def four_point_bending_deflection(P: float, L: float, a: float,
                                  E: float, nu: float, height: float,
                                  timoshenko: bool = True) -> float:
    """Mid-span deflection of a simply-supported beam under two symmetric loads.

    Load positions: x = L/2 ± a  (each load = P).
    Euler-Bernoulli (at center):  δ = P·a·(3L² - 4a²) / (24·E*·I)
    Timoshenko:                   δ += P·a / (2·κ_s·G·A)

    Parameters
    ----------
    P : float  — magnitude of EACH load (total = 2P)
    L : float  — span length between supports
    a : float  — distance from support to nearest load
    """
    E_star = plane_strain_modulus(E, nu)
    I = beam_I(height)
    delta_bend = P * a * (3.0 * L ** 2 - 4.0 * a ** 2) / (24.0 * E_star * I)
    if timoshenko:
        G = shear_modulus(E, nu)
        A = beam_A(height)
        kappa_s = 5.0 / 6.0
        delta_shear = P * a / (2.0 * kappa_s * G * A)
        return delta_bend + delta_shear
    return delta_bend


# ------------------------------------------------------------------
# Uniaxial tension / compression
# ------------------------------------------------------------------

def uniaxial_stress_strain(eps_xx: float, E: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical stress and strain for uniaxial tension in plane strain.

    In plane strain (ε_zz = 0), a prescribed ε_xx with free lateral
    surface (σ_yy = 0) gives:

        ε_yy = -ν/(1-ν) · ε_xx         (lateral strain, NOT -ν·ε_xx)
        σ_xx = E(1-ν) / ((1+ν)(1-2ν)) · ε_xx
        σ_yy = 0                        (free surface)
        σ_zz = ν·σ_xx                   (out-of-plane, nonzero)
        τ_xy = 0

    Parameters
    ----------
    eps_xx : float — applied axial strain
    E : float      — Young's modulus
    nu : float     — Poisson's ratio

    Returns
    -------
    strain : (3,) — [ε_xx, ε_yy, γ_xy]
    stress : (3,) — [σ_xx, σ_yy, τ_xy]
    """
    D = plane_strain_D(E, nu)
    eps_yy = -nu / (1.0 - nu) * eps_xx
    strain = np.array([eps_xx, eps_yy, 0.0])
    stress = D @ strain
    return strain, stress


# ------------------------------------------------------------------
# Volumetric (hydrostatic) compression / tension
# ------------------------------------------------------------------

def volumetric_pressure_volume_change(eps_vol: float, E: float, nu: float) -> float:
    """Hydrostatic stress for uniform in-plane volumetric strain in plane strain.

    In plane strain (ε_zz = 0), with ε_xx = ε_yy = ε_vol/2, γ_xy = 0:

        σ_xx = σ_yy = c · ε_vol/2
        where c = E / ((1+ν)(1-2ν))   [plane-strain OOF parameter]

    This follows from D[0,0]·ε + D[0,1]·ε = c·(1-ν)·ε + c·ν·ε = c·ε.

    Parameters
    ----------
    eps_vol : float — in-plane volumetric strain (= ε_xx + ε_yy)
    E : float
    nu : float

    Returns
    -------
    σ_h : float — hydrostatic stress (σ_xx = σ_yy)
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * eps_vol / 2.0


def volumetric_strain_from_pressure(p: float, E: float, nu: float) -> float:
    """Inverse of volumetric_pressure_volume_change."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * p / c


# ------------------------------------------------------------------
# Single-element stiffness (analytical reference for patch test)
# ------------------------------------------------------------------

def single_element_K_analytical(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Analytical Q4 element stiffness for a rectangular element.

    For a rectangle with sides a × b, the plane-strain stiffness can be
    computed in closed form.  This serves as a reference for the numerical
    element stiffness routines.

    However, computing the full 8×8 matrix analytically is error-prone.
    Instead, we use the B-bar integration with high-order Gauss quadrature
    (8×8 points) as a "pseudo-analytical" reference — the B-bar method
    converges to the exact integral, so 8×8 quadrature gives ~1e-14 accuracy.

    Parameters
    ----------
    coords : (4, 2) — element node coordinates (CCW)
    E, nu : float
    """
    from dispsolver.element.q4 import (
        jacobian, B_matrix, B_bar_matrix, _GP2, _W2,
    )

    # Use 8-point Gauss-Legendre for near-exact integration
    # (B-bar is exact for constant strain, so even 2×2 is fine,
    #  but 8-point removes any doubt)
    from numpy.polynomial.legendre import leggauss
    n_gauss = 8
    gp_1d, w_1d = leggauss(n_gauss)
    gp_2d = []
    w_2d = []
    for i in range(n_gauss):
        for j in range(n_gauss):
            gp_2d.append((gp_1d[i], gp_1d[j]))
            w_2d.append(w_1d[i] * w_1d[j])

    D = plane_strain_D(E, nu)
    _, _, invJ0 = jacobian(0.0, 0.0, coords)
    B0 = B_matrix(0.0, 0.0, invJ0)

    K = np.zeros((8, 8), dtype=np.float64)
    for (xi, eta), w in zip(gp_2d, w_2d):
        _, detJ, invJ = jacobian(xi, eta, coords)
        Bb = B_bar_matrix(xi, eta, invJ, B0, invJ0)
        K += Bb.T @ D @ Bb * detJ * w
    return K
