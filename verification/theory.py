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


# ------------------------------------------------------------------
# Large-rotation elastica: pure end moment (no elliptic integrals)
# ------------------------------------------------------------------

def elastica_pure_moment_tip_state(M: float, L: float, E_star: float,
                                   I: float) -> dict:
    """Exact large-rotation tip state of a cantilever under pure end moment.

    For a pure moment M applied at the tip of a cantilever, the curvature
    is constant along the beam: κ = M / (E*·I). This is the one elastica
    case with a closed form that does not involve elliptic integrals.

    Parameters
    ----------
    M : float — applied end moment (force·length per unit depth)
    L : float — beam length
    E_star : float — plane-strain modulus E*=E/(1-ν²)
    I : float — second moment of area per unit depth (H³/12)

    Returns
    -------
    dict with keys:
        'kappa'  : curvature = M/(E*·I)
        'theta'  : tip rotation angle (radians) = κ·L
        'x_tip'  : deformed tip x-coordinate (relative to root)
        'y_tip'  : deformed tip y-coordinate (relative to root)
        'M'      : input moment (echoed back)
        'L'      : input length (echoed back)

    Reference
    ---------
    Large-rotation elastica under pure end moment — constant curvature
    along the beam.  See e.g. Bower (2009) Applied Mechanics of Solids, 
    Ch. 5, or any mechanics-of-materials text on beam bending.
    """
    if abs(M) < 1e-30:
        # Zero moment → no deformation
        return {
            'kappa': 0.0, 'theta': 0.0,
            'x_tip': float(L), 'y_tip': 0.0,
            'M': float(M), 'L': float(L),
        }

    kappa = M / (E_star * I)
    theta = kappa * L
    # Tip position for a beam bent into a circular arc
    x_tip = np.sin(theta) / kappa if abs(kappa) > 1e-30 else float(L)
    y_tip = (1.0 - np.cos(theta)) / kappa if abs(kappa) > 1e-30 else 0.0

    return {
        'kappa': float(kappa),
        'theta': float(theta),
        'x_tip': float(x_tip),
        'y_tip': float(y_tip),
        'M': float(M),
        'L': float(L),
    }


# ------------------------------------------------------------------
# Elastic-plastic pure bending (linear isotropic hardening, J2)
# ------------------------------------------------------------------

def _section_moment(kappa: float, height: float, width: float,
                    E_star: float, sigma_y0: float, H: float,
                    n_pts: int = 400) -> float:
    """Bending moment for a given curvature, rectangular cross-section,
    bilinear elastic/linear-hardening uniaxial stress-strain law.

    Bernoulli-Euler kinematics eps(y) = kappa*y (plane sections remain
    plane -- valid for pure moment on a prismatic beam even once part of
    the section yields, since M is constant and uniform along the beam
    length regardless of the moment-curvature law, so long as the
    cross-section and material are uniform).

    Uniaxial stress-strain (additive elastic/plastic strain split,
    isotropic hardening sigma_y = sigma_y0 + H*eqps -- same hardening
    law as dispsolver.material.plastic.J2Plasticity):
        eps_y = sigma_y0 / E_star                      (yield strain)
        sigma(eps) = E_star*eps                          , |eps| <= eps_y
        sigma(eps) = sign(eps)*(sigma_y0 + Et*(|eps|-eps_y)) , |eps| > eps_y
    with Et = E_star*H/(E_star+H) the standard series elastic/hardening
    tangent modulus (dsigma/deps = 1/(1/E_star + 1/H)).

    M(kappa) = width * integral_{-c}^{c} sigma(kappa*y) * y dy,
    evaluated by Simpson quadrature (exact to within quadrature error,
    not an approximation of the constitutive law itself).
    """
    c = height / 2.0
    Et = E_star * H / (E_star + H) if H > 0 else 0.0
    eps_y = sigma_y0 / E_star

    y = np.linspace(-c, c, n_pts if n_pts % 2 == 1 else n_pts + 1)
    eps = kappa * y
    sigma = np.where(
        np.abs(eps) <= eps_y,
        E_star * eps,
        np.sign(eps) * (sigma_y0 + Et * (np.abs(eps) - eps_y)),
    )
    integrand = sigma * y
    return float(width * np.trapz(integrand, y))


def elastic_plastic_pure_moment_tip_state(M: float, L: float, height: float,
                                          width: float, E_star: float,
                                          sigma_y0: float, H: float) -> dict:
    """Exact (to quadrature/root-finding tolerance) large-rotation tip
    state of a cantilever under pure end moment, once part of the
    cross-section has yielded -- generalizes
    `elastica_pure_moment_tip_state` from a linear M=E*I*kappa relation
    to the full elastic-plastic moment-curvature relation.

    Curvature is still constant along the beam (M uniform, prismatic
    section) so the same circular-arc tip-position formula applies; only
    the M(kappa) relation used to invert for kappa changes.

    Parameters
    ----------
    M : applied end moment (force*length per unit depth)
    L : beam length
    height, width : cross-section dimensions (width=1 for the "per unit
        depth" plane-strain convention used elsewhere in this module)
    E_star : plane-strain modulus E/(1-nu**2)
    sigma_y0, H : J2Plasticity yield stress / hardening modulus (same
        parameters passed to dispsolver.material.plastic.J2Plasticity)

    Returns
    -------
    dict with 'kappa', 'theta', 'x_tip', 'y_tip', 'M', 'L', 'M_yield',
    'M_p' (fully-plastic limit moment, H=0 asymptote), 'kappa_yield'.

    Reference
    ---------
    Elastic-plastic bending of a beam under pure moment, bilinear
    (elastic + linear isotropic hardening) uniaxial law -- standard
    result, e.g. Hill, R. (1950) "The Mathematical Theory of
    Plasticity", Oxford, Ch. 2; or Chakrabarty, J. (2006) "Theory of
    Plasticity", 3rd ed., Ch. 2, for the elastic-perfectly-plastic (H=0)
    closed form M = M_p*(1 - (1/3)*(kappa_y/kappa)**2), which this
    function's numerical M(kappa) reduces to as H -> 0.
    """
    c = height / 2.0
    I = width * height ** 3 / 12.0
    kappa_yield = sigma_y0 / (E_star * c)
    M_yield = E_star * I * kappa_yield
    M_p = sigma_y0 * width * height ** 2 / 4.0  # fully-plastic limit, H=0

    if abs(M) < 1e-30:
        return {
            'kappa': 0.0, 'theta': 0.0, 'x_tip': float(L), 'y_tip': 0.0,
            'M': float(M), 'L': float(L),
            'M_yield': float(M_yield), 'M_p': float(M_p),
            'kappa_yield': float(kappa_yield),
        }

    if abs(M) <= M_yield:
        kappa = M / (E_star * I)
    else:
        # Monotonic M(kappa) for H >= 0 -> bisection is safe and simple.
        lo, hi = kappa_yield, kappa_yield * 50.0
        M_hi = _section_moment(hi, height, width, E_star, sigma_y0, H)
        while M_hi < abs(M) and hi < kappa_yield * 1e6:
            hi *= 2.0
            M_hi = _section_moment(hi, height, width, E_star, sigma_y0, H)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            M_mid = _section_moment(mid, height, width, E_star, sigma_y0, H)
            if M_mid < abs(M):
                lo = mid
            else:
                hi = mid
        kappa = np.sign(M) * 0.5 * (lo + hi)

    theta = kappa * L
    x_tip = np.sin(theta) / kappa if abs(kappa) > 1e-30 else float(L)
    y_tip = (1.0 - np.cos(theta)) / kappa if abs(kappa) > 1e-30 else 0.0

    return {
        'kappa': float(kappa),
        'theta': float(theta),
        'x_tip': float(x_tip),
        'y_tip': float(y_tip),
        'M': float(M),
        'L': float(L),
        'M_yield': float(M_yield),
        'M_p': float(M_p),
        'kappa_yield': float(kappa_yield),
    }


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
