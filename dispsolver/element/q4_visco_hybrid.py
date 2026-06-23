"""
q4_visco_hybrid.py
==================
Q1P0 hybrid (mean-dilatation) element for small-strain LINEAR viscoelasticity.

The volumetric response is treated with the B-bar mean-dilatation operator
(equivalent to a constant-pressure Q1P0 field), eliminating volumetric locking,
while the deviatoric response relaxes through the material's Prony series with
WLF time-temperature superposition.

This is a *stress-based* element (history-dependent dissipation cannot be
expressed as a single energy potential, so the JAX energy-autodiff hybrid path
used for hyperelasticity does not apply here).

Returns the element internal force (8,), tangent stiffness (8,8), and the
updated per-Gauss-point internal state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import q4


def compute_visco_hybrid_contributions(
    coords: np.ndarray,         # (4, 2) node coordinates
    u_elem: np.ndarray,         # (8,) element displacements
    state_elem: np.ndarray,     # (n_gp, 4*(M+1)) internal state
    material,                   # LinearViscoelastic
    dt: float,
    temperature: float = 20.0,
    thickness: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_gp = len(q4._GP2)
    K_bulk = material.K

    # Prony coefficients are constant over the element for a given (dt, T)
    a_coef, gamma_coef, G_alg = material.prony_coeffs(dt, temperature)

    # Small-strain isotropic plane-strain tangent with relaxed shear modulus.
    # lam_eff = K - 2/3 G_alg  (so bulk modulus is exactly K)
    lam_eff = K_bulk - (2.0 / 3.0) * G_alg
    D = np.array([
        [lam_eff + 2.0 * G_alg, lam_eff, 0.0],
        [lam_eff, lam_eff + 2.0 * G_alg, 0.0],
        [0.0, 0.0, G_alg],
    ], dtype=np.float64)

    # Reference geometry for the mean-dilatation (B-bar) operator
    _, _, invJ0 = q4.jacobian(0.0, 0.0, coords)
    B0 = q4.B_matrix(0.0, 0.0, invJ0)

    f_int = np.zeros(8, dtype=np.float64)
    K_e = np.zeros((8, 8), dtype=np.float64)
    state_new = np.zeros_like(state_elem)

    for gp in range(n_gp):
        xi, eta = q4._GP2[gp]
        _, detJ, invJ = q4.jacobian(xi, eta, coords)
        w = detJ * q4._W2[gp] * thickness

        # Mean-dilatation B-bar: deviatoric strain local, volumetric = element mean
        Bb = q4.B_bar_matrix(xi, eta, invJ, B0, invJ0)
        eps = Bb @ u_elem                      # [eps_xx, eps_yy, gamma_xy]
        theta = eps[0] + eps[1]                # mean dilatation (plane strain)

        # Tensorial deviatoric strain (plane strain: eps_zz = 0)
        e_dev = np.array([
            eps[0] - theta / 3.0,
            eps[1] - theta / 3.0,
            -theta / 3.0,
            0.5 * eps[2],
        ], dtype=np.float64)

        s_dev, sg_new = material.dev_update(e_dev, state_elem[gp], a_coef, gamma_coef)
        p = K_bulk * theta

        # Total stress (Voigt [xx, yy, xy]); s_dev[3] is the xy shear stress
        sig = np.array([s_dev[0] + p, s_dev[1] + p, s_dev[3]], dtype=np.float64)

        f_int += Bb.T @ sig * w
        K_e += (Bb.T @ D @ Bb) * w
        state_new[gp] = sg_new

    return f_int, K_e, state_new
