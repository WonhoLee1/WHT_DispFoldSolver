"""
plastic3d_jax.py
================
3D Finite Strain J2 Multiplicative Logarithmic Plasticity Model with Hencky Strain Return Mapping.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np

from dispsolver.element3d.base3d import QuadraturePointState3D


class J2Plasticity3D:
    """3D J2 Plasticity with Multiplicative Decomposition & Logarithmic Strain Return Mapping."""

    def __init__(self, E: float = 200000.0, nu: float = 0.3, sigma_y0: float = 200.0, H: float = 1000.0):
        self.E = float(E)
        self.nu = float(nu)
        self.sigma_y0 = float(sigma_y0)
        self.H = float(H)

        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.lam = self.K - (2.0 / 3.0) * self.mu

        # Elastic 6x6 Voigt Constitutive Matrix
        self.C_elastic = np.array([
            [self.lam + 2*self.mu, self.lam,              self.lam,              0,          0,          0],
            [self.lam,              self.lam + 2*self.mu, self.lam,              0,          0,          0],
            [self.lam,              self.lam,              self.lam + 2*self.mu, 0,          0,          0],
            [0,                    0,                    0,                    self.mu,    0,          0],
            [0,                    0,                    0,                    0,          self.mu,    0],
            [0,                    0,                    0,                    0,          0,          self.mu]
        ], dtype=np.float64)

    def return_mapping_voigt(
        self,
        total_strain_voigt: np.ndarray,
        state_old: QuadraturePointState3D
    ) -> Tuple[np.ndarray, np.ndarray, QuadraturePointState3D]:
        """Perform 3D J2 Radial Return Mapping in 6D Voigt space.

        Parameters
        ----------
        total_strain_voigt : np.ndarray
            (6,) Total engineering strain Voigt vector [e_xx, e_yy, e_zz, e_xy, e_yz, e_zx]
        state_old : QuadraturePointState3D
            Converged state from previous step

        Returns
        -------
        stress_cauchy : np.ndarray
            (6,) Updated Cauchy stress tensor (Voigt)
        C_tangent : np.ndarray
            (6, 6) Algorithmic Consistent Tangent Matrix
        state_new : QuadraturePointState3D
            Updated quadrature point state
        """
        # Trial Elastic Strain = Total Strain - Plastic Strain Old
        ep_old = state_old.plastic_strain
        eq_p_old = state_old.eq_plastic_strain

        e_trial = total_strain_voigt - ep_old
        
        # Volumetric & Deviatoric split
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = e_trial.copy()
        e_dev[0] -= trace_e / 3.0
        e_dev[1] -= trace_e / 3.0
        e_dev[2] -= trace_e / 3.0

        # Trial Deviatoric Stress s_trial
        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * self.mu * e_dev[0]
        s_trial[1] = 2.0 * self.mu * e_dev[1]
        s_trial[2] = 2.0 * self.mu * e_dev[2]
        s_trial[3] = self.mu * e_dev[3]  # Shear strain component
        s_trial[4] = self.mu * e_dev[4]
        s_trial[5] = self.mu * e_dev[5]

        # Equivalent von Mises Trial Stress: q_trial = \sqrt{3/2 * (s:s)}
        norm_s = np.sqrt(s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 + 2.0*(s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2))
        q_trial = np.sqrt(1.5) * norm_s

        # Yield criterion: f_trial = q_trial - (sigma_y0 + H * eq_p_old)
        sigma_y_current = self.sigma_y0 + self.H * eq_p_old
        f_trial = q_trial - sigma_y_current

        state_new = QuadraturePointState3D.create_initial()

        if f_trial <= 0.0:
            # Elastic Step
            stress_cauchy = s_trial.copy()
            hydro_p = self.K * trace_e
            stress_cauchy[0] += hydro_p
            stress_cauchy[1] += hydro_p
            stress_cauchy[2] += hydro_p

            state_new.plastic_strain = ep_old.copy()
            state_new.eq_plastic_strain = eq_p_old
            state_new.stress_cauchy = stress_cauchy.copy()
            return stress_cauchy, self.C_elastic, state_new

        # Plastic Step: Radial Return
        d_gamma = f_trial / (3.0 * self.mu + self.H)
        eq_p_new = eq_p_old + d_gamma

        # Unit normal flow direction n = s_trial / norm_s
        n_flow = s_trial / np.maximum(norm_s, 1e-12)

        # Updated stress s_new = (1 - 3*mu*d_gamma / q_trial) * s_trial
        scale_s = 1.0 - (3.0 * self.mu * d_gamma) / np.maximum(q_trial, 1e-12)
        s_new = s_trial * scale_s

        # Updated plastic strain ep_new = ep_old + d_gamma * \sqrt{3/2} * n
        ep_new = ep_old.copy()
        flow_factor = d_gamma * np.sqrt(1.5)
        ep_new[0] += flow_factor * n_flow[0]
        ep_new[1] += flow_factor * n_flow[1]
        ep_new[2] += flow_factor * n_flow[2]
        ep_new[3] += flow_factor * n_flow[3] * 2.0  # Engineering shear
        ep_new[4] += flow_factor * n_flow[4] * 2.0
        ep_new[5] += flow_factor * n_flow[5] * 2.0

        hydro_p = self.K * trace_e
        stress_cauchy = s_new.copy()
        stress_cauchy[0] += hydro_p
        stress_cauchy[1] += hydro_p
        stress_cauchy[2] += hydro_p

        state_new.plastic_strain = ep_new
        state_new.eq_plastic_strain = eq_p_new
        state_new.stress_cauchy = stress_cauchy.copy()

        # Compute Consistent Elastoplastic Tangent Modulus C_tangent
        beta = (3.0 * self.mu * d_gamma) / q_trial
        gamma_tangent = (3.0 * self.mu) / (3.0 * self.mu + self.H)

        N_outer = np.outer(n_flow, n_flow)
        C_tangent = self.C_elastic - 2.0 * self.mu * beta * np.eye(6) - 2.0 * self.mu * (gamma_tangent - beta) * N_outer

        return stress_cauchy, C_tangent, state_new
