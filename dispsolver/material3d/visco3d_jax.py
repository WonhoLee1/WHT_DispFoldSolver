"""
visco3d_jax.py
==============
Simo-Hughes (1998) 3D Finite Strain Viscoelasticity Model with Prony Series Relaxation.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np

from dispsolver.element3d.base3d import QuadraturePointState3D


class Viscoelastic3D:
    """3D Simo-Hughes Large Strain Viscoelasticity with Prony Series Memory Tensors."""

    def __init__(
        self,
        E_instant: float = 10.0,
        nu: float = 0.45,
        prony_g: Optional[List[float]] = None,
        prony_tau: Optional[List[float]] = None
    ):
        self.E_instant = float(E_instant)
        self.nu = float(nu)

        if prony_g is None or prony_tau is None:
            # Default single Maxwell mode
            self.prony_g = [0.3]
            self.prony_tau = [5.0]
        else:
            self.prony_g = [float(g) for g in prony_g]
            self.prony_tau = [float(t) for t in prony_tau]

        self.num_prony = len(self.prony_g)
        self.g_infinity = 1.0 - sum(self.prony_g)

        self.mu_0 = self.E_instant / (2.0 * (1.0 + self.nu))
        self.K_0 = self.E_instant / (3.0 * (1.0 - 2.0 * self.nu))
        self.lam_0 = self.K_0 - (2.0 / 3.0) * self.mu_0

        # Instantaneous Elastic 6x6 Voigt Constitutive Matrix
        self.C_instant = np.array([
            [self.lam_0 + 2*self.mu_0, self.lam_0,                 self.lam_0,                 0,            0,            0],
            [self.lam_0,                 self.lam_0 + 2*self.mu_0, self.lam_0,                 0,            0,            0],
            [self.lam_0,                 self.lam_0,                 self.lam_0 + 2*self.mu_0, 0,            0,            0],
            [0,                         0,                         0,                         self.mu_0,    0,            0],
            [0,                         0,                         0,                         0,            self.mu_0,    0],
            [0,                         0,                         0,                         0,            0,            self.mu_0]
        ], dtype=np.float64)

    def update_state_voigt(
        self,
        strain_voigt: np.ndarray,
        dstrain_voigt: np.ndarray,
        state_old: QuadraturePointState3D,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, QuadraturePointState3D]:
        """Perform 3D Prony Series Viscoelastic constitutive update.

        Parameters
        ----------
        strain_voigt : np.ndarray
            (6,) Total strain Voigt vector at t^{n+1}
        dstrain_voigt : np.ndarray
            (6,) Strain increment Voigt vector \Delta \epsilon
        state_old : QuadraturePointState3D
            Previous state container
        dt : float
            Time step increment \Delta t

        Returns
        -------
        stress_cauchy : np.ndarray
            (6,) Viscoelastic Cauchy stress tensor
        C_tangent : np.ndarray
            (6, 6) Algorithmic Consistent Tangent Matrix
        state_new : QuadraturePointState3D
            Updated quadrature point state
        """
        # Instantaneous Elastic Stress Response
        stress_instant = self.C_instant @ strain_voigt
        
        # Volumetric & Deviatoric split
        trace_sig = stress_instant[0] + stress_instant[1] + stress_instant[2]
        s_instant = stress_instant.copy()
        s_instant[0] -= trace_sig / 3.0
        s_instant[1] -= trace_sig / 3.0
        s_instant[2] -= trace_sig / 3.0

        dstress_inst = self.C_instant @ dstrain_voigt
        ds_inst = dstress_inst.copy()
        trace_ds = dstress_inst[0] + dstress_inst[1] + dstress_inst[2]
        ds_inst[0] -= trace_ds / 3.0
        ds_inst[1] -= trace_ds / 3.0
        ds_inst[2] -= trace_ds / 3.0

        state_new = QuadraturePointState3D.create_initial(num_prony=self.num_prony)

        # Update Prony Memory Overstresses Q_k
        Q_new = np.zeros((self.num_prony, 6), dtype=np.float64)
        sum_Q = np.zeros(6, dtype=np.float64)

        for k in range(self.num_prony):
            g_k = self.prony_g[k]
            tau_k = self.prony_tau[k]
            
            if dt > 0.0:
                gamma_k = dt / np.maximum(tau_k, 1e-14)
                exp_factor = np.exp(-gamma_k)
                if gamma_k < 1e-6:
                    A_k = 1.0 - 0.5 * gamma_k
                else:
                    A_k = (1.0 - exp_factor) / gamma_k
            else:
                exp_factor = 1.0
                A_k = 1.0

            Q_old_k = state_old.visco_overstress[k] if (state_old.visco_overstress is not None and state_old.visco_overstress.shape[0] > k) else np.zeros(6, dtype=np.float64)
            Q_new_k = exp_factor * Q_old_k + g_k * A_k * ds_inst
            
            Q_new[k] = Q_new_k
            sum_Q += Q_new_k

        # Total Stress: \sigma = g_\infty * s_instant + p*I + \sum Q_k
        stress_visco = self.g_infinity * s_instant + sum_Q
        p_hydro = trace_sig / 3.0
        stress_visco[0] += p_hydro
        stress_visco[1] += p_hydro
        stress_visco[2] += p_hydro

        state_new.visco_overstress = Q_new
        state_new.stress_cauchy = stress_visco.copy()

        # Consistent Tangent Modulus
        sum_alpha = sum(self.prony_g)
        C_tangent = self.C_instant.copy()
        # Scale deviatoric part by (g_infinity + sum_alpha)
        scale_dev = self.g_infinity + sum_alpha
        for r in range(6):
            for c in range(6):
                if r < 3 and c < 3:
                    if r == c:
                        C_tangent[r, c] = self.K_0 + (4.0 / 3.0) * self.mu_0 * scale_dev
                    else:
                        C_tangent[r, c] = self.K_0 - (2.0 / 3.0) * self.mu_0 * scale_dev
                elif r >= 3 and r == c:
                    C_tangent[r, c] = self.mu_0 * scale_dev

        return stress_visco, C_tangent, state_new
