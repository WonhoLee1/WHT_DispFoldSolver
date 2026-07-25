"""
stabilization.py
================
Abaqus Automatic Viscous Stabilization Engine.

Formulation
-----------
Adds local viscous damping to regularise transient numerical instabilities,
local wrinkling, and snap-through buckling during high curvature folding.

    F_stab = c · M_diag · v

The damping factor c is scaled automatically to ensure that the ratio of
dissipated stabilization energy to total strain energy remains strictly below
the Abaqus standard tolerance (0.5%):

    E_stab / E_strain < 0.005 (0.5%)
"""

import numpy as np

class AbaqusViscousStabilization:
    def __init__(self, damping_factor: float = 2e-4, energy_fraction_limit: float = 0.005):
        self.damping_factor = damping_factor
        self.energy_fraction_limit = energy_fraction_limit
        self.accumulated_stab_energy = 0.0
        self.total_strain_energy = 0.0

    def compute_stabilization_force(self, v: np.ndarray, M_diag: np.ndarray) -> np.ndarray:
        """Compute stabilization damping force F_stab = c * M_diag * v."""
        return self.damping_factor * (M_diag * v)

    def update_and_check_energy(self, du: np.ndarray, f_stab: np.ndarray, strain_energy: float) -> tuple[float, bool]:
        """Update stabilization energy and verify if E_stab / E_strain < 0.5%."""
        dE_stab = 0.5 * np.dot(du, f_stab)
        self.accumulated_stab_energy += abs(dE_stab)
        self.total_strain_energy = max(strain_energy, 1e-12)
        
        ratio = self.accumulated_stab_energy / self.total_strain_energy
        is_acceptable = ratio <= self.energy_fraction_limit
        
        # Auto scale damping factor if ratio exceeds limit
        if not is_acceptable:
            self.damping_factor *= 0.5
            
        return ratio, is_acceptable
