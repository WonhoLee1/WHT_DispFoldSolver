"""
stabilization.py
================
Abaqus Automatic Viscous Stabilization Engine (*STATIC, STABILIZE).

Formulation
-----------
Adds local viscous damping to regularise transient numerical instabilities,
local wrinkling, and snap-through buckling during high curvature folding.

    F_stab = c · M_diag · v

where v = Δu / Δt is the pseudo-velocity over the current static increment.
The corresponding tangent stiffness contribution is:

    K_stab = (c / Δt) · M_diag

The damping factor c is scaled or verified to ensure that the ratio of
dissipated stabilization energy (ALLSD) to total strain energy remains strictly below
the Abaqus standard tolerance (default 0.05 / 5% or user-defined ALLSDVTOL):

    ALLSD / ALLIE < ALLSDVTOL
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.sparse import issparse, diags, csr_matrix


class AbaqusViscousStabilization:
    """Abaqus-compatible automatic viscous stabilization state container."""

    def __init__(
        self,
        damping_factor: float = 2e-4,
        energy_fraction_limit: float = 0.05,
        enabled: bool = True
    ):
        self.damping_factor = float(damping_factor)
        self.energy_fraction_limit = float(energy_fraction_limit)
        self.enabled = bool(enabled)
        self.accumulated_stab_energy: float = 0.0
        self.total_strain_energy: float = 0.0

    def compute_stabilization_force(self, v: np.ndarray, M_diag: np.ndarray) -> np.ndarray:
        """Compute stabilization damping force F_stab = c * M_diag * v."""
        return self.damping_factor * (M_diag * v)

    def update_and_check_energy(self, du: np.ndarray, f_stab: np.ndarray, strain_energy: float) -> Tuple[float, bool]:
        """Update stabilization energy and verify if E_stab / E_strain < tolerance."""
        dE_stab = 0.5 * float(np.dot(du, f_stab))
        self.accumulated_stab_energy += abs(dE_stab)
        self.total_strain_energy = max(float(strain_energy), 1e-12)

        ratio = self.accumulated_stab_energy / self.total_strain_energy
        is_acceptable = ratio <= self.energy_fraction_limit

        # Auto scale damping factor if ratio exceeds limit
        if not is_acceptable:
            self.damping_factor *= 0.5

        return ratio, is_acceptable


class StaticStabilizationIncrement:
    """Manages stabilization transformation over a single static Newton increment."""

    def __init__(self, solver: Any, dim: int, dt: float, stab: AbaqusViscousStabilization):
        self.solver = solver
        self.dim = dim
        self.dt = max(float(dt), 1e-30)
        self.stab = stab
        self.u_n = np.array(solver.u, dtype=np.float64, copy=True)
        from dispsolver.solver.implicit_dynamics import lumped_mass
        self.m = lumped_mass(solver, dim)
        self.c = float(stab.damping_factor)
        self.c_eff = self.c / self.dt
        self.k_stab_diag = self.c_eff * self.m

    def transform(self, res: tuple, u: np.ndarray) -> tuple:
        """Add stabilization damping force to f_int and damping stiffness to K."""
        K, f_int = res[0], res[1]
        rest = res[2:]
        du = np.asarray(u, dtype=np.float64) - self.u_n
        f_stab = self.k_stab_diag * du
        f_eff = f_int + f_stab

        if issparse(K):
            K_eff = (K + diags(self.k_stab_diag, format="csr")).tocsr()
        else:
            K_eff = np.array(K, copy=True)
            np.fill_diagonal(K_eff, K_eff.diagonal() + self.k_stab_diag)

        return (K_eff, f_eff) + rest

    def commit(self) -> None:
        """On converged increment: accumulate dissipated stabilization energy (ALLSD)."""
        du = np.asarray(self.solver.u, dtype=np.float64) - self.u_n
        f_stab = self.k_stab_diag * du
        dE_stab = 0.5 * float(np.dot(du, f_stab))
        self.stab.accumulated_stab_energy += abs(dE_stab)
        self.solver.energy_allsd = self.stab.accumulated_stab_energy
        self.solver.allsd = self.solver.energy_allsd


def apply_step_stabilization(solver: Any, step: Any) -> None:
    """Configure solver stabilization from step definition (*STATIC, STABILIZE)."""
    stab_cfg = getattr(step, "stabilize", None)
    if not hasattr(solver, "energy_allsd"):
        solver.energy_allsd = 0.0
        solver.allsd = 0.0

    if not stab_cfg:
        # Check if already present on solver or disable
        if hasattr(solver, "_stabilization") and solver._stabilization is not None:
            # Preserve accumulated energy across steps but disable damping
            solver._stabilization.enabled = False
        return

    if isinstance(stab_cfg, dict):
        enabled = bool(stab_cfg.get("enabled", True))
        factor = float(stab_cfg.get("factor", 2e-4))
        allsdvtol = float(stab_cfg.get("allsdvtol", 0.05))
    elif isinstance(stab_cfg, bool):
        enabled = stab_cfg
        factor = 2e-4
        allsdvtol = 0.05
    elif isinstance(stab_cfg, (int, float)):
        enabled = True
        factor = float(stab_cfg)
        allsdvtol = 0.05
    else:
        enabled = True
        factor = 2e-4
        allsdvtol = 0.05

    if hasattr(solver, "_stabilization") and isinstance(solver._stabilization, AbaqusViscousStabilization):
        solver._stabilization.enabled = enabled
        solver._stabilization.damping_factor = factor
        solver._stabilization.energy_fraction_limit = allsdvtol
    else:
        solver._stabilization = AbaqusViscousStabilization(
            damping_factor=factor,
            energy_fraction_limit=allsdvtol,
            enabled=enabled
        )
        # Link accumulated energy
        solver._stabilization.accumulated_stab_energy = float(getattr(solver, "energy_allsd", 0.0))
