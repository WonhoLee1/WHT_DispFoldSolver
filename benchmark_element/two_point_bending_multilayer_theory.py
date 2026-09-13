"""
two_point_bending_multilayer_theory.py
======================================
Analytical closed-form Elastica solutions and Multilayer Composite Beam Mechanics
for Two-Point Bending of Flexible Display Panels & Thin Substrates.

Theoretical Foundations:
------------------------
1. Suresh T. Gulati, Jamie Westbrook, Stephen Carley, Hemanth Vepakomma,
   and Toshihiko Ono (Corning Incorporated),
   "45.2: Two Point Bending of Thin Glass Substrate",
   SID 2004 Digest, ISSN 0004-0966X/04/3502-0001, pp. 1-2.

2. Multilayer Composite Laminate Theory (Parallel Axis Theorem & Shear-Lag):
   - Fully bonded limit: (EI)_full = sum_k (E'_k * I_k + E'_k * A_k * d_k^2)
   - Frictionless slip limit: (EI)_slip = sum_k (E'_k * I_k)
   - Finite interlayer shear compliance (PSA layer shear modulus G_psa):
     (EI)_eff = (EI)_slip + eta * [(EI)_full - (EI)_slip]
     where eta in [0, 1] is the shear transmission efficiency factor.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.integrate as integrate

from verification.two_point_bending import GulatiTwoPointBendingTheory


class LayerSpec:
    """Specification of a single layer in a composite stackup."""

    def __init__(
        self,
        name: str,
        thickness: float,  # mm
        E: float,          # MPa
        nu: float = 0.3,   # Poisson's ratio
        G: Optional[float] = None,  # Shear modulus in MPa (if None, isotropic E / (2*(1+nu)))
        color: str = "#1f77b4"
    ):
        self.name = str(name)
        self.t = float(thickness)
        self.E = float(E)
        self.nu = float(nu)
        self.G = float(G) if G is not None else float(E / (2.0 * (1.0 + nu)))
        self.color = color

    @property
    def E_eff(self) -> float:
        """Plane-strain effective modulus E / (1 - nu^2)."""
        return self.E / (1.0 - self.nu ** 2)


class MultilayerTwoPointBendingTheory(GulatiTwoPointBendingTheory):
    """Analytical & semi-analytical solution for two-point bending of multilayer flexible displays."""

    def __init__(
        self,
        layers: List[LayerSpec],
        D: float = 20.0,            # mm (parallel plate spacing)
        b: float = 1.0,             # mm (unit width in 2D plane strain)
        plane_strain: bool = True,
        shear_efficiency: float = 0.45  # eta: shear transmission efficiency of PSA (0=slip, 1=bonded)
    ):
        self.layers = list(layers)
        self.b = float(b)
        self.eta = float(shear_efficiency)

        # Compute total thickness
        t_total = sum(l.t for l in self.layers)
        self.total_thickness = t_total

        # Layer coordinates along thickness: y in [-t_total/2, t_total/2]
        self.layer_y_bounds = []
        cur_y = -0.5 * t_total
        for l in self.layers:
            self.layer_y_bounds.append((cur_y, cur_y + l.t))
            cur_y += l.t

        # Centroid (neutral axis) of fully bonded laminate
        sum_EA = sum(l.E_eff * (self.b * l.t) for l in self.layers)
        sum_EA_y = sum(l.E_eff * (self.b * l.t) * (0.5 * (y0 + y1))
                       for l, (y0, y1) in zip(self.layers, self.layer_y_bounds))
        self.y_neutral_full = sum_EA_y / sum_EA

        # Fully bonded bending stiffness (EI)_full
        self.EI_full = 0.0
        self.EI_slip = 0.0
        for l, (y0, y1) in zip(self.layers, self.layer_y_bounds):
            I_k = (self.b * l.t ** 3) / 12.0
            A_k = self.b * l.t
            y_mid = 0.5 * (y0 + y1)
            d_k = y_mid - self.y_neutral_full

            # Individual bending contribution
            self.EI_slip += l.E_eff * I_k
            # Parallel axis theorem contribution
            self.EI_full += l.E_eff * (I_k + A_k * (d_k ** 2))

        # Effective composite bending stiffness
        self.EI_eff = self.EI_slip + self.eta * (self.EI_full - self.EI_slip)

        # Equivalent homogeneous modulus E_eq such that (EI)_eff = E_eq * I_geom
        I_geom = (self.b * self.total_thickness ** 3) / 12.0
        E_eq = self.EI_eff / I_geom

        # Initialize base Gulati theory with equivalent parameters
        super().__init__(
            E=E_eq,
            nu=0.3,  # Nominal
            t=self.total_thickness,
            D=D,
            plane_strain=plane_strain
        )

    def compute_thickness_stress_profile(
        self,
        theta_rad: float = np.pi / 2.0,
        n_points_per_layer: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Compute normal bending stress sigma_xx(y) through thickness at a given loop angle theta.
        
        At theta = pi/2 (apex), the bending moment is maximal:
        M = K_GULATI * EI_eff / R_apex, where 1/R_apex = 2.396 / (D - t).
        
        Returns:
            (ys, sigmas, layer_names)
        """
        # Curvature at theta
        kappa = (2.396 / self.D_eff) * np.sqrt(np.maximum(np.sin(theta_rad), 0.0))

        ys_list = []
        sigmas_list = []
        names_list = []

        for l, (y0, y1) in zip(self.layers, self.layer_y_bounds):
            y_pts = np.linspace(y0, y1, n_points_per_layer)
            y_mid = 0.5 * (y0 + y1)

            # In a multilayer with partial slip eta:
            # Strain has two components:
            # 1. Global bending around laminate neutral axis (weighted by eta):
            #    eps_global = -kappa * (y - y_neutral_full) * eta
            # 2. Local individual bending around layer neutral axis (weighted by (1 - eta)):
            #    eps_local = -kappa * (y - y_mid) * (1.0 - eta)
            # Total strain: eps_xx = eps_global + eps_local
            eps_xx = -kappa * (self.eta * (y_pts - self.y_neutral_full) + (1.0 - self.eta) * (y_pts - y_mid))
            sigma_xx = l.E_eff * eps_xx

            ys_list.extend(y_pts)
            sigmas_list.extend(sigma_xx)
            names_list.extend([l.name] * len(y_pts))

        return np.array(ys_list), np.array(sigmas_list), names_list

    def compute_interlayer_shear_stress(
        self,
        theta_rad: float
    ) -> float:
        """Estimate peak interlayer shear stress tau_xy in the compliant adhesive (PSA) layers.
        
        tau_xy ~= (dM/ds) * (Q_eff / I_eff), where dM/ds relates to transverse shear force V.
        """
        # In pure 2-point bending between plates, transverse shear force is concentrated
        # near the transition from free loop to parallel plate contact.
        # Theoretical maximum shear occurs near contact liftoff theta ~= 25-35 deg.
        V_eff = (self.EI_eff / (self.D_eff ** 2)) * np.cos(theta_rad) * np.sqrt(np.maximum(np.sin(theta_rad), 0.0))
        # Normalized shear traction across adhesive interface
        tau_max = 1.5 * abs(V_eff) / (self.b * self.total_thickness)
        return float(tau_max)


def get_standard_display_stackup(model_type: str = "3layer") -> List[LayerSpec]:
    """Generate canonical flexible display stackup configurations.
    
    Models:
      - '3layer': PET (50 um) / PSA (50 um) / PET (50 um) -> total 0.15 mm
      - '5layer': PET (50 um) / PSA (25 um) / PI (50 um) / PSA (25 um) / PET (50 um) -> total 0.20 mm
    """
    model_type = model_type.lower()
    if model_type in ("3layer", "symmetric_3layer"):
        return [
            LayerSpec("Cover PET", thickness=0.050, E=4000.0, nu=0.35, color="#1f77b4"),
            LayerSpec("Opt PSA",   thickness=0.050, E=0.5,    nu=0.49, G=0.168, color="#ff7f0e"),
            LayerSpec("Base PET",  thickness=0.050, E=4000.0, nu=0.35, color="#2ca02c"),
        ]
    elif model_type in ("5layer", "display_5layer"):
        return [
            LayerSpec("Cover Film (PET)", thickness=0.050, E=4000.0, nu=0.35, color="#1f77b4"),
            LayerSpec("OCA Layer 1 (PSA)", thickness=0.025, E=0.5,    nu=0.49, G=0.168, color="#ff7f0e"),
            LayerSpec("Substrate (PI)",    thickness=0.050, E=5000.0, nu=0.34, color="#d62728"),
            LayerSpec("OCA Layer 2 (PSA)", thickness=0.025, E=0.5,    nu=0.49, G=0.168, color="#ff7f0e"),
            LayerSpec("Backplate (PET)",   thickness=0.050, E=4000.0, nu=0.35, color="#2ca02c"),
        ]
    else:
        raise ValueError(f"Unknown stackup model: {model_type}")
