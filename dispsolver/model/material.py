"""Material definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class Material:
    """Constitutive material definition with mechanical properties."""
    name: str
    mat_type: str = "ELASTIC"  # "ELASTIC", "J2_PLASTICITY", "NEO_HOOKEAN", "VISCOELASTIC"
    density: float = 0.0
    elastic: Optional[Tuple[float, float]] = None  # (E, nu)
    plastic: Optional[List[Tuple[float, float]]] = None  # [(sigma_y, eps_p), ...]
    hyperelastic: Optional[Dict[str, Any]] = None  # {'model': 'ARRUDA_BOYCE', 'c10': ..., 'd1': ...}
    viscoelastic_prony: Optional[List[Tuple[float, float, float]]] = None  # [(g, k, tau), ...]
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_numba_props(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Numba DOD materials."""
        from dispsolver.material3d.numba_materials import (
            MAT_LINEAR_ELASTIC,
            MAT_J2_PLASTICITY,
            MAT_NEO_HOOKEAN,
            MAT_CUSTOM_ELASTIC
        )
        
        props: Dict[str, Any] = {}
        upper_type = self.mat_type.upper()
        
        if upper_type == "ELASTIC" or upper_type == "LINEAR_ELASTIC":
            E, nu = self.elastic if self.elastic else (1.0, 0.0)
            props["E"] = float(E)
            props["nu"] = float(nu)
            props["mat_type"] = MAT_LINEAR_ELASTIC
        elif upper_type in ["J2_PLASTICITY", "PLASTIC"]:
            E, nu = self.elastic if self.elastic else (1.0, 0.0)
            props["E"] = float(E)
            props["nu"] = float(nu)
            props["mat_type"] = MAT_J2_PLASTICITY
            if self.plastic and len(self.plastic) > 0:
                props["yield_stress"] = float(self.plastic[0][0])
                if len(self.plastic) > 1:
                    # Estimate linear hardening modulus H from initial plastic points
                    sy0, ep0 = self.plastic[0]
                    sy1, ep1 = self.plastic[1]
                    dep = ep1 - ep0
                    props["hardening_modulus"] = float((sy1 - sy0) / dep) if dep > 1e-12 else 0.0
                else:
                    props["hardening_modulus"] = float(self.extra_params.get("hardening_modulus", 0.0))
            else:
                props["yield_stress"] = float(self.extra_params.get("yield_stress", 1e9))
                props["hardening_modulus"] = float(self.extra_params.get("hardening_modulus", 0.0))
        elif upper_type in ["NEO_HOOKEAN", "HYPERELASTIC"]:
            props["mat_type"] = MAT_NEO_HOOKEAN
            if self.elastic:
                E, nu = self.elastic
                props["E"] = float(E)
                props["nu"] = float(nu)
                # Compute Neo-Hookean C10 and D1 from E and nu
                mu = E / (2.0 * (1.0 + nu))
                K = E / (3.0 * (1.0 - 2.0 * nu))
                props["c10"] = float(mu / 2.0)
                props["d1"] = float(2.0 / K) if K > 1e-12 else 1e-6
            elif self.hyperelastic:
                props["c10"] = float(self.hyperelastic.get("c10", self.hyperelastic.get("C10", 0.5)))
                props["d1"] = float(self.hyperelastic.get("d1", self.hyperelastic.get("D1", 0.001)))
                props["E"] = float(4.0 * props["c10"] * (1.0 + 0.49))  # approximate
                props["nu"] = 0.49
        else:
            props["mat_type"] = MAT_CUSTOM_ELASTIC
            
        for k, v in self.extra_params.items():
            if k not in props:
                props[k] = v
                
        return props
