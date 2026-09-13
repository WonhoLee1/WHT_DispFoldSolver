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

    def Elastic(self, table: Any = None, E: Optional[float] = None, nu: Optional[float] = None) -> None:
        """Abaqus-style Elastic property definition (supports table tuple or E, nu keywords)."""
        if E is not None and nu is not None:
            self.elastic = (float(E), float(nu))
        elif isinstance(table, (list, tuple)) and len(table) > 0:
            if isinstance(table[0], (list, tuple)):
                self.elastic = (float(table[0][0]), float(table[0][1]))
            else:
                self.elastic = (float(table[0]), float(table[1]))

    def Plastic(self, table: Any) -> None:
        """Abaqus-style Plastic property definition."""
        self.plastic = [(float(row[0]), float(row[1])) for row in table]

    def Hyperelastic(self, model: str = "NEO_HOOKEAN", **kwargs) -> None:
        """Abaqus-style Hyperelastic property definition."""
        self.hyperelastic = {"model": model, **kwargs}

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

        # Fallback attribute extraction
        E_val = getattr(self, "E", None)
        nu_val = getattr(self, "nu", None)
        if self.elastic is not None:
            E_val, nu_val = self.elastic
        elif E_val is None:
            E_val = 1.0
        if nu_val is None:
            nu_val = 0.3
        
        if upper_type == "ELASTIC" or upper_type == "LINEAR_ELASTIC":
            props["E"] = float(E_val)
            props["nu"] = float(nu_val)
            props["mat_type"] = MAT_LINEAR_ELASTIC
        elif upper_type in ["J2_PLASTICITY", "PLASTIC"]:
            props["E"] = float(E_val)
            props["nu"] = float(nu_val)
            props["mat_type"] = MAT_J2_PLASTICITY
            sy0 = getattr(self, "sigma_y0", getattr(self, "yield_stress", None))
            H_mod = getattr(self, "H", getattr(self, "hardening_modulus", None))
            if self.plastic and len(self.plastic) > 0:
                props["yield_stress"] = float(self.plastic[0][0])
                if len(self.plastic) > 1:
                    sy0_p, ep0 = self.plastic[0]
                    sy1_p, ep1 = self.plastic[1]
                    dep = ep1 - ep0
                    props["hardening_modulus"] = float((sy1_p - sy0_p) / dep) if dep > 1e-12 else 0.0
                else:
                    props["hardening_modulus"] = float(H_mod) if H_mod is not None else float(self.extra_params.get("hardening_modulus", 0.0))
            else:
                props["yield_stress"] = float(sy0) if sy0 is not None else float(self.extra_params.get("yield_stress", 1e9))
                props["hardening_modulus"] = float(H_mod) if H_mod is not None else float(self.extra_params.get("hardening_modulus", 0.0))
        elif upper_type in ["NEO_HOOKEAN", "HYPERELASTIC"]:
            props["mat_type"] = MAT_NEO_HOOKEAN
            c10_val = getattr(self, "C10", getattr(self, "c10", None))
            d1_val = getattr(self, "D1", getattr(self, "d1", None))
            if self.hyperelastic:
                c10_val = self.hyperelastic.get("c10", self.hyperelastic.get("C10", c10_val))
                d1_val = self.hyperelastic.get("d1", self.hyperelastic.get("D1", d1_val))
            
            if c10_val is not None and d1_val is not None:
                props["c10"] = float(c10_val)
                props["d1"] = float(d1_val)
                # Compute effective E and nu for solver dispatch
                mu = 2.0 * float(c10_val)
                K = 2.0 / float(d1_val) if float(d1_val) > 1e-12 else 1e6
                props["E"] = float((9.0 * K * mu) / (3.0 * K + mu))
                props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu)))
            elif self.elastic:
                props["E"] = float(E_val)
                props["nu"] = float(nu_val)
                mu = E_val / (2.0 * (1.0 + nu_val))
                K = E_val / (3.0 * (1.0 - 2.0 * nu_val))
                props["c10"] = float(mu / 2.0)
                props["d1"] = float(2.0 / K) if K > 1e-12 else 1e-6
            else:
                props["c10"] = 0.1
                props["d1"] = 0.01
                props["E"] = 0.6
                props["nu"] = 0.49
        else:
            props["mat_type"] = MAT_CUSTOM_ELASTIC
            
        for k, v in self.extra_params.items():
            if k not in props:
                props[k] = v
                
        return props
