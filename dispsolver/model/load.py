"""Load definitions for commercial CAE model hierarchy (Abaqus parity).

Provides:
- ConcentratedForce (Cload): Nodal point forces and moments.
- Pressure: Surface distributed normal pressure (*PRESSURE in Abaqus).
- Gravity: Gravitational body force acceleration vector.
- BodyForce: Volumetric body force per unit volume.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Sequence
import numpy as np


@dataclass
class Load:
    """Base class for all loads in CAE model hierarchy."""
    name: str
    create_step_name: str = "Initial"


@dataclass
class ConcentratedForce(Load):
    """Concentrated nodal point force or moment (*CLOAD in Abaqus).
    
    Applies forces (cf1, cf2, cf3) and/or moments (cm1, cm2, cm3) to a node set.
    """
    region: Union[str, Any] = ""  # Set name or GeneralSet
    cf1: Optional[float] = None
    cf2: Optional[float] = None
    cf3: Optional[float] = None
    cm1: Optional[float] = None
    cm2: Optional[float] = None
    cm3: Optional[float] = None
    amplitude: Optional[Union[str, Any]] = None

    def get_force_vector(self, t: float = 0.0, step_time: float = 0.0, model: Any = None) -> np.ndarray:
        """Get (3,) or (6,) load magnitude vector scaled by amplitude at time t."""
        scale = 1.0
        if self.amplitude is not None:
            if hasattr(self.amplitude, "evaluate"):
                scale = float(self.amplitude.evaluate(t, step_time))
            elif isinstance(self.amplitude, str) and model is not None and hasattr(model, "amplitudes"):
                amp_obj = model.amplitudes.get(self.amplitude)
                if amp_obj is not None and hasattr(amp_obj, "evaluate"):
                    scale = float(amp_obj.evaluate(t, step_time))
        
        f1 = (self.cf1 if self.cf1 is not None else 0.0) * scale
        f2 = (self.cf2 if self.cf2 is not None else 0.0) * scale
        f3 = (self.cf3 if self.cf3 is not None else 0.0) * scale
        return np.array([f1, f2, f3], dtype=np.float64)


@dataclass
class Pressure(Load):
    """Distributed normal pressure load on surfaces (*PRESSURE in Abaqus).
    
    Positive pressure acts inward against the outward surface normal:
        f_surf = - integral_A (P * n * N^T) dA
    """
    region: Union[str, Any] = ""  # Surface name or GeneralSet containing exterior faces/segments
    magnitude: float = 0.0
    amplitude: Optional[Union[str, Any]] = None
    distribution: str = "UNIFORM"

    def get_magnitude(self, t: float = 0.0, step_time: float = 0.0, model: Any = None) -> float:
        """Get pressure magnitude scaled by amplitude at time t."""
        scale = 1.0
        if self.amplitude is not None:
            if hasattr(self.amplitude, "evaluate"):
                scale = float(self.amplitude.evaluate(t, step_time))
            elif isinstance(self.amplitude, str) and model is not None and hasattr(model, "amplitudes"):
                amp_obj = model.amplitudes.get(self.amplitude)
                if amp_obj is not None and hasattr(amp_obj, "evaluate"):
                    scale = float(amp_obj.evaluate(t, step_time))
        return self.magnitude * scale


@dataclass
class Gravity(Load):
    """Uniform gravitational acceleration field (*DLOAD, GRAV in Abaqus).
    
    Generates body forces f_body = integral_V (rho * g * N^T) dV.
    """
    comp1: float = 0.0  # g_x
    comp2: float = 0.0  # g_y
    comp3: float = 0.0  # g_z
    amplitude: Optional[Union[str, Any]] = None
    region: Optional[Union[str, Any]] = None  # None applies to entire model

    def get_acceleration_vector(self, t: float = 0.0, step_time: float = 0.0, model: Any = None) -> np.ndarray:
        """Get (3,) acceleration vector scaled by amplitude at time t."""
        scale = 1.0
        if self.amplitude is not None:
            if hasattr(self.amplitude, "evaluate"):
                scale = float(self.amplitude.evaluate(t, step_time))
            elif isinstance(self.amplitude, str) and model is not None and hasattr(model, "amplitudes"):
                amp_obj = model.amplitudes.get(self.amplitude)
                if amp_obj is not None and hasattr(amp_obj, "evaluate"):
                    scale = float(amp_obj.evaluate(t, step_time))
        return np.array([self.comp1 * scale, self.comp2 * scale, self.comp3 * scale], dtype=np.float64)


@dataclass
class BodyForce(Load):
    """Volumetric body force per unit volume (*DLOAD, CENT/BX/BY/BZ in Abaqus)."""
    region: Optional[Union[str, Any]] = None
    b1: float = 0.0
    b2: float = 0.0
    b3: float = 0.0
    amplitude: Optional[Union[str, Any]] = None
