"""Boundary condition definitions including prescribed displacements and user-defined kinematic functions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Union
import numpy as np


@dataclass
class BoundaryCondition:
    """Base class for all boundary conditions in CAE model hierarchy."""
    name: str
    region: str  # Qualified Set name (e.g. "PART_INST.NSET_NAME")


@dataclass
class DisplacementBC(BoundaryCondition):
    """Prescribed displacement boundary condition (*BOUNDARY in Abaqus)."""
    u1: Optional[float] = None
    u2: Optional[float] = None
    u3: Optional[float] = None
    ur1: Optional[float] = None
    ur2: Optional[float] = None
    ur3: Optional[float] = None
    amplitude: Optional[Union[str, Any]] = None  # Amplitude name or object


@dataclass
class VelocityBC(BoundaryCondition):
    """Prescribed velocity boundary condition (*BOUNDARY, TYPE=VELOCITY)."""
    v1: Optional[float] = None
    v2: Optional[float] = None
    v3: Optional[float] = None
    amplitude: Optional[Union[str, Any]] = None


@dataclass
class UserFunctionBC(BoundaryCondition):
    """User-defined kinematic boundary condition supporting Python callable or Numba @njit C-kernel function.
    
    Function Signature:
        user_func(coords: np.ndarray, t: float, step_time: float) -> Tuple[Optional[float], Optional[float], Optional[float]]
    where coords is (x, y, z) or (x, y) initial nodal position.
    """
    py_func: Optional[Callable[[np.ndarray, float, float], Any]] = None
    numba_func: Optional[Any] = None

    def evaluate_node(self, coords: np.ndarray, t: float, step_time: float = 0.0) -> Any:
        """Evaluate prescribed displacement (u1, u2, u3) for a node given its coordinates."""
        if self.numba_func is not None:
            return self.numba_func(coords, t, step_time)
        if self.py_func is not None:
            return self.py_func(coords, t, step_time)
        return (None, None, None)
