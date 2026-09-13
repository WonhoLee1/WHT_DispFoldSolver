"""
base2d.py
========
Abstract Base Class and State Management for 2D Solid Elements.
Mirroring 3D element architecture for full codebase parity.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class QuadraturePointState2D:
    """2D Gauss Integration Point Internal State Variables."""
    F: np.ndarray                # (2, 2) In-plane Deformation Gradient Tensor
    F_elastic: np.ndarray        # (2, 2) Elastic In-plane Deformation Gradient
    plastic_strain: np.ndarray   # (4,) Voigt Plastic Strain [ep_xx, ep_yy, ep_zz, ep_xy]
    eq_plastic_strain: float     # Equivalent Plastic Strain \bar{\epsilon}^p
    backstress: np.ndarray       # (4,) Kinematic Hardening Backstress
    visco_overstress: np.ndarray # (N_prony, 4) Viscoelastic Memory Tensors
    stress_initial: np.ndarray   # (4,) Initial Voigt Stress [s_xx, s_yy, s_zz, s_xy]
    stress_cauchy: np.ndarray    # (4,) Cauchy Voigt Stress
    stress_pk2: np.ndarray       # (4,) 2nd Piola-Kirchhoff Voigt Stress

    @classmethod
    def create_initial(cls, num_prony: int = 0) -> QuadraturePointState2D:
        """Factory method to create an identity initial state."""
        return cls(
            F=np.eye(2, dtype=np.float64),
            F_elastic=np.eye(2, dtype=np.float64),
            plastic_strain=np.zeros(4, dtype=np.float64),
            eq_plastic_strain=0.0,
            backstress=np.zeros(4, dtype=np.float64),
            visco_overstress=np.zeros((num_prony, 4), dtype=np.float64) if num_prony > 0 else np.zeros((0, 4), dtype=np.float64),
            stress_initial=np.zeros(4, dtype=np.float64),
            stress_cauchy=np.zeros(4, dtype=np.float64),
            stress_pk2=np.zeros(4, dtype=np.float64)
        )


class SolidElement2D(ABC):
    """Abstract Base Class for all 2D Solid Finite Elements."""

    def __init__(self, elem_type: str, num_nodes: int, num_quad_points: int):
        self.elem_type = elem_type
        self.num_nodes = num_nodes
        self.num_quad_points = num_quad_points
        self.dofs_per_node = 2

    @abstractmethod
    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate shape functions N_i(xi, eta)."""
        pass

    @abstractmethod
    def shape_function_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate natural derivatives dN_i/d(xi, eta)."""
        pass

    @abstractmethod
    def get_quadrature_points(self) -> List[Tuple[np.ndarray, float]]:
        """Return integration points and weights: list of (xi_vec, weight)."""
        pass
