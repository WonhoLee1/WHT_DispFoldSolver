"""
base3d.py
========
Abstract Base Class and State Management for 3D Solid Elements.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class QuadraturePointState3D:
    """3D Gauss Integration Point Internal State Variables."""
    F: np.ndarray              # (3, 3) Deformation Gradient Tensor
    F_elastic: np.ndarray      # (3, 3) Elastic Deformation Gradient Tensor
    plastic_strain: np.ndarray # (6,) Voigt Plastic Strain Tensor [ep_xx, ep_yy, ep_zz, ep_xy, ep_yz, ep_zx]
    eq_plastic_strain: float   # Equivalent Plastic Strain \bar{\epsilon}^p
    backstress: np.ndarray     # (6,) Kinematic Hardening Backstress
    visco_overstress: np.ndarray # (N_prony, 6) Viscoelastic Memory Tensors
    stress_initial: np.ndarray # (6,) Initial Stress Tensor (Voigt) for Geostatic balance
    stress_cauchy: np.ndarray  # (6,) Cauchy Stress Tensor (Voigt)
    stress_pk2: np.ndarray     # (6,) 2nd Piola-Kirchhoff Stress Tensor (Voigt)

    @classmethod
    def create_initial(cls, num_prony: int = 0) -> QuadraturePointState3D:
        """Factory method to create an identity initial state."""
        return cls(
            F=np.eye(3, dtype=np.float64),
            F_elastic=np.eye(3, dtype=np.float64),
            plastic_strain=np.zeros(6, dtype=np.float64),
            eq_plastic_strain=0.0,
            backstress=np.zeros(6, dtype=np.float64),
            visco_overstress=np.zeros((num_prony, 6), dtype=np.float64) if num_prony > 0 else np.zeros((0, 6), dtype=np.float64),
            stress_initial=np.zeros(6, dtype=np.float64),
            stress_cauchy=np.zeros(6, dtype=np.float64),
            stress_pk2=np.zeros(6, dtype=np.float64)
        )


class SolidElement3D(ABC):
    """Abstract Base Class for all 3D Solid Finite Elements."""

    def __init__(self, elem_type: str, num_nodes: int, num_quad_points: int):
        self.elem_type = elem_type
        self.num_nodes = num_nodes
        self.num_quad_points = num_quad_points
        self.dofs_per_node = 3

    @abstractmethod
    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate shape functions N_i(xi, eta, zeta).

        Parameters
        ----------
        xi : np.ndarray
            Natural coordinates (3,) -> [xi, eta, zeta]

        Returns
        -------
        np.ndarray
            (num_nodes,) shape function values.
        """
        pass

    @abstractmethod
    def shape_derivatives_natural(self, xi: np.ndarray) -> np.ndarray:
        r"""Evaluate derivatives dN_i / d\xi_j in natural coordinates.

        Parameters
        ----------
        xi : np.ndarray
            Natural coordinates (3,) -> [xi, eta, zeta]

        Returns
        -------
        np.ndarray
            (3, num_nodes) array of derivatives dN_i / d(xi, eta, zeta).
        """
        pass

    def jacobian(self, xi: np.ndarray, node_coords: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Compute 3x3 Jacobian matrix J, its determinant detJ, and dN/dX in physical coordinates.

        Parameters
        ----------
        xi : np.ndarray
            Natural coordinates (3,)
        node_coords : np.ndarray
            Physical nodal initial coordinates (num_nodes, 3)

        Returns
        -------
        J : np.ndarray
            (3, 3) Jacobian matrix
        detJ : float
            Determinant of Jacobian
        dN_dX : np.ndarray
            (3, num_nodes) Physical derivatives dN_i / d(X, Y, Z)
        """
        dN_dxi = self.shape_derivatives_natural(xi)  # (3, num_nodes)
        J = dN_dxi @ node_coords                     # (3, 3)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(f"Inverted or degenerate 3D element! detJ = {detJ}")
        invJ = np.linalg.inv(J)
        dN_dX = invJ @ dN_dxi                        # (3, num_nodes)
        return J, detJ, dN_dX

    def b_matrix(self, dN_dX: np.ndarray) -> np.ndarray:
        """Assemble 3D Strain-Displacement Operator B (6, 3 * num_nodes).

        Parameters
        ----------
        dN_dX : np.ndarray
            Physical derivatives dN_i / d(X, Y, Z) of shape (3, num_nodes)

        Returns
        -------
        np.ndarray
            B-matrix of shape (6, 3 * num_nodes) in 6D Voigt order [xx, yy, zz, xy, yz, zx]
        """
        nn = self.num_nodes
        B = np.zeros((6, 3 * nn), dtype=np.float64)
        for i in range(nn):
            dNx, dNy, dNz = dN_dX[0, i], dN_dX[1, i], dN_dX[2, i]
            col = 3 * i
            # Voigt order: [xx, yy, zz, xy, yz, zx]
            B[0, col + 0] = dNx
            B[1, col + 1] = dNy
            B[2, col + 2] = dNz
            
            B[3, col + 0] = dNy
            B[3, col + 1] = dNx

            B[4, col + 1] = dNz
            B[4, col + 2] = dNy

            B[5, col + 0] = dNz
            B[5, col + 2] = dNx
        return B
