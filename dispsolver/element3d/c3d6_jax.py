"""
c3d6_jax.py
===========
3D 6-Node Linear Triangular Wedge/Prism Element (C3D6).

Reference:
- Abaqus Theory Guide §3.2.1 ("Solid isoparametric elements")
- Zienkiewicz, Taylor & Zhu, The Finite Element Method: Its Basis and Fundamentals.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np

from .base3d import SolidElement3D, QuadraturePointState3D


class Wedge6Element(SolidElement3D):
    """3D 6-Node Linear Wedge/Prism Element (C3D6).
    
    6 Nodes:
      Bottom triangle (zeta = -1): nodes 0, 1, 2
      Top triangle    (zeta = +1): nodes 3, 4, 5
    
    6 Gauss Points:
      3 Gauss points on triangular cross-section x 2 Gauss points along axis (zeta).
    """

    # 6 Gauss points (3-point Hammer triangle x 2-point Gauss-Legendre zeta)
    _SQ3_INV = 1.0 / np.sqrt(3.0)  # ~ 0.577350269

    GAUSS_POINTS_NATURAL = np.array([
        [1.0 / 6.0, 1.0 / 6.0, -_SQ3_INV],
        [2.0 / 3.0, 1.0 / 6.0, -_SQ3_INV],
        [1.0 / 6.0, 2.0 / 3.0, -_SQ3_INV],
        [1.0 / 6.0, 1.0 / 6.0,  _SQ3_INV],
        [2.0 / 3.0, 1.0 / 6.0,  _SQ3_INV],
        [1.0 / 6.0, 2.0 / 3.0,  _SQ3_INV],
    ], dtype=np.float64)

    # Reference triangle area is 0.5. Each 3-point rule weight is 1/6 (sum = 0.5).
    # 2-point Gauss-Legendre weight is 1.0. Product weight = 1/6 * 1.0 = 1/6.
    GAUSS_WEIGHTS = np.full(6, 1.0 / 6.0, dtype=np.float64)

    def __init__(self):
        super().__init__(elem_type="C3D6", num_nodes=6, num_quad_points=6)

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 6-node linear wedge shape functions N_i(xi, eta, zeta)."""
        x, y, z = xi[0], xi[1], xi[2]
        lam1 = 1.0 - x - y
        lam2 = x
        lam3 = y

        half_m = 0.5 * (1.0 - z)
        half_p = 0.5 * (1.0 + z)

        N = np.zeros(6, dtype=np.float64)
        # Bottom triangular face (zeta = -1)
        N[0] = lam1 * half_m
        N[1] = lam2 * half_m
        N[2] = lam3 * half_m

        # Top triangular face (zeta = +1)
        N[3] = lam1 * half_p
        N[4] = lam2 * half_p
        N[5] = lam3 * half_p

        return N

    def shape_derivatives_natural(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate dN_i / d(xi, eta, zeta) (3, 6) in natural coordinates."""
        x, y, z = xi[0], xi[1], xi[2]
        lam1 = 1.0 - x - y
        lam2 = x
        lam3 = y

        half_m = 0.5 * (1.0 - z)
        half_p = 0.5 * (1.0 + z)

        dN = np.zeros((3, 6), dtype=np.float64)

        # dN / dxi (xi = x)
        dN[0, 0] = -half_m
        dN[0, 1] =  half_m
        dN[0, 2] =  0.0
        dN[0, 3] = -half_p
        dN[0, 4] =  half_p
        dN[0, 5] =  0.0

        # dN / deta (eta = y)
        dN[1, 0] = -half_m
        dN[1, 1] =  0.0
        dN[1, 2] =  half_m
        dN[1, 3] = -half_p
        dN[1, 4] =  0.0
        dN[1, 5] =  half_p

        # dN / dzeta (zeta = z)
        dN[2, 0] = -0.5 * lam1
        dN[2, 1] = -0.5 * lam2
        dN[2, 2] = -0.5 * lam3
        dN[2, 3] =  0.5 * lam1
        dN[2, 4] =  0.5 * lam2
        dN[2, 5] =  0.5 * lam3

        return dN

    def compute_element_stiffness_and_force(
        self,
        node_coords: np.ndarray,
        u_elem: np.ndarray,
        C_material: np.ndarray,
        states: Optional[List[QuadraturePointState3D]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute C3D6 element stiffness matrix K_e (18, 18) and internal force vector (18)."""
        K_elem = np.zeros((18, 18), dtype=np.float64)
        f_int = np.zeros(18, dtype=np.float64)

        for pt_idx in range(6):
            pt_nat = self.GAUSS_POINTS_NATURAL[pt_idx]
            weight = self.GAUSS_WEIGHTS[pt_idx]

            _, detJ, dN_dX = self.jacobian(pt_nat, node_coords)
            if detJ <= 0.0:
                raise ValueError(f"Negative or zero Jacobian in C3D6 element: detJ = {detJ}")

            dV = detJ * weight
            B = self.b_matrix(dN_dX)  # (6, 18)

            strain = B @ u_elem
            stress = C_material @ strain

            f_int += (B.T @ stress) * dV
            K_elem += (B.T @ C_material @ B) * dV

        return K_elem, f_int
