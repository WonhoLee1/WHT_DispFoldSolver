"""
c3d10_jax.py
============
3D 10-Node Standard Quadratic Tetrahedron Element (C3D10).

Note: Abaqus C3D10M modifies shape functions for contact consistency (strictly positive
surface weights) and includes hourglass control. This is the standard 10-node quadratic
tetrahedron formulation (C3D10).
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np

from .base3d import SolidElement3D, QuadraturePointState3D


class Tetra10Element(SolidElement3D):
    """3D 10-Node Quadratic Tetrahedron Element (C3D10)."""

    # 4-point Gauss Quadrature natural coordinates (xi, eta, zeta) & weights
    a = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0  # ~ 0.58541020
    b = (5.0 - np.sqrt(5.0)) / 20.0        # ~ 0.13819660

    GAUSS_POINTS_NATURAL = np.array([
        [a, b, b],
        [b, a, b],
        [b, b, a],
        [b, b, b]
    ], dtype=np.float64)

    GAUSS_WEIGHTS = np.array([1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0], dtype=np.float64)

    def __init__(self):
        super().__init__(elem_type="C3D10", num_nodes=10, num_quad_points=4)

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 10-node quadratic tetrahedral shape functions N_i(xi, eta, zeta)."""
        x, y, z = xi[0], xi[1], xi[2]
        w = 1.0 - x - y - z  # Fourth barycentric coordinate \lambda_4

        N = np.zeros(10, dtype=np.float64)
        # Corner nodes 1..4
        N[0] = w * (2.0 * w - 1.0)
        N[1] = x * (2.0 * x - 1.0)
        N[2] = y * (2.0 * y - 1.0)
        N[3] = z * (2.0 * z - 1.0)

        # Mid-side nodes 5..10
        N[4] = 4.0 * w * x  # Edge 1-2
        N[5] = 4.0 * x * y  # Edge 2-3
        N[6] = 4.0 * y * w  # Edge 3-1
        N[7] = 4.0 * w * z  # Edge 1-4
        N[8] = 4.0 * x * z  # Edge 2-4
        N[9] = 4.0 * y * z  # Edge 3-4

        return N

    def shape_derivatives_natural(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate dN_i / d(xi, eta, zeta) (3, 10) in natural coordinates."""
        x, y, z = xi[0], xi[1], xi[2]
        w = 1.0 - x - y - z

        dN = np.zeros((3, 10), dtype=np.float64)

        # dN / dxi
        dN[0, 0] = 1.0 - 4.0 * w
        dN[0, 1] = 4.0 * x - 1.0
        dN[0, 2] = 0.0
        dN[0, 3] = 0.0
        dN[0, 4] = 4.0 * (w - x)
        dN[0, 5] = 4.0 * y
        dN[0, 6] = -4.0 * y
        dN[0, 7] = -4.0 * z
        dN[0, 8] = 4.0 * z
        dN[0, 9] = 0.0

        # dN / deta
        dN[1, 0] = 1.0 - 4.0 * w
        dN[1, 1] = 0.0
        dN[1, 2] = 4.0 * y - 1.0
        dN[1, 3] = 0.0
        dN[1, 4] = -4.0 * x
        dN[1, 5] = 4.0 * x
        dN[1, 6] = 4.0 * (w - y)
        dN[1, 7] = -4.0 * z
        dN[1, 8] = 0.0
        dN[1, 9] = 4.0 * z

        # dN / dzeta
        dN[2, 0] = 1.0 - 4.0 * w
        dN[2, 1] = 0.0
        dN[2, 2] = 0.0
        dN[2, 3] = 4.0 * z - 1.0
        dN[2, 4] = -4.0 * x
        dN[2, 5] = 0.0
        dN[2, 6] = -4.0 * y
        dN[2, 7] = 4.0 * (w - z)
        dN[2, 8] = 4.0 * x
        dN[2, 9] = 4.0 * y

        return dN

    def compute_element_stiffness_and_force(
        self,
        node_coords: np.ndarray,
        u_elem: np.ndarray,
        C_material: np.ndarray,
        states: List[QuadraturePointState3D]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Tet10 element stiffness matrix K_e (30, 30) and internal force."""
        K_elem = np.zeros((30, 30), dtype=np.float64)
        f_int = np.zeros(30, dtype=np.float64)

        for pt_idx in range(4):
            pt_nat = self.GAUSS_POINTS_NATURAL[pt_idx]
            weight = self.GAUSS_WEIGHTS[pt_idx]

            _, detJ, dN_dX = self.jacobian(pt_nat, node_coords)
            dV = detJ * weight

            B = self.b_matrix(dN_dX)  # (6, 30)
            strain = B @ u_elem
            stress = C_material @ strain

            K_elem += B.T @ C_material @ B * dV
            f_int += B.T @ stress * dV

        return K_elem, f_int
