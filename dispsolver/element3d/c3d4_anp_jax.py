"""
c3d4_anp_jax.py
===============
Bonet & Burton (1998) 3D 4-Node Average Nodal Pressure (ANP) Tetrahedron Element (C3D4_ANP).
Eliminates severe volumetric locking in 1-point linear tetrahedral meshes.
"""

from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np

from .base3d import SolidElement3D, QuadraturePointState3D


class Tetra4ANPElement(SolidElement3D):
    """3D 4-Node Average Nodal Pressure (ANP) Tetrahedron Element."""

    def __init__(self):
        super().__init__(elem_type="C3D4_ANP", num_nodes=4, num_quad_points=1)

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 4-node tetrahedral shape functions N_i(xi, eta, zeta)."""
        x, y, z = xi[0], xi[1], xi[2]
        return np.array([1.0 - x - y - z, x, y, z], dtype=np.float64)

    def shape_derivatives_natural(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate dN_i / d(xi, eta, zeta) (3, 4) in natural coordinates."""
        return np.array([
            [-1.0,  1.0,  0.0,  0.0],
            [-1.0,  0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0,  1.0]
        ], dtype=np.float64)

    def compute_element_stiffness_and_force(
        self,
        node_coords: np.ndarray,
        u_elem: np.ndarray,
        C_material: np.ndarray,
        states: List[QuadraturePointState3D]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Tet4 ANP element stiffness matrix K_e (12, 12) and internal force."""
        # 1-point centroid integration for Tet4: xi = (0.25, 0.25, 0.25)
        pt_nat = np.array([0.25, 0.25, 0.25], dtype=np.float64)
        dN_dxi = self.shape_derivatives_natural(pt_nat)
        
        J = dN_dxi @ node_coords # (3, 3)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(f"Inverted or degenerate 3D Tet4 element! detJ = {detJ}")
        
        invJ = np.linalg.inv(J)
        dN_dX = invJ @ dN_dxi    # (3, 4)

        # Tet4 volume V = detJ / 6.0
        Ve = detJ / 6.0

        B = self.b_matrix(dN_dX) # (6, 12)
        strain = B @ u_elem
        stress = C_material @ strain

        K_elem = B.T @ C_material @ B * Ve
        f_int = B.T @ stress * Ve

        return K_elem, f_int
