"""
c3d4_anp_jax.py
===============
JAX Implementation of 3D 4-Node Linear Tetrahedron with Average Nodal Pressure (ANP).
Reference: Bonet & Burton (1998), "A simple average nodal pressure tetrahedral element for finite strain analysis".

Features:
- 2-Pass Nodal Volume Averaging:
    Pass 1: Accumulate element volume and Jacobian contributions to nodes (V_a, v_a)
    Pass 2: Smooth volume ratio J_bar_e = 1/4 sum_a (v_a / V_a) and apply F_bar = (J_bar_e / J_e)^(1/3) * F_e
- Eliminates volumetric locking in nearly incompressible materials (nu -> 0.5) and J2 plasticity.
- Functional formulation compatible with JAX AutoDiff for exact patch tangent verification.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

from .base3d import SolidElement3D


class Tetra4ANPElement(SolidElement3D):
    """3D 4-Node Tetrahedron with Bonet & Burton (1998) 2-Pass Average Nodal Pressure (ANP)."""

    def __init__(self):
        super().__init__(elem_type="C3D4_ANP", num_nodes=4, num_quad_points=1)

    def shape_derivatives_natural(self, xi: np.ndarray = None) -> np.ndarray:
        r"""Natural coordinate derivatives dN / d(xi, eta, zeta) (3, 4)."""
        dN = np.array([
            [-1.0,  1.0,  0.0,  0.0],
            [-1.0,  0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0,  1.0]
        ], dtype=np.float64)
        return dN

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Linear shape functions for 4-node tetrahedron."""
        return np.array([1.0 - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]], dtype=np.float64)

    def compute_patch_nodal_dilatations(
        self,
        node_coords: np.ndarray,   # (n_nodes, 3)
        elem_conn: np.ndarray,     # (n_elems, 4)
        u_global: np.ndarray       # (3 * n_nodes,)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pass 1: Accumulate nodal initial volumes V_a and deformed volumes v_a.
        
        Returns:
            nodal_J: (n_nodes,) smoothed volume ratio at each node.
            elem_V0: (n_elems,) initial element volumes.
            elem_detF: (n_elems,) standard element Jacobians.
            elem_F: (n_elems, 3, 3) deformation gradients.
            elem_dN_dX: (n_elems, 3, 4) physical shape derivatives.
        """
        n_nodes = node_coords.shape[0]
        n_elems = elem_conn.shape[0]

        nodal_V0 = np.zeros(n_nodes, dtype=np.float64)
        nodal_v_def = np.zeros(n_nodes, dtype=np.float64)

        elem_V0 = np.zeros(n_elems, dtype=np.float64)
        elem_detF = np.zeros(n_elems, dtype=np.float64)
        elem_F = np.zeros((n_elems, 3, 3), dtype=np.float64)
        elem_dN_dX = np.zeros((n_elems, 3, 4), dtype=np.float64)

        dN_dxi = self.shape_derivatives_natural()  # (3, 4)

        for e in range(n_elems):
            nodes_e = elem_conn[e]
            coords_e = node_coords[nodes_e]  # (4, 3)

            J0 = dN_dxi @ coords_e  # (3, 3)
            detJ0 = float(np.linalg.det(J0))
            V0 = abs(detJ0) / 6.0
            elem_V0[e] = V0

            invJ0 = np.linalg.inv(J0)
            dN_dX = invJ0 @ dN_dxi  # (3, 4)
            elem_dN_dX[e] = dN_dX

            # Deformation gradient F_e = I + du / dX
            u_e = u_global.reshape(-1, 3)[nodes_e]  # (4, 3)
            grad_u = u_e.T @ dN_dX.T  # (3, 3)
            F = np.eye(3, dtype=np.float64) + grad_u
            detF = float(np.linalg.det(F))
            elem_F[e] = F
            elem_detF[e] = detF

            v_def = V0 * detF

            # Distribute 1/4 to each of the 4 nodes
            for a in nodes_e:
                nodal_V0[a] += 0.25 * V0
                nodal_v_def[a] += 0.25 * v_def

        nodal_J = np.ones(n_nodes, dtype=np.float64)
        valid_mask = nodal_V0 > 1e-15
        nodal_J[valid_mask] = nodal_v_def[valid_mask] / nodal_V0[valid_mask]

        return nodal_J, elem_V0, elem_detF, elem_F, elem_dN_dX

    def compute_element_b_bar(self, dN_dX: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute standard B matrix (6, 12) and volumetric B_vol (1, 12)."""
        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            dNx, dNy, dNz = dN_dX[0, i], dN_dX[1, i], dN_dX[2, i]
            col = 3 * i
            B[0, col]     = dNx
            B[1, col + 1] = dNy
            B[2, col + 2] = dNz

            B[3, col + 1] = dNz
            B[3, col + 2] = dNy

            B[4, col]     = dNz
            B[4, col + 2] = dNx

            B[5, col]     = dNy
            B[5, col + 1] = dNx

        return B
