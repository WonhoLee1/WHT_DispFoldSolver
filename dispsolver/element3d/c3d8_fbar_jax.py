"""
c3d8_fbar_jax.py
================
de Souza Neto et al. (1996) 3D 8-Node Multiplicative F-bar Element (C3D8_FBAR).
Removes volumetric locking in nearly incompressible finite strain plasticity and viscoelasticity.
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np

from .base3d import SolidElement3D, QuadraturePointState3D


class Hexa8FbarElement(SolidElement3D):
    """3D 8-Node Multiplicative F-bar Element for Finite Strain Incompressible Materials."""

    NODE_NATURAL_COORDS = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0]
    ], dtype=np.float64)

    GAUSS_1D = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)

    def __init__(self):
        super().__init__(elem_type="C3D8_FBAR", num_nodes=8, num_quad_points=8)

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 8-node trilinear shape functions N_i(xi, eta, zeta)."""
        xi_n = self.NODE_NATURAL_COORDS
        N = 0.125 * (1.0 + xi_n[:, 0] * xi[0]) * (1.0 + xi_n[:, 1] * xi[1]) * (1.0 + xi_n[:, 2] * xi[2])
        return N

    def shape_derivatives_natural(self, xi: np.ndarray) -> np.ndarray:
        r"""Evaluate derivatives dN_i / d\xi_j (3, 8) in natural coordinates."""
        xi_n = self.NODE_NATURAL_COORDS
        dN = np.zeros((3, 8), dtype=np.float64)
        dN[0, :] = 0.125 * xi_n[:, 0] * (1.0 + xi_n[:, 1] * xi[1]) * (1.0 + xi_n[:, 2] * xi[2])
        dN[1, :] = 0.125 * xi_n[:, 1] * (1.0 + xi_n[:, 0] * xi[0]) * (1.0 + xi_n[:, 2] * xi[2])
        dN[2, :] = 0.125 * xi_n[:, 2] * (1.0 + xi_n[:, 0] * xi[0]) * (1.0 + xi_n[:, 1] * xi[1])
        return dN

    def compute_volume_averaged_jbar(self, node_coords: np.ndarray, u_elem: np.ndarray) -> Tuple[float, float]:
        """Compute volume-averaged volume ratio J_bar = V_deformed / V_initial."""
        V0 = 0.0
        V_def = 0.0

        current_coords = node_coords + u_elem.reshape(8, 3)

        for xi in self.GAUSS_1D:
            for eta in self.GAUSS_1D:
                for zeta in self.GAUSS_1D:
                    pt_nat = np.array([xi, eta, zeta], dtype=np.float64)
                    dN_dxi = self.shape_derivatives_natural(pt_nat)
                    
                    J_0 = dN_dxi @ node_coords
                    detJ0 = float(np.linalg.det(J_0))

                    J_curr = dN_dxi @ current_coords
                    detJ_curr = float(np.linalg.det(J_curr))

                    dV0 = detJ0 * 1.0  # Weight = 1.0
                    V0 += dV0
                    V_def += detJ_curr * 1.0

        J_bar = V_def / np.maximum(V0, 1e-14)
        return J_bar, V0

    def compute_fbar_b_matrix(self, dN_dX: np.ndarray, dN_dX_bar: np.ndarray) -> np.ndarray:
        """Compute F-bar modified B-matrix: B_bar = B_dev + 1/3 * B_vol_mean."""
        B_std = self.b_matrix(dN_dX)      # (6, 24)
        B_mean = self.b_matrix(dN_dX_bar) # (6, 24)

        B_bar = B_std.copy()
        # Replace volumetric component (rows 0, 1, 2) with mean volumetric strain
        vol_mean = (B_mean[0, :] + B_mean[1, :] + B_mean[2, :]) / 3.0
        vol_std = (B_std[0, :] + B_std[1, :] + B_std[2, :]) / 3.0

        for row in range(3):
            B_bar[row, :] = B_std[row, :] - vol_std + vol_mean

        return B_bar

    def compute_element_stiffness_and_force(
        self,
        node_coords: np.ndarray,
        u_elem: np.ndarray,
        C_material: np.ndarray,
        states: List[QuadraturePointState3D]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute F-bar element stiffness matrix K_e (24, 24) and internal force."""
        # 1. Compute element volume-averaged dN/dX_bar
        dN_dX_mean = np.zeros((3, 8), dtype=np.float64)
        V0_total = 0.0

        for xi in self.GAUSS_1D:
            for eta in self.GAUSS_1D:
                for zeta in self.GAUSS_1D:
                    pt_nat = np.array([xi, eta, zeta], dtype=np.float64)
                    _, detJ, dN_dX = self.jacobian(pt_nat, node_coords)
                    dV = detJ * 1.0
                    dN_dX_mean += dN_dX * dV
                    V0_total += dV

        dN_dX_bar = dN_dX_mean / V0_total

        # 2. Quadrature Integration with B_bar
        K_elem = np.zeros((24, 24), dtype=np.float64)
        f_int = np.zeros(24, dtype=np.float64)

        for xi in self.GAUSS_1D:
            for eta in self.GAUSS_1D:
                for zeta in self.GAUSS_1D:
                    pt_nat = np.array([xi, eta, zeta], dtype=np.float64)
                    _, detJ, dN_dX = self.jacobian(pt_nat, node_coords)
                    
                    B_bar = self.compute_fbar_b_matrix(dN_dX, dN_dX_bar)
                    strain_bar = B_bar @ u_elem
                    stress = C_material @ strain_bar

                    dV = detJ * 1.0
                    K_elem += B_bar.T @ C_material @ B_bar * dV
                    f_int += B_bar.T @ stress * dV

        return K_elem, f_int
