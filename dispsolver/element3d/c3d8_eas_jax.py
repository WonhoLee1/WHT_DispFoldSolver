"""
c3d8_eas_jax.py
===============
Simo & Armero (1992) 3D 8-Node Enhanced Assumed Strain (EAS) Element.
Mitigates shear and volumetric locking in 3D bending and incompressible plasticity.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
import jax
import jax.numpy as jnp

from .base3d import SolidElement3D, QuadraturePointState3D


class Hexa8EASElement(SolidElement3D):
    """3D 8-Node Enhanced Assumed Strain Element (C3D8I) with 9 Internal Modes."""

    # 8-node natural nodal coordinates \xi_i, \eta_i, \zeta_i \in {-1, +1}
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

    # 2x2x2 Gauss Quadrature points (8 points)
    GAUSS_1D = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
    GAUSS_WEIGHTS_1D = np.array([1.0, 1.0], dtype=np.float64)

    def __init__(self, num_eas_modes: int = 9):
        super().__init__(elem_type="C3D8I", num_nodes=8, num_quad_points=8)
        self.num_eas_modes = num_eas_modes

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

    def compute_transformation_matrix_t0(self, J0: np.ndarray) -> np.ndarray:
        r"""Compute 6D Voigt Transformation Matrix T_0 at element center (Simo-Armero 1992)."""
        J11, J12, J13 = J0[0, 0], J0[0, 1], J0[0, 2]
        J21, J22, J23 = J0[1, 0], J0[1, 1], J0[1, 2]
        J31, J32, J33 = J0[2, 0], J0[2, 1], J0[2, 2]

        T0 = np.array([
            [J11**2, J21**2, J31**2, 2.0*J11*J21, 2.0*J21*J31, 2.0*J31*J11],
            [J12**2, J22**2, J32**2, 2.0*J12*J22, 2.0*J22*J32, 2.0*J32*J12],
            [J13**2, J23**2, J33**2, 2.0*J13*J23, 2.0*J23*J33, 2.0*J33*J13],
            [J11*J12, J21*J22, J31*J32, J11*J22+J12*J21, J21*J32+J22*J31, J31*J12+J32*J11],
            [J12*J13, J22*J23, J32*J33, J12*J23+J13*J22, J22*J33+J23*J32, J32*J13+J33*J12],
            [J13*J11, J23*J21, J33*J31, J13*J21+J11*J23, J23*J31+J21*J33, J33*J11+J31*J13]
        ], dtype=np.float64)
        return T0

    def enhanced_strain_matrix_m(self, xi: np.ndarray) -> np.ndarray:
        """9-mode natural strain interpolation matrix M(xi) (6, 9)."""
        x, y, z = xi[0], xi[1], xi[2]
        M = np.zeros((6, 9), dtype=np.float64)
        # Normal modes
        M[0, 0] = x
        M[1, 1] = y
        M[2, 2] = z
        # Shear modes
        M[3, 3] = x
        M[3, 4] = y
        M[4, 5] = y
        M[4, 6] = z
        M[5, 7] = z
        M[5, 8] = x
        return M

    def compute_element_stiffness_and_force(
        self,
        node_coords: np.ndarray,
        u_elem: np.ndarray,
        C_material: np.ndarray,
        states: List[QuadraturePointState3D]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute 3D element stiffness matrix K_e (24, 24) with EAS static condensation.

        Parameters
        ----------
        node_coords : np.ndarray
            (8, 3) initial nodal coordinates
        u_elem : np.ndarray
            (24,) current element nodal displacement vector
        C_material : np.ndarray
            (6, 6) Voigt Elastic Constitutive Matrix
        states : List[QuadraturePointState3D]
            8 Quadrature point state objects

        Returns
        -------
        K_condensed : np.ndarray
            (24, 24) Condensated Element Stiffness Matrix
        f_int : np.ndarray
            (24,) Internal Force Vector
        alpha_opt : np.ndarray
            (9,) Optimized internal strain parameters
        """
        nn = self.num_nodes
        num_eas = self.num_eas_modes
        
        # Central Jacobian J0 and Transformation Matrix T0
        J0, detJ0, _ = self.jacobian(np.zeros(3), node_coords)
        T0 = self.compute_transformation_matrix_t0(J0)
        invT0_T = np.linalg.inv(T0).T  # (6, 6)

        K_uu = np.zeros((24, 24), dtype=np.float64)
        K_ua = np.zeros((24, num_eas), dtype=np.float64)
        K_aa = np.zeros((num_eas, num_eas), dtype=np.float64)
        f_int = np.zeros(24, dtype=np.float64)

        # 2x2x2 Gauss Loop
        ip_idx = 0
        for xi in self.GAUSS_1D:
            for eta in self.GAUSS_1D:
                for zeta in self.GAUSS_1D:
                    pt_nat = np.array([xi, eta, zeta], dtype=np.float64)
                    J, detJ, dN_dX = self.jacobian(pt_nat, node_coords)
                    B = self.b_matrix(dN_dX)  # (6, 24)
                    
                    # Enhanced B-matrix \tilde{B} = (detJ0 / detJ) * invT0^T * M(xi)
                    M = self.enhanced_strain_matrix_m(pt_nat)  # (6, 9)
                    B_tilde = (detJ0 / detJ) * (invT0_T @ M)   # (6, 9)

                    # Compute strain & linear elasticity stress
                    strain_comp = B @ u_elem
                    stress = C_material @ strain_comp       # (6,)

                    # Accumulate stiffness submatrices
                    weight = 1.0  # GAUSS_WEIGHTS_1D[i] * GAUSS_WEIGHTS_1D[j] * GAUSS_WEIGHTS_1D[k] = 1.0 * 1.0 * 1.0
                    dV = detJ * weight

                    K_uu += B.T @ C_material @ B * dV
                    K_ua += B.T @ C_material @ B_tilde * dV
                    K_aa += B_tilde.T @ C_material @ B_tilde * dV
                    f_int += B.T @ stress * dV

                    ip_idx += 1

        # Static Condensation: K_cond = K_uu - K_ua * K_aa^-1 * K_au
        inv_K_aa = np.linalg.inv(K_aa)
        K_condensed = K_uu - K_ua @ inv_K_aa @ K_ua.T
        alpha_opt = -inv_K_aa @ (K_ua.T @ u_elem)

        return K_condensed, f_int, alpha_opt
