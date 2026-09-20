"""
Solver Thermal Adapter Module.

Computes equivalent nodal thermal force vectors for finite elements:
    f_th = int_V B^T @ C @ eps_th dV
where eps_th is the isotropic thermal strain vector:
    eps_th = alpha * (T - T0) * I

Fully Numba-accelerated with 2x2x2 (3D) and 2x2 (2D) Gauss quadrature.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def compute_element_thermal_force_3d(
    elem_coords: np.ndarray,
    alpha: float,
    delta_T: float,
    E: float,
    nu: float,
    f_th_elem: np.ndarray
) -> None:
    """
    Compute 3D 8-node brick element equivalent thermal load vector (24 DOFs).

    Parameters
    ----------
    elem_coords : np.ndarray (8, 3)
        Nodal coordinates of the C3D8 element
    alpha : float
        Coefficient of thermal expansion
    delta_T : float
        Temperature change (T - T0)
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    f_th_elem : np.ndarray (24,)
        Output thermal force vector
    """
    f_th_elem.fill(0.0)
    if abs(alpha * delta_T) < 1e-20:
        return

    # Constitutive matrix for isotropic 3D
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    # For isotropic thermal strain, sigma_th = C @ eps_th
    # eps_th = [eps0, eps0, eps0, 0, 0, 0]^T, where eps0 = alpha * delta_T
    # sigma_th_xx = (lam + 2*mu)*eps0 + lam*eps0 + lam*eps0 = (3*lam + 2*mu)*eps0 = 3*K*eps0
    K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
    sigma_th_normal = 3.0 * K_bulk * alpha * delta_T

    # 2x2x2 Gauss points and weights
    g_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    weights = np.array([1.0, 1.0])

    for i in range(2):
        xi = g_pts[i]
        w_xi = weights[i]
        for j in range(2):
            eta = g_pts[j]
            w_eta = weights[j]
            for k in range(2):
                zeta = g_pts[k]
                w_zeta = weights[k]

                # Shape function derivatives w.r.t natural coordinates
                # N_a = 1/8 * (1 + xi_a*xi) * (1 + eta_a*eta) * (1 + zeta_a*zeta)
                dNdxi = np.zeros((8, 3), dtype=np.float64)
                xi_nodes = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
                eta_nodes = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
                zeta_nodes = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

                for a in range(8):
                    dNdxi[a, 0] = 0.125 * xi_nodes[a] * (1.0 + eta_nodes[a] * eta) * (1.0 + zeta_nodes[a] * zeta)
                    dNdxi[a, 1] = 0.125 * eta_nodes[a] * (1.0 + xi_nodes[a] * xi) * (1.0 + zeta_nodes[a] * zeta)
                    dNdxi[a, 2] = 0.125 * zeta_nodes[a] * (1.0 + xi_nodes[a] * xi) * (1.0 + eta_nodes[a] * eta)

                # Jacobian: J = elem_coords^T @ dNdxi => (3, 3)
                J = np.zeros((3, 3), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        s = 0.0
                        for a in range(8):
                            s += elem_coords[a, r] * dNdxi[a, c]
                        J[r, c] = s

                # Determinant of Jacobian
                detJ = (
                    J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
                    - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
                    + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
                )
                if detJ <= 0.0:
                    continue

                # Inverse of J
                invJ = np.zeros((3, 3), dtype=np.float64)
                inv_det = 1.0 / detJ
                invJ[0, 0] = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) * inv_det
                invJ[0, 1] = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) * inv_det
                invJ[0, 2] = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) * inv_det

                invJ[1, 0] = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) * inv_det
                invJ[1, 1] = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) * inv_det
                invJ[1, 2] = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) * inv_det

                invJ[2, 0] = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) * inv_det
                invJ[2, 1] = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) * inv_det
                invJ[2, 2] = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) * inv_det

                # Physical derivatives: dNdx = dNdxi @ invJ^T
                dNdx = np.zeros((8, 3), dtype=np.float64)
                for a in range(8):
                    for c in range(3):
                        s = 0.0
                        for r in range(3):
                            s += dNdxi[a, r] * invJ[r, c]
                        dNdx[a, c] = s

                dV = w_xi * w_eta * w_zeta * detJ

                # For isotropic thermal stress, sigma_th = sigma_th_normal * [1, 1, 1, 0, 0, 0]
                # B^T @ sigma_th for node a:
                # [dNdx[a, 0]*sigma_th_normal, dNdx[a, 1]*sigma_th_normal, dNdx[a, 2]*sigma_th_normal]
                for a in range(8):
                    f_th_elem[3 * a + 0] += dNdx[a, 0] * sigma_th_normal * dV
                    f_th_elem[3 * a + 1] += dNdx[a, 1] * sigma_th_normal * dV
                    f_th_elem[3 * a + 2] += dNdx[a, 2] * sigma_th_normal * dV


@njit(fastmath=True, cache=True)
def compute_element_thermal_force_2d(
    elem_coords: np.ndarray,
    alpha: float,
    delta_T: float,
    E: float,
    nu: float,
    f_th_elem: np.ndarray
) -> None:
    """
    Compute 2D 4-node quad element equivalent thermal load vector for plane strain (8 DOFs).

    Parameters
    ----------
    elem_coords : np.ndarray (4, 2)
        Nodal coordinates of the CPE4 element
    alpha : float
        Coefficient of thermal expansion
    delta_T : float
        Temperature change (T - T0)
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    f_th_elem : np.ndarray (8,)
        Output thermal force vector
    """
    f_th_elem.fill(0.0)
    if abs(alpha * delta_T) < 1e-20:
        return

    # In plane strain: eps_33 = 0, but eps_33^th = alpha * delta_T
    # sigma_th = C @ eps_th = [sigma_xx, sigma_yy, 0]^T
    # where sigma_xx = sigma_yy = E / (1 - 2*nu) * alpha * delta_T
    K_bulk_2d = E / (1.0 - 2.0 * nu)  # Plane strain volumetric thermal stiffness
    sigma_th_normal = K_bulk_2d * alpha * delta_T

    g_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    weights = np.array([1.0, 1.0])

    xi_nodes = np.array([-1.0, 1.0, 1.0, -1.0])
    eta_nodes = np.array([-1.0, -1.0, 1.0, 1.0])

    for i in range(2):
        xi = g_pts[i]
        w_xi = weights[i]
        for j in range(2):
            eta = g_pts[j]
            w_eta = weights[j]

            dNdxi = np.zeros((4, 2), dtype=np.float64)
            for a in range(4):
                dNdxi[a, 0] = 0.25 * xi_nodes[a] * (1.0 + eta_nodes[a] * eta)
                dNdxi[a, 1] = 0.25 * eta_nodes[a] * (1.0 + xi_nodes[a] * xi)

            J = np.zeros((2, 2), dtype=np.float64)
            for r in range(2):
                for c in range(2):
                    s = 0.0
                    for a in range(4):
                        s += elem_coords[a, r] * dNdxi[a, c]
                    J[r, c] = s

            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if detJ <= 0.0:
                continue

            inv_det = 1.0 / detJ
            invJ = np.array([
                [J[1, 1] * inv_det, -J[0, 1] * inv_det],
                [-J[1, 0] * inv_det, J[0, 0] * inv_det]
            ], dtype=np.float64)

            dNdx = np.zeros((4, 2), dtype=np.float64)
            for a in range(4):
                for c in range(2):
                    s = 0.0
                    for r in range(2):
                        s += dNdxi[a, r] * invJ[r, c]
                    dNdx[a, c] = s

            dV = w_xi * w_eta * detJ

            for a in range(4):
                f_th_elem[2 * a + 0] += dNdx[a, 0] * sigma_th_normal * dV
                f_th_elem[2 * a + 1] += dNdx[a, 1] * sigma_th_normal * dV
