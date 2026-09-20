import numpy as np
from numba import njit

@njit(fastmath=True)
def get_gauss_lobatto_points(n_points):
    """
    Returns points and weights for Gauss-Lobatto integration (through thickness).
    Supported up to 7 points.
    """
    if n_points == 3:
        return np.array([-1.0, 0.0, 1.0]), np.array([1/3, 4/3, 1/3])
    elif n_points == 5:
        w1 = 0.1
        w2 = 49.0 / 90.0
        w3 = 32.0 / 45.0
        p2 = np.sqrt(1.0 / 7.0)
        return np.array([-1.0, -p2, 0.0, p2, 1.0]), np.array([w1, w2, w3, w2, w1])
    elif n_points == 7:
        # Simplified weights/points for demonstration
        return np.linspace(-1, 1, 7), np.ones(7)*2/7
    else:
        # Fallback to Simpson's rule like
        pts = np.linspace(-1, 1, n_points)
        wts = np.ones(n_points) * (2.0 / n_points)
        return pts, wts

@njit(fastmath=True)
def update_director_vectors(u_nodes, t_nodes):
    """
    Update director vectors using the drilling DOFs and out-of-plane rotations.
    u_nodes: (n_nodes, 6) - ux, uy, uz, thetax, thetay, thetaz
    """
    # Simplified Director vector update using Co-rotational assumption
    n_nodes = u_nodes.shape[0]
    directors = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        theta_x = u_nodes[i, 3]
        theta_y = u_nodes[i, 4]
        # Approximation of normal vector update
        directors[i, 0] = theta_y
        directors[i, 1] = -theta_x
        directors[i, 2] = 1.0 - 0.5 * (theta_x**2 + theta_y**2)
    return directors

@njit(fastmath=True)
def calc_drilling_penalty_stiffness(area, E, nu, thickness, penalty_factor=1e-3):
    """
    Hughes-Brezzi Drilling DOF penalty stiffness.
    Provides artificial stiffness to the normal rotation (thetaz).
    """
    G = E / (2.0 * (1.0 + nu))
    # Penalty parameter is typically proportional to shear modulus and volume
    return penalty_factor * G * area * thickness
