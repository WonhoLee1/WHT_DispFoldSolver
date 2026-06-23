"""
test_contact.py
===============
Verify JAX-based PenaltyContactConstraint:
1. Contact forces trigger only below the threshold distance d_0.
2. Nodal forces are symmetric (action-reaction).
3. Tangent stiffness matrix matches numerical finite difference gradient of forces.
"""

import numpy as np
import pytest
from dispsolver.mesh import Mesh
from dispsolver.constraint.contact_jax import PenaltyContactConstraint


def test_contact_activation_and_symmetry():
    # Setup a simple mesh with 2 nodes
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    # Element is not strictly needed for penalty constraint, but we need node index maps
    
    # Distance threshold: d_0 = 0.5
    # Initial distance is 1.0 (>= 1.5 * d_0), so this is a valid candidate pair
    contact = PenaltyContactConstraint(mesh, contact_nodes=[0, 1], k_contact=1e6, d_0=0.5)
    
    # Check no contact when far: displacements such that x1=0.0, x2=0.8 (distance = 0.8 > 0.5)
    u_far = np.array([0.0, 0.0, -0.2, 0.0]) # Node 1 moved to x=0.8
    f_int = np.zeros(4)
    K_T = np.zeros((4, 4))
    contact.apply_penalty(u_far, f_int, K_T)
    assert np.allclose(f_int, 0.0)
    assert np.allclose(K_T, 0.0)
    
    # Check contact when close: displacements such that x1=0.0, x2=0.3 (distance = 0.3 < 0.5)
    u_close = np.array([0.0, 0.0, -0.7, 0.0]) # Node 1 moved to x=0.3
    f_int = np.zeros(4)
    K_T = np.zeros((4, 4))
    contact.apply_penalty(u_close, f_int, K_T)
    
    # Penetration: gap = 0.3 - 0.5 = -0.2. Force magnitude = k * |gap| = 1e6 * 0.2 = 200,000.
    # Node 0 should be pushed to the left (-x), Node 1 to the right (+x)
    # Due to f_int = grad(E_contact), where E_contact = 0.5*k*gap^2:
    # dE/du_0 = -k*|gap| * dir = -200,000 * (-1, 0) = (+200,000, 0) -> f_int[0] > 0
    # dE/du_1 = -k*|gap| * dir_1 = -200,000 * (1, 0) = (-200,000, 0) -> f_int[2] < 0
    # Let's check magnitude and signs:
    assert np.abs(f_int[0]) > 1e-3
    assert np.abs(f_int[2]) > 1e-3
    assert np.allclose(f_int[0], -f_int[2]), "Forces must be symmetric (Action-Reaction)"
    assert f_int[0] > 0.0, "Node 0 should feel repulsive force in +x"
    assert f_int[2] < 0.0, "Node 1 should feel repulsive force in -x"


def test_contact_tangent_finite_difference():
    # Setup mesh
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 0.0, 1.0) # Vertical distance 1.0
    
    # d_0 = 0.6. Candidate pair since init dist = 1.0 >= 1.5 * 0.6 = 0.9.
    k_contact = 1e5
    d_0 = 0.6
    contact = PenaltyContactConstraint(mesh, contact_nodes=[0, 1], k_contact=k_contact, d_0=d_0)
    
    # Displace nodes to trigger contact (distance < 0.6)
    # Node 0: moves up by 0.3 -> (0.0, 0.3)
    # Node 1: moves down by 0.2 -> (0.0, 0.8)
    # Current distance = 0.5 < 0.6
    u = np.array([0.0, 0.3, 0.0, -0.2])
    
    # 1. Get analytical force and stiffness from constraint
    f_analytic = np.zeros(4)
    K_analytic = np.zeros((4, 4))
    contact.apply_penalty(u, f_analytic, K_analytic)
    
    # 2. Compute numerical stiffness via central finite difference of forces:
    # K_num[:, j] = (f_int(u + eps*e_j) - f_int(u - eps*e_j)) / (2*eps)
    eps = 1e-7
    K_numerical = np.zeros((4, 4))
    
    for j in range(4):
        u_plus = u.copy()
        u_plus[j] += eps
        f_plus = np.zeros(4)
        K_dummy = np.zeros((4, 4))
        contact.apply_penalty(u_plus, f_plus, K_dummy)
        
        u_minus = u.copy()
        u_minus[j] -= eps
        f_minus = np.zeros(4)
        contact.apply_penalty(u_minus, f_minus, K_dummy)
        
        K_numerical[:, j] = (f_plus - f_minus) / (2.0 * eps)
        
    # Check that analytical stiffness matches numerical finite difference
    np.testing.assert_allclose(K_analytic, K_numerical, rtol=1e-5, atol=1e-5,
                               err_msg="Analytical tangent stiffness does not match finite difference")
