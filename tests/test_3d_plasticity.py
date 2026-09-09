"""
test_3d_plasticity.py
=====================
TDD Test Suite for 3D J2 Multiplicative Plasticity & Tetra4 ANP Formulation.
"""

import pytest
import numpy as np

from dispsolver.material3d import J2Plasticity3D
from dispsolver.element3d import QuadraturePointState3D
from dispsolver.element3d.c3d4_anp_jax import Tetra4ANPElement


def test_j2_plasticity3d_elastic_and_yield():
    """Verify 3D J2 plasticity elasticity and radial return mapping for yielding."""
    mat = J2Plasticity3D(E=200000.0, nu=0.3, sigma_y0=200.0, H=1000.0)
    state_init = QuadraturePointState3D.create_initial()

    # 1. Small strain -> Pure Elastic Step
    strain_elastic = np.array([0.0005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_e, C_t_e, state_e = mat.return_mapping_voigt(strain_elastic, state_init)

    assert state_e.eq_plastic_strain == 0.0
    assert np.allclose(C_t_e, mat.C_elastic)
    assert pytest.approx(stress_e[0]) == (mat.lam + 2*mat.mu) * 0.0005

    # 2. Large tensile strain -> Yielding Step (f_trial > 0)
    strain_plastic = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_p, C_t_p, state_p = mat.return_mapping_voigt(strain_plastic, state_init)

    # Plastic strain must accumulate
    assert state_p.eq_plastic_strain > 0.0
    assert state_p.plastic_strain[0] > 0.0
    # Consistent tangent matrix must be symmetric
    assert np.allclose(C_t_p, C_t_p.T, atol=1e-6)


def test_c3d4_anp_element_stiffness():
    """Verify Bonet-Burton C3D4_ANP 4-node Tetra element stiffness rank and symmetry."""
    elem = Tetra4ANPElement()
    assert elem.num_nodes == 4
    assert elem.num_quad_points == 1

    # 4-node unit tetrahedron coordinates
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    u_elem = np.zeros(12, dtype=np.float64)
    states = [QuadraturePointState3D.create_initial()]

    mat = J2Plasticity3D(E=200000.0, nu=0.3)
    K_elem, f_int = elem.compute_element_stiffness_and_force(
        node_coords, u_elem, mat.C_elastic, states
    )

    assert K_elem.shape == (12, 12)
    assert f_int.shape == (12,)
    assert np.allclose(K_elem, K_elem.T, atol=1e-8)

    # 3D Tet element must have rank 6 (12 - 6 rigid body modes = 6)
    rank = np.linalg.matrix_rank(K_elem, tol=1e-5)
    assert rank == 6
