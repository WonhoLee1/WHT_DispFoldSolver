"""
test_3d_plasticity.py
=====================
TDD Test Suite for 3D J2 Multiplicative Plasticity & Tetra4 ANP Formulation.
"""

import pytest
import numpy as np

from dispsolver.material3d import J2Plasticity3D
from dispsolver.element3d import QuadraturePointState3D
from dispsolver.element3d.c3d4_jax import Tetra4Element


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


def test_C3D4_element_stiffness():
    """Verify Bonet-Burton C3D4 4-node Tetra element stiffness rank and symmetry."""
    elem = Tetra4Element()
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


def test_3d_viscoelastic_prony_response_and_tangent():
    """Verify 3D Prony viscoelasticity stress relaxation and algorithmic consistent tangent."""
    from dispsolver.material3d.numba_materials import (
        material_dispatch_3d,
        MAT_VISCOELASTIC_PRONY,
        get_default_sdv_count,
    )

    E, nu = 1000.0, 0.25
    g1 = 0.5
    tau1 = 2.0
    props = np.array([E, nu, g1, tau1], dtype=np.float64)

    n_sdvs = get_default_sdv_count(MAT_VISCOELASTIC_PRONY)
    assert n_sdvs == 12
    sdv_prev = np.zeros(n_sdvs, dtype=np.float64)

    strain = np.array([0.01, -0.005, 0.0, 0.02, 0.0, 0.0], dtype=np.float64)
    F = np.eye(3, dtype=np.float64)
    detF = 1.0

    # 1. Very small dt (instantaneous elastic limit: dt -> 0)
    S_inst, C_inst, sdv_inst, err_inst = material_dispatch_3d(
        MAT_VISCOELASTIC_PRONY, props, sdv_prev, strain, F, detF, dt=1e-8
    )
    assert err_inst == 0
    assert np.allclose(C_inst, C_inst.T, atol=1e-10)

    # 2. Hold strain constant for large relaxation time (dt -> inf, delta_strain = 0)
    S_rel, C_rel, sdv_rel, err_rel = material_dispatch_3d(
        MAT_VISCOELASTIC_PRONY, props, sdv_inst, strain, F, detF, dt=100.0
    )
    assert err_rel == 0
    assert np.allclose(C_rel, C_rel.T, atol=1e-10)

    # In relaxed limit under constant strain, deviatoric shear stress relaxes to exactly (1 - g1) = 0.5 of instantaneous
    assert pytest.approx(S_rel[3], rel=1e-6) == (1.0 - g1) * S_inst[3]

    # 3. Intermediate dt step: finite difference check of algorithmic tangent C_mat
    dt = 1.0
    S0, C_alg, _, _ = material_dispatch_3d(
        MAT_VISCOELASTIC_PRONY, props, sdv_prev, strain, F, detF, dt=dt
    )
    h = 1e-7
    for j in range(6):
        strain_pert = strain.copy()
        strain_pert[j] += h
        S_pert, _, _, _ = material_dispatch_3d(
            MAT_VISCOELASTIC_PRONY, props, sdv_prev, strain_pert, F, detF, dt=dt
        )
        col_fd = (S_pert - S0) / h
        assert np.allclose(C_alg[:, j], col_fd, rtol=1e-5, atol=1e-5)

