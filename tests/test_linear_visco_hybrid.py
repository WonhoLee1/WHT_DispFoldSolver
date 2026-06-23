"""
test_linear_visco_hybrid.py
===========================
Linear viscoelastic Q1P0 hybrid element (mean-dilatation) + WLF TTS.
"""

import numpy as np
import pytest

from dispsolver.mesh import Mesh
from dispsolver.material import LinearViscoelastic, J2Plasticity
from dispsolver.solver import DynamicSolver
from dispsolver.element.q4_visco_hybrid import compute_visco_hybrid_contributions


# ------------------------------------------------------------------
# Material-level: instantaneous vs relaxed moduli
# ------------------------------------------------------------------

def test_prony_instantaneous_and_relaxed():
    E, nu = 100.0, 0.45
    mat = LinearViscoelastic(E, nu, g_i=[0.6], tau_i=[1.0])
    G0 = E / (2.0 * (1.0 + nu))

    # dt << tau  -> gamma ~ 1 -> algorithmic shear ~ instantaneous G0
    _, _, G_alg_fast = mat.prony_coeffs(dt=1e-6)
    assert G_alg_fast == pytest.approx(G0, rel=1e-3)

    # dt >> tau  -> a_i ~ 0, gamma_i ~ 0 -> algorithmic shear ~ relaxed G_inf
    _, _, G_alg_slow = mat.prony_coeffs(dt=1e6)
    assert G_alg_slow == pytest.approx(mat.G_inf, rel=1e-3)
    assert mat.G_inf == pytest.approx(G0 * 0.4, rel=1e-12)


def test_dev_update_stress_relaxation():
    """Hold a fixed deviatoric strain; deviatoric stress must relax G0 -> G_inf."""
    E, nu = 100.0, 0.3
    mat = LinearViscoelastic(E, nu, g_i=[0.7], tau_i=[1.0])
    G0 = mat.G0

    e_dev = np.array([0.01, -0.005, -0.005, 0.0])  # constant tensorial dev strain
    state = mat.initial_internal_vars()

    # First step (fast) -> ~instantaneous
    a, g, _ = mat.prony_coeffs(dt=1e-4)
    s0, state = mat.dev_update(e_dev, state, a, g)
    s0_xx = s0[0]
    assert s0_xx == pytest.approx(2.0 * G0 * e_dev[0], rel=2e-2)

    # Many large steps holding strain -> relaxes toward 2*G_inf*e
    a, g, _ = mat.prony_coeffs(dt=5.0)
    for _ in range(50):
        s, state = mat.dev_update(e_dev, state, a, g)
    assert s[0] == pytest.approx(2.0 * mat.G_inf * e_dev[0], rel=1e-2)
    assert abs(s[0]) < abs(s0_xx)  # stress decreased


# ------------------------------------------------------------------
# Element-level: symmetry & finiteness
# ------------------------------------------------------------------

def test_element_tangent_symmetric():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mat = LinearViscoelastic(100.0, 0.45, g_i=[0.5], tau_i=[1.0])
    u = np.array([0.0, 0.0, 0.01, 0.0, 0.012, 0.02, 0.0, 0.018])
    state = np.zeros((4, mat.n_internal_vars))

    f, K, sn = compute_visco_hybrid_contributions(coords, u, state, mat, dt=0.1)
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(K))
    assert np.allclose(K, K.T, atol=1e-9)          # tangent symmetric
    assert sn.shape == (4, mat.n_internal_vars)


# ------------------------------------------------------------------
# Solver integration: single hybrid element
# ------------------------------------------------------------------

def test_hybrid_solver_single_element():
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 1.0, 1.0)
    mesh.add_node(3, 0.0, 1.0)
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4")

    mat = LinearViscoelastic(1000.0, 0.49, g_i=[0.6], tau_i=[1.0])  # near-incompressible

    solver = DynamicSolver(
        mesh, mat, rho=1000.0,
        material_params={},
        element_type="Q4_UP",
        verbose=False, tol=1e-8,
    )
    assert solver.state.shape == (1, 4, mat.n_internal_vars)

    bc_dofs, bc_vals = [], []
    for nid in [0, 1]:
        bc_dofs += [nid * 2, nid * 2 + 1]; bc_vals += [0.0, 0.0]
    for nid in [2, 3]:
        bc_dofs += [nid * 2, nid * 2 + 1]; bc_vals += [0.0, 0.05]
    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    n_iter = solver.solve_step(dt=0.1)
    assert n_iter >= 0
    assert np.any(solver.state != 0.0)


def test_hybrid_solver_wlf_shift():
    """WLF shift changes the effective relaxation; state still evolves & converges."""
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0); mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 1.0, 1.0); mesh.add_node(3, 0.0, 1.0)
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4")

    wlf = {"C1": 17.44, "C2": 51.6, "T_ref": 20.0}
    mat = LinearViscoelastic(1000.0, 0.45, g_i=[0.6], tau_i=[1.0], wlf_params=wlf)

    # aT < 1 above T_ref (faster relaxation); aT > 1 below
    aT_hot = mat.prony_coeffs(dt=1.0, temperature=60.0)
    aT_cold = mat.prony_coeffs(dt=1.0, temperature=-10.0)
    # algorithmic modulus hotter -> more relaxed (smaller) than colder
    assert aT_hot[2] < aT_cold[2]

    solver = DynamicSolver(
        mesh, mat, rho=1000.0, material_params={},
        element_type="Q4_UP", verbose=False, tol=1e-8,
    )
    solver.temperature = 60.0
    bc_dofs, bc_vals = [], []
    for nid in [0, 1]:
        bc_dofs += [nid * 2, nid * 2 + 1]; bc_vals += [0.0, 0.0]
    for nid in [2, 3]:
        bc_dofs += [nid * 2, nid * 2 + 1]; bc_vals += [0.0, 0.05]
    solver.set_prescribed_dofs(bc_dofs, bc_vals)
    n_iter = solver.solve_step(dt=0.1)
    assert n_iter >= 0
    assert np.any(solver.state != 0.0)


def test_mixed_element_type_per_pid():
    """PET(J2)=Q4 batch  +  PSA(linear visco)=Q4_UP hybrid in one mesh."""
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0); mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 1.0, 1.0); mesh.add_node(3, 0.0, 1.0)
    mesh.add_node(4, 1.0, 2.0); mesh.add_node(5, 0.0, 2.0)
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4", pid=0)   # PET (J2)
    mesh.add_element(1, [3, 2, 4, 5], "QUAD4", pid=1)   # PSA (linear visco)

    materials = {
        0: J2Plasticity(E=2000.0, nu=0.3, sigma_y0=50.0, H=100.0),
        1: LinearViscoelastic(50.0, 0.49, g_i=[0.7], tau_i=[1.0]),
    }
    params = {0: {}, 1: {}}
    element_type = {0: "Q4", 1: "Q4_UP"}   # per-pid

    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=params,
        element_type=element_type, verbose=False, tol=1e-7,
    )
    assert solver.element_type == "MIXED"
    assert solver.use_multi_material_batch

    bc_dofs, bc_vals = [], []
    for nid in [0, 1]:
        bc_dofs += [nid * 2, nid * 2 + 1]; bc_vals += [0.0, 0.0]
    for nid in [4, 5]:
        bc_dofs += [nid * 2, nid * 2 + 1]; bc_vals += [0.0, 0.1]
    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    n_iter = solver.solve_step(dt=0.1)
    assert n_iter >= 0
    # J2 element state (eqps/Fp) and visco element state both evolved
    assert np.any(solver.state[0, :, :5] != 0.0)
    assert np.any(solver.state[1, :, :8] != 0.0)
