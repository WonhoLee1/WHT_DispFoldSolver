"""
test_visco_solver.py
====================
Verify viscoelastic material adapter and multi-material solver integration.
"""

import numpy as np
import pytest
from dispsolver.mesh import Mesh
from dispsolver.material import NeoHookean, ViscoelasticMaterial, J2Plasticity
from dispsolver.solver import DynamicSolver


def test_viscoelastic_single_element_solver():
    # 1. Mesh setup: Single element
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 1.0, 1.0)
    mesh.add_node(3, 0.0, 1.0)
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4")

    # 2. Material setup: Viscoelastic NeoHookean
    base_mat = NeoHookean()
    visco_mat = ViscoelasticMaterial(base_mat, g_i=[0.5], tau_i=[1.0])
    params = {"E": 1000.0, "nu": 0.3}

    # 3. Solver setup
    solver = DynamicSolver(
        mesh,
        visco_mat,
        rho=1000.0,
        material_params=params,
        verbose=True,
        tol=1e-8
    )

    # 4. Boundary conditions: Stretch top nodes in Y-direction
    bc_dofs = []
    bc_vals = []
    # Bottom nodes fixed
    for nid in [0, 1]:
        bc_dofs.extend([nid * 2, nid * 2 + 1])
        bc_vals.extend([0.0, 0.0])
    # Top nodes pulled
    for nid in [2, 3]:
        bc_dofs.extend([nid * 2, nid * 2 + 1])
        bc_vals.extend([0.0, 0.1])  # 10% stretch

    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    # 5. Solve step with dt=0.5
    n_iter = solver.solve_step(dt=0.5)
    assert n_iter >= 0, "Viscoelastic solver step failed to converge"
    
    # 6. Verify state evolution: Prony overstress should be non-zero
    assert solver.state is not None
    # Shape of self.state: (n_elem, n_gp, max_vars)
    # n_elem=1, n_gp=4, n_vars=12 for ViscoelasticMaterial with M=1
    assert solver.state.shape == (1, 4, 12)
    assert np.any(solver.state != 0.0), "Viscoelastic state variables did not evolve"


def test_multi_material_solver():
    # 1. Mesh setup: Two elements
    # Elem 0 (PET, pid=0): [0, 1] x [0, 1]
    # Elem 1 (PSA, pid=1): [0, 1] x [1, 2]
    mesh = Mesh()
    # Bottom nodes
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    # Middle nodes
    mesh.add_node(2, 1.0, 1.0)
    mesh.add_node(3, 0.0, 1.0)
    # Top nodes
    mesh.add_node(4, 1.0, 2.0)
    mesh.add_node(5, 0.0, 2.0)

    # Element 0 (PET): pid=0
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4", pid=0)
    # Element 1 (PSA): pid=1
    mesh.add_element(1, [3, 2, 4, 5], "QUAD4", pid=1)

    # 2. Multi-material definition
    pet_mat = J2Plasticity(E=2000.0, nu=0.3, sigma_y0=50.0)
    psa_mat = ViscoelasticMaterial(NeoHookean(), g_i=[0.8], tau_i=[1.0])

    materials = {
        0: pet_mat,
        1: psa_mat
    }

    params = {
        0: {},  # J2 parameters are set in init
        1: {"E": 10.0, "nu": 0.49}
    }

    # 3. Solver setup with material dictionary
    solver = DynamicSolver(
        mesh,
        materials,
        rho=1000.0,
        material_params=params,
        verbose=True,
        tol=1e-8
    )

    # 4. State shape check:
    # max_vars is max(5 for J2, 12 for Visco M=1) = 12
    assert solver.state is not None
    assert solver.state.shape == (2, 4, 12)

    # 5. Boundary conditions: pull top edge
    bc_dofs = []
    bc_vals = []
    # Bottom nodes fixed
    for nid in [0, 1]:
        bc_dofs.extend([nid * 2, nid * 2 + 1])
        bc_vals.extend([0.0, 0.0])
    # Top nodes pulled
    for nid in [4, 5]:
        bc_dofs.extend([nid * 2, nid * 2 + 1])
        bc_vals.extend([0.0, 0.2])

    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    # 6. Solve step
    n_iter = solver.solve_step(dt=0.1)
    assert n_iter >= 0, "Multi-material solver step failed to converge"
    
    # 7. Check state evolution
    # Element 0 (J2) state check (vars 0-4 are eqps, F_p_inv)
    j2_state = solver.state[0]
    assert np.any(j2_state[:, :5] != 0.0)
    
    # Element 1 (Visco) state check (vars 0-11)
    visco_state = solver.state[1]
    assert np.any(visco_state[:, :12] != 0.0)
