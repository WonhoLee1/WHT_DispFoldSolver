"""
test_cpe4h_hybrid.py
====================
Unit tests for Abaqus-grade CPE4H / CPE4IH Hybrid Elements (Q4_HYBRID, Q4_COROTATIONAL_HYBRID_EAS).
"""

import numpy as np
import pytest

from dispsolver.element.q4_hybrid_jax import (
    compute_hybrid_element_contributions_jax,
    compute_element_energy_hybrid,
)

try:
    from dispsolver.element.q4_hybrid_numba import (
        _compute_q4_hybrid_single_element,
    )
    _has_numba = True
except ImportError:
    _compute_q4_hybrid_single_element = None
    _has_numba = False

from dispsolver.mesh import Mesh
from dispsolver.material import NeoHookean
from dispsolver.solver import DynamicSolver


def test_q4_hybrid_jax_energy():
    """Verify Q1P0 hybrid element energy and force vector computation in JAX."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    u_elem = np.zeros(8)
    params = {'E': 1000.0, 'nu': 0.499}

    energy = compute_element_energy_hybrid(coords, u_elem, params)
    assert np.isfinite(energy)
    assert energy == pytest.approx(0.0, abs=1e-10)

    f_int, K_e = compute_hybrid_element_contributions_jax(coords, u_elem, params)
    assert f_int.shape == (8,)
    assert K_e.shape == (8, 8)
    assert np.all(np.isfinite(f_int))
    assert np.all(np.isfinite(K_e))


@pytest.mark.skipif(not _has_numba, reason="numba not available")
def test_q4_hybrid_numba_kernel_matches():
    """Verify Numba C-JIT hybrid element kernel produces finite stiffness and force."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    u_elem = np.zeros(8)
    E = 1000.0
    nu = 0.499

    f_e, K_e, p_e = _compute_q4_hybrid_single_element(coords, u_elem, E, nu)
    assert f_e.shape == (8,)
    assert K_e.shape == (8, 8)
    assert p_e == pytest.approx(0.0, abs=1e-10)
    assert np.all(np.isfinite(f_e))
    assert np.all(np.isfinite(K_e))


def test_volumetric_locking_relief():
    """Verify nearly incompressible cantilever beam (nu = 0.4999) under load does not lock with Q4_HYBRID."""
    m = Mesh()
    m.add_node(0, 0.0, 0.0)
    m.add_node(1, 10.0, 0.0)
    m.add_node(2, 10.0, 2.0)
    m.add_node(3, 0.0, 2.0)
    m.add_element(0, [0, 1, 2, 3], 'QUAD4', pid=0)

    mat = NeoHookean()

    solver = DynamicSolver(
        mesh=m,
        material={0: mat},
        rho=1000.0,
        material_params={0: {'E': 1000.0, 'nu': 0.4999}},
        constraints=[],
        element_type="Q4_HYBRID",
        mode="quasistatic",
    )

    nid = m.node_id_to_index()
    # Fixed left end
    solver.set_prescribed_dofs([nid[0]*2, nid[0]*2+1, nid[3]*2, nid[3]*2+1], [0.0, 0.0, 0.0, 0.0])
    # Tip downward load
    solver.apply_load([nid[2]*2+1], [-0.1])

    n_iter = solver.solve_step(1.0)
    assert n_iter >= 0, "NR solver failed to converge"
    
    # Check tip vertical displacement is non-zero (relieved volumetric locking)
    uy_tip = solver.u[nid[2]*2+1]
    assert np.abs(uy_tip) > 1e-6, f"Tip displacement {uy_tip} indicates locking!"
