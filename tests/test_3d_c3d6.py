"""
test_3d_c3d6.py
===============
Verification and Unit Tests for 3D 6-Node Linear Wedge/Prism Element (C3D6).

Tests:
1. Full Rank 12 Spectrum: 18 DOFs - 6 rigid body modes = exactly 12 positive eigenvalues.
2. JAX vs Numba Parity: Exact agreement between Python/JAX Wedge6Element and Numba kernel.
3. Algorithmic Tangent Consistency: Finite-difference Jacobian match (relative error < 1e-5).
4. DynamicSolver3D Integration: Fastpath assembly and Newton-Raphson solve convergence.
"""

import pytest
import numpy as np

from dispsolver.element3d import (
    Wedge6Element,
    compute_c3d6_element_numba,
    assemble_mesh_c3d6_numba
)
from dispsolver.material3d.numba_materials import MAT_LINEAR_ELASTIC, MAT_J2_PLASTICITY
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def make_standard_wedge6_coords():
    """Create nodal coordinates for a canonical 6-node triangular prism/wedge.
    
    Bottom triangle (z = 0):
      Node 0: (0, 0, 0)
      Node 1: (1, 0, 0)
      Node 2: (0, 1, 0)
    Top triangle (z = 2):
      Node 3: (0, 0, 2)
      Node 4: (1, 0, 2)
      Node 5: (0, 1, 2)
    """
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0]
    ], dtype=np.float64)
    return coords


def test_c3d6_eigenvalue_spectrum_and_rank():
    """Verify that C3D6 element has full rank 12 (18 DOFs - 6 rigid body modes)."""
    coords = make_standard_wedge6_coords()
    u_elem = np.zeros(18, dtype=np.float64)

    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((6, 1), dtype=np.float64)

    K_elem, f_int, err = compute_c3d6_element_numba(
        coords, u_elem, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs
    )
    assert err == 0
    assert K_elem.shape == (18, 18)

    # Symmetry check
    assert np.allclose(K_elem, K_elem.T, atol=1e-8)

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(K_elem)

    # Exactly 6 zero eigenvalues (rigid body modes in 3D)
    zero_modes = eigvals[:6]
    positive_modes = eigvals[6:]

    assert np.all(np.abs(zero_modes) < 1e-4), f"Rigid body modes exceed threshold: {zero_modes}"
    assert len(positive_modes) == 12
    assert np.all(positive_modes > 10.0), f"Positive eigenvalues too small: {positive_modes[:5]}"
    assert np.min(eigvals) >= -1e-6, "Negative eigenvalues detected!"


def test_c3d6_jax_numba_parity():
    """Verify exact parity between JAX Wedge6Element and Numba compute_c3d6_element_numba."""
    coords = make_standard_wedge6_coords()
    np.random.seed(42)
    u_elem = 0.002 * np.random.randn(18)

    E_val = 200000.0
    nu_val = 0.3
    lam = (E_val * nu_val) / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    mu = E_val / (2.0 * (1.0 + nu_val))
    C_mat = np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],
        [lam,        lam + 2*mu, lam,        0,  0,  0],
        [lam,        lam,        lam + 2*mu, 0,  0,  0],
        [0,          0,          0,          mu, 0,  0],
        [0,          0,          0,          0,  mu, 0],
        [0,          0,          0,          0,  0,  mu]
    ], dtype=np.float64)

    # Numba
    props = np.array([E_val, nu_val], dtype=np.float64)
    sdvs = np.zeros((6, 1), dtype=np.float64)
    K_nb, f_nb, err = compute_c3d6_element_numba(
        coords, u_elem, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs
    )
    assert err == 0

    # JAX
    wedge_jax = Wedge6Element()
    K_jax, f_jax = wedge_jax.compute_element_stiffness_and_force(coords, u_elem, C_mat)

    assert np.allclose(f_nb, f_jax, rtol=1e-10, atol=1e-10)
    assert np.allclose(K_nb, K_jax, rtol=1e-10, atol=1e-10)


def test_c3d6_finite_difference_tangent():
    """Verify consistent algorithmic tangent K_elem = df_int/du via central finite differences."""
    coords = make_standard_wedge6_coords()

    np.random.seed(123)
    u_elem = 0.001 * np.random.randn(18)

    props = np.array([200000.0, 0.3, 400.0, 1000.0], dtype=np.float64)  # J2 Plasticity
    sdvs = np.zeros((6, 4), dtype=np.float64)

    K_elem, f0, err = compute_c3d6_element_numba(
        coords, u_elem, mat_type=MAT_J2_PLASTICITY, props=props, sdvs=sdvs
    )
    assert err == 0

    # Numerical Jacobian via central differences
    K_fd = np.zeros((18, 18), dtype=np.float64)
    eps = 1e-7

    for j in range(18):
        u_plus = u_elem.copy()
        u_plus[j] += eps
        sdvs_plus = sdvs.copy()
        _, f_plus, _ = compute_c3d6_element_numba(
            coords, u_plus, mat_type=MAT_J2_PLASTICITY, props=props, sdvs=sdvs_plus
        )

        u_minus = u_elem.copy()
        u_minus[j] -= eps
        sdvs_minus = sdvs.copy()
        _, f_minus, _ = compute_c3d6_element_numba(
            coords, u_minus, mat_type=MAT_J2_PLASTICITY, props=props, sdvs=sdvs_minus
        )

        K_fd[:, j] = (f_plus - f_minus) / (2.0 * eps)

    diff_norm = np.linalg.norm(K_elem - K_fd)
    ref_norm = np.linalg.norm(K_elem)
    rel_err = diff_norm / ref_norm

    assert rel_err < 1e-5, f"C3D6 algorithmic tangent error too large: {rel_err:.2e}"


def test_c3d6_dynamic_solver3d_integration():
    """Verify DynamicSolver3D setup, assembly, and solve step with C3D6 wedge elements."""
    mesh = Mesh3D()
    coords = make_standard_wedge6_coords()

    for i in range(6):
        mesh.add_node(i + 1, coords[i, 0], coords[i, 1], coords[i, 2])

    node_ids = list(range(1, 7))
    mesh.add_element(1, node_ids, elem_type="C3D6", pid=0)

    solver = DynamicSolver3D(
        mesh,
        material_params={"E": 200000.0, "nu": 0.3},
        nlgeom=True
    )

    # Check topology
    assert 7 in solver.elem_kernel_groups
    assert solver.elem_conn_groups[7].shape == (1, 6)
    assert solver.rows_topo.shape[0] == 1 * 18 * 18

    # Fix bottom triangular face nodes 1, 2, 3 (z = 0)
    for nid in [1, 2, 3]:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # Prescribe vertical tension on top node 4
    solver.fix_dof(4, 2, 0.02)

    # Assemble system
    K_glob, f_glob = solver.assemble_system(solver.u, dt=1.0)
    assert K_glob.shape == (18, 18)
    assert f_glob.shape == (18,)

    # Solve step
    success = solver.solve_step(dt=1.0)
    assert success, "DynamicSolver3D solve_step failed with C3D6 element!"

    nid_map = mesh.node_id_to_index()
    top_z_dof = 3 * nid_map[4] + 2
    assert np.isclose(solver.u[top_z_dof], 0.02, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
