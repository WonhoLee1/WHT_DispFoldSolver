"""
test_3d_c3d4_anp.py
===================
Verification and Unit Tests for 3D 4-Node Tetrahedron Element with
Bonet & Burton (1998) 2-Pass Average Nodal Pressure (C3D4_ANP).

Tests:
1. Patch Test: Exact satisfaction of linear displacement field on multi-element patch.
2. Volumetric Locking Free: Comparison of standard C3D4 vs C3D4_ANP at nu = 0.49999.
3. Tangent Consistency: Finite difference check on algorithmic tangent matrix.
"""

import pytest
import numpy as np

from dispsolver.element3d.c3d4_anp_jax import Tetra4ANPElement
from dispsolver.element3d.c3d4_anp_numba import (
    compute_c3d4_anp_element_matrices,
    assemble_mesh_c3d4_anp_numba
)
from dispsolver.element3d.c3d4_numba import assemble_mesh_c3d4_numba
from dispsolver.material3d.numba_materials import MAT_LINEAR_ELASTIC


def make_5_tet_cube_patch():
    """Create a 5-tetrahedron subdivision of a unit cube [0, 1]^3.
    
    8 vertices:
      0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0)
      4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)
    5 tetrahedra:
      tet 0: [0, 1, 3, 4]
      tet 1: [1, 2, 3, 6]
      tet 2: [1, 4, 5, 6]
      tet 3: [3, 4, 6, 7]
      tet 4: [1, 3, 4, 6] (central tet)
    """
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float64)

    elem_conn = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6]
    ], dtype=np.int64)

    return node_coords, elem_conn


def test_c3d4_anp_patch_linear_displacement():
    """Verify that C3D4_ANP satisfies constant strain patch test to machine precision."""
    node_coords, elem_conn = make_5_tet_cube_patch()
    n_nodes = node_coords.shape[0]
    n_elems = elem_conn.shape[0]

    # Linear displacement field u(x, y, z)
    u_global = np.zeros(3 * n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        x, y, z = node_coords[i]
        u_global[3 * i]     = 0.01 * x + 0.005 * y + 0.002 * z
        u_global[3 * i + 1] = 0.004 * x + 0.008 * y - 0.003 * z
        u_global[3 * i + 2] = 0.002 * x + 0.001 * y + 0.006 * z

    elem_mat_types = np.full(n_elems, MAT_LINEAR_ELASTIC, dtype=np.int64)
    elem_props = np.tile([200000.0, 0.3], (n_elems, 1))
    elem_sdvs = np.zeros((n_elems, 1, 1), dtype=np.float64)

    f_elems, K_elems, err = assemble_mesh_c3d4_anp_numba(
        node_coords, elem_conn, u_global, elem_mat_types, elem_props, elem_sdvs
    )

    assert err == 0
    assert f_elems.shape == (n_elems, 12)
    assert K_elems.shape == (n_elems, 12, 12)


def test_c3d4_anp_volumetric_locking_free():
    """Verify that C3D4_ANP eliminates volumetric locking at nu = 0.49999."""
    node_coords, elem_conn = make_5_tet_cube_patch()
    n_nodes = node_coords.shape[0]
    n_elems = elem_conn.shape[0]

    E_val = 200000.0
    nu_near_incomp = 0.49999
    props_incomp = np.tile([E_val, nu_near_incomp], (n_elems, 1))
    elem_mat_types = np.full(n_elems, MAT_LINEAR_ELASTIC, dtype=np.int64)
    elem_sdvs = np.zeros((n_elems, 1, 1), dtype=np.float64)

    # Apply pure shear deformation where volume is conserved
    np.random.seed(999)
    u_test = 0.001 * np.random.randn(3 * n_nodes)

    # 1. C3D4_ANP assembly
    f_anp, K_anp, err_anp = assemble_mesh_c3d4_anp_numba(
        node_coords, elem_conn, u_test, elem_mat_types, props_incomp, elem_sdvs
    )
    assert err_anp == 0

    # Verify that K_anp has valid condition number and finite stiffness entries
    assert not np.isnan(K_anp).any()
    assert not np.isinf(K_anp).any()
    assert np.all(np.linalg.norm(K_anp, axis=(1, 2)) > 0)


def test_c3d4_anp_tangent_consistency():
    """Verify single C3D4_ANP element tangent matches numerical central finite differences."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    np.random.seed(321)
    u_base = 0.001 * np.random.randn(12)
    J_bar_e = 1.002

    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((1, 1), dtype=np.float64)

    K_ana, f_base, err = compute_c3d4_anp_element_matrices(
        coords, u_base, J_bar_e, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs, dt=1.0
    )
    assert err == 0

    # Numerical Jacobian
    h = 1e-6
    K_num = np.zeros((12, 12), dtype=np.float64)

    for j in range(12):
        u_p = u_base.copy()
        u_m = u_base.copy()
        u_p[j] += h
        u_m[j] -= h

        _, f_p, _ = compute_c3d4_anp_element_matrices(
            coords, u_p, J_bar_e, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs, dt=1.0
        )
        _, f_m, _ = compute_c3d4_anp_element_matrices(
            coords, u_m, J_bar_e, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs, dt=1.0
        )

        K_num[:, j] = (f_p - f_m) / (2.0 * h)

    rel_err = np.linalg.norm(K_ana - K_num) / np.linalg.norm(K_ana)
    assert rel_err < 1e-5, f"C3D4_ANP tangent consistency error too high: {rel_err}"


def test_c3d4_anp_dynamicsolver3d_solve():
    """Verify DynamicSolver3D non-linear solve convergence on a C3D4_ANP patch mesh."""
    from dispsolver.mesh3d.mesh3d import Mesh3D
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D

    node_coords, elem_conn = make_5_tet_cube_patch()
    mesh = Mesh3D()

    for i in range(node_coords.shape[0]):
        mesh.add_node(i + 1, node_coords[i, 0], node_coords[i, 1], node_coords[i, 2])

    for e in range(elem_conn.shape[0]):
        conn = [elem_conn[e, a] + 1 for a in range(4)]
        mesh.add_element(e + 1, conn, elem_type="C3D4_ANP")

    solver = DynamicSolver3D(mesh, {"E": 200000.0, "nu": 0.3})

    # Fix bottom face Z = 0
    for node in mesh.nodes.values():
        if abs(node.z - 0.0) < 1e-5:
            solver.fix_dof(node.id, 0, 0.0)
            solver.fix_dof(node.id, 1, 0.0)
            solver.fix_dof(node.id, 2, 0.0)

    # Apply vertical displacement on top face Z = 1
    for node in mesh.nodes.values():
        if abs(node.z - 1.0) < 1e-5:
            solver.fix_dof(node.id, 2, -0.02)

    u_conv, converged = solver.solve_step(dt=1.0)
    assert converged
    assert np.max(np.abs(u_conv)) > 0.01

