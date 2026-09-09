"""
test_3d_numba.py
================
Unit & Performance tests for Numba OpenMP C3D8 3D Element Kernels.
"""

import numpy as np
import pytest

from dispsolver.element3d.c3d8_numba import (
    HAS_NUMBA,
    compute_c3d8_element_numba,
    assemble_mesh_c3d8_numba
)

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")


def test_c3d8_numba_element_stiffness():
    """Test Numba C3D8 element stiffness matrix calculation for a 1x1x1 cube."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float64)

    u_elem = np.zeros(24, dtype=np.float64)

    E, nu = 200000.0, 0.3
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    C_mat = np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],
        [lam,        lam + 2*mu, lam,        0,  0,  0],
        [lam,        lam,        lam + 2*mu, 0,  0,  0],
        [0,          0,          0,          mu, 0,  0],
        [0,          0,          0,          0,  mu, 0],
        [0,          0,          0,          0,  0,  mu]
    ], dtype=np.float64)

    K_elem, f_int = compute_c3d8_element_numba(coords, u_elem, C_mat)

    # 1. Symmetry check: K_elem == K_elem.T
    assert np.allclose(K_elem, K_elem.T, atol=1e-10)

    # 2. Rigid body modes check: 6 zero eigenvalues
    evals = np.linalg.eigvalsh(K_elem)
    n_rigid = np.sum(np.abs(evals) < 1e-8)
    assert n_rigid == 6, f"Expected 6 rigid body modes, got {n_rigid}"

    # 3. Pure tension check: u_x = 0.001 * x
    u_tension = np.zeros(24, dtype=np.float64)
    for i in range(8):
        u_tension[3*i] = 0.001 * coords[i, 0]

    _, f_tens = compute_c3d8_element_numba(coords, u_tension, C_mat)
    sig_xx_expected = (lam + 2*mu) * 0.001
    # Surface area for face x=1 is 1.0, total tensile force = sig_xx_expected
    force_x1 = sum(f_tens[3*i] for i in [1, 2, 5, 6])
    assert abs(force_x1 - sig_xx_expected) < 1e-6


def test_c3d8_numba_parallel_assembly():
    """Test parallel OpenMP assembly across 100 3D elements."""
    n_x, n_y, n_z = 5, 5, 4
    n_nodes = (n_x + 1) * (n_y + 1) * (n_z + 1)
    node_coords = np.zeros((n_nodes, 3), dtype=np.float64)

    idx = 0
    for k in range(n_z + 1):
        for j in range(n_y + 1):
            for i in range(n_x + 1):
                node_coords[idx] = [i, j, k]
                idx += 1

    elem_conn = []
    for k in range(n_z):
        for j in range(n_y):
            for i in range(n_x):
                n0 = k * (n_x + 1) * (n_y + 1) + j * (n_x + 1) + i
                n1 = n0 + 1
                n3 = n0 + (n_x + 1)
                n2 = n3 + 1
                n4 = n0 + (n_x + 1) * (n_y + 1)
                n5 = n4 + 1
                n7 = n4 + (n_x + 1)
                n6 = n7 + 1
                elem_conn.append([n0, n1, n2, n3, n4, n5, n6, n7])

    elem_conn = np.array(elem_conn, dtype=np.int32)
    u_global = np.zeros(3 * n_nodes, dtype=np.float64)

    E, nu = 200000.0, 0.3
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C_mat = np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],
        [lam,        lam + 2*mu, lam,        0,  0,  0],
        [lam,        lam,        lam + 2*mu, 0,  0,  0],
        [0,          0,          0,          mu, 0,  0],
        [0,          0,          0,          0,  mu, 0],
        [0,          0,          0,          0,  0,  mu]
    ], dtype=np.float64)

    f_elems, K_elems = assemble_mesh_c3d8_numba(node_coords, elem_conn, u_global, C_mat)

    assert f_elems.shape == (100, 24)
    assert K_elems.shape == (100, 24, 24)


def test_3d_dod_multimaterial_assembly():
    """Verify Phase 1: DOD Pre-allocated Assembly and Multi-Material UMAT Dispatch."""
    from dispsolver.mesh3d import Mesh3D
    from dispsolver.solver3d import DynamicSolver3D
    from dispsolver.material3d.numba_materials import MAT_LINEAR_ELASTIC, MAT_J2_PLASTICITY

    mesh = Mesh3D()
    # 2 elements stacked in X direction: 12 nodes
    # Elem 1: x in [0, 1], Elem 2: x in [1, 2]
    coords = [
        # x=0: nodes 1..4
        (1, 0.0, 0.0, 0.0), (2, 0.0, 1.0, 0.0), (3, 0.0, 1.0, 1.0), (4, 0.0, 0.0, 1.0),
        # x=1: nodes 5..8
        (5, 1.0, 0.0, 0.0), (6, 1.0, 1.0, 0.0), (7, 1.0, 1.0, 1.0), (8, 1.0, 0.0, 1.0),
        # x=2: nodes 9..12
        (9, 2.0, 0.0, 0.0), (10, 2.0, 1.0, 0.0), (11, 2.0, 1.0, 1.0), (12, 2.0, 0.0, 1.0)
    ]
    for nid, x, y, z in coords:
        mesh.add_node(nid, x, y, z)

    # Element 1: PID 0 (Linear Elastic)
    mesh.add_element(1, [1, 5, 6, 2, 4, 8, 7, 3], "C3D8_FBAR", pid=0)
    # Element 2: PID 1 (J2 Plasticity)
    mesh.add_element(2, [5, 9, 10, 6, 8, 12, 11, 7], "C3D8_FBAR", pid=1)

    materials = {
        0: {"type": "linear_elastic", "E": 100000.0, "nu": 0.25},
        1: {"type": "j2_plasticity", "E": 200000.0, "nu": 0.3, "sigma_y0": 300.0, "H": 1000.0}
    }

    solver = DynamicSolver3D(mesh, materials=materials, nlgeom=True)

    # 1. Verify DOD Pre-allocated data structures
    assert solver.rows_topo is not None
    assert solver.cols_topo is not None
    assert len(solver.rows_topo) == 2 * 576
    assert len(solver.cols_topo) == 2 * 576
    assert solver.elem_mat_types[0] == MAT_LINEAR_ELASTIC
    assert solver.elem_mat_types[1] == MAT_J2_PLASTICITY
    assert solver.elem_props[0, 0] == 100000.0
    assert solver.elem_props[1, 0] == 200000.0
    assert solver.elem_props[1, 2] == 300.0  # sigma_y0

    # 2. Assemble system at zero displacement
    u_zeros = np.zeros(solver.num_dofs, dtype=np.float64)
    K_g, f_int = solver.assemble_system(u_zeros, dt=1.0)

    assert K_g.shape == (36, 36)
    assert f_int.shape == (36,)
    assert np.allclose(f_int, 0.0)
    assert np.allclose(K_g.toarray(), K_g.toarray().T, atol=1e-8)

    # 3. Assemble system under prescribed deformation to test UMAT stress & force scattering
    u_deformed = np.zeros(solver.num_dofs, dtype=np.float64)
    nid_map = mesh.node_id_to_index()
    # Stretch in X
    for nid in [9, 10, 11, 12]:
        u_deformed[3 * nid_map[nid] + 0] = 0.01

    K_g2, f_int2 = solver.assemble_system(u_deformed, dt=1.0)
    assert np.linalg.norm(f_int2) > 0.0
    # Net internal force on free end must be positive tension
    f_end_x = sum(f_int2[3 * nid_map[nid] + 0] for nid in [9, 10, 11, 12])
    assert f_end_x > 0.0

