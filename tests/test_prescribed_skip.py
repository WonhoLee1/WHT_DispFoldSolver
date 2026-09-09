"""
test_prescribed_skip.py
=======================
Tests for multi-material batch prescribed element skipping in DynamicSolver.
"""

import numpy as np
import pytest

from dispsolver.mesh.mesh import Mesh, Node, Element
from dispsolver.material.plastic import J2Plasticity
from dispsolver.material.neohookean import NeoHookean
from dispsolver.solver.dynamic import DynamicSolver


def create_two_block_mesh():
    """Create a 2-block mesh: Block 0 (active, x in [0, 10]), Block 1 (prescribed, x in [10, 20])."""
    mesh = Mesh()
    xs = [0.0, 5.0, 10.0, 15.0, 20.0]
    ys = [0.0, 0.5, 1.0]

    node_map = {}
    nid = 1
    for y in ys:
        for x in xs:
            mesh.add_node(nid, x, y)
            node_map[(x, y)] = nid
            nid += 1

    # Elements: 4 columns x 2 rows = 8 elements
    # Col 0, 1 -> pid 0 (J2Plasticity)
    # Col 2, 3 -> pid 1 (NeoHookean / Plate)
    eid = 1
    for row in range(2):
        y0, y1 = ys[row], ys[row + 1]
        for col in range(4):
            x0, x1 = xs[col], xs[col + 1]
            n1 = node_map[(x0, y0)]
            n2 = node_map[(x1, y0)]
            n3 = node_map[(x1, y1)]
            n4 = node_map[(x0, y1)]
            pid = 0 if col < 2 else 1
            mesh.add_element(eid, [n1, n2, n3, n4], "CPE4", pid=pid)
            eid += 1

    return mesh, xs, ys, node_map


def test_prescribed_skip_identification():
    """Verify that fully prescribed elements are correctly identified."""
    mesh, xs, ys, node_map = create_two_block_mesh()

    mat0 = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=100.0, H=100.0)
    mat1 = NeoHookean()
    materials = {0: mat0, 1: mat1}
    mat_params = {0: {}, 1: {"E": 2000.0, "nu": 0.3}}

    solver = DynamicSolver(
        mesh=mesh,
        material=materials,
        material_params=mat_params,
        rho=1.0,
        skip_fully_prescribed=True,
        mode="quasistatic",
    )

    # Prescribe all nodes with x >= 10.0
    conn, nid_to_idx, sorted_nids, elem_ids = mesh.connectivity_array()
    bc_dofs = []
    for (x, y), nid in node_map.items():
        if x >= 10.0:
            idx = nid_to_idx[nid]
            bc_dofs.extend([idx * 2, idx * 2 + 1])

    solver.set_prescribed_dofs(np.array(bc_dofs, dtype=np.int32), np.zeros(len(bc_dofs)))

    # Elements in Col 2 and 3 (all nodes x >= 10) should be fully prescribed
    assert solver._elem_is_fully_prescribed is not None
    for e in range(solver.n_elem):
        coords = solver.elem_coords[e]
        if np.all(coords[:, 0] >= 10.0):
            assert solver._elem_is_fully_prescribed[e] == True
        else:
            assert solver._elem_is_fully_prescribed[e] == False

    # pid 1 elements (Col 2 and 3) should have 0 active elements
    assert len(solver._pid_active_elem_indices[1]) == 0
    # pid 0 elements should all be active
    assert len(solver._pid_active_elem_indices[0]) == 4


def test_prescribed_skip_solution_equivalence():
    """Verify that solver with skip_fully_prescribed=True yields identical u to False."""
    mesh, xs, ys, node_map = create_two_block_mesh()

    mat0 = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=100.0, H=100.0)
    mat1 = NeoHookean()
    materials = {0: mat0, 1: mat1}
    mat_params = {0: {}, 1: {"E": 2000.0, "nu": 0.3}}

    conn, nid_to_idx, sorted_nids, elem_ids = mesh.connectivity_array()

    # Prescribe all nodes with x >= 10 with a displacement of 1e-4 in Y
    bc_dofs = []
    bc_vals = []
    for (x, y), nid in node_map.items():
        if x >= 10.0:
            idx = nid_to_idx[nid]
            bc_dofs.extend([idx * 2, idx * 2 + 1])
            bc_vals.extend([0.0, 1e-4])
        elif x == 0.0:
            # Fix x=0
            idx = nid_to_idx[nid]
            bc_dofs.extend([idx * 2, idx * 2 + 1])
            bc_vals.extend([0.0, 0.0])

    bc_dofs = np.array(bc_dofs, dtype=np.int32)
    bc_vals = np.array(bc_vals, dtype=np.float64)

    # Solve with skip_fully_prescribed = False
    s_full = DynamicSolver(
        mesh=mesh,
        material=materials,
        material_params=mat_params,
        rho=1.0,
        skip_fully_prescribed=False,
        mode="quasistatic",
        tol=1e-4,
    )
    s_full.set_prescribed_dofs(bc_dofs, bc_vals)
    ret_full = s_full.solve_step(1.0)
    assert ret_full >= 0

    # Solve with skip_fully_prescribed = True
    s_skip = DynamicSolver(
        mesh=mesh,
        material=materials,
        material_params=mat_params,
        rho=1.0,
        skip_fully_prescribed=True,
        mode="quasistatic",
        tol=1e-4,
    )
    s_skip.set_prescribed_dofs(bc_dofs, bc_vals)
    ret_skip = s_skip.solve_step(1.0)
    assert ret_skip >= 0

    # Displacements must match to machine precision
    np.testing.assert_allclose(s_skip.u, s_full.u, atol=1e-10, rtol=1e-8)
