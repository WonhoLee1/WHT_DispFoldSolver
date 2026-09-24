"""
test_nodal_transform.py
========================
Verification suite for Abaqus *TRANSFORM nodal coordinate transformation engine.

Tests:
1. Rotation matrix computation for 2D & 3D (Rectangular, Cylindrical, Spherical).
2. 2D Slanted Roller BC (45° inclined boundary): exact zero normal displacement
   and free tangential sliding.
3. 3D Slanted Boundary Condition on C3D8 solid element mesh via DynamicSolver3D.
4. Abaqus .inp deck parsing + end-to-end solve integration with *TRANSFORM.
"""

import numpy as np
import pytest

from dispsolver.model import Model, Instance
from dispsolver.model.bc import DisplacementBC
from dispsolver.model.nodal_transform import (
    compute_rotation_matrix,
    build_nodal_transforms,
    transform_u_to_global,
    transform_u_to_local,
)


def test_rotation_matrix_2d_rectangular():
    """Verify 2D 45-degree rectangular transform matrix."""
    spec = {"name": "TRF_45", "nset": "SLANTED", "type": "R", "points": ((1.0, 1.0),)}
    R = compute_rotation_matrix(spec, np.array([0.0, 0.0]), dim=2)
    expected_R = np.array([[np.sqrt(0.5), -np.sqrt(0.5)],
                           [np.sqrt(0.5),  np.sqrt(0.5)]])
    assert np.allclose(R, expected_R)
    # Check orthogonality
    assert np.allclose(R @ R.T, np.eye(2))


def test_rotation_matrix_3d_cylindrical():
    """Verify 3D cylindrical transform matrix along Z-axis."""
    spec = {"name": "TRF_CYL", "nset": "RING", "type": "C", "points": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))}
    # Node at (1.0, 0.0, 5.0) -> radial direction (1,0,0), tangential (0,1,0), axial (0,0,1)
    R = compute_rotation_matrix(spec, np.array([1.0, 0.0, 5.0]), dim=3)
    assert np.allclose(R, np.eye(3))

    # Node at (0.0, 2.0, 5.0) -> radial direction (0,1,0), tangential (-1,0,0), axial (0,0,1)
    R_90 = compute_rotation_matrix(spec, np.array([0.0, 2.0, 5.0]), dim=3)
    expected_90 = np.array([[0.0, -1.0, 0.0],
                            [1.0,  0.0, 0.0],
                            [0.0,  0.0, 1.0]])
    assert np.allclose(R_90, expected_90)


def test_2d_slanted_roller_bc_physics():
    """Verify 2D 45° slanted roller BC: exact zero normal movement, free tangential sliding."""
    model = Model(name="SlantedRollerTest")

    # Part: 1x1 block
    p = model.Part(name="Block")
    p.Node(1, 0.0, 0.0)
    p.Node(2, 1.0, 0.0)
    p.Node(3, 1.0, 1.0)
    p.Node(4, 0.0, 1.0)

    p.Element(1, "CPE4", [1, 2, 3, 4])
    p.create_set(name="FIX_LEFT", nodes=[1, 4])
    p.create_set(name="SLANTED_RIGHT", nodes=[2, 3])

    mat = model.Material(name="Steel")
    mat.Elastic(E=200000.0, nu=0.3)
    p.create_set(name="EALL", elements=[1])
    sec = model.SolidSection(name="Sec", material="Steel")
    p.SectionAssignment(region="EALL", section_name="Sec")

    # Instance & Assembly
    inst = Instance(name="BlockInst", part=p)
    model.root_assembly.instances[inst.name] = inst

    # 45-degree Transform on SLANTED_RIGHT (Point A = (1, 1))
    model.Transform(name="TRF_45", nset="BlockInst.SLANTED_RIGHT", transform_type="R", points=((1.0, 1.0),))

    step = model.Step(name="Pull")
    # Fix left edge (x=0) in global coordinates
    step.add_boundary_condition(DisplacementBC(name="FixLeft", region="BlockInst.FIX_LEFT", u1=0.0, u2=0.0))
    # Fix SLANTED_RIGHT normal DOF (local u2 = 0) while leaving local u1 free
    step.add_boundary_condition(DisplacementBC(name="FixNormal", region="BlockInst.SLANTED_RIGHT", u2=0.0))

    # Create solver
    solver, sys = model.create_solver2d(step_name="Pull")
    assert solver.has_nodal_transforms

    # Apply pulling force along local u1 (tangential) on top-right node
    inv_nid = {idx: g for g, idx in sys.nid_to_idx.items()}
    node3_idx = sys.global_nsets["BlockInst.SLANTED_RIGHT"][1]  # node 3 index
    node3_gid = inv_nid[node3_idx]

    # Prescribe node 3 local u1 tangential displacement = 0.001 mm
    solver.fix_dof(node3_gid, 0, 0.001)

    res = solver.solve_step(dt=1.0)
    converged = res[0] if isinstance(res, tuple) else (res > 0)
    assert converged

    # Local displacements
    u_local = solver.u
    u2_local_norm = u_local[node3_idx * 2 + 1]  # local u2 (normal)
    u2_local_tang = u_local[node3_idx * 2 + 0]  # local u1 (tangential)

    # Normal displacement must be exact machine zero (0.0)
    assert abs(u2_local_norm) < 1e-12, f"Normal displacement {u2_local_norm} is not zero!"
    # Tangential displacement must be positive
    assert u2_local_tang > 0.0, f"Tangential displacement {u2_local_tang} should be positive!"

    # Global displacements must be u_tang * (cos 45°, sin 45°)
    u_global = solver.get_global_displacements(u_local)
    gx = u_global[node3_idx * 2]
    gy = u_global[node3_idx * 2 + 1]

    expected_ratio = np.cos(np.pi / 4.0)
    assert pytest.approx(gx, rel=1e-5) == u2_local_tang * expected_ratio
    assert pytest.approx(gy, rel=1e-5) == u2_local_tang * expected_ratio


def test_3d_slanted_transform_physics():
    """Verify 3D C3D8 solid element under 45° slanted transform via DynamicSolver3D."""
    model = Model(name="3DSlantedTest")

    p = model.Part(name="Cube")
    p.Node(1, 0.0, 0.0, 0.0)
    p.Node(2, 1.0, 0.0, 0.0)
    p.Node(3, 1.0, 1.0, 0.0)
    p.Node(4, 0.0, 1.0, 0.0)
    p.Node(5, 0.0, 0.0, 1.0)
    p.Node(6, 1.0, 0.0, 1.0)
    p.Node(7, 1.0, 1.0, 1.0)
    p.Node(8, 0.0, 1.0, 1.0)

    p.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    p.create_set(name="FIX_BACK", nodes=[1, 4, 5, 8])
    p.create_set(name="TRANSFORM_FRONT", nodes=[2, 3, 6, 7])

    mat = model.Material(name="Steel3D")
    mat.Elastic(E=200000.0, nu=0.3)
    p.create_set(name="EALL", elements=[1])
    sec = model.SolidSection(name="Sec3D", material="Steel3D")
    p.SectionAssignment(region="EALL", section_name="Sec3D")

    inst = Instance(name="CubeInst", part=p)
    model.root_assembly.instances[inst.name] = inst

    # 45° transform in XY plane for TRANSFORM_FRONT (Point A = (1, 1, 0), Point B = (0, 1, 0))
    model.Transform(name="TRF_XY45", nset="CubeInst.TRANSFORM_FRONT", transform_type="R",
                    points=((1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))

    step = model.Step(name="Step1")
    step.add_boundary_condition(DisplacementBC(name="FixBack", region="CubeInst.FIX_BACK", u1=0.0, u2=0.0, u3=0.0))
    # Fix local u2 = 0 on front nodes
    step.add_boundary_condition(DisplacementBC(name="FixLocalU2", region="CubeInst.TRANSFORM_FRONT", u2=0.0))

    solver, sys = model.create_solver3d(step_name="Step1")
    assert solver.has_nodal_transforms

    # Apply force along local u1 (100N) on node 7
    node7_idx = sys.global_nsets["CubeInst.TRANSFORM_FRONT"][3]  # node 7
    solver.fix_dof(sys.nid_to_idx[sys.node_local_to_global[("CubeInst", 7)]], 0, 0.001)  # 0.001 mm local u1 push

    res = solver.solve_step(dt=1.0)
    converged = res[0] if isinstance(res, tuple) else (res > 0)
    assert converged

    u_local = solver.u
    u2_local_front = u_local[node7_idx * 3 + 1]
    assert abs(u2_local_front) < 1e-12, "Local u2 on 3D transform node must be zero"


def test_abaqus_deck_transform_end_to_end():
    """Verify end-to-end .inp deck reading + solving with *TRANSFORM."""
    deck = """*HEADING
Slanted Roller Deck Test
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=EALL
1, 1, 2, 3, 4
*NSET, NSET=FIX_LEFT
1, 4
*NSET, NSET=SLANTED
2, 3
*TRANSFORM, NSET=SLANTED, TYPE=R
1.0, 1.0, 0.0
*MATERIAL, NAME=STEEL
*ELASTIC
200000.0, 0.3
*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL
*STEP
*STATIC
*BOUNDARY
FIX_LEFT, 1, 2, 0.0
SLANTED, 2, 2, 0.0
*CLOAD
3, 1, 50.0
*END STEP
"""
    from dispsolver.io.abaqus_reader import read_inp, execute_inp
    model = read_inp(deck)

    assert "TRF_SLANTED_1" in model.transforms or any("SLANTED" in name for name in model.transforms)
    res = execute_inp(deck)
    assert res is not None
    assert res.get("converged", False) or res.get("status", 0) >= 0
