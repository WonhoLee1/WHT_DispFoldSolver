"""Unit and integration test suite for dispsolver.model CAE architecture."""

import pytest
import numpy as np

from dispsolver.model import (
    Model,
    Part,
    Material,
    SolidSection,
    ShellSection,
    Transform3D,
    NodeSet,
    ElementSet,
    Surface,
    Step,
    DisplacementBC
)


def test_transform3d_affine_math():
    """Verify 3D translation and Rodrigues rotation."""
    coords = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Pure translation
    t1 = Transform3D(translation=[10.0, -5.0, 2.0])
    out1 = t1.apply(coords)
    assert np.allclose(out1[0], [11.0, -5.0, 2.0])
    
    # 90 deg rotation about Z axis: (1, 0, 0) -> (0, 1, 0)
    t2 = Transform3D(rotation_axis=[0, 0, 1], rotation_angle_deg=90.0)
    out2 = t2.apply(coords)
    assert np.allclose(out2[0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(out2[1], [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(out2[2], [0.0, 0.0, 1.0], atol=1e-12)


def test_part_and_section_assignment():
    """Verify Part creation and SectionAssignment to multiple ElementSets."""
    model = Model(name="TestModel", dim=2)
    
    # Materials
    mat_pet = model.Material(name="PET", mat_type="ELASTIC")
    mat_pet.elastic = (4000.0, 0.3)
    
    mat_psa = model.Material(name="PSA", mat_type="NEO_HOOKEAN")
    mat_psa.hyperelastic = {"c10": 0.25, "d1": 0.001}
    
    # Sections
    sec_pet = model.SolidSection(name="SecPET", material="PET", thickness=0.1)
    sec_psa = model.SolidSection(name="SecPSA", material="PSA", thickness=0.05)
    
    # Part
    part = model.Part(name="BilayerBeam", dim=2)
    part.add_node(1, [0.0, 0.0])
    part.add_node(2, [10.0, 0.0])
    part.add_node(3, [10.0, 0.1])
    part.add_node(4, [0.0, 0.1])
    part.add_node(5, [10.0, 0.15])
    part.add_node(6, [0.0, 0.15])
    
    part.add_element(1, "CPE4", [1, 2, 3, 4])
    part.add_element(2, "CPE4", [4, 3, 5, 6])
    
    part.create_element_set("ELSET_PET", [1])
    part.create_element_set("ELSET_PSA", [2])
    part.create_node_set("NSET_LEFT", [1, 4, 6])
    
    part.assign_section("ELSET_PET", "SecPET")
    part.assign_section("ELSET_PSA", "SecPSA")
    
    assert len(part.section_assignments) == 2
    assert part.section_assignments[0].region == "ELSET_PET"
    assert part.section_assignments[1].region == "ELSET_PSA"


def test_assembly_multi_instance_flattening():
    """Verify assembly of multiple instances of the same part with transformations."""
    model = Model(name="AssemblyTest", dim=3)
    
    mat = model.Material(name="Steel", mat_type="ELASTIC")
    mat.elastic = (210000.0, 0.3)
    sec = model.SolidSection(name="SteelSec", material="Steel")
    
    # Unit cube part (0,0,0) to (1,1,1)
    part = model.Part(name="Cube", dim=3)
    coords = [
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ]
    for i, c in enumerate(coords, start=1):
        part.add_node(i, c)
    part.add_element(1, "C3D8", list(range(1, 9)))
    part.create_element_set("ALL_ELEMS", [1])
    part.create_node_set("BOTTOM_NODES", [1, 2, 3, 4])
    part.assign_section("ALL_ELEMS", "SteelSec")
    
    # Instantiate Cube twice:
    # Instance 1 at (0, 0, 0)
    inst1 = model.root_assembly.create_instance(name="Cube_1", part=part)
    
    # Instance 2 translated by (10, 0, 0)
    inst2 = model.root_assembly.create_instance(name="Cube_2", part=part)
    inst2.translate([10.0, 0.0, 0.0])
    
    # Flatten
    sys = model.build_solver_system()
    
    assert sys.coords.shape == (16, 3)
    assert sys.elem_conn_0based.shape == (2, 8)
    assert sys.num_dofs == 16 * 3
    
    # Instance 1 should be at [0, 1]
    assert np.allclose(sys.coords[0], [0, 0, 0])
    # Instance 2 node 1 should be at [10, 0, 0]
    inst2_node1_gid = sys.node_local_to_global[("Cube_2", 1)]
    inst2_node1_idx = sys.nid_to_idx[inst2_node1_gid]
    assert np.allclose(sys.coords[inst2_node1_idx], [10.0, 0.0, 0.0])
    
    # Qualified sets
    assert "Cube_1.BOTTOM_NODES" in sys.global_nsets
    assert "Cube_2.BOTTOM_NODES" in sys.global_nsets
    assert len(sys.global_nsets["Cube_1.BOTTOM_NODES"]) == 4
    assert len(sys.global_nsets["Cube_2.BOTTOM_NODES"]) == 4
    
    # CSR Sparsity topology check
    # 2 elements, each has 8 nodes * 3 DOFs = 24 DOFs -> 24*24 = 576 entries
    assert len(sys.rows_topo) == 2 * 576
    assert len(sys.cols_topo) == 2 * 576


def test_model_to_solver3d_integration():
    """Verify that Model creates a fully functioning DynamicSolver3D instance."""
    model = Model(name="Solver3DIntegrationTest", dim=3)
    
    mat = model.Material(name="Aluminum", mat_type="ELASTIC")
    mat.elastic = (70000.0, 0.33)
    sec = model.SolidSection(name="AluSec", material="Aluminum")
    
    part = model.Part(name="CantileverBar", dim=3)
    # Simple 2-element bar along X
    # Nodes: x in [0, 1, 2], y in [0, 1], z in [0, 1]
    nid = 1
    node_grid = {}
    for i in range(3):
        for j in range(2):
            for k in range(2):
                part.add_node(nid, [float(i), float(j), float(k)])
                node_grid[(i, j, k)] = nid
                nid += 1
                
    # Element 1 (i=0 to 1), Element 2 (i=1 to 2)
    for i in range(2):
        n1 = node_grid[(i,   0, 0)]
        n2 = node_grid[(i+1, 0, 0)]
        n3 = node_grid[(i+1, 1, 0)]
        n4 = node_grid[(i,   1, 0)]
        n5 = node_grid[(i,   0, 1)]
        n6 = node_grid[(i+1, 0, 1)]
        n7 = node_grid[(i+1, 1, 1)]
        n8 = node_grid[(i,   1, 1)]
        conn = [n1, n2, n3, n4, n5, n6, n7, n8]
        part.add_element(i+1, "C3D8_CR", conn)
        
    part.create_element_set("ALL_ELEMS", [1, 2])
    # Root face at x=0
    fixed_nids = [node_grid[(0, j, k)] for j in range(2) for k in range(2)]
    part.create_node_set("ROOT_FACE", fixed_nids)
    part.assign_section("ALL_ELEMS", "AluSec")
    
    inst = model.root_assembly.create_instance("BAR", part=part)
    
    # Step with Boundary Condition
    step1 = model.Step(name="Step-1", procedure="STATIC")
    bc = DisplacementBC(name="FIX_ROOT", region="BAR.ROOT_FACE", u1=0.0, u2=0.0, u3=0.0)
    step1.add_boundary_condition(bc)
    
    # Build solver
    solver, sys = model.create_solver3d(step_name="Step-1")
    
    # Check that ROOT_FACE DOFs were fixed
    for nid_val in fixed_nids:
        gid = sys.node_local_to_global[("BAR", nid_val)]
        idx = sys.nid_to_idx[gid]
        assert 3 * idx + 0 in solver.fixed_dofs
        assert 3 * idx + 1 in solver.fixed_dofs
        assert 3 * idx + 2 in solver.fixed_dofs
        
    # Solve 1 step (zero external load -> trivial zero displacement solution in 1 iter)
    conv, iters = solver.solve_step(dt=1.0)
    assert conv is True
    assert iters <= 2
    assert np.allclose(solver.u, 0.0)
