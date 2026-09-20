"""
test_abaqus_reader_and_executor.py
==================================
Comprehensive TDD test suite for AbaqusInputReader and execute_inp pipeline.
Tests Part/Assembly hierarchy, 3D affine instance transformations,
Non-Assembly flat models, materials, sections, and solver execution.
"""

import pytest
import numpy as np

from dispsolver.io import read_inp, execute_inp, AbaqusInputReader
from dispsolver.model.model import Model


def test_reader_non_assembly_flat_deck_to_model():
    """Verify parsing a flat (non-assembly) 3D cantilever deck into Model."""
    inp_text = """
*HEADING
Single Cantilever Non-Assembly
*NODE, NSET=ALL_NODES
1, 0.0, 0.0, 0.0
2, 10.0, 0.0, 0.0
3, 10.0, 2.0, 0.0
4, 0.0, 2.0, 0.0
5, 0.0, 0.0, 2.0
6, 10.0, 0.0, 2.0
7, 10.0, 2.0, 2.0
8, 0.0, 2.0, 2.0
*ELEMENT, TYPE=C3D8, ELSET=BEAM_ELEM
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=FIX_ROOT
1, 4, 5, 8
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=BEAM_ELEM, MATERIAL=STEEL
*STEP, NAME=PULL, NLGEOM=NO
*STATIC
0.1, 1.0, 0.01, 0.1
*BOUNDARY
FIX_ROOT, 1, 3, 0.0
2, 1, 1, 0.5
*END STEP
"""
    model = read_inp(inp_text, dim=3)
    assert isinstance(model, Model)
    assert len(model.parts) >= 1

    part = list(model.parts.values())[0]
    assert len(part.nodes) == 8
    assert len(part.elements) == 1
    assert "FIX_ROOT" in part.sets or "FIX_ROOT" in part.node_sets

    assert "STEEL" in model.materials
    mat = model.materials["STEEL"]
    assert mat.elastic == (210000.0, 0.3)

    assert "PULL" in model.steps
    step = model.steps["PULL"]
    assert len(step.boundary_conditions) >= 2


def test_reader_part_assembly_with_instance_transform():
    """Verify parsing an Abaqus Part/Assembly deck with instance translation & rotation."""
    inp_text = """
*HEADING
Two Instances Assembly Model
*PART, NAME=BLOCK_PART
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*END PART
*MATERIAL, NAME=RUBBER
*ELASTIC
10.0, 0.45
*SOLID SECTION, ELSET=EALL, MATERIAL=RUBBER
*ASSEMBLY, NAME=ROOT_ASSY
*INSTANCE, NAME=INST_A, PART=BLOCK_PART
*END INSTANCE
*INSTANCE, NAME=INST_B, PART=BLOCK_PART
5.0, 0.0, 0.0
*END INSTANCE
*END ASSEMBLY
"""
    model = read_inp(inp_text, dim=3)
    assert "BLOCK_PART" in model.parts
    part = model.parts["BLOCK_PART"]
    assert len(part.nodes) == 8

    assembly = model.root_assembly
    assert len(assembly.instances) == 2
    assert "INST_A" in assembly.instances
    assert "INST_B" in assembly.instances

    inst_a = assembly.instances["INST_A"]
    inst_b = assembly.instances["INST_B"]

    # World coordinates check: INST_B is translated by +5 in X
    coords_a = inst_a.get_world_coordinates()
    coords_b = inst_b.get_world_coordinates()
    assert np.allclose(coords_b[:, 0], coords_a[:, 0] + 5.0)

    # Flatten solver system check
    sys = model.build_solver_system()
    assert len(sys.coords) == 16  # 8 nodes * 2 instances
    assert len(sys.elem_conn_0based) == 2


def test_reader_and_executor_end_to_end():
    """Verify executing an .inp model directly via execute_inp() using DynamicSolver3D."""
    inp_text = """
*HEADING
3D Uniaxial Tension Benchmark
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=CUBE
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=FIX_X0
1, 4, 5, 8
*NSET, NSET=PULL_X1
2, 3, 6, 7
*MATERIAL, NAME=MAT1
*ELASTIC
1000.0, 0.0
*SOLID SECTION, ELSET=CUBE, MATERIAL=MAT1
*STEP, NAME=TENSION, NLGEOM=NO
*STATIC
1.0, 1.0
*BOUNDARY
FIX_X0, 1, 3, 0.0
PULL_X1, 1, 1, 0.1
*END STEP
"""
    res = execute_inp(inp_text, dim=3, verbose=False)
    assert res["converged"] is True
    assert res["iterations"] <= 3

    # Under E=1000, nu=0, L=1.0, prescribed ux=0.1 at x=1,
    # axial stress = E * eps = 1000 * 0.1 = 100 MPa
    u = res["u"]
    assert u is not None
    # Maximum displacement should be exactly 0.1
    assert np.isclose(np.max(u), 0.1, atol=1e-6)


def test_reader_and_executor_2d_plane_strain():
    """Verify executing a 2D plane strain .inp model via execute_inp() using DynamicSolver2D."""
    inp_text = """
*HEADING
2D Plane Strain Pure Tension
*NODE
1, 0.0, 0.0
2, 2.0, 0.0
3, 2.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=QUAD_EL
1, 1, 2, 3, 4
*NSET, NSET=LEFT_EDGE
1, 4
*NSET, NSET=RIGHT_EDGE
2, 3
*MATERIAL, NAME=ALU
*ELASTIC
70000.0, 0.33
*SOLID SECTION, ELSET=QUAD_EL, MATERIAL=ALU
1.0
*STEP, NAME=STRETCH, NLGEOM=NO
*STATIC
0.1, 1.0
*BOUNDARY
LEFT_EDGE, 1, 2, 0.0
RIGHT_EDGE, 1, 1, 0.05
*END STEP
"""
    res = execute_inp(inp_text, dim=2, verbose=False)
    assert res["converged"] is True
    assert res["iterations"] <= 3
    u = res["u"]
    assert u is not None
    # Maximum X displacement should be exactly 0.05
    ux = u[0::2]
    assert np.isclose(np.max(ux), 0.05, atol=1e-6)
