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
    # linear elastic, NLGEOM=NO: every increment of the *STATIC ramp converges
    # in a couple of iterations (the deck is now solved incrementally like
    # Abaqus does, not in a single one-shot increment)
    assert res["max_increment_iterations"] <= 3

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
    # linear elastic, NLGEOM=NO: every increment of the *STATIC ramp converges
    # in a couple of iterations (the deck is now solved incrementally like
    # Abaqus does, not in a single one-shot increment)
    assert res["max_increment_iterations"] <= 3
    u = res["u"]
    assert u is not None
    # Maximum X displacement should be exactly 0.05
    ux = u[0::2]
    assert np.isclose(np.max(ux), 0.05, atol=1e-6)


def test_instance_translation_before_rotation():
    """T-before-R: x' = R(axis, angle) @ (x + t) about the axis point."""
    inp_text = """
*HEADING
T-before-R fixture
*PART, NAME=P
*NODE
1, 1.0, 0.0
*END PART
*ASSEMBLY, NAME=A
*INSTANCE, NAME=I, PART=P
1.0, 0.0, 0.0
0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0
*END INSTANCE
*END ASSEMBLY
"""
    model = read_inp(inp_text, dim=2)
    inst = model.root_assembly.instances["I"]
    # (1,0) + t(1,0) = (2,0); R(90deg) -> (0,2). Old R-then-T gave (1,1).
    w = inst.get_world_coordinates()
    assert np.allclose(w[0], [0.0, 2.0], atol=1e-12)
    sys = model.build_solver_system()
    assert np.allclose(sys.coords[0][:2], [0.0, 2.0], atol=1e-12)


def test_legacy_new_pipeline_node_element_equivalence(tmp_path):
    """G0: legacy read_abaqus_input and registry read_inp agree on counts."""
    from dispsolver.io import read_abaqus_input
    inp_text = """
*HEADING
G0 equivalence deck
*NODE
1, 0.0, 0.0
2, 2.0, 0.0
3, 2.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=Q
1, 1, 2, 3, 4
*MATERIAL, NAME=ALU
*ELASTIC
70000.0, 0.33
*SOLID SECTION, ELSET=Q, MATERIAL=ALU
"""
    model = read_inp(inp_text, dim=2)
    sys = model.build_solver_system()
    inp_file = tmp_path / "g0.inp"
    inp_file.write_text(inp_text, encoding="utf-8")
    legacy = read_abaqus_input(str(inp_file))
    assert sys.coords.shape[0] == legacy.mesh.node_count()
    assert len(sys.elem_conn_0based) == legacy.mesh.element_count()


def _fingerprint(model):
    parts = {}
    for pname, part in model.parts.items():
        parts[pname] = (len(part.nodes), len(part.elements))
    return {
        "parts": parts,
        "materials": sorted(model.materials.keys()),
        "steps": sorted(model.steps.keys()),
        "ties": sorted(model.constraints.keys()),
        "amplitudes": sorted(model.amplitudes.keys()),
    }


def test_g3_double_parse_fingerprint_stable():
    """G3: parsing the same deck twice yields identical model fingerprints."""
    inp_text = """
*HEADING
G3 fingerprint deck
*NODE
1, 0.0, 0.0
2, 2.0, 0.0
3, 2.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=Q
1, 1, 2, 3, 4
*NSET, NSET=L
1, 4
*MATERIAL, NAME=ALU
*ELASTIC
70000.0, 0.33
*SOLID SECTION, ELSET=Q, MATERIAL=ALU
*AMPLITUDE, NAME=R
0.0, 0.0
1.0, 1.0
*TIE, NAME=T
L, L
*STEP, NAME=S, NLGEOM=NO
*STATIC
0.1, 1.0
*BOUNDARY, OP=MOD
L, 1, 2, 0.0
*END STEP
"""
    m1 = read_inp(inp_text, dim=2)
    m2 = read_inp(inp_text, dim=2)
    assert _fingerprint(m1) == _fingerprint(m2)
    assert m1.steps["S"].boundary_conditions["BC_L_DOF1_2"].modified_params["op"] == "MOD"


def test_g4_3d_cload_cube_executes_through_inp():
    """G4: inp-driven 3D CLOAD cube converges with the load applied."""
    inp_text = """
*HEADING
G4 3D CLOAD cube
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=C
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=FIX
1, 4, 5, 8
*NSET, NSET=LOADN
2, 3, 6, 7
*MATERIAL, NAME=M
*ELASTIC
1000.0, 0.0
*SOLID SECTION, ELSET=C, MATERIAL=M
*STEP, NAME=S, NLGEOM=NO
*STATIC
1.0, 1.0
*BOUNDARY
FIX, 1, 3, 0.0
*CLOAD
LOADN, 1, 100.0
*END STEP
"""
    res = execute_inp(inp_text, dim=3, verbose=False)
    assert res["converged"] is True
    assert np.isclose(np.max(np.abs(res["u"])), 0.4, atol=1e-6)


def test_g5_op_new_mod_semantics_dump():
    """G5: OP=NEW deactivates propagated BCs/loads, OP=MOD keeps them."""
    inp_text = """
*HEADING
G5 OP deck
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
*NSET, NSET=A
1
*NSET, NSET=B
2
*STEP, NAME=S1
*STATIC
1.0, 1.0
*BOUNDARY
A, 1, 1, 0.0
*CLOAD
A, 1, 5.0
*END STEP
*STEP, NAME=S2
*STATIC
1.0, 1.0
*BOUNDARY, OP=NEW
B, 2, 2, 0.5
*CLOAD, OP=NEW
B, 1, 9.0
*END STEP
"""
    model = read_inp(inp_text, dim=2)
    s2 = model.steps["S2"]
    assert sorted(s2.get_active_bcs().keys()) == ["BC_B_DOF2_2"]
    assert sorted(s2.get_active_loads().keys()) == ["CLOAD_B_DOF1"]
    assert s2.boundary_conditions["BC_A_DOF1_1"].status.name == "DEACTIVATED"


def test_g1_lexer_three_word_quoted_comma_continuation():
    """G1: 3-word keywords, quoted commas, trailing-comma lists tokenize."""
    from dispsolver.io.abaqus_lexer import tokenize_string
    blocks = tokenize_string(
        "** leading comment\n"
        "*HEADING\n"
        "A ** B kept\n"
        "*CONTACT CONTROLS ASSIGNMENT, TYPE=FACTOR\n"
        "set-1,\n"
        "set-2\n"
        "*AMPLITUDE, NAME=\"A, B\", DEFINITION=TABULAR\n"
        "0.0, 0.0\n"
        "1.0, 1.0\n"
        "*NSET, NSET=G\n"
        "1,\n"
        "2, 3\n"
        "*END STEP\n"
    )
    by_kw = {b.keyword: b for b in blocks}
    assert by_kw["CONTACT CONTROLS ASSIGNMENT"].params.get("type") == "FACTOR"
    assert by_kw["AMPLITUDE"].params.get("name") == '"A, B"'
    assert     by_kw["NSET"].data_lines == ["1, 2, 3"]
    assert by_kw["HEADING"].data_lines == ["A ** B kept"]


def test_p2_support_declared_for_every_keyword():
    """P2-2: every registered keyword declares FULL/PARTIAL/STUB/IGNORED."""
    from dispsolver.io import GLOBAL_KEYWORD_REGISTRY
    allowed = {"FULL", "PARTIAL", "STUB", "IGNORED"}
    seen = {}
    for name, spec in GLOBAL_KEYWORD_REGISTRY._handlers.items():
        assert spec.support in allowed, name
        seen[spec.name] = spec.support
    assert seen["MPC"] == "STUB"
    assert seen["DLOAD"] == "PARTIAL"
    assert seen["NODE"] == "FULL"


def test_g6_acceptance_deck_zero_warnings():
    """G6: a clean acceptance deck parses with no warnings and no unsupported."""
    from dispsolver.io.abaqus_reader import _parse_source
    inp_text = """
*HEADING
G6 acceptance deck
*NODE
1, 0.0, 0.0
2, 2.0, 0.0
3, 2.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=Q
1, 1, 2, 3, 4
*NSET, NSET=L
1, 4
*MATERIAL, NAME=ALU
*ELASTIC
70000.0, 0.33
*SOLID SECTION, ELSET=Q, MATERIAL=ALU
*STEP, NAME=S, NLGEOM=NO
*STATIC
0.1, 1.0
*BOUNDARY
L, 1, 2, 0.0
*END STEP
"""
    reader, model = _parse_source(inp_text, dim=2)
    assert reader.warnings == []
    assert reader.unsupported_keywords == []
