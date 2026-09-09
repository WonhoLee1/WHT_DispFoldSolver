"""
test_3d_inp_parser.py
=====================
TDD Test Suite for Abaqus 3D INP Parser Integration.
"""

import pytest
import os
import tempfile

from dispsolver.io.abaqus_lexer import tokenize_string
from dispsolver.io.abaqus_parser import AbaqusParser


def test_abaqus_3d_inp_parsing():
    """Verify parsing 3D nodes (x, y, z) and 3D solid elements (C3D8I, C3D10M) from .inp."""
    inp_content = """
*HEADING
3D Solid Element Test Case
*NODE, NSET=ALLNODES
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8I, ELSET=HEX_ELEM
1, 1, 2, 3, 4, 5, 6, 7, 8
*ELEMENT, TYPE=C3D10M, ELSET=TET_ELEM
2, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10
*MATERIAL, NAME=STEEL
*ELASTIC
200000.0, 0.3
*SOLID SECTION, ELSET=HEX_ELEM, MATERIAL=STEEL
"""

    blocks = tokenize_string(inp_content)
    parser = AbaqusParser()
    model = parser.parse(blocks)

    # Check 3D Node Parsing
    assert len(model.nodes) == 8
    node_8 = [n for n in model.nodes if n.id == 8][0]
    assert node_8.x == 0.0
    assert node_8.y == 1.0
    assert node_8.z == 1.0

    # Check 3D Element Parsing
    assert len(model.elements) == 2
    elem_hex = [e for e in model.elements if e.eid == 1][0]
    assert elem_hex.etype == "C3D8I"
    assert len(elem_hex.node_ids) == 8

    elem_tet = [e for e in model.elements if e.eid == 2][0]
    assert elem_tet.etype == "C3D10M"
    assert len(elem_tet.node_ids) == 10

    # Check Material & Section
    assert "STEEL" in model.materials
    assert model.materials["STEEL"].elastic == (200000.0, 0.3)
