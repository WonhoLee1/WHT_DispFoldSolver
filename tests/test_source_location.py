"""Source locations through the .inp pipeline (plan F3) and the lexer rules
they depend on.

Every keyword block carries (file, physical line), *INCLUDE files keep their
own names and line numbers, data lines keep theirs, and the model objects a
keyword creates carry `source_location` so errors raised at solve time point
at the deck line.  Plus the lexer defects found on the way (2026-10-05):
indented keyword lines were swallowed as data (85 corpus decks), hyphenated
parameter names became bare flags (*SURFACE BEHAVIOR always ran HARD), and
keyword-line continuation was not supported.
"""
import os
import warnings

import numpy as np
import pytest

from dispsolver.io.abaqus_lexer import InpSyntaxError, tokenize, tokenize_string
from dispsolver.io.abaqus_reader import _parse_source, detect_dim
from dispsolver.model.regions import RegionError

CUBE_NODES = """*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# ----------------------------------------------------------------- lexer
def test_include_keeps_file_and_line_numbers(tmp_path):
    _write(tmp_path, "mesh.inp", "** mesh\n" + CUBE_NODES + "*ELEMENT, TYPE=C3D8, ELSET=E\n1,1,2,3,4,5,6,7,8\n")
    main = _write(tmp_path, "main.inp", "*HEADING\nx\n*INCLUDE, INPUT=mesh.inp\n*NSET, NSET=BOT\n1,2,3,4\n")
    by_kw = {b.keyword: b for b in tokenize(main)}
    assert by_kw["NODE"].location == "mesh.inp:2"
    assert by_kw["NODE"].data_line_numbers[0] == 3
    assert by_kw["ELEMENT"].location == "mesh.inp:11"
    assert by_kw["NSET"].location == "main.inp:4"          # not shifted by the included lines


def test_missing_and_cyclic_includes_are_errors_with_location(tmp_path):
    main = _write(tmp_path, "main.inp", "*HEADING\nx\n*INCLUDE, INPUT=nope.inp\n")
    with pytest.raises(InpSyntaxError, match="main.inp:3"):
        tokenize(main)
    _write(tmp_path, "a.inp", "*INCLUDE, INPUT=b.inp\n")
    _write(tmp_path, "b.inp", "*NODE\n1,0,0,0\n*INCLUDE, INPUT=a.inp\n")
    with pytest.raises(InpSyntaxError, match="cyclic"):
        tokenize(str(tmp_path / "a.inp"))


def test_indented_keywords_are_keywords():
    blocks = tokenize_string("*ASSEMBLY,NAME=A\n *INSTANCE, NAME=I, PART=P\n   *END INSTANCE\n*END ASSEMBLY\n")
    assert [b.keyword for b in blocks] == ["ASSEMBLY", "INSTANCE", "END INSTANCE", "END ASSEMBLY"]
    assert blocks[1].params == {"name": "I", "part": "P"}


def test_hyphenated_parameter_names_and_bare_flags():
    b = tokenize_string("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=EXPONENTIAL, NO SEPARATION\n0.1, 2.\n")[0]
    assert b.params == {"pressure-overclosure": "EXPONENTIAL", "no separation": "yes"}
    b = tokenize_string("*HYPERELASTIC, ARRUDA-BOYCE\n1., 3., 0.\n")[0]
    assert b.params == {"arruda-boyce": "yes"}


def test_keyword_line_continuation_and_long_names():
    b = tokenize_string("*ELEMENT, TYPE=C3D8,\n  ELSET=E\n1,1,2,3,4,5,6,7,8\n")[0]
    assert b.params == {"type": "C3D8", "elset": "E"} and b.data_lines == ["1,1,2,3,4,5,6,7,8"]
    b = tokenize_string("*CONTACT CLEARANCE  ASSIGNMENT  EXTRA, NAME=X\n")[0]
    assert b.keyword == "CONTACT CLEARANCE ASSIGNMENT EXTRA"


def test_detect_dim_sees_included_elements(tmp_path):
    _write(tmp_path, "els.inp", "*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n *ELEMENT, TYPE=CPE4, ELSET=E\n1,1,2,3,4\n")
    main = _write(tmp_path, "main.inp", "*HEADING\nx\n*INCLUDE, INPUT=els.inp\n")
    assert detect_dim(main) == 2


# ----------------------------------------------------------------- reader / model / solver
DECK = CUBE_NODES + """*ELEMENT, TYPE=C3D8, ELSET=E
1,1,2,3,4,5,6,7,8
*MATERIAL, NAME=M
*ELASTIC
1000., 0.3
*SOLID SECTION, ELSET=E, MATERIAL=M
*STEP
*STATIC
0.1, 1.0
*BOUNDARY
NOPE, 1, 3
*END STEP
"""


def test_handler_errors_and_unsupported_keywords_cite_file_line(tmp_path):
    # *FOOBAR is not an Abaqus keyword (error); *RADIATION VIEW FACTOR is one
    # this reader does not handle (gap).
    main = _write(tmp_path, "deck.inp", DECK.replace(
        "*STEP\n", "*FOOBAR\n1\n*RADIATION VIEW FACTOR\n*SOLID SECTION, ELSET=MISSING, MATERIAL=M\n*STEP\n"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, _ = _parse_source(main, dim=3, strict=False)
    gaps = reader.physics_gaps()
    unknown = [d for d in reader.diagnostics if d.code == "unknown-keyword"]
    assert len(unknown) == 1 and unknown[0].level == "gap" and str(unknown[0].loc) == "deck.inp:16"
    assert any("*RADIATION VIEW FACTOR (unsupported, deck.inp:18)" == g for g in gaps), gaps
    assert any("SOLID SECTION at deck.inp:19" in g for g in gaps), gaps


def test_model_objects_carry_their_deck_location_into_solver_errors(tmp_path):
    main = _write(tmp_path, "deck.inp", DECK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model = _parse_source(main, dim=3)
        solver, sysm = model.create_solver(dim=3)
    st = model.steps[[s for s in model.steps if s != "Initial"][-1]]
    st.parent_model = model
    bc = next(iter(st.get_active_bcs().values()))
    assert bc.source_location == "deck.inp:19"
    assert model.sections[next(iter(model.sections))].source_location == "deck.inp:15"
    solver._u_step_start = solver.u.copy()
    with pytest.raises(RegionError, match=r"deck\.inp:19"):
        solver._update_step_bcs(st, sysm, 1.0, 1.0)


def test_surface_behavior_reads_its_law_and_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model = _parse_source("*SURFACE INTERACTION, NAME=I\n*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=EXPONENTIAL\n"
                                 "0.01, 50.\n*SURFACE INTERACTION, NAME=J\n"
                                 "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TABULAR\n0., 0.\n10., 0.01\n", dim=3)
    i, j = model.contact_properties["I"], model.contact_properties["J"]
    assert (i.normal_behavior, i.exponential_c0, i.exponential_p0) == ("SOFT_EXPONENTIAL", 0.01, 50.0)
    assert j.normal_behavior == "SOFT_TABULAR" and j.tabular_pressure == [0.0, 10.0]
