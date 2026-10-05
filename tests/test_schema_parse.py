"""Schema-driven PARSE (design M2 / spec A2): dispsolver/io/schema.py on top of
the Abaqus keyword index (dispsolver/io/keyword_index.py)."""
import warnings
from typing import Literal, Optional

import pytest

import dispsolver.io.keywords  # noqa: F401  (registers the keyword classes)
from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.io.keyword_index import AmbiguousName, abaqus_index, resolve_name
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY
from dispsolver.io.schema import (InputError, Keyword, Param, Schema, SCHEMA_KEYWORDS, keyword,
                                  parse_text)
from dispsolver.io.keywords.p0_2 import AmplitudeRecord
from dispsolver.io.keywords.steps import BcResetRecord, BoundaryRecord

CUBE = """*NODE, NSET=ALL
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE=C3D8, ELSET=E
1,1,2,3,4,5,6,7,8
*MATERIAL, NAME=M
*ELASTIC
1000., 0.3
*SOLID SECTION, ELSET=E, MATERIAL=M
"""


def _read(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parse_source(text, dim=3, strict=False)


def _codes(diags, level=None):
    return [c for lv, c, _ in diags if level is None or lv == level]


# ── the index and Abaqus's abbreviation rule ────────────────────────────────

def test_index_resolves_abbreviations_exact_first():
    ix = abaqus_index()
    assert len(ix) > 500
    assert ix.get("ELASTIC").resolve_param("DEP") == "DEPENDENCIES"
    assert ix.get("BOUNDARY").resolve_param("AMP") == "AMPLITUDE"
    assert ix.get("BOUNDARY").resolve_param("op") == "OP"            # exact wins over OP...-prefixes
    assert ix.resolve_keyword("BOUN") == "BOUNDARY"
    assert ix.resolve_keyword("ENDSTEP") == "END STEP"                # blanks are ignored
    with pytest.raises(AmbiguousName):
        ix.get("BOUNDARY").resolve_param("S")                         # SCALE, STEP, SUBMODEL
    assert resolve_name("TYPE", ["TYPE", "TYPED"]) == "TYPE"


# ── parameter policy ────────────────────────────────────────────────────────

def test_param_policy_codes():
    _, d = parse_text("*BOUNDARY, S=1\nN, 1\n")
    assert _codes(d) == ["ambiguous-param"]
    recs, d = parse_text("*BOUNDARY, FIXED\nN, 1\n")
    assert recs == [] and _codes(d, "gap") == ["unsupported-param"]   # Abaqus defines it; not declared
    recs, d = parse_text("*BOUNDARY, OPX=NEW\nN, 1\n")
    assert recs == [] and _codes(d, "error") == ["unknown-param"]     # Abaqus does not define it
    recs, d = parse_text("*AMPLITUDE\n0, 0\n")
    assert recs == [] and _codes(d, "error") == ["missing-param"]
    recs, d = parse_text("*BOUNDARY, TYPE=VELOCITY\nN, 1, 1, 1.\n")
    assert recs == [] and _codes(d, "gap") == ["unsupported-value"]
    recs, d = parse_text("*BOUNDARY, OP=SOMETIMES\nN, 1\n")
    assert recs == [] and _codes(d, "error") == ["bad-value"]
    # AMP and AMPLITUDE are one parameter; Abaqus keeps one setting
    recs, d = parse_text("*BOUNDARY, AMP=A, AMPLITUDE=B\nN, 1\n")
    assert _codes(d, "warning") == ["duplicate-param"] and recs[0].amplitude == "B"


def test_unsupported_param_refuses_the_block():
    # SCALEX ignored would be a different curve, not a smaller one
    recs, d = parse_text("*AMPLITUDE, NAME=A, SCALEX=2.\n0, 0, 1, 1\n")
    assert recs == [] and _codes(d) == ["unsupported-param"]


def test_registration_refuses_names_abaqus_does_not_define():
    with pytest.raises(TypeError, match="not in the Abaqus keyword index"):
        @keyword("BOUNDARY")
        class Bogus(Keyword):
            class Params(Schema):
                bogus: Optional[str] = None
    assert GLOBAL_KEYWORD_REGISTRY.get_handler("BOUNDARY").schema.__name__ == "Boundary"


def test_supported_must_be_a_subset_of_the_choices():
    with pytest.raises(TypeError, match="supported"):
        class P(Schema):
            op: Literal["MOD", "NEW"] = Param("MOD", supported=("REPLACE",))


# ── *BOUNDARY (reference migration) ─────────────────────────────────────────

def test_boundary_records():
    recs, d = parse_text("*BOUNDARY, OP=NEW, AMP=A1\nN1, ENCASTRE\nN2, 1, 2, 0.5\nN3, 3\n"
                         "N4, XASYMM\nN5, 2, , -1.\n", step="S1")
    assert d == []
    assert recs[0] == BcResetRecord("S1")
    b = recs[1:]
    assert [r.dofs for r in b] == [(1, 2, 3, 4, 5, 6), (1, 2), (3,), (2, 3), (2,)]
    assert [r.value for r in b] == [0.0, 0.5, 0.0, 0.0, -1.0]
    assert {r.amplitude for r in b} == {"A1"} and {r.op for r in b} == {"NEW"}
    assert [str(r.loc) for r in b] == ["<string>:2", "<string>:3", "<string>:4", "<string>:5", "<string>:6"]


def test_boundary_bad_lines_are_errors():
    with pytest.raises(InputError, match="neither a degree of freedom"):
        parse_text("*BOUNDARY\nN, X, 1\n")
    with pytest.raises(InputError, match="takes no further values"):
        parse_text("*BOUNDARY\nN, PINNED, 1\n")
    with pytest.raises(InputError, match="at most 4"):
        parse_text("*BOUNDARY\nN, 1, 1, 0., 9\n")


def test_boundary_lands_in_model_and_abbreviated_keyword_resolves():
    reader, model = _read(CUBE + "*BOUN\nALL, PINNED\n*STEP\n*STATIC\n1., 1.\n*BOUNDARY\n1, 3, 3, 0.1\n*END STEP\n")
    assert not [d for d in reader.diagnostics if d.level in ("error", "gap")]
    init = model.initial_step.boundary_conditions["BC_ALL_DOF1_3"]
    assert init is not None
    step = [s for s in model.steps if s != "Initial"][-1]
    assert "BC_1_DOF3_3" in model.steps[step].boundary_conditions


# ── *AMPLITUDE (reference migration) ────────────────────────────────────────

def test_amplitude_records():
    recs, d = parse_text("*AMPLITUDE, NAME=A1, DEF=SMOOTHSTEP\n0, 0, 1, 1\n2, 1\n")
    assert d == [] and recs == [AmplitudeRecord("A1", "SMOOTH STEP", ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0)))]
    recs, d = parse_text("*AMPLITUDE, NAME=A2, DEFINITION=PERIODIC\n1., 0., 0., 1.\n")
    assert recs == [] and _codes(d, "gap") == ["unsupported-value"]
    with pytest.raises(InputError, match="multiple of 2"):
        parse_text("*AMPLITUDE, NAME=A3\n0, 0, 1\n")


def test_amplitude_lands_in_model():
    _, model = _read(CUBE + "*AMPLITUDE, NAME=RAMP2\n0., 0., 1., 2.\n")
    assert "RAMP2" in model.amplitudes


# ── legacy handlers see expanded abbreviations ──────────────────────────────

def test_legacy_handler_receives_full_parameter_name():
    reader, model = _read(CUBE + "*AMPLITUDE, NAME=R\n0., 0., 1., 1.\n"
                                 "*STEP\n*STATIC\n1., 1.\n*CLOAD, AMPL=R\n7, 1, 1.\n*END STEP\n")
    step = [s for s in model.steps if s != "Initial"][-1]
    load = model.steps[step].loads["CLOAD_7_DOF1"]
    assert getattr(load.entity, "amplitude", None) == "R"
    assert not any(d.code in ("unknown-param", "unsupported-param") for d in reader.diagnostics)


def test_schema_keywords_are_registered_with_their_schema():
    for name in ("BOUNDARY", "AMPLITUDE"):
        assert SCHEMA_KEYWORDS[name] is GLOBAL_KEYWORD_REGISTRY.get_handler(name).schema
