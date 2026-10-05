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
    # values past the documented items: reported and ignored (official
    # verification decks carry such lines and Abaqus runs them)
    recs, d = parse_text("*BOUNDARY\nN, 1, 1, 0.5, 9\n")
    assert _codes(d) == ["extra-data-items"] and recs[0].value == 0.5


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


# ── loads (US-005) ──────────────────────────────────────────────────────────

def test_cload_records():
    from dispsolver.io.keywords.loads import ConcentratedLoadRecord, LoadResetRecord
    recs, d = parse_text("*CLOAD, OP=NEW, AMP=A\nN1, 2, -5.\n", step="S")
    assert d == [] and recs[0] == LoadResetRecord("S", "CLOAD")
    assert recs[1] == ConcentratedLoadRecord("S", "N1", 2, -5.0, "A", "NEW")
    with pytest.raises(NotImplementedError, match="degree of freedom 4"):
        parse_text("*CLOAD\nN1, 4, 1.\n")


def test_dload_types_and_blank_gravity_region():
    from dispsolver.io.keywords.loads import BodyForceRecord, GravityRecord, PressureRecord
    recs, d = parse_text("*DLOAD\n, GRAV, 9.81, 0., , -1.\nE, P2, 1.5\nE, BY, 2.\n", step="S")
    assert d == []
    assert recs[0] == GravityRecord("S", "", (0.0, 0.0, -9.81), None)
    assert recs[1] == PressureRecord("S", "DLOAD", "E", "P2", "P2", 1.5, None)
    assert recs[2] == BodyForceRecord("S", "E", "BY", (0.0, 2.0, 0.0), None)
    with pytest.raises(NotImplementedError, match="P2NU"):
        parse_text("*DLOAD\nE, P2NU\n")
    with pytest.raises(InputError, match="direction vector"):
        parse_text("*DLOAD\nE, GRAV, 9.81, 0., -1.\n")


def test_dsload_pressure_only():
    recs, d = parse_text("*DSLOAD\nS1, P, 3.\n", step="S")
    assert d == [] and recs[0].region == "S1" and recs[0].face is None and recs[0].magnitude == 3.0
    # HP used to be landed as a uniform pressure; PNU (no magnitude) was skipped silently
    for line in ("S1, HP, 3., 0., 1.", "S1, PNU", "S1, TRSHR, 1., 1., 0., 0."):
        with pytest.raises(NotImplementedError):
            parse_text("*DSLOAD\n" + line + "\n")


def test_op_new_removes_loads_of_the_same_kind_only():
    _, model = _read(CUBE + "*SURFACE, NAME=TOP\nE, S2\n*STEP\n*STATIC\n1., 1.\n*CLOAD\n7, 1, 1.\n"
                            "*DSLOAD\nTOP, P, 1.\n*END STEP\n*STEP\n*STATIC\n1., 1.\n*DSLOAD, OP=NEW\n"
                            "TOP, P, 2.\n*END STEP\n")
    from dispsolver.model.step import EntityStatus
    s2 = model.steps[[s for s in model.steps if s != "Initial"][-1]]
    # *DSLOAD, OP=NEW in step 2 must not touch the CLOAD (legacy deactivated every load)
    assert s2.loads["CLOAD_7_DOF1"].status != EntityStatus.DEACTIVATED
    assert s2.loads["DSLOAD_TOP_P"].status != EntityStatus.DEACTIVATED   # re-applied after the reset


# ── step keywords (US-005) ──────────────────────────────────────────────────

def test_step_static_dynamic_records():
    from dispsolver.io.keywords.steps import ProcedureRecord, StepRecord
    recs, d = parse_text("*STEP, NAME=S1, NLGEOM, INC=50\nLoad step one\n", context="GLOBAL")
    assert recs == [StepRecord("S1", "Load step one", True, None)]
    assert [c for _, c, _ in d] == ["control-param-ignored"]           # INC: solution control
    recs, d = parse_text("*STATIC\n0.1, 1., 1e-5, 0.2\n")
    assert recs == [ProcedureRecord("Step-1", "STATIC", 0.1, 1.0, 1e-5, 0.2)]
    recs, d = parse_text("*DYNAMIC, HAFTOL=0.1\n0.01, 2.\n")
    assert recs[0].procedure == "DYNAMIC" and recs[0].time_period == 2.0
    recs, d = parse_text("*STATIC, RIKS\n0.1, 1.\n")
    assert recs == [] and _codes(d, "gap") == ["unsupported-param"]
    with pytest.raises(InputError, match="no data lines"):
        parse_text("*END STEP\n1\n")


def test_unsupported_step_parameter_does_not_cascade():
    # a refused *STEP used to push every following keyword out of context
    reader, model = _read(CUBE + "*STEP, PERTURBATION\n*STATIC\n1., 1.\n*END STEP\n")
    codes = [d.code for d in reader.diagnostics if d.level in ("error", "gap")]
    assert codes == ["unsupported-param"] and len(model.steps) == 2


# ── sets and surfaces (US-005) ──────────────────────────────────────────────

def test_nset_elset_records():
    from dispsolver.io.keywords.mesh import SetRecord
    recs, d = parse_text("*NSET, NSET=A\n1, 2, 3\nB\n", context="PART")
    assert recs == [SetRecord("node", "A", None, tokens=("1", "2", "3", "B"))]
    recs, _ = parse_text("*ELSET, ELSET=E, GEN\n1, 9, 2\n", context="PART")
    assert recs[0].ranges == ((1, 9, 2),)
    with pytest.raises(InputError, match="whole number"):
        parse_text("*NSET, NSET=A, GENERATE\n1, 10, 2\n", context="PART")
    recs, d = parse_text("*ELSET, ELSET=E, UNSORTED\n5, 3, 1\n", context="PART")
    assert _codes(d) == [] and recs[0].code == "set-order-sorted"


def test_nset_of_elset_and_set_redefinition_adds():
    _, model = _read(CUBE + "*NSET, NSET=EN, ELSET=E\n*NSET, NSET=A\n1, 2\n*NSET, NSET=A\n3\n")
    part = next(iter(model.parts.values()))
    assert sorted(part.node_sets["EN"].node_ids) == [1, 2, 3, 4, 5, 6, 7, 8]   # used to be empty
    assert sorted(part.node_sets["A"].node_ids) == [1, 2, 3]                   # used to be [3]


def test_surface_records_and_refusals():
    from dispsolver.io.keywords.mesh import ElementSurfaceRecord, NodeSurfaceRecord, SegmentSurfaceRecord
    recs, d = parse_text("*SURFACE, NAME=S\nE, S2\n7, S6\n", context="PART")
    assert recs == [ElementSurfaceRecord("S", (("E", 2), ("7", 6)))]
    recs, _ = parse_text("*SURFACE, NAME=N, TYPE=NODE\nNS, 1.\n", context="PART")
    assert recs == [NodeSurfaceRecord("N", ("NS",))]
    recs, _ = parse_text("*SURFACE, NAME=R, TYPE=SEGMENTS\nSTART, 0., 0.\nLINE, 1., 0.\n", context="PART")
    assert recs == [SegmentSurfaceRecord("R", (0.0, 0.0), (1.0, 0.0))]
    for text in ("*SURFACE, NAME=S\nE\n",                                     # free faces (was: S1)
                 "*SURFACE, NAME=S\nE, SPOS\n",
                 "*SURFACE, NAME=R, TYPE=SEGMENTS\nSTART, 0., 0.\nLINE, 1., 0.\nLINE, 1., 1.\n"):
        with pytest.raises(NotImplementedError):
            parse_text(text, context="PART")
    recs, d = parse_text("*SURFACE, NAME=S, COMBINE=INTERSECTION\nA\nB\n", context="ASSEMBLY")
    assert recs == [] and _codes(d, "gap") == ["unsupported-value"]


def test_surface_unresolved_member_is_an_error():
    reader, _ = _read(CUBE + "*SURFACE, NAME=S\nNOPE, S1\n")
    assert any(d.level == "error" and "NOPE" in d.message for d in reader.diagnostics)


# ── sections and scope (US-005) ─────────────────────────────────────────────

def test_sections():
    from dispsolver.io.keywords.sections import SectionRecord
    recs, _ = parse_text("*SOLID SECTION, ELSET=E, MATERIAL=M\n0.5\n", context="PART")
    assert recs == [SectionRecord("SOLID", "E", "M", None, 0.5)]
    recs, _ = parse_text("*SOLID SECTION, ELSET=E, COMPOSITE\n.1, 3, M, O1\n.2, 3, M, O1\n", context="PART")
    assert recs == [SectionRecord("SOLID", "E", "M", "O1", 1.0)]
    with pytest.raises(NotImplementedError, match="2 different"):
        parse_text("*SOLID SECTION, ELSET=E, COMPOSITE\n.1, 3, M1\n.1, 3, M2\n", context="PART")
    with pytest.raises(NotImplementedError, match="COMPOSITE"):
        parse_text("*SHELL SECTION, ELSET=E, COMPOSITE\n.1, 3, M, -45.\n", context="PART")
    # a gap on a section does not refuse it (its elements would lose their material)
    recs, d = parse_text("*SOLID SECTION, ELSET=E, MATERIAL=M, CONTROLS=EC\n", context="PART")
    assert len(recs) == 1 and _codes(d, "gap") == ["unsupported-param"]


def test_scope_and_instance():
    from dispsolver.io.keywords.scope import InstanceRecord
    recs, _ = parse_text("*INSTANCE, NAME=I1, PART=P\n1., 2.\n0., 0., 0., 0., 0., 1., 90.\n",
                         context="ASSEMBLY")
    assert recs == [InstanceRecord("I1", "P", (1.0, 2.0, 0.0), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0))]
    with pytest.raises(InputError, match="rotation needs 7"):
        parse_text("*INSTANCE, NAME=I1, PART=P\n0., 0., 0.\n0., 0., 0., 1.\n", context="ASSEMBLY")
    # materials are model data: outside the *PART
    mesh, rest = CUBE.split("*MATERIAL")
    material, section = ("*MATERIAL" + rest).split("*SOLID SECTION")
    reader, model = _read("*PART, NAME=P\n" + mesh + "*SOLID SECTION" + section + "*END PART\n" + material +
                          "*ASSEMBLY, NAME=A\n*INSTANCE, NAME=I1, PART=P\n5., 0., 0.\n*END INSTANCE\n"
                          "*INSTANCE, NAME=I2, PART=TYPO\n*END INSTANCE\n*END ASSEMBLY\n")
    assert model.root_assembly.name == "A" and "I1" in model.root_assembly.instances
    assert any(d.code == "undefined-part" for d in reader.diagnostics)


def test_migrated_keywords_read_no_raw_params():
    import inspect
    import dispsolver.io.keywords.loads as loads
    import dispsolver.io.keywords.sections as sections
    for mod in (loads, sections):
        assert "params.get" not in inspect.getsource(mod), mod.__name__
    for name in ("BOUNDARY", "AMPLITUDE", "CLOAD", "DLOAD", "DSLOAD", "NSET", "ELSET", "SURFACE",
                 "SOLID SECTION", "SHELL SECTION", "PART", "END PART", "ASSEMBLY", "END ASSEMBLY",
                 "INSTANCE", "END INSTANCE", "STEP", "END STEP", "STATIC", "DYNAMIC"):
        cls = GLOBAL_KEYWORD_REGISTRY.get_handler(name).schema
        assert cls is not None, name
        assert "params.get" not in inspect.getsource(cls), name
