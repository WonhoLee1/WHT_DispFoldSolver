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
    assert recs == [] and _codes(d, "gap") == ["unknown-param"]       # not in the Keywords Guide
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
    assert _codes(d) == ["set-order-sorted"] and recs[0].tokens == ("5", "3", "1")


def test_nset_of_elset_and_set_redefinition_adds():
    _, model = _read(CUBE + "*NSET, NSET=EN, ELSET=E\n*NSET, NSET=A\n1, 2\n*NSET, NSET=A\n3\n")
    part = next(iter(model.parts.values()))
    assert sorted(part.node_sets["EN"].node_ids) == [1, 2, 3, 4, 5, 6, 7, 8]   # used to be empty
    assert sorted(part.node_sets["A"].node_ids) == [1, 2, 3]                   # used to be [3]


def test_surface_records_and_refusals():
    from dispsolver.io.keywords.mesh import SegmentSurfaceRecord, SurfaceRecord
    recs, d = parse_text("*SURFACE, NAME=S\nE, S2\n7, S6\nOTHER\n", context="PART")
    assert recs == [SurfaceRecord("S", "ELEMENT", (("E", 2), ("7", 6), ("OTHER", None)))]
    recs, _ = parse_text("*SURFACE, NAME=N, TYPE=NODE\nNS, 1.\n", context="PART")
    assert recs == [SurfaceRecord("N", "NODE", (("NS", None),))]
    recs, _ = parse_text("*SURFACE, NAME=R, TYPE=SEGMENTS\nSTART, 0., 0.\nLINE, 1., 0.\n", context="PART")
    assert recs == [SegmentSurfaceRecord("R", (0.0, 0.0), (1.0, 0.0))]
    for text in ("*SURFACE, NAME=S\nE, SPOS\n",
                 "*SURFACE, NAME=N, TYPE=NODE\nNS, 2.5\n",                  # area weights contact
                 "*SURFACE, NAME=R, TYPE=SEGMENTS\nSTART, 0., 0.\nLINE, 1., 0.\nLINE, 1., 1.\n"):
        with pytest.raises(NotImplementedError):
            parse_text(text, context="PART")
    with pytest.raises(InputError, match="no data lines"):
        parse_text("*SURFACE, NAME=S\n", context="ASSEMBLY")
    recs, d = parse_text("*SURFACE, NAME=S, COMBINE=INTERSECTION\nA\nB\n", context="ASSEMBLY")
    assert recs == [] and _codes(d, "gap") == ["unsupported-value"]


def test_surface_free_faces_is_a_gap_in_part():
    reader, _ = _read(CUBE + "*SURFACE, NAME=S\nE\n")                        # used to be S1
    assert [d.code for d in reader.diagnostics if d.level == "gap"] == ["unsupported-feature"]


ASSEMBLY_SURFACES = """*PART, NAME=P
*NODE
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
*SURFACE, NAME=PS
E, S2
*END PART
*ASSEMBLY, NAME=A
*INSTANCE, NAME=P-1, PART=P
*END INSTANCE
*ELSET, ELSET=_S_S2, INSTANCE=P-1, INTERNAL
1
*SURFACE, TYPE=ELEMENT, NAME=S1
_S_S2, S2
*SURFACE, NAME=S2
P-1.PS
*NSET, NSET=NS, INSTANCE=P-1
1, 2
*SURFACE, TYPE=NODE, NAME=S4
NS, 1.
*SURFACE, NAME=S3
P-1.E
*END ASSEMBLY
"""


def test_assembly_surfaces_resolve_names_against_the_model():
    # Abaqus/CAE writes "_Surf_S2, S2" with an assembly element set: that used
    # to be a bad-input error plus a phantom PART_DEFAULT; a face-less element
    # set used to become "all nodes of those elements" without a word
    reader, model = _read(ASSEMBLY_SURFACES)
    a = model.root_assembly
    top = sorted(("P-1", n) for n in (5, 6, 7, 8))
    assert a.surfaces["S1"].inst_faces == [("P-1", 1, 2)] and sorted(a.surfaces["S1"].inst_nodes) == top
    assert a.surfaces["S2"].inst_faces == [("P-1", 1, 2)] and sorted(a.surfaces["S2"].inst_nodes) == top
    assert sorted(a.surfaces["S4"].inst_nodes) == [("P-1", 1), ("P-1", 2)]
    assert "S3" not in a.surfaces
    probs = [(d.level, d.code) for d in reader.diagnostics if d.level in ("error", "gap")]
    assert probs == [("gap", "unsupported-feature")]                          # S3: free faces
    assert list(model.parts) == ["P"]


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
    # a bad transform is reported and dropped; the instance itself is kept
    recs, d = parse_text("*INSTANCE, NAME=I1, PART=P\n0., 0., 0.\n0., 0., 0., 1.\n", context="ASSEMBLY")
    assert recs[-1] == InstanceRecord("I1", "P", (0.0, 0.0, 0.0), None)
    assert _codes(d) == ["bad-instance-transform"]
    # materials are model data: outside the *PART
    mesh, rest = CUBE.split("*MATERIAL")
    material, section = ("*MATERIAL" + rest).split("*SOLID SECTION")
    reader, model = _read("*PART, NAME=P\n" + mesh + "*SOLID SECTION" + section + "*END PART\n" + material +
                          "*ASSEMBLY, NAME=A\n*INSTANCE, NAME=I1, PART=P\n5., 0., 0.\n*END INSTANCE\n"
                          "*INSTANCE, NAME=I2, PART=TYPO\n*END INSTANCE\n*END ASSEMBLY\n")
    assert model.root_assembly.name == "A" and "I1" in model.root_assembly.instances
    assert [d.level for d in reader.diagnostics if d.code == "undefined-part"] == ["error"]


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


# ── reviewer findings (A2 review) ───────────────────────────────────────────

def test_refused_instance_does_not_abort_the_parse():
    # used to raise "Keyword *END INSTANCE illegal in context ASSEMBLY" and lose
    # every diagnostic
    reader, model = _read("*PART, NAME=P\n*END PART\n*ASSEMBLY, NAME=A\n*INSTANCE, PART=P\n"
                          "*NODE\n1, 0., 0., 0.\n*END INSTANCE\n*INSTANCE, NAME=I2, PART=P\n"
                          "0., 0., 0.\n0., 0., 0., 0., 0., 1.\n*END INSTANCE\n*END ASSEMBLY\n")
    codes = [d.code for d in reader.diagnostics if d.level == "error"]
    assert "missing-param" in codes and "refused-scope-skipped" in codes
    assert "bad-instance-transform" in codes and "I2" in model.root_assembly.instances


def test_nlgeom_stays_on_once_switched_on():
    reader, model = _read(CUBE + "*STEP, NLGEOM=YES\n*STATIC\n1., 1.\n*END STEP\n"
                                 "*STEP, NLGEOM=NO\n*STATIC\n1., 1.\n*END STEP\n")
    steps = [model.steps[s] for s in model.steps if s != "Initial"]
    assert [s.nlgeom for s in steps] == [True, True]
    assert any(d.code == "nlgeom-stays-on" for d in reader.diagnostics)


def test_composite_layer_angle_is_a_gap():
    with pytest.raises(NotImplementedError, match="angle"):
        parse_text("*SOLID SECTION, ELSET=E, COMPOSITE\n1., 3, M, 45.\n", context="PART")


def test_yasymm_splits_into_contiguous_dof_runs():
    recs, _ = parse_text("*BOUNDARY\nN, YASYMM\n")
    assert [(r.dofs, r.label) for r in recs] == [((1,), "DOF1_1"), ((3,), "DOF3_3")]


def test_unknown_parameter_is_a_gap_on_both_paths():
    # undocumented is not proof of wrong: Abaqus's own verification decks use
    # *VISCOELASTIC, FORMULATION=, which the Keywords Guide does not list
    reader, _ = _read(CUBE + "*MATERIAL, NAME=M2\n*DENSITY, BOGUS=1\n7.8e-9\n"
                             "*STEP\n*STATIC\n1., 1.\n*CLOAD, BOGUS=1\n7, 1, 1.\n*END STEP\n")
    assert [d.level for d in reader.diagnostics if d.code == "unknown-param"] == ["gap", "gap"]


def test_notes_survive_a_refused_block():
    with pytest.raises(InputError):
        parse_text("*ELSET, ELSET=E, UNSORTED, GENERATE\n1, 10, 2\n", context="PART")


# ── reviewer findings, second pass ──────────────────────────────────────────

def test_assembly_set_without_instance_is_not_an_empty_surface():
    # members (None, label) used to be filtered out: a surface with zero faces
    # and no diagnostic, so a *DSLOAD on it loaded nothing
    deck = ASSEMBLY_SURFACES.replace("*SURFACE, NAME=S3\nP-1.E\n", "*ELSET, ELSET=X\n1\n"
                                     "*SURFACE, TYPE=ELEMENT, NAME=S5\nX, S2\n")
    reader, model = _read(deck)
    assert "S5" not in model.root_assembly.surfaces
    assert [(d.level, d.code) for d in reader.diagnostics if d.level in ("error", "gap")] == \
        [("gap", "unsupported-feature")]


def test_unclosed_refused_scope_is_reported():
    reader, model = _read("*PART, NAME=P\n*END PART\n*ASSEMBLY, NAME=A\n*INSTANCE, PART=P\n"
                          "*STEP\n*STATIC\n1., 1.\n*END STEP\n")
    msgs = [d.message for d in reader.diagnostics if d.code == "refused-scope-skipped"]
    assert len(msgs) == 1 and "never came" in msgs[0]


def test_faceless_member_surface_is_a_gap():
    reader, _ = _read(CUBE + "*SURFACE, NAME=R, TYPE=SEGMENTS\nSTART, 0., 0.\nLINE, 1., 0.\n"
                             "*SURFACE, NAME=U\nR\n")
    assert [d.code for d in reader.diagnostics if d.level == "gap"] == ["unsupported-feature"]

