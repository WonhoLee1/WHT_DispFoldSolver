"""BIND pass (spec A3 / plan_a3_bind_20261005.md P1): dispsolver/model/bind.py."""
import warnings

import pytest

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.io.diagnostics import Diagnostics
from dispsolver.model.bind import AmbiguousRef, SymbolTable, UnresolvedRef, bind_model
from dispsolver.model.model import Model

NODES = """*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE=C3D8
1,1,2,3,4,5,6,7,8
*NSET, NSET=Left
1,4,5,8
*ELSET, ELSET=E
1
*MATERIAL, NAME=M
*ELASTIC
1000., 0.3
*SOLID SECTION, ELSET=E, MATERIAL=M
*SURFACE, NAME=Top, TYPE=ELEMENT
E, S2
"""

STEP = """*AMPLITUDE, NAME=Ramp
0,0,1,1
*STEP, NAME=S1
*STATIC
1., 1.
{body}
*END STEP
"""


def _read(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parse_source(text, dim=3, strict=False)


def _refs(reader, level=None):
    return [d for d in reader.diagnostics
            if d.code in ("unresolved-ref", "ambiguous-ref") and (level is None or d.level == level)]


# ── a model with an explicit assembly: two instances of one part ────────────

def _assembly_model():
    from dispsolver.model.model import Model
    m = Model("T", dim=3)
    p = m.Part("P")
    for i, xyz in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1),
                             (0, 1, 1)], 1):
        p.nodes[i] = xyz
    p.elements[1] = ("C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    p.create_node_set("Shared", [1, 2])
    p.create_node_set("OnlyA", [3])
    p.create_element_set("EL", [1])
    p.create_surface("Sf", [(1, 1)])
    m.root_assembly.Instance("A", p)
    m.root_assembly.Instance("B", p)
    m.root_assembly.create_node_set("AsmSet", [1]) if hasattr(m.root_assembly, "create_node_set") else None
    m.Material("M")
    m.SolidSection("S", "M")
    m.Amplitude("Amp") if hasattr(m, "Amplitude") else None
    m.Step("Step-1")
    return m


@pytest.fixture
def sym():
    return SymbolTable(_assembly_model())


def test_simple_kinds_exact_and_case_folded(sym):
    assert sym.resolve("material", "M") is sym.resolve("material", "m")
    assert sym.resolve("section", "s").name == "S"
    assert sym.resolve("part", "p").name == "P"
    assert sym.resolve("instance", "a").name == "A"
    assert sym.resolve("step", "STEP-1").name == "Step-1"
    for kind, name in (("material", "X"), ("section", "X"), ("part", "X"), ("instance", "X"), ("step", "X"),
                       ("amplitude", "X"), ("orientation", "X"), ("contact_property", "X")):
        with pytest.raises(UnresolvedRef):
            sym.resolve(kind, name)


def test_qualified_and_assembly_prefixed_names(sym):
    asm = sym.assembly_name
    a = sym.resolve("nset", "A.Shared")
    assert a is sym.resolve("nset", "a.shared")
    assert a is sym.resolve("nset", f"{asm}.A.Shared")
    assert sym.resolve("elset", "B.EL").name == "EL"
    assert sym.resolve("surface", "A.sf").name == "Sf"
    assert sym.resolve("node_region", "A.OnlyA") is sym.resolve("nset", "A.OnlyA")


def test_bare_part_level_name_ambiguous_across_instances(sym):
    for kind, name in (("nset", "Shared"), ("elset", "el"), ("surface", "SF"), ("node_region", "shared"),
                       ("elem_region", "EL")):
        with pytest.raises(AmbiguousRef) as e:
            sym.resolve(kind, name)
        assert e.value.candidates == [f"A.{name}", f"B.{name}"]


def test_unknown_names_raise_unresolved(sym):
    for kind in ("nset", "elset", "surface", "node_region", "elem_region"):
        with pytest.raises(UnresolvedRef):
            sym.resolve(kind, "Nope")


def test_node_region_forms_in_multi_instance_model(sym):
    assert sym.resolve("node_region", "ALL") == "ALL"
    assert sym.resolve("elem_region", "all") == "ALL"
    assert sym.resolve("node_region", "A.3") == ("node", "A", 3)
    assert sym.resolve("elem_region", "B.1") == ("element", "B", 1)
    with pytest.raises(UnresolvedRef, match="INSTANCE"):
        sym.resolve("node_region", "3")                         # bare number, two instances
    with pytest.raises(UnresolvedRef):
        sym.resolve("node_region", "A.99")


def test_step_overlay_node_set():
    m = _assembly_model()
    m.root_assembly.instances.pop("B")
    s = SymbolTable(m)
    step = m.steps["Step-1"]
    step.step_nsets["StepSet"] = [1, 2]
    with pytest.raises(UnresolvedRef):
        s.resolve("node_region", "stepset")                      # no step given
    assert s.resolve("node_region", "stepset", step) == [1, 2]
    assert s.resolve("node_region", "2") == ("node", "A", 2)     # bare number, single instance
    # with one instance a bare part-level name is unique
    assert s.resolve("nset", "Shared").name == "Shared"


def test_unknown_kind_is_a_programming_error(sym):
    with pytest.raises(ValueError):
        sym.resolve("bogus", "x")


# ── flat deck: bare part-level names resolve on the default instance ────────

def test_flat_deck_binds_default_instance_and_resolves_bare_names():
    reader, model = _read(NODES + STEP.format(body="*BOUNDARY\nLEFT, 1, 3\n*CLOAD\nleft, 1, 5.\n"))
    assert reader.symbols is not None
    assert reader.symbols.resolve("nset", "LEFT") is model.root_assembly.instances["INST_DEFAULT"].part.node_sets["Left"]
    assert reader.symbols.resolve("node_region", "INST_DEFAULT.1") == ("node", "INST_DEFAULT", 1)
    assert reader.symbols.resolve("surface", "top").name == "Top"
    assert not _refs(reader)


def test_bind_model_returns_table_and_does_not_mutate():
    reader, model = _read(NODES)
    before = (sorted(model.steps), sorted(model.materials), len(model.constraints))
    d = Diagnostics(echo=False)
    t = bind_model(model, d)
    assert isinstance(t, SymbolTable)
    assert list(d) == []
    assert before == (sorted(model.steps), sorted(model.materials), len(model.constraints))


# ── one deck snippet per reference field ────────────────────────────────────

def test_bc_region_typo():
    reader, _ = _read(NODES + STEP.format(body="*BOUNDARY\nLeft_TYPO, 1, 3\n"))
    (d,) = _refs(reader)
    assert (d.level, d.code) == ("error", "unresolved-ref")
    assert "Left_TYPO" in d.message and "boundary condition" in d.message
    assert d.loc is not None and d.loc.line > 0


def test_cload_region_typo_and_valid_amplitude():
    reader, _ = _read(NODES + STEP.format(body="*CLOAD, AMPLITUDE=Ramp\nNopeSet, 1, 5.\n"))
    msgs = [d.message for d in _refs(reader)]
    assert len(msgs) == 1 and "NopeSet" in msgs[0] and "Ramp" not in msgs[0]


def test_amplitude_typo_on_bc_and_load():
    reader, _ = _read(NODES + STEP.format(
        body="*BOUNDARY, AMPLITUDE=Rmap\nLeft, 1, 3\n*CLOAD, AMPLITUDE=Rmp2\nLeft, 1, 5.\n"))
    msgs = " | ".join(d.message for d in _refs(reader, "error"))
    assert "Rmap" in msgs and "Rmp2" in msgs


def test_dload_pressure_region_may_be_surface_or_elset_and_typo_is_error():
    ok, _ = _read(NODES + STEP.format(body="*DSLOAD\nTop, P, 1.\n*DLOAD\nE, P2, 1.\n"))
    assert not _refs(ok)
    bad, _ = _read(NODES + STEP.format(body="*DSLOAD\nTopp, P, 1.\n"))
    assert any("Topp" in d.message for d in _refs(bad))


def test_gravity_blank_region_means_whole_model_but_typo_is_error():
    ok, _ = _read(NODES + STEP.format(body="*DLOAD\n, GRAV, 9.8, 0, 0, -1\n"))
    assert not _refs(ok)
    bad, _ = _read(NODES + STEP.format(body="*DLOAD\nEE, GRAV, 9.8, 0, 0, -1\n"))
    assert any("'EE'" in d.message for d in _refs(bad))


def test_section_material_typo():
    reader, _ = _read(NODES.replace("MATERIAL=M", "MATERIAL=MM"))
    (d,) = _refs(reader)
    assert d.level == "error" and "MM" in d.message and "material" in d.message


def test_section_orientation_typo():
    reader, _ = _read(NODES.replace("MATERIAL=M", "MATERIAL=M, ORIENTATION=Ori"))
    assert any("Ori" in d.message and d.level == "error" for d in _refs(reader))


def test_section_assignment_region_and_section_typos():
    # the *SOLID SECTION handler already refuses an unknown ELSET; the binder covers models
    # built through the API, whose assignments are never checked at creation
    from dispsolver.model.part import SectionAssignment
    m = _assembly_model()
    m.parts["P"].section_assignments.append(SectionAssignment(region="EX", section_name="S"))
    m.parts["P"].section_assignments.append(SectionAssignment(region="EL", section_name="SS"))
    d = Diagnostics(echo=False)
    bind_model(m, d)
    msgs = [x.message for x in d if x.code == "unresolved-ref"]
    assert any("'EX'" in x for x in msgs) and any("'SS'" in x for x in msgs)
    assert len(msgs) == 2


def test_contact_pair_interaction_typo():
    deck = NODES + """*SURFACE, NAME=Bot, TYPE=ELEMENT
E, S1
*SURFACE INTERACTION, NAME=Fric
*CONTACT PAIR, INTERACTION=Frik
Top, Bot
"""
    reader, _ = _read(deck)
    msgs = [d.message for d in _refs(reader, "error")]
    assert any("Frik" in m and "interaction property" in m for m in msgs)
    ok, _ = _read(deck.replace("Frik", "FRIC"))
    assert not _refs(ok)


def test_contact_pair_surface_typo():
    deck = NODES + """*SURFACE INTERACTION, NAME=Fric
*CONTACT PAIR, INTERACTION=Fric
Top, Nowhere
"""
    reader, _ = _read(deck)
    assert any("Nowhere" in d.message for d in _refs(reader, "error"))


def test_tie_surface_typo():
    deck = NODES + "*TIE, NAME=T1\nTop, Bottm\n"
    reader, _ = _read(deck)
    (d,) = _refs(reader, "error")
    assert "Bottm" in d.message and d.loc is not None


def test_transform_nset_typo():
    deck = NODES + "*TRANSFORM, NSET=Lft, TYPE=R\n1,0,0, 0,1,0\n"
    reader, _ = _read(deck)
    assert any("Lft" in d.message and d.level == "error" for d in _refs(reader))


def test_output_request_nset_typo_is_a_warning_not_an_error():
    body = "*NODE OUTPUT, NSET=Lft\nU\n"
    reader, _ = _read(NODES + STEP.format(body=body))
    refs = _refs(reader)
    assert refs and all(d.level == "warning" for d in refs)
    assert any("Lft" in d.message for d in refs)


def test_predefined_field_region_typo():
    deck = NODES + "*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNoSuchSet, 20.\n"
    reader, _ = _read(deck)
    assert any("NoSuchSet" in d.message and d.level == "error" for d in _refs(reader))


def test_coupling_reference_and_surface_typos():
    deck = NODES + "*COUPLING, CONSTRAINT NAME=C1, REF NODE=999, SURFACE=Tpo\n*KINEMATIC\n"
    reader, _ = _read(deck)
    msgs = " | ".join(d.message for d in _refs(reader, "error"))
    assert "Tpo" in msgs and "999" in msgs


def test_internal_failure_is_not_swallowed():
    class Boom:
        steps = None
    with pytest.raises(Exception):
        bind_model(Boom(), Diagnostics(echo=False))
