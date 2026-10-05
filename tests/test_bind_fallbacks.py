"""A3 / plan_a3_bind_20261005.md P2: references that do not resolve RAISE (or report)
instead of silently building a different model. One test per site, plus the
resolving case unchanged."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.model.model import Model


def _read(text, dim=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parse_source(text, dim=dim, strict=False)[0]


def _errors(reader):
    return [d for d in reader.diagnostics if d.level == "error"]


def _quad_part(m):
    p = m.Part("P", dim=2)
    for i, xy in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], 1):
        p.Node(i, *xy)
    p.Element(1, "CPE4", [1, 2, 3, 4])
    return p


# 1 ── contact pair resolves its property through model.contact_properties ──

def _pair(prop):
    from dispsolver.model.interaction import ContactPair
    return ContactPair(name="CP", main="M", secondary="S", interaction_property=prop)


def test_site1_unknown_property_name_raises():
    m = Model("T", dim=2)
    with pytest.raises(ValueError, match="interaction property 'Nope'"):
        _pair("Nope").build_runtime_constraint(mesh=None, nid_to_idx={}, model=m)


def test_site1_property_name_resolves_through_contact_properties():
    from dispsolver.model.interaction import _resolve_contact_property
    m = Model("T", dim=2)
    prop = m.ContactProperty("Fric", friction=0.3)
    assert _resolve_contact_property(_pair("Fric"), "fric", m) is prop     # case-insensitive


# 2 ── 2D pressure: region must be an element set; face must be known ───────

def test_site2_pressure_region_not_elset_raises():
    from dispsolver.solver.traction import SurfaceTraction
    mesh = SimpleNamespace(element_sets={"E": {1}}, elements={1: object()})
    st = SurfaceTraction(elset_name="NotASet", face="P1", pressure=1.0)
    with pytest.raises(ValueError, match="not an element set"):
        st.assemble_forces_and_stiffness(mesh, np.zeros(8), 1.0)


def test_site2_unknown_face_label_raises():
    from dispsolver.solver.traction import _resolve_face_indices
    assert _resolve_face_indices("P2", "CPE4") == [1, 2]
    with pytest.raises(ValueError, match="not a face"):
        _resolve_face_indices("P9", "CPE4")


# 3 ── unknown step name no longer attaches to Initial ──────────────────────

def test_site3_unknown_step_raises_initial_still_maps():
    m = Model("T", dim=2)
    m.Step("S1")
    with pytest.raises(KeyError, match="Typo"):
        m.DisplacementBC("bc", "Typo", "Left", u1=0.0)
    m.DisplacementBC("ok", "Initial", "Left", u1=0.0)
    assert "ok" in m.initial_step.boundary_conditions
    m.DisplacementBC("ok2", "s1", "Left", u1=0.0)                  # case-insensitive, real step


def test_site3_unknown_previous_step_raises():
    m = Model("T", dim=2)
    with pytest.raises(KeyError, match="Ghost"):
        m.Step("S2", previous="Ghost")


# 4 ── unknown interaction property name: raised where the pair is consumed ──
# (a deck may define *SURFACE INTERACTION after *CONTACT PAIR, so the factory
# keeps the written name in `interaction_property_ref` and build resolves it)

def test_site4_unknown_interaction_property_raises_at_build():
    m = Model("T", dim=2)
    cp = m.ContactPair("cp", "Initial", main="a", secondary="b", interactionProperty="Nope")
    assert cp.interaction_property is None and cp.interaction_property_ref == "Nope"
    with pytest.raises(ValueError, match="interaction property 'Nope'"):
        cp.build_runtime_constraint(mesh=None, nid_to_idx={}, model=m)


def test_site4_forward_reference_resolves_and_absent_keeps_default():
    from dispsolver.model.interaction import _resolve_contact_property
    m = Model("T", dim=2)
    cp = m.ContactPair("cp", "Initial", main="a", secondary="b", interactionProperty="Late")
    prop = m.ContactProperty("Late", friction=0.2)                 # defined after the pair
    assert _resolve_contact_property(cp, cp.interaction_property_ref, m) is prop
    cp2 = m.ContactPair("cp2", "Initial", main="a", secondary="b")     # absent -> default kept
    assert cp2.interaction_property.normal_behavior == "HARD"
    assert getattr(cp2, "interaction_property_ref", None) is None


# 5 ── orientation: checked where the section is consumed ──────────────────

def _section_model(orientation_defined):
    m = Model("T", dim=2)
    p = _quad_part(m)
    p.create_element_set("E", [1])
    m.Material("Mat")
    m.HomogeneousSolidSection("S", "Mat", orientation="Ori")      # forward reference is legal
    p.assign_section("E", "S")
    m.root_assembly.Instance("I", p)
    if orientation_defined:
        m.Orientation("Ori", frame=np.eye(2))
    return m


def test_site5_forward_orientation_ok_missing_orientation_raises_at_consumption():
    m = _section_model(orientation_defined=True)                  # defined AFTER the section
    assert m.sections["S"].orientation is not None                # back-filled
    m.root_assembly.build_solver_system(m)
    m = _section_model(orientation_defined=False)                 # creation itself still fine
    with pytest.raises(ValueError, match="orientation 'Ori'"):
        m.root_assembly.build_solver_system(m)


# 6 ── SectionAssignment region / material ─────────────────────────────────

def test_site6_section_assignment_region_missing_raises():
    from dispsolver.model.part import SectionAssignment
    m = Model("T", dim=2)
    p = _quad_part(m)
    m.Material("Mat")
    m.HomogeneousSolidSection("S", "Mat")
    p.section_assignments.append(SectionAssignment(region="NoSuchSet", section_name="S"))
    m.root_assembly.Instance("I", p)
    with pytest.raises(ValueError, match="NoSuchSet"):
        m.root_assembly.build_solver_system(m)


def test_site6_unknown_material_non_deck_raises():
    m = Model("T", dim=2)
    p = _quad_part(m)
    p.create_element_set("E", [1])
    m.HomogeneousSolidSection("S", "NoMat")
    p.assign_section("E", "S")
    m.root_assembly.Instance("I", p)
    with pytest.raises(ValueError, match="NoMat"):
        m.root_assembly.build_solver_system(m)


# 7 ── amplitude name with no model to resolve it ──────────────────────────

def test_site7_amplitude_name_without_model_raises():
    from dispsolver.model.load import ConcentratedForce
    f = ConcentratedForce(name="F", region="R", cf1=1.0, amplitude="Amp")
    with pytest.raises(ValueError, match="amplitude 'Amp'"):
        f.get_force_vector(t=0.5, step_time=0.5, model=None)
    amp = SimpleNamespace(evaluate=lambda t, s: 2.0)
    ok = f.get_force_vector(t=0.5, step_time=0.5, model=SimpleNamespace(amplitudes={"Amp": amp}))
    assert ok[0] == 2.0


def test_site7_pressure_amplitude_name_without_model_raises():
    from dispsolver.model.load import Pressure
    p = Pressure(name="P", region="R", magnitude=1.0, amplitude="Amp")
    with pytest.raises(ValueError, match="amplitude 'Amp'"):
        p.get_magnitude(t=0.5, step_time=0.5, model=None)


# 8 ── nodal transform nset not found ──────────────────────────────────────

def test_site8_transform_nset_missing_raises():
    from dispsolver.model.nodal_transform import build_nodal_transforms
    m = Model("T", dim=2)
    m.Transform("T1", "Nope", "R", points=((0, 0, 0), (1, 0, 0)))
    sys = SimpleNamespace(coords=np.zeros((2, 2)), dim=2, global_nsets={"Other": [0]})
    with pytest.raises(ValueError, match="'Nope'"):
        build_nodal_transforms(sys, m)
    m2 = Model("T", dim=2)
    m2.Transform("T1", "Other", "R", points=((0, 0, 0), (1, 0, 0)))
    build_nodal_transforms(sys, m2)                                # resolving case unchanged


# 9 ── history output region not found: loud warning, no data ──────────────

def test_site9_history_region_missing_warns():
    from dispsolver.output.manager import OutputManager
    mgr = OutputManager.__new__(OutputManager)
    mgr.dim = 2
    mesh = SimpleNamespace(node_sets={"N": [7]}, node_id_to_index=lambda: {7: 1})
    solver = SimpleNamespace(u=np.array([0.0, 0.0, 3.0, 4.0]), mesh=mesh)
    with pytest.warns(UserWarning, match="node set 'Missing'"):
        assert mgr._extract_nodal_history(solver, "Missing", "U1") is None
    assert mgr._extract_nodal_history(solver, "N", "U2") == 4.0    # resolving case unchanged


# 10 ── NGEN / NFILL / NCOPY / ELGEN missing references ────────────────────

_NODES = "*NODE\n1,0,0\n2,4,0\n"


@pytest.mark.parametrize("body, needle", [
    ("*NGEN\n1,9\n", "end node"),
    ("*NFILL, NSET=F\nNoA, NoB, 2, 1\n", "node set 'NoA'"),
    ("*NCOPY, OLD SET=NoSet, CHANGE NUMBER=10, SHIFT\n0.,0.,1.\n", "OLD SET 'NoSet'"),
    ("*ELGEN, ELSET=G\n5, 3, 1, 1\n", "master element 5"),
])
def test_site10_mesh_generation_missing_reference_is_an_error(body, needle):
    text = _NODES + "*ELEMENT, TYPE=CPE4\n1,1,2,2,1\n" + body
    r = _read(text)
    errs = [d for d in _errors(r) if needle in d.message]
    assert errs, [d.message for d in r.diagnostics]


def test_site10_ngen_resolving_case_unchanged():
    r = _read(_NODES + "*NGEN, NSET=L\n1,2\n")
    assert not _errors(r)


# 11 ── *RIGID BODY analytical surface ─────────────────────────────────────

def test_site11_rigid_body_analytical_surface_misses_are_errors():
    text = (_NODES + "*RIGID BODY, REF NODE=1, ANALYTICAL SURFACE=Ghost\n")
    r = _read(text)
    assert any(d.code == "rigid-body-analytical-surface" and "Ghost" in d.message for d in r.diagnostics)

    text = (_NODES + "*SURFACE, NAME=Rs, TYPE=SEGMENTS\nSTART, 0., 0.\nLINE, 1., 0.\n"
            "*RIGID BODY, ANALYTICAL SURFACE=Rs\n")
    r = _read(text)
    assert any(d.code == "rigid-body-ref-node" for d in r.diagnostics)

    text = (_NODES + "*SURFACE, NAME=Rs, TYPE=SEGMENTS\nSTART, 0., 0.\nLINE, 1., 0.\n"
            "*RIGID BODY, REF NODE=1, ANALYTICAL SURFACE=Rs\n")
    r = _read(text)
    assert not [d for d in r.diagnostics if d.code.startswith("rigid-body-")]
