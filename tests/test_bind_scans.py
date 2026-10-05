"""A3 / plan_a3_bind_20261005.md P3: the first-match name scans resolve through
model/regions.py, so *COUPLING, *EQUATION, *TRANSFORM and part-surface contact
references follow the BC/load rules (qualified, unique bare, ambiguity error,
bare number only in a single-instance model)."""
import warnings

import numpy as np
import pytest

from dispsolver.model.model import Model
from dispsolver.model.regions import RegionError


def _part(m, name="P"):
    p = m.Part(name, dim=2)
    for i, xy in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], 1):
        p.Node(i, *xy)
    p.Element(1, "CPE4", [1, 2, 3, 4])
    p.NodeSet("REF", [1])
    p.NodeSet("PAIR", [1, 2])
    p.NodeSet("SIDE", [2, 3])
    return p


def _model(n_inst=1):
    m = Model("T", dim=2)
    p = _part(m)
    asm = m.root_assembly
    for k in range(n_inst):
        asm.Instance(f"I{k + 1}", p)
    return m


def _build(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.build_solver_system()


# ── Coupling ───────────────────────────────────────────────────────────────

def test_coupling_unique_bare_and_qualified_resolve():
    for ref, surf in (("REF", "SIDE"), ("I1.REF", "I1.SIDE"), ("1", "side")):
        m = _model(1)
        m.KinematicCoupling("C", refPoint=ref, surface=surf)
        c = _build(m).constraints[0]
        assert c._bound and c._rp_idx == 0 and sorted(c._slave_indices) == [1, 2]


def test_coupling_qualified_in_two_instances():
    m = _model(2)
    m.KinematicCoupling("C", refPoint="I2.REF", surface="I2.SIDE")
    c = _build(m).constraints[0]
    assert c._rp_idx == 4 and sorted(c._slave_indices) == [5, 6]


def test_coupling_ambiguous_bare_ref_node_set_raises():
    m = _model(2)
    m.KinematicCoupling("C", refPoint="REF", surface="I1.SIDE")
    with pytest.raises(RegionError, match="several instances"):
        _build(m)


def test_coupling_ambiguous_bare_surface_raises():
    m = _model(2)
    m.KinematicCoupling("C", refPoint="I1.REF", surface="SIDE")
    with pytest.raises(RegionError, match="several instances"):
        _build(m)


def test_coupling_bare_node_number_in_multi_instance_raises():
    m = _model(2)
    m.KinematicCoupling("C", refPoint="1", surface="I1.SIDE")
    with pytest.raises(RegionError, match="INSTANCE.1"):
        _build(m)


def test_coupling_ref_node_set_with_two_nodes_raises():
    m = _model(1)
    m.KinematicCoupling("C", refPoint="PAIR", surface="SIDE")
    with pytest.raises(RegionError, match="exactly one node"):
        _build(m)


def test_coupling_ref_node_no_longer_taken_from_element_set():
    m = _model(1)
    m.root_assembly.instances["I1"].part.ElementSet("E1", [1])
    m.KinematicCoupling("C", refPoint="E1", surface="SIDE")
    with pytest.raises(RegionError):
        _build(m)


# ── Equation ───────────────────────────────────────────────────────────────

def test_equation_set_term_pairing_unchanged():
    m = _model(1)
    m.Equation("E0", terms=[("PAIR", 0, 1.0), (3, 0, -1.0)], pair_index=0)
    m.Equation("E1", terms=[("PAIR", 0, 1.0), (3, 0, -1.0)], pair_index=1)
    sys = _build(m)
    eqs = sys.constraints
    assert [c._dofs for c in eqs] == [[0, 4], [2, 4]]
    assert all(c._bound for c in eqs)


def test_equation_ambiguous_bare_set_raises():
    m = _model(2)
    m.Equation("E", terms=[("PAIR", 0, 1.0), ("I1.3", 0, -1.0)], pair_index=0)
    with pytest.raises(RegionError, match="several instances"):
        _build(m)


def test_equation_qualified_set_in_multi_instance():
    m = _model(2)
    m.Equation("E", terms=[("I2.PAIR", 0, 1.0), ("I1.3", 0, -1.0)], pair_index=1)
    c = _build(m).constraints[0]
    assert c._dofs == [2 * 5, 2 * 2]


def test_equation_reader_keeps_set_name_and_binder_refuses_ambiguity():
    from dispsolver.io.abaqus_reader import _parse_source
    deck = """*HEADING
*PART, NAME=P
*NODE
1,0,0
2,1,0
3,1,1
4,0,1
*ELEMENT, TYPE=CPE4
1,1,2,3,4
*NSET, NSET=PAIR
1,2
*SOLID SECTION, ELSET=ALLE, MATERIAL=M
*ELSET, ELSET=ALLE
1
*END PART
*ASSEMBLY, NAME=A
*INSTANCE, NAME=I1, PART=P
*END INSTANCE
*INSTANCE, NAME=I2, PART=P
*END INSTANCE
*END ASSEMBLY
*MATERIAL, NAME=M
*ELASTIC
1000,0.3
*EQUATION
2
PAIR,1,1.
I1.3,1,-1.
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, model = _parse_source(deck, dim=2, strict=False)[:2]
    # two instances of P both define PAIR: refused when the set is resolved
    assert [d.code for d in reader.diagnostics if d.level == "error"].count("ambiguous-ref") == 1
    # one instance: expanded by set size, set NAME kept for bind-time resolution
    deck1 = deck.replace("*INSTANCE, NAME=I2, PART=P\n*END INSTANCE\n", "").replace("I1.3", "3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, model = _parse_source(deck1, dim=2, strict=False)[:2]
    from dispsolver.model.constraint import EquationConstraint
    eqs = [c for c in model.constraints.values() if isinstance(c, EquationConstraint)]
    assert len(eqs) == 2 and [e.pair_index for e in eqs] == [0, 1]
    assert eqs[0].terms[0][0] == "PAIR"


# ── Nodal transform ────────────────────────────────────────────────────────

def test_transform_nset_resolves_and_rejects_ambiguity():
    for n_inst, target, ok in ((1, "SIDE", True), (2, "I2.SIDE", True), (2, "SIDE", False)):
        m = _model(n_inst)
        m.transforms["T1"] = {"name": "T1", "nset": target, "type": "R",
                              "a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 1.0)}
        if ok:
            _build(m)
        else:
            with pytest.raises(RegionError, match="several instances"):
                _build(m)


# ── part surface without an instance (interaction.py) ──────────────────────

def _surf_model(n_inst):
    m = _model(n_inst)
    part = m.root_assembly.instances["I1"].part
    from dispsolver.model.set import Surface
    part.surfaces["S1"] = Surface("S1", node_ids=[1, 2])
    return m


def test_part_surface_owner_unique_and_ambiguous():
    from dispsolver.model.interaction import _part_surface_owner
    m = _surf_model(1)
    inst, part, surf = _part_surface_owner("S1", m)
    assert inst == "I1"
    m2 = _surf_model(2)
    with pytest.raises(RegionError, match="several instances"):
        _part_surface_owner("S1", m2)
    assert _part_surface_owner("Nope", m2) is None
