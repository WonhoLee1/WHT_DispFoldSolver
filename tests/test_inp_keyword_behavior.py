"""Keyword -> solver behavior audit (2026-10-04).

Each test reads a small deck and checks what the SOLVER ends up holding --
fixed dofs, the external load vector, the material and kernel of every
element -- not what the reader stored. Every case here used to be wrong
without any error: a BC on a lower-case set name, a section that silently
became E=4000, gravity without density, an amplitude table read as a
constant 0, a C3D8XYZ run as C3D8, ... See
dev_log/keyword_behavior_audit_20261004.md.

No case solves; they evaluate the step's BCs/loads at the end of the step.
"""
import warnings

import numpy as np
import pytest

from dispsolver.io.abaqus_reader import (UnsupportedDeckError, _parse_source, detect_dim,
                                         execute_inp)
from dispsolver.model.regions import RegionError

NODES = """*NODE
1,0,0,0
2,1,0,0
3,2,0,0
4,0,1,0
5,1,1,0
6,2,1,0
7,0,0,1
8,1,0,1
9,2,0,1
10,0,1,1
11,1,1,1
12,2,1,1
"""
ELEMS = """*ELEMENT, TYPE={etype}, ELSET=EALL
1, 1,2,5,4,7,8,11,10
2, 2,3,6,5,8,9,12,11
*ELSET, ELSET=E1
1
*ELSET, ELSET=E2
2
*NSET, NSET=LEFT
1,4,7,10
*NSET, NSET=RIGHT
3,6,9,12
"""
MATS = """*MATERIAL, NAME=STEEL
*ELASTIC
200000., 0.3
*DENSITY
7.8e-9
*MATERIAL, NAME=SOFT
*ELASTIC
1000., 0.3
*DENSITY
1e-9
"""
SECTION = "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n"
STEP = "*STEP\n*STATIC\n0.1, 1.0\n"


def deck(body="", section=SECTION, head=STEP, etype="C3D8", nodes=NODES):
    return nodes + ELEMS.format(etype=etype) + MATS + section + head + body + "*END STEP\n"


def build(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, model = _parse_source(text, dim=3, strict=True)
        solver, sysm = model.create_solver(dim=3)
    st = model.steps[[s for s in model.steps if s != "Initial"][-1]]
    st.parent_model = model
    return reader, model, solver, sysm, st


def at_end(text):
    reader, model, solver, sysm, st = build(text)
    T = st.time_period
    solver._u_step_start = solver.u.copy()
    solver._update_step_bcs(st, sysm, T, T)
    f = solver._compute_step_external_loads(st, sysm, T, T)
    fixed = {(int(d) // 3 + 1, int(d) % 3 + 1): float(v) for d, v in solver.fixed_dofs.items()}
    return fixed, f.reshape(-1, 3), solver, reader


# ----------------------------------------------------------------- names
def test_set_names_are_case_insensitive():
    fixed, f, _, _ = at_end(deck("*BOUNDARY\nleft, 1, 3\n*CLOAD\nright, 1, 10.\n"))
    assert sorted({n for n, _ in fixed}) == [1, 4, 7, 10]
    assert np.allclose(f[[2, 5, 8, 11], 0], 10.0) and np.isclose(f.sum(), 40.0)


@pytest.mark.parametrize("body", ["*BOUNDARY\nNOPE, 1, 3\n", "*CLOAD\nNOPE, 1, 10.\n"])
def test_unresolved_region_raises(body):
    with pytest.raises(RegionError):
        at_end(deck(body))


def test_bare_node_number_is_the_deck_label():
    # node labels 101..108: CLOAD on "103" means the deck's node 103 (the
    # third node), never the solver's internal id 103 (or 3)
    text = """*NODE
101,0,0,0
102,1,0,0
103,1,1,0
104,0,1,0
105,0,0,1
106,1,0,1
107,1,1,1
108,0,1,1
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 101,102,103,104,105,106,107,108
""" + MATS + SECTION + STEP + "*CLOAD\n103, 2, 5.\n*END STEP\n"
    _, f, _, _ = at_end(text)
    assert np.isclose(f[2, 1], 5.0) and np.isclose(f.sum(), 5.0)


# ----------------------------------------------------------------- sections
def test_section_with_missing_elset_is_a_gap_and_refused():
    text = deck(section="*SOLID SECTION, ELSET=NOPE, MATERIAL=SOFT\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, model = _parse_source(text, dim=3)
    assert any("SOLID SECTION" in g for g in reader.physics_gaps())
    with pytest.raises(ValueError, match="without a section"):
        model.create_solver(dim=3)
    with pytest.raises(UnsupportedDeckError):
        execute_inp(text, verbose=False)


def test_section_with_undefined_material_raises():
    with pytest.raises(ValueError, match="does not define"):
        build(deck(section="*SOLID SECTION, ELSET=EALL, MATERIAL=NOPE\n"))


def test_element_without_section_raises():
    with pytest.raises(ValueError, match="without a section"):
        build(deck(section="*SOLID SECTION, ELSET=E1, MATERIAL=SOFT\n"))


CUBE = """*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE=C3D8, ELSET=SEC
1, 1,2,3,4,5,6,7,8
*NSET, NSET=BOT
1,2,3,4
*NSET, NSET=TOP
5,6,7,8
"""


def two_parts(step_body="", asm_extra=""):
    return ("*PART, NAME=PA\n" + CUBE + "*SOLID SECTION, ELSET=SEC, MATERIAL=STEEL\n*END PART\n"
            "*PART, NAME=PB\n" + CUBE + "*SOLID SECTION, ELSET=SEC, MATERIAL=SOFT\n*END PART\n"
            "*ASSEMBLY, NAME=A\n*INSTANCE, NAME=IA, PART=PA\n*END INSTANCE\n"
            "*INSTANCE, NAME=IB, PART=PB\n0., 0., 1.\n*END INSTANCE\n" + asm_extra + "*END ASSEMBLY\n"
            + MATS + STEP + step_body + "*END STEP\n")


def test_same_elset_name_in_two_parts_keeps_each_material():
    _, _, solver, _, _ = build(two_parts())
    assert np.allclose(np.asarray(solver.elem_props)[:, 0], [200000.0, 1000.0])


def test_ambiguous_bare_set_name_raises_and_qualified_works():
    with pytest.raises(RegionError, match="several instances"):
        at_end(two_parts("*BOUNDARY\nBOT, 1, 3\n"))
    fixed, f, _, _ = at_end(two_parts("*BOUNDARY\nIA.BOT, 1, 3\n*CLOAD\nIB.TOP, 3, 1.\n"))
    assert sorted({n for n, _ in fixed}) == [1, 2, 3, 4]
    assert np.allclose(f[12:16, 2], 1.0)


def test_assembly_set_with_instance_maps_to_that_instance():
    extra = "*NSET, NSET=PULL, INSTANCE=IB\n5, 6, 7, 8\n"
    _, f, _, _ = at_end(two_parts("*CLOAD\nPULL, 3, 1.\n", asm_extra=extra))
    assert np.allclose(f[12:16, 2], 1.0) and np.isclose(f.sum(), 4.0)


# ----------------------------------------------------------------- body loads
def test_gravity_is_density_times_volume_on_its_region():
    _, f, _, _ = at_end(deck("*DLOAD\nE1, GRAV, 9810., 0., 0., -1.\n"))
    assert np.isclose(f[:, 2].sum(), -7.8e-9 * 9810.0 * 1.0)       # one unit cube
    assert np.allclose(f[[2, 5, 8, 11], :], 0.0)                   # element 2's own nodes untouched


def test_gravity_blank_region_is_whole_model():
    _, f, _, _ = at_end(deck("*DLOAD\n, GRAV, 9810., 0., 0., -1.\n"))
    assert np.isclose(f[:, 2].sum(), -7.8e-9 * 9810.0 * 2.0)


def test_gravity_without_density_raises():
    mats_nodens = "*MATERIAL, NAME=ND\n*ELASTIC\n1000., 0.3\n*SOLID SECTION, ELSET=EALL, MATERIAL=ND\n"
    with pytest.raises(ValueError, match="DENSITY"):
        at_end(deck("*DLOAD\nEALL, GRAV, 9810., 0., 0., -1.\n", section=mats_nodens))


def test_body_force_is_consistent():
    _, f, _, _ = at_end(deck("*DLOAD\nEALL, BZ, -2.\n"))
    assert np.isclose(f[:, 2].sum(), -4.0) and np.isclose(f[0, 2], -0.25) and np.isclose(f[1, 2], -0.5)


# ----------------------------------------------------------------- amplitude / BC / CLOAD parsing
def test_amplitude_reads_several_pairs_per_line():
    fixed, _, _, _ = at_end(deck("*BOUNDARY, AMPLITUDE=A1\nRIGHT, 1, 1, 0.5\n",
                                 head="*AMPLITUDE, NAME=A1\n0,0, 1,2\n" + STEP))
    assert np.isclose(fixed[(3, 1)], 1.0)


def test_undefined_amplitude_raises():
    with pytest.raises(ValueError, match="not defined"):
        at_end(deck("*BOUNDARY, AMPLITUDE=NOAMP\nRIGHT, 1, 1, 0.5\n"))


@pytest.mark.parametrize("body,head", [
    ("*BOUNDARY, TYPE=VELOCITY\nRIGHT, 1, 1, 0.5\n", STEP),
    ("*BOUNDARY\nRIGHT, X, 1\n", STEP),
    ("*CLOAD\nRIGHT, 4, 10.\n", STEP),
    ("*BOUNDARY, AMPLITUDE=A1\nRIGHT, 1, 1, 0.5\n", "*AMPLITUDE, NAME=A1, TIME=TOTAL TIME\n0,0, 1,1\n" + STEP),
    ("*FOOBAR\n1\n", STEP),
])
def test_unrepresentable_input_is_a_physics_gap(body, head):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, _ = _parse_source(deck(body, head=head), dim=3)
    assert reader.physics_gaps()


def test_output_only_keywords_are_not_gaps():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, _ = _parse_source(deck("*EL PRINT\nS\n*NODE PRINT\nU\n"), dim=3)
    assert reader.unsupported_keywords and not reader.physics_gaps()


# ----------------------------------------------------------------- elements / temperature / dim
def test_unknown_element_type_raises():
    with pytest.raises(NotImplementedError, match="C3D8XYZ"):
        build(deck(etype="C3D8XYZ"))


def test_partial_formulation_substitution_is_reported():
    _, _, solver, _, _ = build(deck(etype="C3D8RH"))   # build() mutes the warning itself
    assert "hybrid pressure field is not included" in solver.element_substitutions["C3D8RH"]


def test_temperature_with_expansion_is_refused():
    sec = "*MATERIAL, NAME=TH\n*ELASTIC\n200000., 0.3\n*EXPANSION\n1e-5\n*SOLID SECTION, ELSET=EALL, MATERIAL=TH\n"
    with pytest.raises(NotImplementedError, match="EXPANSION"):
        build(deck("*TEMPERATURE\nEALL, 100.\n", section=sec))


def test_dimension_comes_from_the_deck():
    assert detect_dim("*ELEMENT, TYPE=CPE4, ELSET=E\n1,1,2,3,4\n") == 2
    assert detect_dim(deck()) == 3
    with pytest.raises(ValueError, match="mixes"):
        detect_dim("*ELEMENT, TYPE=CPE4\n*ELEMENT, TYPE=C3D8\n")


def test_2d_cload_ramps_over_a_static_step():
    text = """*NODE
1,0,0
2,1,0
3,1,1
4,0,1
*ELEMENT, TYPE=CPE4, ELSET=E
1,1,2,3,4
*MATERIAL, NAME=M
*ELASTIC
1000., 0.3
*SOLID SECTION, ELSET=E, MATERIAL=M
1.
*STEP
*STATIC
0.1, 1.0
*BOUNDARY
1, 1, 2
*CLOAD
3, 1, 8.
*END STEP
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model = _parse_source(text, dim=2)
        solver, sysm = model.create_solver(dim=2)
    st = model.steps[[s for s in model.steps if s != "Initial"][-1]]
    st.parent_model = model
    f_half = solver._compute_step_external_loads(st, sysm, 0.5, 1.0)
    assert np.isclose(f_half.sum(), 4.0)
