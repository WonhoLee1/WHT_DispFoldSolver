"""
tests/test_material_keywords.py
================================
Unit tests for the 21 material keywords migrated to the schema framework (B4).

Verifies:
- Snippet -> Schema parsing (to_records, records)
- Application to Model Material objects
- Mapping to canonical numeric IDs and packed props through dispsolver.material.catalog
"""

from __future__ import annotations

import numpy as np
import pytest

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.io.schema import parse_text
from dispsolver.material.catalog import (
    MAT_CONCRETE_DP,
    MAT_CUSTOM_ELASTIC,
    MAT_DRUCKER_PRAGER,
    MAT_HYPERELASTIC_ARRUDA_BOYCE,
    MAT_HYPERELASTIC_MOONEY_RIVLIN,
    MAT_HYPERELASTIC_NEOHOOKEAN,
    MAT_HYPERELASTIC_OGDEN,
    MAT_HYPERELASTIC_POLYNOMIAL,
    MAT_HYPERELASTIC_YEOH,
    MAT_J2_PLASTICITY,
    MAT_J2_PLASTICITY_TABULAR,
    MAT_LINEAR_ELASTIC,
    MAT_VISCOELASTIC_PRONY,
    material_id,
    pack_props,
)


def _read_snippet(deck: str):
    """Parse snippet string and return (reader, model)."""
    return _parse_source(deck, dim=3, strict=False)


def test_material_keyword_schema():
    """*MATERIAL keyword creates MaterialRecord and opens material context."""
    snippet = "*MATERIAL, NAME=STEEL\n"
    recs, diags = parse_text(snippet, context="GLOBAL")
    assert len(recs) == 1
    assert recs[0].name == "STEEL"
    assert not diags


def test_elastic_isotropic():
    """*ELASTIC isotropic maps to MAT_LINEAR_ELASTIC and packs props."""
    deck = """
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["STEEL"]
    assert mat.elastic == (210000.0, 0.3)
    mid = material_id(mat)
    assert mid == MAT_LINEAR_ELASTIC
    props = pack_props(mat, dim=3)
    assert props[0] == pytest.approx(210000.0)
    assert props[1] == pytest.approx(0.3)


def test_elastic_engineering_constants():
    """*ELASTIC, TYPE=ENGINEERING CONSTANTS maps to custom elastic."""
    deck = """
*MATERIAL, NAME=COMPOSITE
*ELASTIC, TYPE=ENGINEERING CONSTANTS
10000.0, 2000.0, 2000.0, 0.3, 0.3, 0.3, 500.0, 500.0
450.0
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["COMPOSITE"]
    assert mat.extra_params["engineering_constants"] == (10000.0, 2000.0, 2000.0, 0.3, 0.3, 0.3, 500.0, 500.0, 450.0)
    mid = material_id(mat)
    assert mid == MAT_CUSTOM_ELASTIC
    props = pack_props(mat, dim=3)
    assert len(props) >= 21


def test_expansion_thermal():
    """*EXPANSION stores thermal expansion data and zero temperature."""
    deck = """
*MATERIAL, NAME=THERMAL_STEEL
*ELASTIC
210000.0, 0.3
*EXPANSION, ZERO=20.0
1.2e-5
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["THERMAL_STEEL"]
    assert mat.expansion is not None
    assert mat.expansion["type"] == "ISO"
    assert mat.expansion["zero"] == 20.0
    assert mat.expansion["table"] == [(1.2e-5,)]


def test_plastic_isotropic():
    """*PLASTIC: one point is J2 (perfect plasticity), two or more points the tabular kernel."""
    deck = """
*MATERIAL, NAME=ALU
*ELASTIC
70000.0, 0.33
*PLASTIC
200.0, 0.0
250.0, 0.05
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["ALU"]
    # two or more points go to the tabular kernel (it holds the last yield stress constant; the
    # linear-H J2 kernel would keep hardening without bound)
    assert material_id(mat) == MAT_J2_PLASTICITY_TABULAR
    props = pack_props(mat, dim=3)
    assert props[0] == pytest.approx(70000.0)
    assert props[1] == pytest.approx(0.33)
    assert props[2] == 2.0                     # number of points
    assert list(props[5:9]) == [200.0, 0.0, 250.0, 0.05]
    _, m1 = _read_snippet(deck.replace("250.0, 0.05\n", ""))
    assert material_id(m1.materials["ALU"]) == MAT_J2_PLASTICITY


def test_plastic_johnson_cook():
    """*PLASTIC, HARDENING=JOHNSON COOK has no solver material: refused at parse, material refused."""
    deck = """
*MATERIAL, NAME=JC_STEEL
*ELASTIC
200000.0, 0.3
*PLASTIC, HARDENING=JOHNSON COOK
300.0, 400.0, 0.5, 0.0, 1800.0, 293.0
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["JC_STEEL"]
    assert "johnson_cook" not in mat.extra_params
    assert any(d.level == "gap" and "JOHNSON COOK" in d.message for d in reader.diagnostics)
    assert "JOHNSON COOK" in mat.extra_params["refused_definitions"][0]


def test_creep_strain():
    """*CREEP is a gap: no solver material id consumes a creep law."""
    deck = """
*MATERIAL, NAME=CREEP_STEEL
*ELASTIC
150000.0, 0.3
*CREEP, LAW=TIME
1.0e-10, 5.0, -0.5
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["CREEP_STEEL"]
    assert "creep" not in mat.extra_params
    assert any(d.level == "gap" and "creep" in d.message.lower() for d in reader.diagnostics)
    assert "*CREEP" in mat.extra_params["refused_definitions"][0]


def test_hyperelastic_neo_hookean():
    """*HYPERELASTIC, NEO HOOKE maps to MAT_HYPERELASTIC_NEOHOOKEAN."""
    deck = """
*MATERIAL, NAME=RUBBER_NH
*HYPERELASTIC, NEO HOOKE
0.5, 0.01
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["RUBBER_NH"]
    mid = material_id(mat)
    assert mid == MAT_HYPERELASTIC_NEOHOOKEAN
    props = pack_props(mat, dim=3)
    assert props[0] == pytest.approx(1.0)   # mu = 2*C10
    assert props[1] == pytest.approx(200.0) # K = 2/D1


def test_hyperelastic_arruda_boyce():
    """*HYPERELASTIC, ARRUDA-BOYCE maps to MAT_HYPERELASTIC_ARRUDA_BOYCE."""
    deck = """
*MATERIAL, NAME=RUBBER_AB
*HYPERELASTIC, ARRUDA-BOYCE
0.5, 3.0, 0.01
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["RUBBER_AB"]
    mid = material_id(mat)
    assert mid == MAT_HYPERELASTIC_ARRUDA_BOYCE
    props = pack_props(mat, dim=3)
    assert props[0] == pytest.approx(0.5)
    assert props[1] == pytest.approx(3.0)


def test_hyperelastic_yeoh():
    """*HYPERELASTIC, YEOH maps to MAT_HYPERELASTIC_YEOH."""
    deck = """
*MATERIAL, NAME=RUBBER_YEOH
*HYPERELASTIC, YEOH
0.5, 0.1, 0.02, 0.01, 0.0, 0.0
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["RUBBER_YEOH"]
    mid = material_id(mat)
    assert mid == MAT_HYPERELASTIC_YEOH
    props = pack_props(mat, dim=3)
    assert props[0] == pytest.approx(0.5)
    assert props[1] == pytest.approx(0.1)


def test_hyperelastic_mooney_rivlin():
    """*HYPERELASTIC, MOONEY-RIVLIN maps to MAT_HYPERELASTIC_MOONEY_RIVLIN."""
    deck = """
*MATERIAL, NAME=RUBBER_MR
*HYPERELASTIC, MOONEY-RIVLIN
0.4, 0.1, 0.01
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["RUBBER_MR"]
    mid = material_id(mat)
    assert mid == MAT_HYPERELASTIC_MOONEY_RIVLIN


def test_viscoelastic_prony():
    """*VISCOELASTIC, TIME=PRONY maps to MAT_VISCOELASTIC_PRONY."""
    deck = """
*MATERIAL, NAME=VISCO_ELAST
*ELASTIC
100.0, 0.4
*VISCOELASTIC, TIME=PRONY
0.3, 0.0, 1.5
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["VISCO_ELAST"]
    mid = material_id(mat)
    assert mid == MAT_VISCOELASTIC_PRONY
    props = pack_props(mat, dim=3)
    # *ELASTIC default MODULI=LONG TERM: instantaneous shear = long-term / (1 - sum g); bulk unchanged
    mu = props[0] / (2.0 * (1.0 + props[1]))
    K = props[0] / (3.0 * (1.0 - 2.0 * props[1]))
    assert mu == pytest.approx(100.0 / (2.0 * 1.4) / 0.7)
    assert K == pytest.approx(100.0 / (3.0 * 0.2))
    assert (props[2], props[3]) == (0.3, 1.5)
    # MODULI=INSTANTANEOUS: the constants are the kernel's own
    _, m2 = _read_snippet(deck.replace("*ELASTIC", "*ELASTIC, MODULI=INSTANTANEOUS"))
    p2 = pack_props(m2.materials["VISCO_ELAST"], dim=3)
    assert (p2[0], p2[1]) == (100.0, 0.4)


def test_rate_dependent():
    """*RATE DEPENDENT stores strain rate parameters."""
    deck = """
*MATERIAL, NAME=RATE_PLASTIC
*ELASTIC
200000.0, 0.3
*PLASTIC
200.0, 0.0
*RATE DEPENDENT, TYPE=POWER LAW
100.0, 5.0
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["RATE_PLASTIC"]
    assert "rate_dependent" in mat.extra_params
    rd = mat.extra_params["rate_dependent"]
    assert rd["type"] == "POWER_LAW"
    assert rd["table"] == ((100.0, 5.0),)


def test_damage_initiation_and_evolution():
    """*DAMAGE INITIATION and *DAMAGE EVOLUTION store fracture parameters."""
    deck = """
*MATERIAL, NAME=FRACTURE_STEEL
*ELASTIC
200000.0, 0.3
*PLASTIC
300.0, 0.0
*DAMAGE INITIATION, CRITERION=DUCTILE
0.2
*DAMAGE EVOLUTION, TYPE=ENERGY, SOFTENING=LINEAR
50.0
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["FRACTURE_STEEL"]
    assert "damage_init" in mat.extra_params
    assert mat.extra_params["damage_init"]["type"] == 0
    assert mat.extra_params["damage_init"]["p"][0] == pytest.approx(0.2)
    assert "damage_evol" in mat.extra_params
    assert mat.extra_params["damage_evol"]["type"] == 0  # ENERGY_LINEAR
    assert mat.extra_params["damage_evol"]["Gf"] == pytest.approx(50.0)


_DP_HARD = "*DRUCKER PRAGER HARDENING\n20.0, 0.0\n30.0, 0.02\n"


def _dp_deck(dp_line="35.0, 1.0, 20.0", hard=_DP_HARD):
    return f"""
*MATERIAL, NAME=SOIL
*ELASTIC
50000.0, 0.3
*DRUCKER PRAGER
{dp_line}
{hard}"""


def test_drucker_prager():
    """*DRUCKER PRAGER data line is (beta, K, psi); COMPRESSION hardening gives d."""
    import math
    reader, model = _read_snippet(_dp_deck())
    mat = model.materials["SOIL"]
    assert material_id(mat) == MAT_DRUCKER_PRAGER
    props = pack_props(mat, dim=3)
    f = 1.0 - math.tan(math.radians(35.0)) / 3.0
    assert props[0] == pytest.approx(50000.0)
    assert props[1] == pytest.approx(0.3)
    assert props[2] == pytest.approx(f * 20.0)             # d0 = (1 - tan(beta)/3) sigma_c
    assert props[3] == pytest.approx(f * 10.0 / 0.02)      # H
    assert props[4] == pytest.approx(35.0)                 # beta
    assert props[5] == pytest.approx(20.0)                 # psi
    assert [d.code for d in reader.diagnostics.of("error", "gap")] == ["dp-hardening-linear-extrapolation"]


def test_drucker_prager_suboption_order_is_free():
    a, ma = _read_snippet(_dp_deck())
    hard_first = ("\n*MATERIAL, NAME=SOIL\n*ELASTIC\n50000.0, 0.3\n" + _DP_HARD + "*DRUCKER PRAGER\n35.0, 1.0, 20.0\n")
    b, mb = _read_snippet(hard_first)
    assert ma.materials["SOIL"].extra_params["drucker_prager"] == mb.materials["SOIL"].extra_params["drucker_prager"]


def test_drucker_prager_hardening_types():
    import math
    tb = math.tan(math.radians(30.0))
    # TENSION: d = (1/K + tan(beta)/3) sigma_t; SHEAR: the data value IS the cohesion d (cached page)
    for typ, f in (("TENSION", 1.0 + tb / 3.0), ("SHEAR", 1.0)):
        _, model = _read_snippet(_dp_deck("30.0, 0.0, 10.0",
                                          f"*DRUCKER PRAGER HARDENING, TYPE={typ}\n10.0, 0.0\n"))
        dp = model.materials["SOIL"].extra_params["drucker_prager"]
        assert dp["d0"] == pytest.approx(f * 10.0)
        assert dp["H"] == 0.0 and dp["psi"] == 10.0 and dp["beta"] == 30.0


def test_drucker_prager_missing_items_and_hardening():
    # blank beta / psi are zero (Abaqus Input Syntax Rules)
    for line in (", 1.0, 10.0", "30.0, 1.0"):
        reader, _ = _read_snippet(_dp_deck(line))
        assert not reader.diagnostics.of("error"), line
    # K != 1 is a gap (Material has no K)
    reader, _ = _read_snippet(_dp_deck("30.0, 0.8, 10.0"))
    assert reader.diagnostics.of("gap")
    # no hardening -> refused, naming the material and recorded on it
    reader, m = _read_snippet(_dp_deck(hard=""))
    msgs = [d.message for d in reader.diagnostics.of("error")]
    assert any("HARDENING" in x for x in msgs)
    assert any("HARDENING" in r for r in m.materials["SOIL"].extra_params["refused_definitions"])
    # hardening without DRUCKER PRAGER, strain not increasing, more than two points: refused
    for deck in ("\n*MATERIAL, NAME=SOIL\n*ELASTIC\n5e4, 0.3\n" + _DP_HARD,
                 _dp_deck(hard="*DRUCKER PRAGER HARDENING\n20.0, 0.0\n30.0, 0.0\n"),
                 _dp_deck(hard="*DRUCKER PRAGER HARDENING\n20.0, 0.0\n30.0, 0.02\n40.0, 0.04\n")):
        reader, m = _read_snippet(deck)
        assert m.materials["SOIL"].extra_params["refused_definitions"], deck
        assert "drucker_prager" not in m.materials["SOIL"].extra_params


def test_concrete_damaged_plasticity():
    """*CONCRETE DAMAGED PLASTICITY suite maps to MAT_CONCRETE_DP."""
    deck = """
*MATERIAL, NAME=CONCRETE
*ELASTIC
30000.0, 0.2
*CONCRETE DAMAGED PLASTICITY
36.0, 0.1, 1.16, 0.667, 0.0
*CONCRETE COMPRESSION HARDENING
15.0, 0.0
30.0, 0.001
*CONCRETE TENSION STIFFENING
3.0, 0.0
0.3, 0.001
*CONCRETE COMPRESSION DAMAGE
0.0, 0.0
0.5, 0.001
*CONCRETE TENSION DAMAGE
0.0, 0.0
0.9, 0.001
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["CONCRETE"]
    mid = material_id(mat)
    assert mid == MAT_CONCRETE_DP
    props = pack_props(mat, dim=3)
    assert props[0] == pytest.approx(30000.0)
    assert props[1] == pytest.approx(0.2)


def test_mullins_effect():
    """*MULLINS EFFECT parses damage parameters for filled elastomer."""
    deck = """
*MATERIAL, NAME=FILLED_RUBBER
*HYPERELASTIC, NEO HOOKE
0.5, 0.01
*MULLINS EFFECT
2.5, 0.6, 0.15
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["FILLED_RUBBER"]
    assert "mullins" in mat.extra_params
    m = mat.extra_params["mullins"]
    assert m["r"] == 2.5
    assert m["m"] == 0.6
    assert m["beta"] == 0.15


def test_density():
    """*DENSITY parses mass density and assigns to material."""
    deck = """
*MATERIAL, NAME=DENSE_STEEL
*ELASTIC
210000.0, 0.3
*DENSITY
7.85e-9
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["DENSE_STEEL"]
    assert mat.density == 7.85e-9


def test_trs():
    """*TRS parses temperature-time shift data."""
    deck = """
*MATERIAL, NAME=TRS_RUBBER
*ELASTIC
100.0, 0.4
*TRS, DEFINITION=WLF
293.15, 17.44, 51.6
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["TRS_RUBBER"]
    assert "trs" in mat.extra_params
    trs = mat.extra_params["trs"]
    assert trs["definition"] == "WLF"
    assert trs["T_ref"] == 293.15
    assert trs["C1"] == 17.44
    assert trs["C2"] == 51.6


def test_orientation():
    """*ORIENTATION parses local coordinate system and frame."""
    deck = """
*ORIENTATION, NAME=ORI1, SYSTEM=RECTANGULAR
1.0, 0.0, 0.0, 0.0, 1.0, 0.0
"""
    reader, model = _read_snippet(deck)
    assert "ORI1" in model.orientations
    ori = model.orientations["ORI1"]
    assert ori.system == "RECTANGULAR"
    assert ori.frame is not None
    assert np.allclose(ori.frame, np.eye(3))


# ─── Guide data-line layout audit: ELASTIC / EXPANSION / PLASTIC ─────────────

def _errors(deck):
    reader, model = _read_snippet(deck)
    return reader, model, [d for d in reader.diagnostics.of("error", "gap")]


def test_elastic_positional_temperature_and_field():
    # (E, nu, T): temperature-dependent moduli never reach the solver -> refused
    _, m, bad = _errors("*MATERIAL, NAME=A\n*ELASTIC\n200000., 0.3, 20.\n180000., 0.3, 100.\n")
    assert any(d.level == "gap" and "temperature" in d.message for d in bad)
    assert m.materials["A"].extra_params["refused_definitions"]
    # DEPENDENCIES=1: data items are E, nu, T, FV1 -- the field variable is item 4, not item 3
    # field variables never reach the solver either -> refused (positional decode checked by the message)
    _, m, bad = _errors("*MATERIAL, NAME=B\n*ELASTIC, DEPENDENCIES=1\n200000., 0.3, , 0.\n160000., 0.3, , 1.\n")
    assert any(d.level == "gap" and "field-dependent" in d.message for d in bad)
    assert m.materials["B"].extra_params["refused_definitions"]
    # temperature together with a field variable is not represented
    _, _, bad = _errors("*MATERIAL, NAME=C\n*ELASTIC, DEPENDENCIES=1\n5., 0.3, 0., 0.\n5., 0.3, 10., 0.\n")
    assert bad


def test_elastic_missing_nu_is_input_error():
    reader, model, bad = _errors("*MATERIAL, NAME=A\n*ELASTIC\n200000.\n")
    assert not [d for d in bad if d.level == "error"] and model.materials["A"].elastic == (200000.0, 0.0)   # blank nu == 0


def test_elastic_types_not_isotropic_are_gaps():
    for t in ("LAMINA", "ORTHOTROPIC", "ANISOTROPIC", "TRACTION"):
        _, _, bad = _errors(f"*MATERIAL, NAME=A\n*ELASTIC, TYPE={t}\n1., 2., 3., 4., 5., 6.\n")
        assert any(d.level == "gap" for d in bad), t


def test_elastic_engineering_constants_two_lines():
    # G23 on the second line omitted (or that line blank) == 0
    _, m, bad = _errors("*MATERIAL, NAME=A\n*ELASTIC, TYPE=ENGINEERING CONSTANTS\n1., 2., 3., .3, .3, .3, 5., 6.\n")
    assert m.materials["A"].extra_params["engineering_constants"][-1] == 0.0


def test_expansion_layout():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*EXPANSION, ZERO=20.\n1.2e-5, 20.\n1.3e-5, 100.\n")
    assert not bad
    e = m.materials["A"].expansion
    assert e["table"] == [(1.2e-5, 20.), (1.3e-5, 100.)] and e["zero"] == 20.0
    _, _, bad = _errors("*MATERIAL, NAME=B\n*EXPANSION, TYPE=ORTHO\n1e-5, 2e-5, 3e-5\n")
    assert any(d.level == "gap" for d in bad)


def test_plastic_layout_and_unsupported_hardening():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*PLASTIC\n250., 0.\n300., 0.1\n")
    assert not bad and m.materials["A"].plastic == [(250., 0.), (300., 0.1)]
    # perfect plasticity: one line (yield, 0)
    _, m, bad = _errors("*MATERIAL, NAME=P\n*PLASTIC\n250., 0.\n")
    assert not bad and m.materials["P"].plastic == [(250., 0.)]
    # plastic strain blank == 0
    _, m, bad = _errors("*MATERIAL, NAME=Q\n*PLASTIC\n250.\n")
    assert not bad and m.materials["Q"].plastic == [(250., 0.)]
    # the first plastic strain must be zero (Guide)
    _, _, bad = _errors("*MATERIAL, NAME=R\n*PLASTIC\n250., 0.01\n300., 0.1\n")
    assert any(d.level == "error" for d in bad)
    # more than 10 points: the tabular kernel holds 10 (no silent truncation)
    pts = "".join(f"{250. + i}, {0.01 * i}\n" for i in range(11))
    _, _, bad = _errors("*MATERIAL, NAME=S\n*PLASTIC\n" + pts)
    assert any(d.level == "gap" for d in bad)
    # temperature / field dependent families would be flattened into one curve -> refused
    _, _, bad = _errors("*MATERIAL, NAME=T\n*PLASTIC\n250., 0., 20.\n200., 0., 100.\n")
    assert any(d.level == "gap" for d in bad)
    _, _, bad = _errors("*MATERIAL, NAME=F\n*PLASTIC, DEPENDENCIES=1\n250., 0., , 0.\n200., 0., , 1.\n")
    assert any(d.level == "gap" for d in bad)
    for h in ("KINEMATIC", "COMBINED"):
        _, _, bad = _errors(f"*MATERIAL, NAME=K\n*PLASTIC, HARDENING={h}\n250., 0.\n")
        assert any(d.level == "gap" for d in bad), h


def test_plastic_johnson_cook_items():
    _, m, bad = _errors("*MATERIAL, NAME=J\n*PLASTIC, HARDENING=JOHNSON COOK\n500., 400., 0.3, 1.0, 1500., 20.\n")
    assert any(d.level == "gap" for d in bad) and "johnson_cook" not in m.materials["J"].extra_params


# ─── Guide data-line layout audit: CREEP / HYPERELASTIC / VISCOELASTIC / RATE DEPENDENT ───

def test_creep_laws():
    # no solver material consumes a creep law: every LAW is a gap and the material is refused
    for law in ("TIME", "STRAIN", "HYPERB", "USER", "POWER LAW", "DARVEAUX"):
        _, m, bad = _errors(f"*MATERIAL, NAME=A\n*CREEP, LAW={law}\n1e-10, 0.01, 3.\n")
        assert any(d.level == "gap" for d in bad), law
        assert "creep" not in m.materials["A"].extra_params
        assert m.materials["A"].extra_params["refused_definitions"]


def test_creep_anand_two_lines():
    deck = ("*MATERIAL, NAME=AN\n*CREEP, LAW=ANAND\n"
            "10., 1e4, 1e6, 7., 0.2, 1e3, 20., 0.01\n1.5\n")
    _, m, bad = _errors(deck)
    assert any(d.level == "gap" for d in bad) and "creep" not in m.materials["AN"].extra_params


def test_hyperelastic_default_is_polynomial_n1():
    # no model flag: POLYNOMIAL, N=1 (C10, C01, D1) == Mooney-Rivlin, not Neo-Hooke
    _, m, bad = _errors("*MATERIAL, NAME=A\n*HYPERELASTIC\n0.3, 0.1, 0.02\n")
    h = m.materials["A"].hyperelastic
    assert not bad and h["model"] == "MOONEY_RIVLIN" and (h["c10"], h["c01"], h["d1"]) == (0.3, 0.1, 0.02)


def test_hyperelastic_layouts():
    _, m, bad = _errors("*MATERIAL, NAME=Y\n*HYPERELASTIC, YEOH\n0.5, -0.1, 0.02, 0.01, 0.002, 0.0005\n")
    h = m.materials["Y"].hyperelastic
    assert not bad and h["model"] == "YEOH"
    assert (h["c10"], h["c20"], h["c30"], h["d1"], h["d2"], h["d3"]) == (0.5, -0.1, 0.02, 0.01, 0.002, 0.0005)
    _, m, bad = _errors("*MATERIAL, NAME=P\n*HYPERELASTIC, POLYNOMIAL, N=2\n1., 2., 3., 4., 5., 6., 7.\n")
    h = m.materials["P"].hyperelastic
    assert not bad and h["model"] == "POLYNOMIAL" and (h["c20"], h["c11"], h["c02"], h["d1"], h["d2"]) == (3., 4., 5., 6., 7.)
    _, m, bad = _errors("*MATERIAL, NAME=R\n*HYPERELASTIC, REDUCED POLYNOMIAL, N=1\n0.4, 0.01\n")
    h = m.materials["R"].hyperelastic
    assert not bad and h["model"] == "NEO_HOOKEAN" and (h["c10"], h["d1"]) == (0.4, 0.01)
    _, m, bad = _errors("*MATERIAL, NAME=O\n*HYPERELASTIC, OGDEN, N=2\n1., 2., 3., -2., 0.01, 0.\n")
    h = m.materials["O"].hyperelastic
    assert not bad and h["mu_list"] == [1., 3.] and h["alpha_list"] == [2., -2.] and h["d_list"] == [0.01, 0.0]
    # D omitted / all D_i zero == fully incompressible in Abaqus: no incompressible path exists -> refused
    for deck in ("*HYPERELASTIC, NEO HOOKE\n0.4\n", "*HYPERELASTIC, YEOH\n0.5, -0.1, 0.02\n",
                 "*HYPERELASTIC, ARRUDA-BOYCE\n0.4, 3.\n", "*HYPERELASTIC, OGDEN, N=1\n1., 2.\n"):
        _, m, bad = _errors("*MATERIAL, NAME=E\n" + deck)
        assert any(d.level == "gap" and "incompressible" in d.message for d in bad), deck
        assert m.materials["E"].extra_params["refused_definitions"]
    # the coefficient itself has no default; TYPE is not a model selector; MARLOW needs test data
    _, _, bad = _errors("*MATERIAL, NAME=E\n*HYPERELASTIC, NEO HOOKE\n, 0.01\n")
    assert not [d for d in bad if d.level == "error"]
    # OGDEN: blank cells keep their place (the temperature must not land in D1); D2..DN != 0 refused
    _, m, bad = _errors("*MATERIAL, NAME=G\n*HYPERELASTIC, OGDEN, N=1\n0.5, 2.0, , 20.\n")
    assert any(d.level == "gap" for d in bad)                      # D1 blank == 0 (and a temperature)
    _, m, bad = _errors("*MATERIAL, NAME=G\n*HYPERELASTIC, OGDEN, N=2\n1., 2., 3., -2., 0.01, 0.02\n")
    assert any(d.level == "gap" and "D2" in d.message for d in bad)
    for hdr in ("*HYPERELASTIC, TYPE=NEO HOOKE", "*HYPERELASTIC, MARLOW"):
        _, _, bad = _errors(f"*MATERIAL, NAME=T\n{hdr}\n0.4, 0.01\n")
        assert any(d.level == "gap" for d in bad), hdr


def test_viscoelastic_requires_time_and_three_items():
    _, m, bad = _errors("*MATERIAL, NAME=V\n*VISCOELASTIC, TIME=PRONY\n0.2, 0.0, 3.3\n0.1, 0.0, 30.\n")
    assert not bad and m.materials["V"].viscoelastic_prony == [(0.2, 0.0, 3.3), (0.1, 0.0, 30.)]
    # k_i != 0 (bulk relaxation) is never packed -> refused
    _, _, bad = _errors("*MATERIAL, NAME=V\n*VISCOELASTIC, TIME=PRONY\n0.2, 0.1, 3.3\n")
    assert any(d.level == "gap" and "bulk" in d.message for d in bad)
    _, _, bad = _errors("*MATERIAL, NAME=V\n*VISCOELASTIC\n0.2, 0.0, 3.3\n")           # TIME required
    assert any(d.level == "error" for d in bad)
    _, _, bad = _errors("*MATERIAL, NAME=V\n*VISCOELASTIC, TIME=RELAXATION TEST DATA\n")
    assert any(d.level == "gap" for d in bad)


def test_rate_dependent_layout_and_dropped_effect():
    _, m, bad = _errors("*MATERIAL, NAME=R\n*PLASTIC\n250., 0.\n*RATE DEPENDENT, TYPE=JOHNSON COOK\n0.02, 1.\n")
    rd = m.materials["R"].extra_params["rate_dependent"]
    assert rd["C"] == 0.02 and rd["eps_dot_0"] == 1.0
    assert any(d.code == "rate-effect-dropped" for d in bad)   # no kernel consumes it
    _, _, bad = _errors("*MATERIAL, NAME=R\n*RATE DEPENDENT, TYPE=JOHNSON COOK\n0.02\n")
    assert not [d for d in bad if d.level == "error"]
    _, _, bad = _errors("*MATERIAL, NAME=R\n*RATE DEPENDENT, TYPE=CHABOCHE\n1., 2., 3.\n")
    assert any(d.level == "gap" for d in bad)


# ─── Guide data-line layout audit: DAMAGE INITIATION / EVOLUTION / CONCRETE ──

def test_damage_initiation_ductile_layouts():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*DAMAGE INITIATION, CRITERION=DUCTILE\n0.2\n")
    assert not bad and m.materials["A"].extra_params["damage_init"] == {"type": 0, "p": [0.2, 0, 0, 0, 0]}
    # Guide row is (eD, triaxiality); the kernel wants (eta0, e0, eta1, e1), ascending eta
    _, m, bad = _errors("*MATERIAL, NAME=B\n*DAMAGE INITIATION, CRITERION=DUCTILE\n0.1, 0.8\n0.3, 0.\n")
    assert not bad and m.materials["B"].extra_params["damage_init"] == {"type": 1, "p": [0.0, 0.3, 0.8, 0.1, 0]}
    # strain-rate item given: not represented; CRITERION is required
    _, _, bad = _errors("*MATERIAL, NAME=C\n*DAMAGE INITIATION, CRITERION=DUCTILE\n0.2, 0., 5.\n")
    assert any(d.level == "gap" for d in bad)
    _, _, bad = _errors("*MATERIAL, NAME=D\n*DAMAGE INITIATION\n0.2\n")
    assert any(d.level == "error" for d in bad)


def test_damage_initiation_johnson_cook_items():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*DAMAGE INITIATION, CRITERION=JOHNSON COOK\n"
                        "0.1, 0.2, -1.5, 0.01, 0.5, 1500., 20., 2.\n")
    di = m.materials["A"].extra_params["damage_init"]
    assert not bad and di["p"] == [0.1, 0.2, -1.5, 0.01, 0.5]
    assert (di["Tmelt"], di["Troom"], di["eps0"]) == (1500., 20., 2.)
    # d5 / d4 nonzero need theta_melt, theta_transition / eps0 (no Guide default)
    _, _, bad = _errors("*MATERIAL, NAME=B\n*DAMAGE INITIATION, CRITERION=JOHNSON COOK\n0.1, 0.2, -1.5, 0., 0.5\n")
    assert not [d for d in bad if d.level == "error"]
    _, _, bad = _errors("*MATERIAL, NAME=C\n*DAMAGE INITIATION, CRITERION=JOHNSON COOK\n0.1, 0.2, -1.5, 0.01\n")
    assert not [d for d in bad if d.level == "error"]


def test_damage_evolution_requirements():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*DAMAGE EVOLUTION, TYPE=ENERGY\n12.5\n")
    assert [d.code for d in bad] == ["damage-energy-regularization"]       # L fixed 1.0, u_f from sigma_y0
    assert m.materials["A"].extra_params["damage_evol"]["Gf"] == 12.5
    assert m.materials["A"].extra_params["damage_evol"]["type"] == 0       # SOFTENING default LINEAR
    _, m, bad = _errors("*MATERIAL, NAME=B\n*DAMAGE EVOLUTION, TYPE=ENERGY\n,\n")   # empty cell: Gf == 0
    assert m.materials["B"].extra_params["damage_evol"]["Gf"] == 0.0
    _, _, bad = _errors("*MATERIAL, NAME=C\n*DAMAGE EVOLUTION\n5.\n")               # TYPE required
    assert any(d.level == "error" for d in bad)
    # DEGRADATION=MAXIMUM is the default and is accepted; MULTIPLICATIVE is not
    _, _, bad = _errors("*MATERIAL, NAME=D\n*DAMAGE EVOLUTION, TYPE=ENERGY, DEGRADATION=MAXIMUM\n5.\n")
    assert [d.code for d in bad] == ["damage-energy-regularization"]
    _, _, bad = _errors("*MATERIAL, NAME=D\n*DAMAGE EVOLUTION, TYPE=ENERGY, DEGRADATION=MULTIPLICATIVE\n5.\n")
    assert any(d.code == "unsupported-value" for d in bad)


def test_cdp_tables_are_strain_value_pairs_and_psi_required():
    deck = ("*MATERIAL, NAME=C\n*ELASTIC\n30000., 0.2\n*CONCRETE DAMAGED PLASTICITY\n36.\n"
            "*CONCRETE COMPRESSION HARDENING\n15., 0.\n30., 0.001\n"
            "*CONCRETE TENSION STIFFENING\n3., 0.\n0.3, 0.001\n"
            "*CONCRETE TENSION DAMAGE\n0., 0.\n0.9, 0.001\n")
    _, m, bad = _errors(deck)
    cd = m.materials["C"].extra_params["concrete_dp"]
    # Guide lines are (stress, strain); the material kernel wants (strain, stress)
    assert cd["compression"] == [(0., 15.), (0.001, 30.)] and cd["tension"] == [(0., 3.), (0.001, 0.3)]
    assert cd["dt_table"] == [(0., 0.), (0.001, 0.9)]
    assert cd["dc_table"] == [(0., 0.)]           # no *CONCRETE COMPRESSION DAMAGE -> no damage
    assert any(d.code == "cdp-backbone-approximation" for d in bad)
    _, _, bad = _errors("*MATERIAL, NAME=D\n*CONCRETE DAMAGED PLASTICITY\n, 0.1\n")
    assert not [d for d in bad if d.level == "error"]
    _, _, bad = _errors("*MATERIAL, NAME=E\n*CONCRETE DAMAGED PLASTICITY\n36., 0.3\n")   # eccentricity 0.3 != default
    assert any(d.level == "gap" for d in bad)


# ─── Guide data-line layout audit: MULLINS / DENSITY / TRS / ORIENTATION ─────

def test_mullins_data_line_and_beta_default():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*MULLINS EFFECT\n2.5, 0.6\n")
    mu = m.materials["A"].extra_params["mullins"]
    assert (mu["r"], mu["m"], mu["beta"]) == (2.5, 0.6, 0.0)       # blank beta: 0.0 (Abaqus/Standard)
    assert any(d.code == "mullins-not-connected" for d in bad)     # no kernel consumes it
    _, _, bad = _errors("*MATERIAL, NAME=B\n*MULLINS EFFECT\n2.5\n")
    assert not [d for d in bad if d.level == "error"]
    # R / M / BETA parameters belong to TEST DATA INPUT
    _, _, bad = _errors("*MATERIAL, NAME=C\n*MULLINS EFFECT, R=2.5\n2.5, 0.6, 0.1\n")
    assert any(d.level == "gap" for d in bad)


def test_density_single_value():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*DENSITY\n7.8e-9\n")
    assert not bad and m.materials["A"].density == 7.8e-9
    _, _, bad = _errors("*MATERIAL, NAME=B\n*DENSITY\n7.8e-9, 20.\n7.7e-9, 100.\n")
    assert any(d.level == "gap" for d in bad)                      # temperature table not represented


def test_trs_wlf_line():
    _, m, bad = _errors("*MATERIAL, NAME=A\n*VISCOELASTIC, TIME=PRONY\n0.2, 0.1, 3.\n*TRS, DEFINITION=WLF\n20., 17.4, 51.6\n")
    assert m.materials["A"].extra_params["trs"] == {"definition": "WLF", "T_ref": 20., "C1": 17.4, "C2": 51.6}
    assert any(d.code == "trs-not-connected" for d in bad)
    _, _, bad = _errors("*MATERIAL, NAME=B\n*TRS, DEFINITION=ARRHENIUS\n20., 1e5\n")
    assert any(d.level == "gap" for d in bad)
    _, _, bad = _errors("*MATERIAL, NAME=C\n*TRS\n20., 17.4\n")
    assert not [d for d in bad if d.level == "error"]


def test_orientation_origin_and_rotation():
    import numpy as np
    # point a on axis 1, b in the 1-2 plane, origin c = (1, 1, 0): vectors are taken relative to c
    _, m, bad = _errors("*ORIENTATION, NAME=O\n2., 1., 0., 1., 2., 0., 1., 1., 0.\n")
    assert not [d for d in bad if d.level == "error"]
    assert np.allclose(m.orientations["O"].frame, np.eye(3))
    # additional rotation alpha about a local direction turns the frame (right-hand rule): 30 deg about axis 1
    _, m, bad = _errors("*ORIENTATION, NAME=P\n1., 0., 0., 0., 1., 0.\n1, 30.\n")
    assert not [d for d in bad if d.level in ("error", "gap")]
    c, s30 = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    assert np.allclose(m.orientations["P"].frame, [[1, 0, 0], [0, c, s30], [0, -s30, c]])
    # ... but an angle without the direction it turns about is an input error, not a guess
    _, _, bad = _errors("*ORIENTATION, NAME=Q\n1., 0., 0., 0., 1., 0.\n, 30.\n")
    assert any(d.level == "error" for d in bad)
    # NODES: only the first line holds node numbers (the second line is direction, alpha)
    _, m, bad = _errors("*ORIENTATION, NAME=N, DEFINITION=NODES\n1, 2, 3\n1, 0.\n")
    assert m.orientations["N"].node_ids == (1, 2, 3)
    # Z RECTANGULAR is an Abaqus system; ZCYLINDRICAL is not
    _, m, bad = _errors("*ORIENTATION, NAME=Z, SYSTEM=Z RECTANGULAR\n1., 0., 0., 0., 1., 0.\n")
    assert not [d for d in bad if d.level == "error"]


# ─── Abaqus Input Syntax Rules: blank / omitted numeric data items are zero ──

def test_blank_numeric_items_are_zero_across_keywords():
    # "Abaqus will use values of zero for any required numeric data that are omitted unless a
    # default value is specified"
    _, m, bad = _errors("*MATERIAL, NAME=A\n*ELASTIC\n1.E8,\n")                       # nu
    assert not bad and m.materials["A"].elastic == (1.0e8, 0.0)
    _, m, bad = _errors("*MATERIAL, NAME=B\n*VISCOELASTIC, TIME=PRONY\n0.5,,10.\n")    # k
    assert not bad and m.materials["B"].viscoelastic_prony == [(0.5, 0.0, 10.)]
    _, m, bad = _errors("*MATERIAL, NAME=D\n*DRUCKER PRAGER\n30.\n*DRUCKER PRAGER HARDENING\n10.\n")
    dp = m.materials["D"].extra_params["drucker_prager"]
    assert not [x for x in bad if x.level == "error"] and dp["psi"] == 0.0 and dp["H"] == 0.0
    _, m, bad = _errors("*MATERIAL, NAME=E\n*CONCRETE DAMAGED PLASTICITY\n,\n*CONCRETE COMPRESSION HARDENING\n15.\n"
                        "*CONCRETE TENSION STIFFENING\n3.\n")
    assert m.materials["E"].extra_params["concrete_dp"]["psi"] == 0.0
    # items with their own documented default keep it
    _, m, bad = _errors("*MATERIAL, NAME=F\n*MULLINS EFFECT\n2.5, 0.6\n")
    assert m.materials["F"].extra_params["mullins"]["beta"] == 0.0                     # Standard default
    _, _, bad = _errors("*MATERIAL, NAME=G\n*DRUCKER PRAGER\n30., 0., 10.\n*DRUCKER PRAGER HARDENING\n10., 0.\n")
    assert not [x for x in bad if x.level in ("error", "gap")]                         # K == 0 -> 1.0


def test_schema_numeric_row_field_blank_is_zero_but_str_still_raises():
    from dispsolver.io.schema import RowSchema, parse_rows, InputError

    class R(RowSchema):
        name: str
        a: float
        n: int

    rows = parse_rows("X", R, ["ab, 1.5", "cd,,3", "ef"], [None] * 3)
    assert [(r.name, r.a, r.n) for r in rows] == [("ab", 1.5, 0), ("cd", 0.0, 3), ("ef", 0.0, 0)]
    with pytest.raises(InputError):
        parse_rows("X", R, [",1.5,2"], [None])        # str item blank: still missing


def test_distribution_name_data_line_is_a_gap_not_a_parse_error():
    for deck in ("*MATERIAL, NAME=A\n*ELASTIC\ndistElastic\n", "*MATERIAL, NAME=A\n*DENSITY\ndistRho\n",
                 "*MATERIAL, NAME=A\n*EXPANSION\ndistAlpha\n", "*ORIENTATION, NAME=O\ndistOrient\n"):
        _, _, bad = _errors(deck)
        assert [d.level for d in bad] == ["gap"] and "distribution" in bad[0].message, deck


def test_refused_definition_is_named_when_nothing_is_left_to_compute():
    import warnings
    from dispsolver.material.resolve import resolve_material
    _, m, _ = _errors("*MATERIAL, NAME=A1\n*HYPERELASTIC, MARLOW\n0.4, 0.01\n")
    mat = m.materials["A1"]
    assert "*HYPERELASTIC, MARLOW" in mat.extra_params["refused_definitions"][0]
    with pytest.raises(NotImplementedError) as ei:
        resolve_material(mat, 3, "material 'A1'")
    msg = str(ei.value)
    assert "definition was refused" in msg and "*HYPERELASTIC, MARLOW" in msg and "nothing to compute" in msg


def test_material_with_a_refused_definition_is_refused_whatever_else_it_has():
    from dispsolver.material.resolve import resolve_material
    _, m, _ = _errors("*MATERIAL, NAME=K\n*ELASTIC\n1000., 0.3\n*PLASTIC, HARDENING=KINEMATIC\n10., 0.\n")
    with pytest.raises(NotImplementedError) as ei:
        resolve_material(m.materials["K"], 3, "material of pid 0")
    assert "HARDENING=KINEMATIC" in str(ei.value) and "different model" in str(ei.value)


def test_non_finite_packed_props_are_refused():
    import numpy as np
    from dispsolver.material.resolve import resolve_material
    _, m, _ = _errors("*MATERIAL, NAME=N\n*ELASTIC\n1000., 0.3\n")
    m.materials["N"].elastic = (float("nan"), 0.3)
    with pytest.raises(NotImplementedError, match="not finite"):
        resolve_material(m.materials["N"], 3, "material of pid 0")
    # a multilinear Drucker-Prager hardening is refused by the reader before any NaN exists
    deck = ("*MATERIAL, NAME=D\n*ELASTIC\n3e5, 0.3\n*DRUCKER PRAGER\n40., 1., 20.\n"
            "*DRUCKER PRAGER HARDENING\n10., 0.\n12., 0.01\n14., 0.02\n")
    _, m, bad = _errors(deck)
    assert any(d.code == "dp-hardening-multilinear" for d in bad)
    with pytest.raises(NotImplementedError, match="definition was refused"):
        resolve_material(m.materials["D"], 3, "material of pid 0")




# ─── Reviewer round: CDP packing, applier refusals, MODULI, Prony, combined definitions ───

_CDP_DECK = ("*MATERIAL, NAME=C\n*ELASTIC\n30000., 0.2\n*CONCRETE DAMAGED PLASTICITY\n36.\n"
             "*CONCRETE COMPRESSION HARDENING\n15., 0.\n30., 0.001\n20., 0.004\n"
             "*CONCRETE TENSION STIFFENING\n3., 0.\n0.3, 0.001\n"
             "*CONCRETE COMPRESSION DAMAGE\n0., 0.\n0.5, 0.004\n"
             "*CONCRETE TENSION DAMAGE\n0., 0.\n0.9, 0.001\n")


def test_cdp_packs_every_slot_from_the_deck():
    from dispsolver.material.resolve import resolve_material
    _, m, bad = _errors(_CDP_DECK)
    mid, p = resolve_material(m.materials["C"], 3, "material of pid 0")
    assert mid == MAT_CONCRETE_DP
    # [E, nu, beta, psi, nc, (ec, sig)*nc, nt, (et, sig)*nt, ndt, (et, dt)*ndt, ndc, (ec, dc)*ndc]
    assert list(p[:4]) == [30000., 0.2, 20.0, 36.0]        # psi from the deck, beta = the documented backbone 20 deg
    assert list(p[4:11]) == [3, 0., 15., 0.001, 30., 0.004, 20.]
    assert list(p[11:16]) == [2, 0., 3., 0.001, 0.3]
    assert list(p[16:21]) == [2, 0., 0., 0.001, 0.9]
    assert list(p[21:26]) == [2, 0., 0., 0.004, 0.5]
    assert p[26] == 0.0


def test_cdp_applier_refusals_are_recorded_on_the_material():
    # 7-point table: ConcreteDP rejects it
    pts = "".join(f"{15. + i}, {0.001 * i}\n" for i in range(7))
    deck = _CDP_DECK.replace("15., 0.\n30., 0.001\n20., 0.004\n", pts)
    reader, m, bad = _errors(deck)
    assert any("table rejected" in r for r in m.materials["C"].extra_params["refused_definitions"])
    # compression hardening without tension stiffening
    deck = ("*MATERIAL, NAME=C\n*ELASTIC\n30000., 0.2\n*CONCRETE DAMAGED PLASTICITY\n36.\n"
            "*CONCRETE COMPRESSION HARDENING\n15., 0.\n")
    _, m, bad = _errors(deck)
    assert any("TENSION STIFFENING" in r for r in m.materials["C"].extra_params["refused_definitions"])
    assert "concrete_dp" not in m.materials["C"].extra_params
    # tables without *CONCRETE DAMAGED PLASTICITY
    _, m, bad = _errors("*MATERIAL, NAME=C\n*ELASTIC\n30000., 0.2\n*CONCRETE TENSION STIFFENING\n3., 0.\n")
    assert m.materials["C"].extra_params["refused_definitions"]


def test_dp_refusals_are_recorded_and_order_free():
    _, m, bad = _errors("*MATERIAL, NAME=S\n*ELASTIC\n5e4, 0.3\n*DRUCKER PRAGER\n30., 1., 10.\n"
                        "*DRUCKER PRAGER HARDENING\n10., 0.\n12., 0.\n")
    ex = m.materials["S"].extra_params
    assert any("strain must increase" in r for r in ex["refused_definitions"])
    assert [d.code for d in bad].count("dp-hardening-strain") == 1       # reported once, no second error
    assert "drucker_prager" not in ex


def test_moduli_long_term_scales_the_shear_constants():
    from dispsolver.material.resolve import resolve_material
    deck = ("*MATERIAL, NAME=H\n*HYPERELASTIC, ARRUDA-BOYCE{mod}\n0.4, 3., 0.1\n"
            "*VISCOELASTIC, TIME=PRONY\n0.2, 0., 3.\n0.1, 0., 30.\n")
    _, m, bad = _errors(deck.format(mod=""))
    mid, p = resolve_material(m.materials["H"], 3, "material of pid 0")
    assert mid == MAT_HYPER_VISCOELASTIC if "MAT_HYPER_VISCOELASTIC" in globals() else mid == 6
    assert p[1] == pytest.approx(0.4 / (1.0 - 0.3)) and p[2] == 3.0       # mu scaled, lambda_m / K untouched
    _, m, _ = _errors(deck.format(mod=", MODULI=INSTANTANEOUS"))
    _, p2 = resolve_material(m.materials["H"], 3, "material of pid 0")
    assert p2[1] == pytest.approx(0.4)
    # explicit MODULI=LONG TERM is the default and accepted
    _, m, bad = _errors(deck.format(mod=", MODULI=LONG TERM"))
    assert not [d for d in bad if d.level in ("gap", "error")]


def test_prony_capacity_and_bulk_relaxation_are_refused():
    from dispsolver.material.resolve import resolve_material
    two = "*MATERIAL, NAME=P\n*ELASTIC\n100., 0.4\n*VISCOELASTIC, TIME=PRONY\n0.2, 0., 1.\n0.1, 0., 10.\n"
    _, m, bad = _errors(two)
    with pytest.raises(NotImplementedError, match="2 Prony terms, the kernel packs 1"):
        resolve_material(m.materials["P"], 3, "material of pid 0")
    seven = ("*MATERIAL, NAME=Q\n*HYPERELASTIC, NEO HOOKE\n0.4, 0.1\n*VISCOELASTIC, TIME=PRONY\n"
             + "".join(f"0.01, 0., {i + 1}.\n" for i in range(7)))
    _, m, bad = _errors(seven)
    with pytest.raises(NotImplementedError, match="7 Prony terms, the kernel packs 6"):
        resolve_material(m.materials["Q"], 3, "material of pid 0")


def test_combined_constitutive_definitions_are_refused_whatever_the_order():
    from dispsolver.material.resolve import resolve_material
    for order in (("*PLASTIC\n250., 0.\n", "*HYPERELASTIC, NEO HOOKE\n0.4, 0.1\n"),
                  ("*HYPERELASTIC, NEO HOOKE\n0.4, 0.1\n", "*PLASTIC\n250., 0.\n")):
        _, m, _ = _errors("*MATERIAL, NAME=X\n*ELASTIC\n1e3, 0.3\n" + "".join(order))
        with pytest.raises(NotImplementedError, match="does not consume"):
            resolve_material(m.materials["X"], 3, "material of pid 0")
    # TRS / MULLINS next to a Prony series are stored but consumed by no id
    _, m, _ = _errors("*MATERIAL, NAME=T\n*ELASTIC\n1e3, 0.3\n*VISCOELASTIC, TIME=PRONY\n0.2, 0., 1.\n"
                      "*TRS\n20., 17.4, 51.6\n")
    with pytest.raises(NotImplementedError, match=r"\*TRS"):
        resolve_material(m.materials["T"], 3, "material of pid 0")


def test_ductile_damage_with_a_long_plastic_table_is_refused():
    from dispsolver.material.resolve import resolve_material
    deck = ("*MATERIAL, NAME=D\n*ELASTIC\n7e4, 0.3\n*PLASTIC\n200., 0.\n250., 0.05\n280., 0.1\n"
            "*DAMAGE INITIATION, CRITERION=DUCTILE\n0.3\n*DAMAGE EVOLUTION, TYPE=ENERGY\n10.\n")
    _, m, _ = _errors(deck)
    with pytest.raises(NotImplementedError, match="extrapolates linearly"):
        resolve_material(m.materials["D"], 3, "material of pid 0")
    # initiation without evolution (valid Abaqus: no softening) is a clear refusal, not a ValueError
    _, m, _ = _errors("*MATERIAL, NAME=D\n*ELASTIC\n7e4, 0.3\n*PLASTIC\n200., 0.\n"
                      "*DAMAGE INITIATION, CRITERION=DUCTILE\n0.3\n")
    with pytest.raises(NotImplementedError, match="initiation alone"):
        m.materials["D"].to_numba_props()


def test_arruda_boyce_nonpositive_lambda_m_is_refused():
    from dispsolver.material.resolve import resolve_material
    _, m, _ = _errors("*MATERIAL, NAME=A\n*HYPERELASTIC, ARRUDA-BOYCE\n0.4, , 0.1\n")
    with pytest.raises(NotImplementedError, match="lambda_m"):
        resolve_material(m.materials["A"], 3, "material of pid 0")


def test_section_orientation_on_an_anisotropic_material_is_refused_only_when_the_frame_varies():
    from dispsolver.material.resolve import resolve_material
    deck = ("*NODE\n1, 0, 0, 0\n*ELEMENT, TYPE=C3D8, ELSET=E\n1, 1, 1, 1, 1, 1, 1, 1, 1\n"
            "*ORIENTATION, NAME=O\n1., 0., 0., 0., 1., 0.\n*MATERIAL, NAME=ORTHO\n"
            "*ELASTIC, TYPE=ENGINEERING CONSTANTS\n10., 2., 2., .3, .3, .3, 1., 1.\n1.\n"
            "*MATERIAL, NAME=ISO\n*ELASTIC\n10., .3\n"
            "*SOLID SECTION, ELSET=E, MATERIAL=ORTHO, ORIENTATION=O\n")
    # a RECTANGULAR orientation is a fixed frame: the assembly rotates the stiffness into it, nothing to refuse
    reader, model = _read_snippet(deck)
    assert not any(d.code == "orientation-not-connected" for d in reader.diagnostics)
    # a frame that varies by element (cylindrical) is still refused, with the reason
    reader, model = _read_snippet(deck.replace("*ORIENTATION, NAME=O\n", "*ORIENTATION, NAME=O, SYSTEM=CYLINDRICAL\n"))
    assert any(d.code == "orientation-not-connected" and d.level == "gap" and "CYLINDRICAL" in d.message
               for d in reader.diagnostics)
    assert model.materials["ORTHO"].extra_params["refused_definitions"]
    deck_iso = deck.replace("MATERIAL=ORTHO", "MATERIAL=ISO")
    reader, model = _read_snippet(deck_iso)
    assert not any(d.code == "orientation-not-connected" for d in reader.diagnostics)
    assert any(d.code == "orientation-ignored-isotropic" and d.level == "info" for d in reader.diagnostics)


def test_blank_dload_dsload_magnitude_is_zero():
    deck = ("*NODE\n1, 0, 0, 0\n*STEP\n*STATIC\n*DLOAD\nE, P1\n*END STEP\n")
    reader, _ = _read_snippet(deck)
    assert not [d for d in reader.diagnostics if d.level == "error" and "magnitude" in d.message]


# -- opus re-review: the last two silent paths ------------------------------

def _resolve_first(deck):
    from dispsolver.material.resolve import resolve_material
    reader, model = _read_snippet(deck)
    mat = next(iter(model.materials.values()))
    return reader, mat, lambda: resolve_material(mat, 3, "probe")


def test_ductile_damage_with_two_point_plastic_is_refused():
    # the damage kernel hardens linearly without bound; Abaqus holds 300 after eps=0.1
    _, _, run = _resolve_first("*MATERIAL, NAME=M\n*ELASTIC\n200000., 0.3\n*PLASTIC\n200., 0.\n300., 0.1\n"
                               "*DAMAGE INITIATION, CRITERION=DUCTILE\n0.3\n*DAMAGE EVOLUTION, TYPE=ENERGY\n10.\n")
    with pytest.raises(NotImplementedError, match="extrapolates linearly"):
        run()


def test_field_dependent_elastic_is_refused():
    # deck field variables never reach the solver: the first row (E=1000) would always run
    reader, _ = _read_snippet("*MATERIAL, NAME=M\n*ELASTIC, DEPENDENCIES=1\n1000., 0.3, , -1.\n500., 0.3, , 1.\n")
    assert any("field-dependent moduli" in d.message for d in reader.diagnostics)


def test_elastic_with_hyperelastic_is_refused():
    _, _, run = _resolve_first("*MATERIAL, NAME=M\n*ELASTIC\n1000., 0.3\n*HYPERELASTIC, NEO HOOKE\n0.5, 0.01\n")
    with pytest.raises(NotImplementedError, match=r"together with \*HYPERELASTIC"):
        run()


def test_creep_validates_its_data_before_refusing():
    reader, _ = _read_snippet("*MATERIAL, NAME=M\n*ELASTIC\n1000., 0.3\n*CREEP, LAW=ANAND\n1.,2.,3.\n")
    assert any("ANAND needs two data lines" in d.message for d in reader.diagnostics)   # input error first



def test_ductile_damage_with_perfect_plasticity_resolves():
    # one yield stress (single point, or a flat table) is perfect plasticity: the kernel's H is 0 there
    tail = "*DAMAGE INITIATION, CRITERION=DUCTILE\n0.3\n*DAMAGE EVOLUTION, TYPE=ENERGY\n10.\n"
    for plastic in ("200., 0.\n", "200., 0.\n200., 0.1\n"):
        _, _, run = _resolve_first("*MATERIAL, NAME=M\n*ELASTIC\n200000., 0.3\n*PLASTIC\n" + plastic + tail)
        res = run()
        props = res[1] if isinstance(res, tuple) else res.props
        assert props[2] == 200.0 and props[3] == 0.0, plastic
