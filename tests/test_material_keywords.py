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
10000.0, 2000.0, 2000.0, 0.3, 0.3, 0.3, 500.0, 500.0, 500.0
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["COMPOSITE"]
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
    """*PLASTIC isotropic hardening maps to MAT_J2_PLASTICITY."""
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
    mid = material_id(mat)
    assert mid == MAT_J2_PLASTICITY
    props = pack_props(mat, dim=3)
    assert props[0] == pytest.approx(70000.0)
    assert props[1] == pytest.approx(0.33)
    assert props[2] == pytest.approx(200.0)  # initial yield stress


def test_plastic_johnson_cook():
    """*PLASTIC, HARDENING=JOHNSON COOK parses parameters and reports gap."""
    deck = """
*MATERIAL, NAME=JC_STEEL
*ELASTIC
200000.0, 0.3
*PLASTIC, HARDENING=JOHNSON COOK
300.0, 400.0, 0.5, 0.0, 1800.0, 293.0
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["JC_STEEL"]
    assert "johnson_cook" in mat.extra_params
    jc = mat.extra_params["johnson_cook"]
    assert jc["A"] == 300.0
    assert jc["B"] == 400.0
    assert any(d.code == "rate-effect-dropped" for d in reader.diagnostics)


def test_creep_strain():
    """*CREEP, LAW=STRAIN parses Norton creep parameters."""
    deck = """
*MATERIAL, NAME=CREEP_STEEL
*ELASTIC
150000.0, 0.3
*CREEP, LAW=STRAIN
1.0e-10, 5.0, -0.5
"""
    reader, model = _read_snippet(deck)
    mat = model.materials["CREEP_STEEL"]
    assert "creep" in mat.extra_params
    assert mat.extra_params["creep"]["law"] == "STRAIN"
    assert mat.extra_params["creep"]["A"] == pytest.approx(1.0e-10)
    assert mat.extra_params["creep"]["n"] == pytest.approx(5.0)


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
    assert props[0] == pytest.approx(100.0)
    assert props[1] == pytest.approx(0.4)


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
    assert not reader.diagnostics.of("error", "gap")


def test_drucker_prager_hardening_types():
    import math
    tb = math.tan(math.radians(30.0))
    for typ, f in (("TENSION", 1.0 + tb / 3.0), ("SHEAR", math.sqrt(3.0))):
        _, model = _read_snippet(_dp_deck("30.0, 0.0, 10.0",
                                          f"*DRUCKER PRAGER HARDENING, TYPE={typ}\n10.0, 0.0\n"))
        dp = model.materials["SOIL"].extra_params["drucker_prager"]
        assert dp["d0"] == pytest.approx(f * 10.0)
        assert dp["H"] == 0.0 and dp["psi"] == 10.0 and dp["beta"] == 30.0


def test_drucker_prager_missing_items_and_hardening():
    # beta / psi have no Guide default -> input error
    for line in ("", ", 1.0, 10.0", "30.0, 1.0"):
        reader, _ = _read_snippet(_dp_deck(line))
        assert reader.diagnostics.of("error"), line
    # K != 1 is a gap (Material has no K)
    reader, _ = _read_snippet(_dp_deck("30.0, 0.8, 10.0"))
    assert reader.diagnostics.of("gap")
    # no hardening -> refused, naming the material
    reader, _ = _read_snippet(_dp_deck(hard=""))
    msgs = [d.message for d in reader.diagnostics.of("error")]
    assert any("SOIL" in m and "HARDENING" in m for m in msgs)


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
*MULLINS EFFECT, R=2.5, M=0.6, BETA=0.15
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
