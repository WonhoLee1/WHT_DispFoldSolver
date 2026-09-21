"""SIMAINPRefResources verification-deck reading checks.

Decks in benchmark_inp/verification/ were copied from the local
SIMAINPRefResources archive named by the SIMACAEVERRefMap pages.
Reader coverage only; material constitutive implementation belongs
to the material team (unsupported material keywords below are their
honest backlog, not reader defects).
"""
import os

import pytest

from dispsolver.io.abaqus_reader import _parse_source, _AbortParse

VERIF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "benchmark_inp", "verification")
OFFICIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "benchmark_inp", "official")


def _read(path, dim=3):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parse_source(path, dim=dim)


def test_jcrateplasticuni_reads():
    reader, model = _read(os.path.join(VERIF, "jcrateplasticuni.inp"))
    assert sum(len(p.nodes) for p in model.parts.values()) == 8
    assert "MAT" in model.materials
    rd = model.materials["MAT"].extra_params.get("rate_dependent", {})
    assert rd.get("type") == "JOHNSON_COOK"
    assert "RATE DEPENDENT" not in reader.unsupported_keywords


def test_concrete_damage_reads_structure():
    reader, model = _read(os.path.join(VERIF, "cdpse_cpe4r.inp"))
    assert sum(len(p.nodes) for p in model.parts.values()) == 4
    mat = model.materials["alle"]
    for key in ["cdp_compression", "cdp_tension", "cdp_dc", "cdp_dt"]:
        assert key in mat.extra_params
    assert any("CONCRETE_DAMAGED_PLASTICITY" in kw
               for kw in reader.unsupported_keywords)


def test_hyperelastic_model_mapping():
    deck = """*HEADING
*MATERIAL, NAME=M1A
*HYPERELASTIC, OGDEN, N=2
0.1, 2.0, 0.0, 0.2, -2.0, 0.0
*MATERIAL, NAME=M1B
*HYPERELASTIC, MOONEY-RIVLIN
0.3, 0.1, 0.02
"""
    from dispsolver.io.abaqus_reader import read_inp
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = read_inp(deck, dim=3)
    ha = model.materials["M1A"].hyperelastic or {}
    assert ha.get("model") == "OGDEN"
    assert ha.get("n") == 2
    hb = model.materials["M1B"].hyperelastic or {}
    assert hb.get("model") == "MOONEY_RIVLIN"
    assert hb.get("c10") == 0.3


def test_damage_and_jc_plastic_landings():
    deck = """*HEADING
*MATERIAL, NAME=M2
*ELASTIC
210000., 0.3
*PLASTIC, HARDENING=JOHNSON COOK
500., 300., 0.4, 1.0, 1793., 293.
*DAMAGE INITIATION, CRITERION=DUCTILE
0.3
*DAMAGE EVOLUTION, TYPE=ENERGY, SOFTENING=EXPONENTIAL
10.0
*DRUCKER PRAGER, ANGLE=25.0
100.0
"""
    from dispsolver.io.abaqus_reader import read_inp
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = read_inp(deck, dim=3)
    mat = model.materials["M2"]
    assert mat.mat_type == "DRUCKER_PRAGER"
    assert mat.extra_params["johnson_cook"]["A"] == 500.0
    assert mat.extra_params["damage_init"]["type"] == 0
    assert mat.extra_params["damage_evol"] == {"type": 1, "Gf": 10.0, "L": 1.0}


def test_initial_conditions_velocity_field_stored():
    deck = """*HEADING
*PART, NAME=P1
*NODE, NSET=ALLN
1, 0., 0., 0.
*END PART
*INITIAL CONDITIONS, TYPE=VELOCITY
ALLN, 1.0, 0.0, 0.0
*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=1
ALLN, 0.5
"""
    from dispsolver.io.abaqus_reader import _parse_source as _ps
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader, model = _ps(deck, dim=3)
    types = [pf.field_type for pf in model.initial_step.predefined_fields.values()]
    assert "VELOCITY" in types
    assert "FIELD" in types
    assert any("solver consumption not connected" in w for w in reader.warnings)


def test_honest_aborts():
    cases = [
        (os.path.join(VERIF, "heplmu_yeoh_cgax4rh.inp"), "INCLUDE"),
        (os.path.join(VERIF, "x_heplmu_yeoh_cax4r.inp"), "INCLUDE"),
    ]
    for path, needle in cases:
        with pytest.raises(_AbortParse) as exc:
            _read(path)
        assert needle in str(exc.value), path


def test_transform_landing_stored():
    reader, model = _read(os.path.join(VERIF, "viscocylinder_cpe4i.inp"))
    assert "TRF_ALL_1" in model.transforms
    spec = model.transforms["TRF_ALL_1"]
    assert spec["type"] == "C"
    assert spec["nset"] == "ALL"
    assert "TRANSFORM" not in reader.unsupported_keywords
    assert any("solver-frame enforcement not connected" in w
               for w in reader.warnings)


def test_include_expansion_lands():
    reader, model = _read(os.path.join(VERIF, "temp_elastic.inp"))
    assert "INCLUDE" not in reader.unsupported_keywords
    assert len(model.materials) >= 1


def test_orientation_lands_and_wires_section():
    reader, model = _read(os.path.join(VERIF, "clayorthelaspoten3d.inp"))
    assert "RECT" in model.orientations
    ori = model.orientations["RECT"]
    assert ori.system == "RECTANGULAR"
    assert ori.frame is not None
    import numpy as np
    assert np.allclose(ori.frame, np.eye(3))
    assert "ORIENTATION" not in reader.unsupported_keywords


def test_orientation_nodes_form_lands():
    reader, model = _read(os.path.join(VERIF, "mhecoo2eis.inp"))
    assert "OR1" in model.orientations
    assert model.orientations["OR1"].definition == "NODES"
    assert len(model.orientations["OR1"].node_ids) >= 2


def test_equation_lands_binds_assembles():
    import numpy as np
    reader, model = _read(os.path.join(VERIF, "marlow_volumecomp.inp"))
    from dispsolver.model.constraint import EquationConstraint
    eqs = [c for c in model.constraints.values()
           if isinstance(c, EquationConstraint)]
    assert len(eqs) == 3
    sys = model.build_solver_system()
    bound = [c for c in sys.constraints if isinstance(c, EquationConstraint)]
    assert all(c._bound for c in bound)
    u = np.zeros(sys.coords.shape[0] * sys.dim)
    for c in bound:
        f, (rr, cc, dd), err = c.assemble(u)
        assert err is False
        assert float(np.linalg.norm(f)) == 0.0
        assert len(dd) == len(c._dofs) ** 2
    assert "EQUATION" not in reader.unsupported_keywords
