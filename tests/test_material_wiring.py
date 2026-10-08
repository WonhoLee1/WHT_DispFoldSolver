"""
tests/test_material_wiring.py
=============================
A5 (material half): the 2D and 3D solvers get every material id and props
vector from dispsolver/material/catalog.py (via dispsolver/material/resolve.py).

(a) every material type the model/reader can produce resolves to a catalog
    MaterialSpec whose dims include the solver dimension, or raises
    NotImplementedError -- never a silent linear-elastic id (audit O7);
(b) a static source scan: no solver / model module assigns material ids from
    material types any more (the patterns A5 removed stay removed);
(c) 2D and 3D give the same id and props for the same material, AND those
    are the values the catalog / kernel layouts say (not just equal to each other);
(d) a Material object's id is the catalog's id; a dict whose explicit integer
    mat_type disagrees with the catalog raises;
(e) the 296 distinct materials of the benchmark_inp decks get the id/props the
    pre-A5 chain (9f5e6e9) gave them, except an explicit, reasoned list.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from dispsolver.material.catalog import MATERIALS, material_id, pack_props
from dispsolver.material.resolve import resolve_material
from dispsolver.material3d.numba_materials import (
    MAT_LINEAR_ELASTIC,
    MAT_HYPERELASTIC_NEOHOOKEAN,
    MAT_J2_PLASTICITY,
)
from dispsolver.model.material import Material

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "dispsolver"


def _mat(mat_type=None, extra=None, **kw):
    m = Material(name="M", **kw)
    if mat_type:
        m.mat_type = mat_type
    m.extra_params.update(extra or {})
    return m


# every mat_type string model/material.py can assign, with a minimal valid
# Material; None = no catalog model exists (must raise NotImplementedError).
_NH = {"model": "NEOHOOKEAN", "c10": 0.2, "d1": 0.01}
_CASES = {
    "ELASTIC": (lambda: _mat(elastic=(2e5, 0.3)), MAT_LINEAR_ELASTIC),
    "J2_PLASTICITY": (lambda: _mat("J2_PLASTICITY", elastic=(2e5, 0.3), plastic=[(250.0, 0.0), (350.0, 0.1)]), MAT_J2_PLASTICITY),
    "TABULAR_PLASTIC": (lambda: _mat("TABULAR_PLASTIC", elastic=(2e5, 0.3), extra={"tabular_plastic": [(250.0, 0.0), (350.0, 0.1)]}), 12),
    "NEO_HOOKEAN": (lambda: _mat("NEO_HOOKEAN", hyperelastic=dict(_NH)), MAT_HYPERELASTIC_NEOHOOKEAN),
    "HYPERELASTIC": (lambda: _mat("HYPERELASTIC", hyperelastic=dict(_NH)), MAT_HYPERELASTIC_NEOHOOKEAN),
    "MOONEY_RIVLIN": (lambda: _mat("MOONEY_RIVLIN", hyperelastic={"model": "MOONEY_RIVLIN", "c10": 0.1, "c01": 0.05, "d1": 0.01}), 8),
    "OGDEN": (lambda: _mat("OGDEN", hyperelastic={"model": "OGDEN", "mu_list": [0.2], "alpha_list": [2.0], "d_list": [0.01]}), 9),
    "POLYNOMIAL": (lambda: _mat("POLYNOMIAL", hyperelastic={"model": "POLYNOMIAL", "c10": 0.1, "d1": 0.01}), 10),
    "ARRUDA_BOYCE": (lambda: _mat("ARRUDA_BOYCE", hyperelastic={"model": "ARRUDA_BOYCE", "mu": 0.2, "lambda_m": 3.0, "K": 10.0}), 4),
    "YEOH": (lambda: _mat("YEOH", hyperelastic={"model": "YEOH", "c10": 0.1, "d1": 0.01}), 5),
    "VISCOELASTIC": (lambda: _mat("VISCOELASTIC", elastic=(1e3, 0.45), viscoelastic_prony=[(0.2, 0.0, 1.0)]), 3),
    "HYPER_VISCOELASTIC": (lambda: _mat("HYPER_VISCOELASTIC", hyperelastic=dict(_NH), viscoelastic_prony=[(0.2, 0.0, 1.0)]), 6),
    "DRUCKER_PRAGER": (lambda: _mat("DRUCKER_PRAGER", elastic=(2e5, 0.3), extra={"drucker_prager": {"d0": 100.0, "beta": 20.0, "psi": 20.0}}), 14),
    "CONCRETE_DP": (lambda: _mat("CONCRETE_DP", elastic=(3e4, 0.2), extra={"concrete_dp": {
        "compression": [(10.0, 0.0)], "tension": [(2.0, 0.0)], "dt_table": [(0.0, 0.0)], "dc_table": [(0.0, 0.0)]}}), 15),
    "JOHNSON_COOK": (lambda: _mat("JOHNSON_COOK", elastic=(2e5, 0.3)), None),
    "BERGSTROM_BOYCE": (lambda: _mat("BERGSTROM_BOYCE", elastic=(2e5, 0.3)), None),
    "CREEP": (lambda: _mat("CREEP", elastic=(2e5, 0.3)), None),
}


def test_every_mat_type_the_model_can_assign_is_covered_here():
    src = (PKG / "model" / "material.py").read_text(encoding="utf-8")
    assigned = set(re.findall(r'self\.mat_type = "([A-Z_]+)"', src))
    assigned |= set(re.findall(r'self\.mat_type = "([A-Z_]+)" if', src))
    assigned |= {"VISCOELASTIC", "HYPER_VISCOELASTIC"}
    missing = assigned - set(_CASES)
    assert not missing, f"model/material.py assigns mat_type(s) {sorted(missing)} that tests/test_material_wiring.py does not cover"


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mat_type", sorted(_CASES))
def test_material_resolves_or_raises_never_silent_linear(mat_type, dim):
    make, expected = _CASES[mat_type]
    m = make()
    if expected is None:
        with pytest.raises(NotImplementedError, match=mat_type):
            resolve_material(m, dim)
        return
    mid, props = resolve_material(m, dim)
    assert mid in MATERIALS and dim in MATERIALS[mid].dims
    assert mid == expected, f"{mat_type} resolved to {MATERIALS[mid].name}"
    if mat_type != "ELASTIC":
        assert mid != MAT_LINEAR_ELASTIC, f"{mat_type} must not be linear elastic"
    assert props.shape == (64,) and np.isfinite(props).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_unknown_material_names_itself_and_its_deck_location(dim):
    m = _mat("FOO_BAR", elastic=(1.0, 0.3))
    m.name = "MYSTERY"
    m.source_location = "deck.inp:42"
    with pytest.raises(NotImplementedError) as ei:
        resolve_material(m, dim, "material of pid 7")
    msg = str(ei.value)
    assert "MYSTERY" in msg and "deck.inp:42" in msg and "FOO_BAR" in msg


@pytest.mark.parametrize("dim", [2, 3])
def test_unknown_dict_material_raises(dim):
    with pytest.raises(NotImplementedError, match="WHATEVER"):
        resolve_material({"mat_type": "WHATEVER", "E": 1.0, "nu": 0.3}, dim)


def test_2d_and_3d_give_the_same_id_and_props():
    for name, (make, expected) in _CASES.items():
        if expected is None:
            continue
        m = make()
        id2, p2 = resolve_material(m, 2)
        id3, p3 = resolve_material(m, 3)
        assert id2 == id3 == expected, name
        np.testing.assert_array_equal(p2, p3, err_msg=name)
        if name in _EXPECTED_PROPS:
            np.testing.assert_allclose(p3[:len(_EXPECTED_PROPS[name])], _EXPECTED_PROPS[name], rtol=1e-12, err_msg=name)
    for d in ({"E": 2e5, "nu": 0.3}, {"type": "neo_hookean", "C10": 0.1, "D1": 0.01},
              {"type": "j2_plasticity", "E": 4e3, "nu": 0.35, "yield_stress": 80.0, "hardening_modulus": 200.0}):
        assert resolve_material(d, 2)[0] == resolve_material(d, 3)[0] == material_id(d)


# Expected props[:n] per material, derived by hand from the kernel layouts in
# material3d/numba_materials.py (NOT from pack_props), for the _CASES materials.
_EXPECTED_PROPS = {
    "ELASTIC": [2e5, 0.3],
    "J2_PLASTICITY": [2e5, 0.3, 250.0, 1000.0],                    # H = (350-250)/(0.1-0)
    "TABULAR_PLASTIC": [2e5, 0.3, 2.0, 1e9, 0.0, 250.0, 0.0, 350.0, 0.1],
    "NEO_HOOKEAN": [0.4, 200.0, 0.2, 0.01],                         # mu=2*c10, K=2/d1, c10, d1
    "MOONEY_RIVLIN": [0.1, 0.05, 0.01],
    "OGDEN": [1.0, 0.01, 0.2, 2.0],
    "ARRUDA_BOYCE": [0.2, 3.0, 10.0],
    "YEOH": [0.1, 0.0, 0.0, 0.01],
    "VISCOELASTIC": [1e3, 0.45, 0.2, 1.0],                          # E, nu, g1, tau1
    "HYPER_VISCOELASTIC": [1.0, 0.4, 200.0, 0.0, 1.0, 0.2, 1.0],    # base=NH, mu, K, -, n, g1, tau1
    "DRUCKER_PRAGER": [2e5, 0.3, 100.0, 0.0, 20.0, 20.0],
}
_EXPECTED_PROPS["HYPERELASTIC"] = _EXPECTED_PROPS["NEO_HOOKEAN"]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mat_type", sorted(_EXPECTED_PROPS))
def test_props_content_lands_in_the_kernel_slots(mat_type, dim):
    mid, props = resolve_material(_CASES[mat_type][0](), dim)
    exp = _EXPECTED_PROPS[mat_type]
    np.testing.assert_allclose(props[:len(exp)], exp, rtol=1e-12, err_msg=mat_type)
    assert not props[len(exp):].any(), f"{mat_type}: unexpected extra props {props[len(exp):][props[len(exp):] != 0]}"


def test_viscoelastic_object_is_prony_not_neohookean_with_made_up_mu():
    """Audit: *ELASTIC 1000,0.45 + *VISCOELASTIC Prony (0.3,0,2.0) used to
    become Neo-Hookean-based id 6 with mu=0.2/K=10 invented."""
    m = _mat("VISCOELASTIC", elastic=(1000.0, 0.45), viscoelastic_prony=[(0.3, 0.0, 2.0)])
    for dim in (2, 3):
        mid, props = resolve_material(m, dim)
        assert mid == material_id(m) == 3
        np.testing.assert_array_equal(props[:4], [1000.0, 0.45, 0.3, 2.0])
        np.testing.assert_array_equal(props, pack_props(m, dim))


def test_hyperelastic_with_empty_prony_keeps_its_own_constants():
    """*HYPERELASTIC NEO HOOKE C10=80 D1=20 + *VISCOELASTIC FREQUENCY=TABULAR
    (no Prony terms): catalog id 1 with the deck's C10, not id 3 with defaults."""
    m = _mat("HYPER_VISCOELASTIC", hyperelastic={"model": "NEOHOOKEAN", "c10": 80.0, "d1": 20.0},
             viscoelastic_prony=[])
    for dim in (2, 3):
        mid, props = resolve_material(m, dim)
        assert mid == material_id(m) == MAT_HYPERELASTIC_NEOHOOKEAN
        np.testing.assert_allclose(props[:4], [160.0, 0.1, 80.0, 20.0])


def test_arruda_boyce_hyper_viscoelastic_uses_the_kernel_layout():
    """kernel: p0=mu, p1=lambda_m, p2=K at props[1:4] (the pre-A5 packing put K in p1)."""
    m = _mat("HYPER_VISCOELASTIC", hyperelastic={"model": "ARRUDA_BOYCE", "mu": 0.5, "lambda_m": 5.0, "K": 9.0},
             viscoelastic_prony=[(0.2, 0.0, 1.0), (0.1, 0.0, 10.0)])
    mid, props = resolve_material(m, 3)
    assert mid == 6
    np.testing.assert_allclose(props[:9], [4.0, 0.5, 5.0, 9.0, 2.0, 0.2, 1.0, 0.1, 10.0])


def test_hyper_viscoelastic_with_an_unpackable_base_raises():
    m = _mat("HYPER_VISCOELASTIC", hyperelastic={"model": "YEOH", "c10": 0.1, "d1": 0.01},
             viscoelastic_prony=[(0.2, 0.0, 1.0)])
    with pytest.raises(NotImplementedError, match="base model"):
        resolve_material(m, 3)


@pytest.mark.parametrize("dim", [2, 3])
def test_dict_with_disagreeing_integer_mat_type_raises_naming_both_ids(dim):
    # to_numba_props-style hyper-viscoelastic dict: id 6 stamped, content resolves to Prony 3
    d = {"mat_type": 6, "E": 1e3, "nu": 0.45, "prony": [(0.3, 0.0, 2.0)]}
    with pytest.raises(NotImplementedError, match=r"mat_type 6.*id 3"):
        resolve_material(d, dim)
    # Voce key present, but the stamped id says linear elastic
    with pytest.raises(NotImplementedError, match=r"mat_type 0.*id 7"):
        resolve_material({"mat_type": 0, "E": 2e5, "nu": 0.3, "R_inf": 100.0, "b_exp": 10.0}, dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_dict_with_agreeing_or_no_mat_type_resolves(dim):
    assert resolve_material({"mat_type": 2, "E": 2e5, "nu": 0.3, "yield_stress": 250.0}, dim)[0] == 2
    assert resolve_material({"E": 2e5, "nu": 0.3}, dim)[0] == MAT_LINEAR_ELASTIC


@pytest.mark.parametrize("dim", [2, 3])
def test_material_without_elastic_raises_instead_of_inventing_E(dim):
    m = _mat("J2_PLASTICITY", plastic=[(250.0, 0.0)])
    m.name = "NOELAS"
    m.source_location = "d.inp:3"
    with pytest.raises(NotImplementedError, match=r"NOELAS.*d\.inp:3|d\.inp:3.*NOELAS") as ei:
        resolve_material(m, dim)
    assert "no *ELASTIC" in str(ei.value) and "needs E, nu" in str(ei.value)
    with pytest.raises(NotImplementedError, match="no \*ELASTIC"):
        resolve_material({"mat_type": 0}, dim)


def test_unsupported_material_message_is_the_same_sentence_with_a_location():
    m = _mat("JOHNSON_COOK", elastic=(2e5, 0.3))
    m.name = "JC"
    m.source_location = "deck.inp:7"
    with pytest.raises(NotImplementedError) as ei:
        resolve_material(m, 3, "material of pid 1")
    assert "is not implemented in 3D" in str(ei.value) and "deck.inp:7" in str(ei.value)
    assert "carries id" not in str(ei.value)


def test_reduced_polynomial_and_voce_without_b_exp_raise_instead_of_other_models():
    red = _mat("HYPERELASTIC", hyperelastic={"model": "REDUCED POLYNOMIAL", "c10": 0.1, "d1": 0.01})
    with pytest.raises(NotImplementedError, match="REDUCED POLYNOMIAL|is not implemented in 3D"):
        resolve_material(red, 3)
    voce = _mat("J2_PLASTICITY", elastic=(2e5, 0.3), plastic=[(250.0, 0.0)], extra={"R_inf": 100.0})
    with pytest.raises(NotImplementedError, match="R_inf and b_exp"):
        resolve_material(voce, 3)


def _ortho():
    return _mat("ELASTIC", elastic=(1.0, 0.3), extra={"engineering_constants": (1e5, 1e5, 1e5, .3, .3, .3, 4e4, 4e4, 4e4)})


def test_orthotropic_is_not_implemented_in_2d():
    with pytest.raises(NotImplementedError, match="not implemented in 2D"):
        resolve_material(_ortho(), 2)
    with pytest.raises(NotImplementedError, match="not implemented in 2D"):
        _solver_ids(2, _ortho())


def test_orthotropic_id17_on_a_non_c3d20_element_raises_naming_the_element():
    with pytest.raises(NotImplementedError, match="C3D8"):
        _solver_ids(3, _ortho())


def _solver_ids(dim, mat):
    if dim == 2:
        from dispsolver.mesh2d.mesh2d import Mesh2D
        from dispsolver.solver2d.dynamic2d import DynamicSolver2D
        mesh = Mesh2D()
        for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
            mesh.add_node(i + 1, x, y)
        mesh.add_element(1, [1, 2, 3, 4], "CPE4", pid=0)
        s = DynamicSolver2D(mesh, materials={0: mat}, nlgeom=False)
    else:
        from dispsolver.mesh3d.mesh3d import Mesh3D
        from dispsolver.solver3d.dynamic3d import DynamicSolver3D
        mesh = Mesh3D()
        for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]):
            mesh.add_node(i + 1, x, y, z)
        mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8", pid=0)
        s = DynamicSolver3D(mesh, materials={0: mat})
    return int(s.elem_mat_types[0]), np.array(s.elem_props[0])


@pytest.mark.parametrize("mat_type", ["ELASTIC", "J2_PLASTICITY", "NEO_HOOKEAN", "ARRUDA_BOYCE",
                                      "YEOH", "DRUCKER_PRAGER", "HYPER_VISCOELASTIC"])
def test_solvers_2d_and_3d_assign_the_same_id(mat_type):
    make, expected = _CASES[mat_type]
    id2, p2 = _solver_ids(2, make())
    id3, p3 = _solver_ids(3, make())
    assert id2 == id3 == expected
    np.testing.assert_array_equal(p2, p3)
    np.testing.assert_allclose(p3[:len(_EXPECTED_PROPS[mat_type])], _EXPECTED_PROPS[mat_type], rtol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_solver_refuses_unmapped_material_instead_of_linear_elastic(dim):
    with pytest.raises(NotImplementedError, match="CREEP"):
        _solver_ids(dim, _CASES["CREEP"][0]())


def test_2d_solver_no_longer_turns_hyperelastic_into_linear_elastic():
    """Audit O7: solver2d used to map every non-J2/tabular/Voce material to id 0."""
    assert _solver_ids(2, _CASES["NEO_HOOKEAN"][0]())[0] == MAT_HYPERELASTIC_NEOHOOKEAN
    assert _solver_ids(2, {"mat_type": MAT_HYPERELASTIC_NEOHOOKEAN, "c10": 0.1, "d1": 0.01})[0] == MAT_HYPERELASTIC_NEOHOOKEAN


# -----------------------------------------------------------------------------
# (e) golden comparison against the pre-A5 chain
# -----------------------------------------------------------------------------
_GOLDEN = ROOT / "tests" / "fixtures" / "material_ids_9f5e6e9.json"


def _explained_change(m, old, new):
    """Reason code if the change from the pre-A5 (id, props) is one of the
    reasoned corrections / refusals, else None."""
    extra = m.extra_params
    err = new.get("error")
    # old chain: any ENGINEERING CONSTANTS -> id 17 whatever else the material was
    # (viscoelasticity silently dropped); 17 exists for C3D20 only.
    if err and "has no *ELASTIC" in new["msg"] and not m.elastic:
        return "no *ELASTIC: refused (old: invented E=1.0)"
    if err and old.get("id") == 17 and "engineering_constants" in extra:
        return "orthotropic id 17 on a non-C3D20 element now raises"
    if err and m.mat_type in ("CREEP", "JOHNSON_COOK", "BERGSTROM_BOYCE"):
        return "no solver model for this mat_type (old: id 99 / near-zero stiffness)"
    if err and m.hyperelastic and "Unknown hyperelastic model" in new["msg"]:
        return "hyperelastic form the catalog does not know: REDUCED POLYNOMIAL, COMPRESSIBLE... (old: silently polynomial / Neo-Hooke)"
    if err and m.mat_type == "HYPER_VISCOELASTIC" and "base model" in new["msg"]:
        return "hyper-viscoelastic with a base the kernel layout cannot carry (old: wrong slots)"
    if err:
        return None
    if m.mat_type == "VISCOELASTIC" and old.get("id") == 6 and new["id"] == 3 and m.viscoelastic_prony and not m.hyperelastic:
        return "plain Prony: catalog id 3 (old: Neo-Hooke id 6 with an invented mu=0.2, K=10)"
    if m.mat_type == "HYPER_VISCOELASTIC" and not m.viscoelastic_prony and new["id"] == material_id(m) != old.get("id"):
        return "hyperelastic with empty Prony (FREQUENCY=TABULAR): its own catalog id (old: Prony 3 with defaults)"
    if m.mat_type == "HYPER_VISCOELASTIC" and old.get("id") == new["id"] == 6:
        diff = [i for i in range(64) if old["props"][i] != new["props"][i]]
        if (m.hyperelastic or {}).get("model", "").upper().startswith("ARRUDA") and set(diff) <= {2, 3}:
            return "Arruda-Boyce hyper-viscoelastic: kernel slots mu, lambda_m, K (old: K in the lambda slot, K=0)"
    return None


def test_deck_materials_match_the_pre_a5_chain_except_the_reasoned_list():
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    import material_golden_samples as S
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert len(golden) >= 250
    by_deck = {}
    for key in golden:
        by_deck.setdefault(key.split("::")[0], []).append(key)
    unexplained, explained = [], {}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for deck, keys in by_deck.items():
            mats = S.deck_materials(str(ROOT / deck))
            for key in keys:
                m, old = mats[key], golden[key]
                try:
                    mid, p = S.probe_3d(m)
                    new = {"id": mid, "props": [float(x) for x in p]}
                except Exception as exc:
                    new = {"error": type(exc).__name__, "msg": str(exc)}
                if "error" in old and "error" in new:
                    continue
                if "error" not in old and "error" not in new and old == new:
                    continue
                why = _explained_change(m, old, new)
                if why is None:
                    unexplained.append((key, old.get("id", old), new.get("id", new.get("error"))))
                else:
                    explained.setdefault(why, []).append(key)
    assert not unexplained, f"changes with no reason: {unexplained[:10]}"
    print({k: len(v) for k, v in explained.items()})
    assert explained, "the golden comparison found nothing to explain: fixture stale?"


# -----------------------------------------------------------------------------
# (b) static source scan
# -----------------------------------------------------------------------------
_SCANNED = [
    PKG / "solver2d" / "dynamic2d.py",
    PKG / "solver3d" / "dynamic3d.py",
    PKG / "model" / "assembly.py",
    PKG / "model" / "model.py",
]
# MAT_* names a solver module may still mention: layout/predicate uses of ids
# the catalog already assigned, plus the C3D20-private orthotropic id 17
# (not in the catalog; reported to agent B).
_ALLOWED_MAT_NAMES = {"MAT_LINEAR_ELASTIC", "MAT_CUSTOM_ELASTIC", "MAT_ORTHOTROPIC_CYLINDRICAL",
                      "MAT_J2_DUCTILE_DAMAGE"}
_FORBIDDEN = [
    (r"str\(mtype\)\.lower\(\)", "per-type string sniffing of mat_type"),
    (r"mat_obj\.get\(\s*[\"']mat_type[\"']", "reading mat_type out of a material dict to pick an id"),
    (r"[\"'](?:neo_hookean|neohookean|mooney_rivlin|mooney|ogden|arruda_boyce|arruda|yeoh|tabular_plastic|radioss_law36|j2_plasticity)[\"']\s*(?:,|\])",
     "material-type name lists"),
    (r"elem_mat_types\[[^\]]*\]\s*=\s*\d+\b", "a literal material id written into elem_mat_types"),
    (r"elem_mat_types\[[^\]]*\]\s*=\s*MAT_(?!CUSTOM_ELASTIC\b)\w+", "a MAT_* id written into elem_mat_types"),
    (r"elem_mat_types\[[^\]]*\]\s*=\s*int\(\s*mat_obj", "an id read straight off a material object"),
]


@pytest.mark.parametrize("path", _SCANNED, ids=lambda p: p.name)
def test_no_material_id_mapping_outside_the_catalog(path):
    text = path.read_text(encoding="utf-8")
    for pattern, what in _FORBIDDEN:
        hits = [ln for ln in text.splitlines() if re.search(pattern, ln, flags=re.IGNORECASE)]
        assert not hits, f"{path.name}: {what}: {hits[:3]}"
    names = set(re.findall(r"\bMAT_[A-Z0-9_]+\b", text))
    stray = names - _ALLOWED_MAT_NAMES
    assert not stray, f"{path.name} mentions material ids {sorted(stray)}; map through dispsolver.material.catalog"


def test_solvers_call_the_catalog_wiring():
    for path in (PKG / "solver2d" / "dynamic2d.py", PKG / "solver3d" / "dynamic3d.py"):
        text = path.read_text(encoding="utf-8")
        assert "resolve_material" in text, path.name
        assert "is_linear_elastic" in text, path.name


def test_to_numba_props_callers_are_the_known_ones():
    """Where a Material is flattened into a props dict; new callers must be reviewed
    (the dict's mat_type is an id assigned in model/material.py, owned by B)."""
    callers = set()
    for path in PKG.rglob("*.py"):
        if "to_numba_props" in path.read_text(encoding="utf-8", errors="ignore"):
            callers.add(path.relative_to(PKG).as_posix())
    assert callers <= {"model/material.py", "model/assembly.py", "material/resolve.py",
                       "solver2d/dynamic2d.py", "solver3d/dynamic3d.py"}, sorted(callers)


@pytest.mark.parametrize("dim", [2, 3])
def test_deck_path_props_dict_resolves_from_its_material_object(dim):
    """Assembly hands the solvers to_numba_props() dicts (contact code reads them);
    they carry their Material, so plain *VISCOELASTIC is still catalog id 3."""
    from dispsolver.material.resolve import MaterialProps
    m = _mat("VISCOELASTIC", elastic=(1000.0, 0.45), viscoelastic_prony=[(0.3, 0.0, 2.0)])
    d = MaterialProps(m.to_numba_props(), m)
    assert d["mat_type"] == 6                      # the model's own (different) opinion
    mid, props = resolve_material(d, dim)
    assert mid == 3
    np.testing.assert_array_equal(props[:4], [1000.0, 0.45, 0.3, 2.0])
    assert _solver_ids(dim, d)[0] == 3


# -- review finding 6: the flat damage_* dict layout (DynamicSolver3D(materials={pid: dict}))

@pytest.mark.parametrize("with_type", [True, False])
def test_flat_ductile_damage_dict_keeps_its_data(with_type):
    from dispsolver.material.resolve import resolve_material
    d = {"E": 200000.0, "nu": 0.3, "sigma_y0": 300.0, "H": 1000.0,
         "damage_init_type": 2, "damage_Troom": 25.0, "damage_Tmelt": 1500.0, "damage_eps0": 2.0}
    if with_type:
        d["mat_type"] = 13
    mid, props = resolve_material(d, 3, "probe")
    assert mid == 13
    assert props[4] == 2.0                                   # Johnson-Cook initiation, not constant strain
    assert list(props[13:17]) == [20.0, 25.0, 1500.0, 2.0]   # T, T_room, T_melt, eps0 from the dict


def test_hyper_viscoelastic_dict_with_nested_hyperelastic_block_raises():
    from dispsolver.material.resolve import resolve_material
    d = {"mat_type": 6, "prony": [(0.2, 1.0)],
         "hyperelastic": {"model": "ARRUDA_BOYCE", "mu": 0.5, "lambda_m": 3.0, "K": 10.0}}
    with pytest.raises(NotImplementedError, match="nested 'hyperelastic'"):
        resolve_material(d, 3, "probe")

