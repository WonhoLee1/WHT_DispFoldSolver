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
(c) 2D and 3D give the same id and props for the same material.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from dispsolver.material.catalog import MATERIALS, material_id
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
    "VISCOELASTIC": (lambda: _mat("VISCOELASTIC", elastic=(1e3, 0.45), viscoelastic_prony=[(0.2, 0.0, 1.0)]), 6),
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
        assert id2 == id3, name
        np.testing.assert_array_equal(p2, p3, err_msg=name)
    for d in ({"E": 2e5, "nu": 0.3}, {"type": "neo_hookean", "C10": 0.1, "D1": 0.01},
              {"type": "j2_plasticity", "E": 4e3, "nu": 0.35, "yield_stress": 80.0, "hardening_modulus": 200.0}):
        assert resolve_material(d, 2)[0] == resolve_material(d, 3)[0] == material_id(d)


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


@pytest.mark.parametrize("dim", [2, 3])
def test_solver_refuses_unmapped_material_instead_of_linear_elastic(dim):
    with pytest.raises(NotImplementedError, match="CREEP"):
        _solver_ids(dim, _CASES["CREEP"][0]())


def test_2d_solver_no_longer_turns_hyperelastic_into_linear_elastic():
    """Audit O7: solver2d used to map every non-J2/tabular/Voce material to id 0."""
    assert _solver_ids(2, _CASES["NEO_HOOKEAN"][0]())[0] == MAT_HYPERELASTIC_NEOHOOKEAN
    assert _solver_ids(2, {"mat_type": MAT_HYPERELASTIC_NEOHOOKEAN, "c10": 0.1, "d1": 0.01})[0] == MAT_HYPERELASTIC_NEOHOOKEAN


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
