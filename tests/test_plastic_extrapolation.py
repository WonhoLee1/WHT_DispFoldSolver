"""*PLASTIC, EXTRAPOLATION=LINEAR|CONSTANT -- how the isotropic table continues past its last point.

Independent expected values from a closed-form uniaxial-stress state (small strain, E=1000): the total
axial strain is eps = sigma/E + eps_p, and the table is sigma_y(eps_p) = 10 + 100*eps_p up to eps_p = 0.1.
  CONSTANT (Abaqus default): sigma_y stays 20 past the table      -> sigma = 20
  LINEAR: sigma_y keeps the last slope (100): 20 + 100*(eps_p-0.1) -> sigma = (20 + 100*(eps-0.1)) / (1 + 100/E)
"""
import numpy as np
import pytest

from dispsolver.io import execute_inp

_DECK = """*HEADING
pull past the end of the hardening table
*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE=C3D8, ELSET=EL
1, 1,2,3,4,5,6,7,8
*MATERIAL, NAME=M
*ELASTIC
1000.0, 0.3
*PLASTIC{param}
10.0, 0.0
20.0, 0.1
*SOLID SECTION, ELSET=EL, MATERIAL=M
*STEP, NAME=PULL, NLGEOM=NO
*STATIC
0.1, 1.0, 1e-4, 0.1
*BOUNDARY
1, 1, 3, 0.0
4, 1, 1, 0.0
4, 3, 3, 0.0
5, 1, 2, 0.0
8, 1, 1, 0.0
2, 1, 1, 0.5
3, 1, 1, 0.5
6, 1, 1, 0.5
7, 1, 1, 0.5
*END STEP
"""


def _stress(tmp_path, param):
    path = tmp_path / "deck.inp"
    path.write_text(_DECK.format(param=param))
    res = execute_inp(str(path), dim=3, verbose=False)
    assert res["converged"]
    f = np.asarray(res["solver"].f_int).reshape(-1, 3)
    return float(f[[1, 2, 5, 6], 0].sum())          # right face, area 1


def test_default_is_constant(tmp_path):
    assert _stress(tmp_path, "") == pytest.approx(20.0, abs=0.3)


def test_explicit_constant(tmp_path):
    assert _stress(tmp_path, ", EXTRAPOLATION=CONSTANT") == pytest.approx(20.0, abs=0.3)


def test_linear_continues_last_slope(tmp_path):
    expected = (20.0 + 100.0 * (0.5 - 0.1)) / (1.0 + 100.0 / 1000.0)
    assert _stress(tmp_path, ", EXTRAPOLATION=LINEAR") == pytest.approx(expected, rel=0.03)


def test_nonhybrid_incompressible_warns():
    """Abaqus 6.13 diagnostic: nu above 0.48 on a non-hybrid element warns (error limit 0.495)."""
    import warnings
    from dispsolver.model.element_catalog import ElementCatalog, _ELEMENT_DESCRIPTORS
    # find an element type without an incompressible alternative
    names = [d.type_name for d in _ELEMENT_DESCRIPTORS if d.dimension == 3]
    hits = []
    for name in names:
        desc = ElementCatalog._BY_DIM[3][name]
        if getattr(desc, "kernel_yes_incompressible", None) is None and not desc.unsupported:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    ElementCatalog.resolve(name, 3, True, nu=0.499, linear_elastic=True)
                except NotImplementedError:
                    continue
                if any("non-hybrid" in str(x.message) for x in w):
                    hits.append(name)
                    break
    assert hits, "no non-hybrid 3D element produced the incompressibility warning"
