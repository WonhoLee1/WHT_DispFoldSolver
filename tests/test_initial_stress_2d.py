"""2D initial stress (PRD: dev_log/prd_initial_stress_20261018.md).

Independent expected value, as in test_initial_stress_elements: for an isoparametric element
sum_i f_i,a * X_i,b = s0_ab * A (plane strain, unit thickness), A the element area. The elements are
affine images of the unit square / triangle so A is |det F| (times 1 or 1/2) in closed form.
"""
import numpy as np
import pytest

from dispsolver.io import read_inp

S0 = np.array([3.0, -2.0, 5.0, 1.5])            # S11 S22 S33 S12 (Abaqus plane-strain order)
S0_MAT = np.array([[S0[0], S0[3]], [S0[3], S0[1]]])
F_AFF = np.array([[1.3, 0.2], [0.1, 0.9]])
T_AFF = np.array([0.5, -0.3])


def _mid(a, b):
    return tuple((np.array(a) + np.array(b)) / 2.0)


_QUAD = [(0, 0), (1, 0), (1, 1), (0, 1)]
_TRI = [(0, 0), (1, 0), (0, 1)]
_QUAD8 = _QUAD + [_mid(_QUAD[a], _QUAD[b]) for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
_TRI6 = _TRI + [_mid(_TRI[a], _TRI[b]) for a, b in [(0, 1), (1, 2), (2, 0)]]

_SHAPES = {
    "CPE4": (_QUAD, 1.0), "CPE4I": (_QUAD, 1.0), "CPE4R": (_QUAD, 1.0), "CPE4H": (_QUAD, 1.0),
    "CPE3": (_TRI, 0.5), "CPE6": (_TRI6, 0.5), "CPE6M": (_TRI6, 0.5),
    "CPE8": (_QUAD8, 1.0), "CPE8R": (_QUAD8, 1.0),
}


def _deck(elem_type: str, nodes_ref, ic_row: str) -> str:
    nodes = [F_AFF @ np.array(p) + T_AFF for p in nodes_ref]
    n = len(nodes)
    lines = ["*HEADING", "2D initial stress", "*NODE"]
    lines += [f"{i + 1}, {float(x[0])!r}, {float(x[1])!r}" for i, x in enumerate(nodes)]
    lines += [f"*ELEMENT, TYPE={elem_type}, ELSET=EL", "1, " + ", ".join(str(i + 1) for i in range(n)),
              "*NSET, NSET=ALLN", ", ".join(str(i + 1) for i in range(n)),
              "*MATERIAL, NAME=M", "*ELASTIC", "1000.0, 0.25",
              "*SOLID SECTION, ELSET=EL, MATERIAL=M",
              "*INITIAL CONDITIONS, TYPE=STRESS", ic_row,
              "*STEP, NAME=S1, NLGEOM=NO", "*STATIC", "1.0, 1.0", "*BOUNDARY", "ALLN, 1, 2, 0.0", "*END STEP", ""]
    return "\n".join(lines)


def _read(tmp_path, deck):
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    return read_inp(str(path), dim=2)


@pytest.mark.parametrize("nlgeom", [False, True], ids=["nlgeom_off", "nlgeom_on"])
@pytest.mark.parametrize("elem_type", list(_SHAPES))
def test_uniform_initial_stress_gives_s0_times_area(tmp_path, elem_type, nlgeom):
    ref, a_ref = _SHAPES[elem_type]
    row = "EL, " + ", ".join(repr(float(v)) for v in S0)
    model = _read(tmp_path, _deck(elem_type, ref, row))
    solver, sysm = model.create_solver2d("S1" if not nlgeom else None)
    solver.assemble_system(np.zeros(solver.num_dofs))
    f = np.asarray(solver.f_int).reshape(-1, 2)
    X = np.asarray(sysm.coords)[:, :2]
    area = a_ref * abs(np.linalg.det(F_AFF))
    np.testing.assert_allclose(f.T @ X, S0_MAT * area, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-9)


def test_omitted_components_are_zero_and_more_than_four_is_an_error(tmp_path):
    model = _read(tmp_path, _deck("CPE4", _QUAD, "EL, 1.0, 2.0"))
    solver, sysm = model.create_solver2d("S1")
    solver.assemble_system(np.zeros(solver.num_dofs))
    f = np.asarray(solver.f_int).reshape(-1, 2)
    X = np.asarray(sysm.coords)[:, :2]
    area = abs(np.linalg.det(F_AFF))
    np.testing.assert_allclose(f.T @ X, np.array([[1.0, 0.0], [0.0, 2.0]]) * area, atol=1e-9)
    model = _read(tmp_path, _deck("CPE4", _QUAD, "EL, 1, 2, 3, 4, 5"))
    with pytest.raises(ValueError, match="at most 4"):
        model.create_solver2d("S1")


def test_axisymmetric_elements_are_refused(tmp_path):
    # the axisymmetric kernels integrate over the ring (2 pi r); the plane term here would be wrong
    model = _read(tmp_path, _deck("CAX4", _QUAD, "EL, 10.0"))
    with pytest.raises(NotImplementedError, match="initial stress"):
        model.create_solver2d("S1")
