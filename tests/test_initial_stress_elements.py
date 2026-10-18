"""Initial stress reaches the internal force of every 3D solid element (PRD: dev_log/prd_initial_stress_20261018.md).

Independent expected value: for an isoparametric element f_i = int dN_i/dx . s0 dV and
sum_i x_i,b dN_i/dx_a = delta_ab pointwise, therefore

    sum_i f_i,a * X_i,b = s0_ab * V        for every a, b

with V the element volume. The elements are affine images of the unit cube / wedge / tetrahedron, so V is
|det F| (times 1, 1/2, 1/6) in closed form and no code under test enters the expected value.
"""
import numpy as np
import pytest

from dispsolver.io import read_inp

# S11 S22 S33 S12 S13 S23 (Abaqus order)
S0 = np.array([3.0, -2.0, 5.0, 1.5, -0.7, 2.2])
S0_MAT = np.array([[S0[0], S0[3], S0[4]],
                   [S0[3], S0[1], S0[5]],
                   [S0[4], S0[5], S0[2]]])
# a shear + stretch + translation, so the element is neither axis aligned nor orthogonal
F_AFF = np.array([[1.3, 0.2, -0.1],
                  [0.1, 0.9, 0.25],
                  [-0.2, 0.15, 1.1]])
T_AFF = np.array([0.5, -0.3, 0.2])


def _mid(a, b):
    return tuple((np.array(a) + np.array(b)) / 2.0)


_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_WEDGE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)]
_TET = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
_TET10 = _TET + [_mid(_TET[a], _TET[b]) for a, b in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]]
_HEX20 = _CUBE + [_mid(_CUBE[a], _CUBE[b]) for a, b in
                  [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]]

# element type -> (reference nodes, volume of the reference shape)
_SHAPES = {
    "C3D8": (_CUBE, 1.0), "C3D8H": (_CUBE, 1.0), "C3D8I": (_CUBE, 1.0), "C3D8R": (_CUBE, 1.0),
    "C3D8_FBAR": (_CUBE, 1.0), "C3D4_ANP": (_TET, 1.0 / 6.0),
    "C3D20": (_HEX20, 1.0), "C3D20R": (_HEX20, 1.0),
    "C3D6": (_WEDGE, 0.5),
    "C3D4": (_TET, 1.0 / 6.0), "C3D10": (_TET10, 1.0 / 6.0), "C3D10M": (_TET10, 1.0 / 6.0),
}


def _deck(elem_type: str) -> str:
    ref, _ = _SHAPES[elem_type]
    nodes = [F_AFF @ np.array(p) + T_AFF for p in ref]
    n = len(nodes)
    lines = ["*HEADING", "initial stress element matrix", "*NODE"]
    lines += [f"{i + 1}, {float(x[0])!r}, {float(x[1])!r}, {float(x[2])!r}" for i, x in enumerate(nodes)]
    lines += [f"*ELEMENT, TYPE={elem_type}, ELSET=EL", "1, " + ", ".join(str(i + 1) for i in range(n)),
              "*NSET, NSET=ALLN", ", ".join(str(i + 1) for i in range(n)),
              "*MATERIAL, NAME=M", "*ELASTIC", "1000.0, 0.25",
              "*SOLID SECTION, ELSET=EL, MATERIAL=M",
              "*INITIAL CONDITIONS, TYPE=STRESS", "EL, " + ", ".join(repr(float(v)) for v in S0),
              "*STEP, NAME=S1, NLGEOM=NO", "*STATIC", "1.0, 1.0", "*BOUNDARY", "ALLN, 1, 3, 0.0", "*END STEP", ""]
    return "\n".join(lines)


@pytest.mark.parametrize("nlgeom", [False, True], ids=["nlgeom_off", "nlgeom_on"])
@pytest.mark.parametrize("elem_type", list(_SHAPES))
def test_uniform_initial_stress_gives_s0_times_volume(tmp_path, elem_type, nlgeom):
    if elem_type == "C3D4_ANP" and nlgeom:
        pytest.skip("the average-nodal-pressure kernel cannot be co-rotated, so it has no NLGEOM path")
    path = tmp_path / "deck.inp"
    path.write_text(_deck(elem_type))
    model = read_inp(str(path), dim=3)
    solver, sysm = model.create_solver3d("S1" if not nlgeom else None)
    solver.assemble_system(np.zeros(solver.num_dofs))
    f = np.asarray(solver.f_int).reshape(-1, 3)
    X = np.asarray(sysm.coords)
    ref_nodes, v_ref = _SHAPES[elem_type]
    volume = v_ref * abs(np.linalg.det(F_AFF))
    moment = f.T @ X                       # M_ab = sum_i f_i,a X_i,b
    np.testing.assert_allclose(moment, S0_MAT * volume, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-9)   # a closed body is in equilibrium
