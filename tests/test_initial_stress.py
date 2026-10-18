"""*INITIAL CONDITIONS, TYPE=STRESS reaches the solver (PRD: dev_log/prd_initial_stress_20261018.md).

Expected values come from the Keywords Guide (data-line layout, omitted components are zero) and
from continuum mechanics (a uniform stress s0 on a face of area A gives a nodal force sum s0*A),
never from the code under test.
"""
import numpy as np
import pytest

from dispsolver.io import read_inp

_CUBE_NODES = """*NODE
1, 0,0,0
2, 1,0,0
3, 1,1,0
4, 0,1,0
5, 0,0,1
6, 1,0,1
7, 1,1,1
8, 0,1,1
"""


def _cube_deck(ic_row: str, elem_type: str = "C3D8") -> str:
    return (
        "*HEADING\ninitial stress probe\n"
        + _CUBE_NODES
        + f"*ELEMENT, TYPE={elem_type}, ELSET=EL\n1, 1,2,3,4,5,6,7,8\n"
        "*NSET, NSET=ALLN\n1,2,3,4,5,6,7,8\n"
        "*MATERIAL, NAME=M\n*ELASTIC\n1000.0, 0.25\n"
        "*SOLID SECTION, ELSET=EL, MATERIAL=M\n"
        "*INITIAL CONDITIONS, TYPE=STRESS\n"
        f"{ic_row}\n"
        "*STEP, NAME=S1, NLGEOM=NO\n*STATIC\n1.0, 1.0\n"
        "*BOUNDARY\nALLN, 1, 3, 0.0\n*END STEP\n"
    )


def _read(tmp_path, deck: str, dim: int):
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    return read_inp(str(path), dim=dim)


def test_s2_full_six_components_reach_every_gauss_point(tmp_path):
    model = _read(tmp_path, _cube_deck("EL, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0"), dim=3)
    solver, _ = model.create_solver3d()
    expected = np.array([10.0, 20.0, 30.0, 4.0, 5.0, 6.0])  # S11 S22 S33 S12 S13 S23
    np.testing.assert_array_equal(solver.elem_stress_init[0], np.tile(expected, (8, 1)))


def test_s2_omitted_components_are_zero(tmp_path):
    model = _read(tmp_path, _cube_deck("EL, 1.0, 2.0, 3.0"), dim=3)
    solver, _ = model.create_solver3d()
    expected = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(solver.elem_stress_init[0], np.tile(expected, (8, 1)))


def test_s2_more_than_six_components_is_an_error(tmp_path):
    model = _read(tmp_path, _cube_deck("EL, 1,2,3,4,5,6,7"), dim=3)
    with pytest.raises(ValueError, match="at most 6"):
        model.create_solver3d()


@pytest.mark.parametrize("elem_type", ["C3D8", "C3D8H"])
def test_s3_internal_force_at_zero_displacement_is_stress_times_area(tmp_path, elem_type):
    # Uniform S11 = 10 on a unit cube: the x=1 face carries a nodal force sum of +10 (stress * area),
    # the x=0 face -10, and nothing transverse. Continuum mechanics, independent of the element code.
    model = _read(tmp_path, _cube_deck("EL, 10.0", elem_type), dim=3)
    solver, sysm = model.create_solver3d()
    solver.assemble_system(np.zeros(solver.num_dofs))
    f = np.asarray(solver.f_int).reshape(-1, 3)
    x = np.asarray(sysm.coords)[:, 0]
    assert f[x > 0.5, 0].sum() == pytest.approx(10.0, abs=1e-9)
    assert f[x < 0.5, 0].sum() == pytest.approx(-10.0, abs=1e-9)
    assert np.abs(f[:, 1:]).max() < 1e-9


@pytest.mark.parametrize("elem_type", ["C3D8I", "C3D8R"])
def test_s3_element_that_ignores_initial_stress_is_refused(tmp_path, elem_type):
    # Measured: under NLGEOM these two kernels return f_int = 0 at u = 0 for S11 = 10, i.e. they drop the
    # field. Solving without it is a different model, so the solver must say so (PRD R4).
    model = _read(tmp_path, _cube_deck("EL, 10.0", elem_type), dim=3)
    with pytest.raises(NotImplementedError, match="initial stress"):
        model.create_solver3d()


def test_s4_2d_refuses_initial_stress_explicitly(tmp_path):
    # No 2D kernel reads an initial stress yet; ignoring it would solve a different model, and the
    # old path died with an AttributeError that said nothing about why.
    deck = (
        "*HEADING\n2D initial stress probe\n"
        "*NODE\n1, 0,0\n2, 1,0\n3, 1,1\n4, 0,1\n"
        "*ELEMENT, TYPE=CPE4, ELSET=EL\n1, 1,2,3,4\n"
        "*NSET, NSET=ALLN\n1,2,3,4\n"
        "*MATERIAL, NAME=M\n*ELASTIC\n1000.0, 0.25\n"
        "*SOLID SECTION, ELSET=EL, MATERIAL=M\n"
        "*INITIAL CONDITIONS, TYPE=STRESS\nEL, 10.0\n"
        "*STEP, NAME=S1, NLGEOM=NO\n*STATIC\n1.0, 1.0\n"
        "*BOUNDARY\nALLN, 1, 2, 0.0\n*END STEP\n"
    )
    model = _read(tmp_path, deck, dim=2)
    with pytest.raises(NotImplementedError, match="2D.*initial stress"):
        model.create_solver2d()


@pytest.mark.parametrize("elem_type", ["C3D8", "C3D8H"])
def test_s3_nlgeom_off_internal_force_is_stress_times_area(tmp_path, elem_type):
    # The step decides the kernel (NLGEOM=NO runs different ones from NLGEOM=YES); the same
    # continuum-mechanics check must hold there.
    model = _read(tmp_path, _cube_deck("EL, 10.0", elem_type), dim=3)
    solver, sysm = model.create_solver3d("S1")
    solver.assemble_system(np.zeros(solver.num_dofs))
    f = np.asarray(solver.f_int).reshape(-1, 3)
    x = np.asarray(sysm.coords)[:, 0]
    assert f[x > 0.5, 0].sum() == pytest.approx(10.0, abs=1e-9)
    assert f[x < 0.5, 0].sum() == pytest.approx(-10.0, abs=1e-9)
    assert np.abs(f[:, 1:]).max() < 1e-9


def test_s3_nlgeom_off_c3d8i_ignores_initial_stress_and_is_refused(tmp_path):
    # Measured: C3D8I without NLGEOM returns f_int = 0 for S11 = 10 (kernel drops the field).
    model = _read(tmp_path, _cube_deck("EL, 10.0", "C3D8I"), dim=3)
    with pytest.raises(NotImplementedError, match="initial stress"):
        model.create_solver3d("S1")
