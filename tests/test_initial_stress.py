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
    # input is Abaqus order S11 S22 S33 S12 S13 S23; the kernels store xx yy zz xy yz xz, so S13 and S23 swap.
    # That the swap is right is proven by test_initial_stress_elements (sum f_ia X_ib = s_ab V), not here.
    expected = np.array([10.0, 20.0, 30.0, 4.0, 6.0, 5.0])
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


def test_s3_kernel_that_does_not_apply_the_field_is_refused(tmp_path, monkeypatch):
    # PRD R4: a kernel the catalog does not mark as applying the field is refused, never silently
    # solved without it. (Every real kernel now applies it, so empty one catalog entry for the test.)
    from dataclasses import replace
    from dispsolver.model.element_catalog import KERNELS
    monkeypatch.setitem(KERNELS[3], 16, replace(KERNELS[3][16], initial_stress=""))
    model = _read(tmp_path, _cube_deck("EL, 10.0", "C3D8"), dim=3)
    with pytest.raises(NotImplementedError, match="initial stress"):
        model.create_solver3d("S1")      # NLGEOM=NO C3D8 runs kernel 16


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


def _moment(solver, sysm):
    solver.assemble_system(np.zeros(solver.num_dofs))
    return np.asarray(solver.f_int).reshape(-1, 3).T @ np.asarray(sysm.coords)


def test_scalar_value_is_a_hydrostatic_stress(tmp_path):
    # unit cube, s0 = p I  ->  sum_i f_ia X_ib = p * V * delta_ab, V = 1
    model = _read(tmp_path, _cube_deck("EL, 7.0"), dim=3)
    # the reader stores a row of values; swap in the scalar form of the Python API
    pf = next(iter(model.initial_step.predefined_fields.values()))
    pf.values = 7.0
    solver, sysm = model.create_solver3d()
    np.testing.assert_allclose(_moment(solver, sysm), 7.0 * np.eye(3), atol=1e-9)


def test_callable_value_uses_abaqus_component_order(tmp_path):
    model = _read(tmp_path, _cube_deck("EL, 0.0"), dim=3)
    pf = next(iter(model.initial_step.predefined_fields.values()))
    pf.values = lambda x, y, z: [1.0, 2.0, 3.0, 0.4, 0.5, 0.6]       # S11 S22 S33 S12 S13 S23
    solver, sysm = model.create_solver3d()
    expected = np.array([[1.0, 0.4, 0.5], [0.4, 2.0, 0.6], [0.5, 0.6, 3.0]])   # unit cube, V = 1
    np.testing.assert_allclose(_moment(solver, sysm), expected, atol=1e-9)
