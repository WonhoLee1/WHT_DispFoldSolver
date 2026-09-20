# -*- coding: utf-8 -*-
"""Tests for ductile/JC damage initiation, evolution, and MAT13 dispatch."""
import numpy as np

from dispsolver.material.damage_initiation import (
    fracture_strain_const,
    fracture_strain_triax_table,
    fracture_strain_jc,
    failure_displacement,
    evolution_linear,
    evolution_exponential,
)
from dispsolver.material3d.numba_materials import (
    material_dispatch_3d,
    get_default_sdv_count,
    MAT_J2_DUCTILE_DAMAGE,
)

E, NU, SY0, H = 200000.0, 0.3, 250.0, 2000.0


def _props_const(eD=0.05, Gf=10.0, L=1.0, evol=0):
    return np.array([E, NU, SY0, H, 0, eD, 0, 0, 0, 0, Gf, L, evol],
                    dtype=np.float64)


def _uniax(e):
    return np.array([e, -NU * e, -NU * e, 0.0, 0.0, 0.0], dtype=np.float64)


def _call(props, E_voigt, sdv, F=None, detF=1.0, dt=1.0):
    if F is None:
        F = np.eye(3)
    return material_dispatch_3d(MAT_J2_DUCTILE_DAMAGE, props, sdv,
                                E_voigt, F, detF, dt)


def test_helpers_exact():
    assert fracture_strain_const(0.05) == 0.05
    assert fracture_strain_triax_table(0.0, 0.0, 0.1, 1.0, 0.5) == 0.1
    assert fracture_strain_triax_table(2.0, 0.0, 0.1, 1.0, 0.5) == 0.5
    np.testing.assert_allclose(
        fracture_strain_triax_table(0.5, 0.0, 0.1, 1.0, 0.5), 0.3, atol=1e-15)
    np.testing.assert_allclose(fracture_strain_jc(1.0 / 3.0, 0.05, 0.0, 0.0), 0.05)
    np.testing.assert_allclose(failure_displacement(10.0, 250.0), 0.08)
    assert evolution_linear(0.04, 0.08) == 0.5
    np.testing.assert_allclose(evolution_exponential(1e9, 0.08), 1.0)


def test_sdv_count():
    assert get_default_sdv_count(MAT_J2_DUCTILE_DAMAGE) == 10


def test_initiation_accumulation_linear():
    props = _props_const(eD=0.05)
    sdv = np.zeros(10)
    F = np.eye(3)
    for k in range(1, 6):
        S, C, sdv, err = _call(props, _uniax(0.001 * k), sdv, F)
        assert err == 0
    np.testing.assert_allclose(sdv[7], sdv[6] / 0.05, rtol=1e-9)
    assert sdv[9] == 0.0


def test_jc_reduces_to_const():
    pj = np.array([E, NU, SY0, H, 2, 0.05, 0, 0, 0, 0, 10.0, 1.0, 0],
                  dtype=np.float64)
    pc = _props_const(eD=0.05)
    sj, Cj, _, _ = _call(pj, _uniax(0.004), np.zeros(10))
    sc, Cc, _, _ = _call(pc, _uniax(0.004), np.zeros(10))
    np.testing.assert_allclose(sj, sc, atol=1e-12)
    np.testing.assert_allclose(Cj, Cc, atol=1e-12)


def test_error_flags():
    p = _props_const()
    p[10] = 0.0
    _, _, _, err = _call(p, _uniax(0.004), np.zeros(10))
    assert err == 1
    # D4/D5 nonzero is supported since the rate/T update (no error).
    pj = np.array([E, NU, SY0, H, 2, 0.05, 0, 0, 0.5, 0, 10.0, 1.0, 0,
                   20.0, 20.0, 1000.0, 1.0],
                  dtype=np.float64)
    _, _, _, err = _call(pj, _uniax(0.004), np.zeros(10))
    assert err == 0


def _props_jc_rateT(D4=0.0, D5=0.0):
    return np.array([E, NU, SY0, H, 2, 0.05, 0.0, 0.0, D4, D5, 10.0, 1.0, 0,
                     20.0, 20.0, 1000.0, 1.0], dtype=np.float64)


def test_jc_rate_ordering():
    # Faster straining raises eD (D4 > 0): same strain -> less damage.
    urs = []
    for dt in (1.0, 0.01):
        _, _, sdv, _ = _call(_props_jc_rateT(D4=0.5), _uniax(0.004),
                             np.zeros(10), dt=dt)
        urs.append(sdv[7])
    assert urs[0] > urs[1] > 0.0


def test_jc_temperature_ordering():
    # Hotter raises eD (D5 > 0): same strain -> less damage.
    urs = []
    for T in (20.0, 500.0):
        p = _props_jc_rateT(D5=0.5)
        p[13] = T
        _, _, sdv, _ = _call(p, _uniax(0.004), np.zeros(10))
        urs.append(sdv[7])
    assert urs[0] > urs[1] > 0.0


def test_jc_combined_closed_form():
    # Keep the rate inside the physical branch (rate > eps0) so the D4 log
    # term stays positive and the closed form holds exactly.
    D1, D4, D5 = 0.05, 0.5, 0.5
    p = _props_jc_rateT(D4=D4, D5=D5)
    p[13] = 500.0
    dt, e = 1e-4, 0.004
    _, _, sdv, _ = _call(p, _uniax(e), np.zeros(10), dt=dt)
    rate = sdv[6] / dt
    assert rate > 1.0
    tstar = (500.0 - 20.0) / (1000.0 - 20.0)
    ref = D1 * (1.0 + D4 * np.log(rate / 1.0)) * (1.0 + D5 * tstar)
    np.testing.assert_allclose(sdv[7], sdv[6] / ref, rtol=1e-9)


def test_no_healing_on_unload():
    props = _props_const(eD=0.002, Gf=1.0, L=1.0)
    sdv = np.zeros(10)
    S1, _, sdv, _ = _call(props, _uniax(0.004), sdv)
    D = sdv[9]
    assert D > 0.0
    S2, _, sdv2, _ = _call(props, _uniax(0.003), sdv)
    S3, _, _, _ = _call(props, _uniax(0.003), sdv)
    np.testing.assert_allclose(S2, S3, atol=1e-12)
    np.testing.assert_allclose(sdv2[9], D, atol=1e-15)


def test_drive_to_failure_softens():
    props = _props_const(eD=0.002, Gf=1.0, L=1.0)
    sdv = np.zeros(10)
    prev = 1e300
    for k in range(1, 40):
        S, _, sdv, err = _call(props, _uniax(0.001 * k), sdv)
        assert err == 0
        if sdv[9] >= 1.0:
            break
    assert sdv[9] >= 1.0
    assert abs(S[0]) < prev
    assert abs(S[0]) < 0.05 * SY0


def _fd_check(props, E_voigt, sdv, h=1e-8):
    S0, C0, _, err = _call(props, E_voigt, sdv)
    assert err == 0
    rng = np.random.default_rng(4)
    errs = []
    for _ in range(3):
        d = rng.standard_normal(6) * h
        Sp, _, _, _ = _call(props, E_voigt + d, sdv)
        Sm, _, _, _ = _call(props, E_voigt - d, sdv)
        ds = Sp - Sm
        errs.append(float(np.linalg.norm(ds - C0 @ (2 * d)) / max(np.linalg.norm(ds), 1e-30)))
    return max(errs)


def test_tangent_fd_undamaged_and_damaged():
    props = _props_const(eD=0.05, Gf=10.0, L=1.0)
    assert _fd_check(props, _uniax(0.004), np.zeros(10)) < 1e-6
    # Damaged FD must sit strictly inside the plastic regime: evolve history
    # at a smaller strain, then FD at a larger one. FD at the history's own
    # strain straddles the yield-surface kink (f ~= 0) and is meaningless.
    pd = _props_const(eD=0.002, Gf=1.0, L=1.0)
    _, _, hist, _ = _call(pd, _uniax(0.0038), np.zeros(10), F=np.eye(3))
    assert 0.0 < hist[9] < 1.0
    assert _fd_check(pd, _uniax(0.005), hist) < 1e-5


def test_material_api_and_packing():
    from dispsolver.model.material import Material
    mat = Material(name="Dmg")
    mat.Elastic(E=E, nu=NU)
    mat.Plastic(table=[(SY0, 0.0), (SY0 + H * 0.01, 0.01)])
    mat.DamageInitiation(type="DUCTILE", table=((0.02,),))
    mat.DamageEvolution(type="ENERGY_LINEAR", Gf=1.0, L=1.0)
    props = mat.to_numba_props()
    assert props["damage_init_type"] == 0
    assert props["damage_init_p"][0] == 0.02
    assert props["damage_Gf"] == 1.0
    assert props["damage_L"] == 1.0
    assert props["damage_evol"] == 0
    np.testing.assert_allclose(props["hardening_modulus"], H, rtol=1e-12)

    from dispsolver.mesh3d import Mesh3D
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D
    from dispsolver.material3d.numba_materials import MAT_J2_DUCTILE_DAMAGE as M13
    mesh = Mesh3D()
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="quasistatic")
    assert int(solver.elem_mat_types[0]) == int(M13)
    np.testing.assert_allclose(solver.elem_props[0, 2], SY0, rtol=1e-12)
    np.testing.assert_allclose(solver.elem_props[0, 10], 1.0, rtol=1e-12)


def test_solver_uniaxial_to_failure():
    from dispsolver.mesh3d import Mesh3D
    from dispsolver.model.material import Material
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D
    mesh = Mesh3D()
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    mat = Material(name="DmgBar")
    mat.Elastic(E=E, nu=NU)
    mat.Plastic(table=[(SY0, 0.0), (SY0 + H * 0.01, 0.01)])
    mat.DamageInitiation(type="DUCTILE", table=((0.01,),))
    mat.DamageEvolution(type="ENERGY_LINEAR", Gf=2.0, L=1.0)
    solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="quasistatic")
    for nid in [1, 4, 5, 8]:
        solver.fix_dof(nid, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)
    nid_map = mesh.node_id_to_index()
    forces = []
    for k in range(1, 21):
        for nid in [2, 3, 6, 7]:
            solver.fix_dof(nid, 0, 0.002 * k)
        conv, _ = solver.solve_step(dt=1.0)
        assert conv
        rf = solver.compute_reaction_forces()
        forces.append(sum(rf[3 * nid_map[n] + 0] for n in [2, 3, 6, 7]))
    forces = np.array(forces)
    peak = int(np.argmax(forces))
    assert 0 < peak < len(forces) - 1
    assert forces[-1] < 0.1 * forces[peak]
