# -*- coding: utf-8 -*-
"""Tests for linear Drucker-Prager plasticity (MAT14).

Conventions: h = tension-positive mean stress, f = q + h*tanb - (d0+H*eq).
"""
import numpy as np

from dispsolver.material3d.numba_materials import (
    material_dispatch_3d,
    get_default_sdv_count,
    MAT_DRUCKER_PRAGER,
    MAT_J2_PLASTICITY,
)

E, NU, D0, H = 200000.0, 0.3, 100.0, 0.0
BETA, PSI = 20.0, 15.0
TB = np.tan(np.deg2rad(BETA))


def _props(beta=BETA, psi=PSI, d0=D0, h=H):
    return np.array([E, NU, d0, h, beta, psi], dtype=np.float64)


def _call(props, E_voigt, sdv, F=None, detF=1.0, dt=1.0):
    if F is None:
        F = np.eye(3)
    return material_dispatch_3d(MAT_DRUCKER_PRAGER, props, sdv,
                                E_voigt, F, detF, dt)


def test_sdv_count():
    assert get_default_sdv_count(MAT_DRUCKER_PRAGER) == 7


def test_beta_zero_reduces_to_j2():
    props = _props(beta=0.0, psi=0.0, d0=250.0, h=2000.0)
    pj = np.array([E, NU, 250.0, 2000.0])
    rng = np.random.default_rng(2)
    for _ in range(3):
        e = rng.standard_normal(6) * 0.002
        e[3:] *= 0.5
        S, C, _, err = _call(props, e, np.zeros(7))
        Sj, Cj, _, _ = material_dispatch_3d(
            MAT_J2_PLASTICITY, pj, np.zeros(7), e, np.eye(3), 1.0, 1.0)
        assert err == 0
        np.testing.assert_allclose(S, Sj, atol=1e-9)
        np.testing.assert_allclose(C, Cj, atol=1e-9)


def _uniaxial_stress(props, e_ax, tol=1e-9):
    # Solve lateral strain for sigma_lat == 0 (bisect, monotone in e_lat).
    lo, hi = -abs(e_ax), abs(e_ax)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        S, _, _, _ = _call(props, np.array([e_ax, mid, mid, 0, 0, 0]),
                           np.zeros(7))
        if S[1] > 0:
            hi = mid
        else:
            lo = mid
        if abs(S[1]) < tol:
            break
    S, _, _, _ = _call(props, np.array([e_ax, mid, mid, 0, 0, 0]),
                       np.zeros(7))
    return S[0]


def test_tension_compression_asymmetry():
    props = _props()
    st = D0 / (1.0 + TB / 3.0)
    sc = D0 / (1.0 - TB / 3.0)
    assert sc > st
    np.testing.assert_allclose(_uniaxial_stress(props, 0.002), st, rtol=0.02)
    np.testing.assert_allclose(-_uniaxial_stress(props, -0.002), sc, rtol=0.02)


def test_hydrostatic_no_flow():
    # Hydrostatic COMPRESSION stays elastic; hydrostatic tension flows
    # to the apex (covered by test_apex_hydrostatic_tension).
    props = _props()
    e = -0.001
    S, _, sdv, err = _call(props, np.array([e, e, e, 0, 0, 0]), np.zeros(7))
    assert err == 0
    np.testing.assert_allclose(sdv[6], 0.0, atol=1e-15)
    np.testing.assert_allclose(S[0], S[1], rtol=1e-12)


def test_apex_hydrostatic_tension():
    props = _props()
    e = 0.002
    S, _, sdv, err = _call(props, np.array([e, e, e, 0, 0, 0]), np.zeros(7))
    assert err == 0
    np.testing.assert_allclose(S[0], S[1], rtol=1e-12)
    np.testing.assert_allclose(S[0], D0 / TB, rtol=0.02)
    q = np.sqrt(0.5 * ((S[0] - S[1]) ** 2))
    assert q < 1e-9 * max(abs(S[0]), 1.0)


def test_dilation_sign():
    props = _props()
    _, _, sdv, _ = _call(props, np.array([0.004, -NU * 0.004, -NU * 0.004,
                                          0, 0, 0]), np.zeros(7))
    assert sdv[6] > 0.0
    assert sdv[0] + sdv[1] + sdv[2] > 0.0


def _fd_check(props, E_voigt, sdv, h=1e-8):
    S0, C0, _, err = _call(props, E_voigt, sdv)
    assert err == 0
    rng = np.random.default_rng(6)
    errs = []
    for _ in range(3):
        d = rng.standard_normal(6) * h
        Sp, _, _, _ = _call(props, E_voigt + d, sdv)
        Sm, _, _, _ = _call(props, E_voigt - d, sdv)
        ds = Sp - Sm
        errs.append(float(np.linalg.norm(ds - C0 @ (2 * d)) / max(np.linalg.norm(ds), 1e-30)))
    return max(errs)


def test_tangent_fd():
    props = _props()
    rng = np.random.default_rng(7)
    e = rng.standard_normal(6) * 0.003
    e[3:] *= 0.5
    assert _fd_check(props, e, np.zeros(7)) < 1e-6


def test_api_and_packing():
    from dispsolver.model.material import Material
    mat = Material(name="DP")
    mat.Elastic(E=E, nu=NU)
    mat.DruckerPrager(beta=20.0, psi=15.0, d0=100.0, H=0.0)
    props = mat.to_numba_props()
    assert props["beta"] == 20.0 and props["psi"] == 15.0

    from dispsolver.mesh3d import Mesh3D
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D
    from dispsolver.material3d.numba_materials import MAT_DRUCKER_PRAGER as M14
    mesh = Mesh3D()
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="quasistatic")
    assert int(solver.elem_mat_types[0]) == int(M14)
    np.testing.assert_allclose(solver.elem_props[0, 4], 20.0)


def test_solver_tension_weaker_than_compression():
    from dispsolver.mesh3d import Mesh3D
    from dispsolver.model.material import Material
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D

    def run(sign):
        mesh = Mesh3D()
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                  (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
        for nid, (x, y, z) in enumerate(coords, start=1):
            mesh.add_node(nid, x, y, z)
        mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
        mat = Material(name="DPBar")
        mat.Elastic(E=E, nu=NU)
        mat.DruckerPrager(beta=20.0, psi=15.0, d0=100.0, H=500.0)
        solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="quasistatic")
        for nid in [1, 4, 5, 8]:
            solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(1, 1, 0.0)
        solver.fix_dof(1, 2, 0.0)
        nid_map = mesh.node_id_to_index()
        peak = 0.0
        for k in range(1, 11):
            for nid in [2, 3, 6, 7]:
                solver.fix_dof(nid, 0, sign * 0.001 * k)
            conv, _ = solver.solve_step(dt=1.0)
            assert conv
            rf = solver.compute_reaction_forces()
            f = abs(sum(rf[3 * nid_map[n] + 0] for n in [2, 3, 6, 7]))
            peak = max(peak, f)
        return peak

    assert run(-1.0) > run(1.0) * 1.1
