# -*- coding: utf-8 -*-
"""Tests for concrete damaged plasticity on the DP backbone (MAT15)."""
import numpy as np

from dispsolver.material3d.numba_materials import (
    material_dispatch_3d,
    get_default_sdv_count,
    MAT_CONCRETE_DP,
)

E, NU, BETA, PSI = 30000.0, 0.2, 20.0, 15.0
COMP = ((0.0, 30.0), (0.001, 35.0), (0.003, 30.0))
TENS = ((0.0, 3.0), (0.0002, 1.5), (0.001, 0.2))
DTT = ((0.0, 0.0), (0.0005, 0.5), (0.001, 0.9))
DCT = ((0.0, 0.0), (0.002, 0.3), (0.005, 0.7))


def _props():
    flat = [E, NU, BETA, PSI]
    for tab in (COMP, TENS, DTT, DCT):
        flat.append(float(len(tab)))
        for (a, b) in tab:
            flat.append(float(a))
            flat.append(float(b))
    return np.array(flat, dtype=np.float64)


def _call(props, E_voigt, sdv, F=None, detF=1.0, dt=1.0):
    if F is None:
        F = np.eye(3)
    return material_dispatch_3d(MAT_CONCRETE_DP, props, sdv,
                                E_voigt, F, detF, dt)


def test_sdv_count_and_bad_tables():
    assert get_default_sdv_count(MAT_CONCRETE_DP) == 10
    p = _props()
    p[4] = 7
    _, _, _, err = _call(p, np.zeros(6), np.zeros(10))
    assert err == 1


def _uniaxial(props, e_ax, sdv0=None, tol=1e-9):
    # Lateral solve for sigma_lat == 0 (true uniaxial stress state).
    lo, hi = -abs(e_ax), abs(e_ax)
    base = np.zeros(10) if sdv0 is None else sdv0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        S, _, _, _ = _call(props, np.array([e_ax, mid, mid, 0, 0, 0]), base)
        if S[1] > 0:
            hi = mid
        else:
            lo = mid
        if abs(S[1]) < tol:
            break
    return _call(props, np.array([e_ax, mid, mid, 0, 0, 0]), base)


def test_compression_hardening_exact():
    # Confined compression: trial principals all <= 0 -> r = 0 exactly.
    # Single closed-form step: f == 0 with current tables, exact (linear).
    props = _props()
    E_v = np.array([-0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
    S, _, sdv, err = _call(props, E_v, np.zeros(10))
    assert err == 0
    ec = sdv[6]
    assert 0.0 < ec < 0.001
    D = 1.0 - (1.0 - sdv[8]) * (1.0 - sdv[9])
    St = S / max(1.0 - D, 1e-300)
    q = abs(St[0] - St[1])
    h = (St[0] + St[1] + St[2]) / 3.0
    tb = np.tan(np.deg2rad(BETA))
    ref = 30.0 + 5000.0 * ec
    np.testing.assert_allclose(q + h * tb, ref, rtol=1e-6)


def test_tension_softening_exact():
    # Confined tension: trial principals all >= 0 -> r = 1 exactly.
    props = _props()
    E_v = np.array([0.00025, 0.0, 0.0, 0.0, 0.0, 0.0])
    S, _, sdv, err = _call(props, E_v, np.zeros(10))
    assert err == 0
    et = sdv[7]
    assert 0.0 < et < 0.0002
    D = 1.0 - (1.0 - sdv[8]) * (1.0 - sdv[9])
    St = S / max(1.0 - D, 1e-300)
    q = abs(St[0] - St[1])
    h = (St[0] + St[1] + St[2]) / 3.0
    tb = np.tan(np.deg2rad(BETA))
    ref = 3.0 - 7500.0 * et
    np.testing.assert_allclose(q + h * tb, ref, rtol=1e-6)


def test_tension_softening_shape():
    # Multi-step confined tension: peak then decline, damage monotone.
    # Mild softening keeps apex well-posed (denom_a > 0); steep-softening
    # snap-back is covered by test_apex_snapback_error + the solver test.
    tens_mild = ((0.0, 3.0), (0.001, 2.0), (0.003, 0.5))
    flat = [E, NU, BETA, PSI]
    for tab in (COMP, tens_mild, DTT, DCT):
        flat.append(float(len(tab)))
        for (a, b) in tab:
            flat.append(float(a))
            flat.append(float(b))
    props = np.array(flat, dtype=np.float64)
    sdv = np.zeros(10)
    peaks, dts = [], []
    for k in range(1, 16):
        S, _, sdv, err = _call(
            props, np.array([0.0002 * k, 0.0, 0.0, 0.0, 0.0, 0.0]), sdv)
        assert err == 0
        peaks.append(S[0])
        dts.append(sdv[8])
    peaks = np.array(peaks)
    ip = int(np.argmax(peaks))
    assert 0 < ip < len(peaks) - 1
    assert peaks[-1] < 0.5 * peaks[ip]
    assert all(b >= a for a, b in zip(dts, dts[1:]))


def test_apex_snapback_error():
    # Steep softening + apex: no apex solution exists (denom_a <= 0).
    # Kernel must report error (solver cuts back) instead of garbage.
    props = _props()
    _, _, _, err = _call(props, np.array([0.002, 0.002, 0.002, 0.0, 0.0, 0.0]),
                         np.zeros(10))
    assert err == 1


def test_unload_stiffness_degraded():
    # Build damage with small confined steps (k=1..6 stay in normal return;
    # deeper softening would hit apex snap-back, covered separately),
    # then unload ONE small step: stays elastic, SDVs frozen exactly,
    # tangent = degraded elasticity exactly.
    props = _props()
    sdv = np.zeros(10)
    for k in range(1, 7):
        S1, C1, sdv, err = _call(props, np.array([0.00005 * k, 0.0, 0.0,
                                                  0.0, 0.0, 0.0]), sdv)
        assert err == 0
    assert sdv[8] > 0.0
    D = 1.0 - (1.0 - sdv[8]) * (1.0 - sdv[9])
    S2, C2, sdv2, err2 = _call(props, np.array([0.00025, 0.0, 0.0, 0.0, 0.0, 0.0]),
                               sdv)
    assert err2 == 0
    np.testing.assert_allclose(sdv2[6], sdv[6], atol=0.0)
    np.testing.assert_allclose(sdv2[7], sdv[7], atol=0.0)
    K = E / (3.0 * (1.0 - 2.0 * NU))
    mu = E / (2.0 * (1.0 + NU))
    np.testing.assert_allclose(C2[0, 0], (1.0 - D) * (K + 4.0 * mu / 3.0),
                               rtol=1e-9)


def test_apex_hydrostatic_tension():
    # Hydrostatic tension -> apex (q=0). Hardening tension table keeps
    # denom_a > 0; softening tables snap back (err=1, documented).
    tens_h = ((0.0, 3.0), (0.001, 4.0))
    flat = [E, NU, BETA, PSI]
    for tab in (COMP, tens_h, DTT, DCT):
        flat.append(float(len(tab)))
        for (a, b) in tab:
            flat.append(float(a))
            flat.append(float(b))
    props = np.array(flat, dtype=np.float64)
    e = 0.002
    S, _, sdv, err = _call(props, np.array([e, e, e, 0.0, 0.0, 0.0]),
                           np.zeros(10))
    assert err == 0
    np.testing.assert_allclose(S[0], S[1], rtol=1e-9)
    np.testing.assert_allclose(S[1], S[2], rtol=1e-9)
    D = 1.0 - (1.0 - sdv[8]) * (1.0 - sdv[9])
    St = S / max(1.0 - D, 1e-300)
    tb = np.tan(np.deg2rad(BETA))
    et = sdv[7]
    ref = 4.0 + 1000.0 * (et - 0.001) if et > 0.001 else 3.0 + 1000.0 * et
    np.testing.assert_allclose(St[0] * tb, ref, rtol=1e-6)


def test_damage_bounded_monotone():
    props = _props()
    sdv = np.zeros(10)
    dts, dcs = [], []
    for k in range(1, 11):
        _, _, sdv, err = _call(props, np.array([-0.001 * k, 0.0002 * k, 0.0002 * k,
                                                0, 0, 0]), sdv)
        assert err == 0
        dts.append(sdv[8])
        dcs.append(sdv[9])
    assert all(0.0 <= v <= 1.0 for v in dts + dcs)
    assert dcs[-1] >= dcs[0]


def _fd_check(props, E_voigt, sdv, h=1e-8):
    S0, C0, _, err = _call(props, E_voigt, sdv)
    assert err == 0
    rng = np.random.default_rng(11)
    errs = []
    for _ in range(3):
        d = rng.standard_normal(6) * h
        Sp, _, _, _ = _call(props, E_voigt + d, sdv)
        Sm, _, _, _ = _call(props, E_voigt - d, sdv)
        ds = Sp - Sm
        errs.append(float(np.linalg.norm(ds - C0 @ (2 * d)) / max(np.linalg.norm(ds), 1e-30)))
    return max(errs)


def test_tangent_fd_mid_segment():
    props = _props()
    _, _, hist, _ = _call(props, np.array([-0.0012, 0.00024, 0.00024, 0, 0, 0]),
                          np.zeros(10))
    assert _fd_check(props, np.array([-0.0016, 0.00032, 0.00032, 0, 0, 0]),
                     hist) < 1e-5


def test_api_and_packing():
    from dispsolver.model.material import Material
    mat = Material(name="C")
    mat.Elastic(E=E, nu=NU)
    mat.ConcreteDP(beta=20.0, psi=15.0, compression=COMP, tension=TENS,
                   dt_table=DTT, dc_table=DCT)
    props = mat.to_numba_props()
    assert props["cdp_tables"]["compression"][0] == (0.0, 30.0)

    from dispsolver.mesh3d import Mesh3D
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D
    from dispsolver.material3d.numba_materials import MAT_CONCRETE_DP as M15
    mesh = Mesh3D()
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="quasistatic")
    assert int(solver.elem_mat_types[0]) == int(M15)
    np.testing.assert_allclose(solver.elem_props[0, 4], 3.0)


def test_solver_tension_softening_converges():
    from dispsolver.mesh3d import Mesh3D
    from dispsolver.model.material import Material
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D
    mesh = Mesh3D()
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    mat = Material(name="CBar")
    mat.Elastic(E=E, nu=NU)
    mat.ConcreteDP(beta=20.0, psi=15.0, compression=COMP, tension=TENS,
                   dt_table=DTT, dc_table=DCT)
    solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="quasistatic")
    for nid in [1, 4, 5, 8]:
        solver.fix_dof(nid, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)
    nid_map = mesh.node_id_to_index()
    forces = []
    for k in range(1, 8):
        for nid in [2, 3, 6, 7]:
            solver.fix_dof(nid, 0, 0.0002 * k)
        conv, _ = solver.solve_step(dt=1.0)
        assert conv
        rf = solver.compute_reaction_forces()
        forces.append(sum(rf[3 * nid_map[n] + 0] for n in [2, 3, 6, 7]))
    forces = np.array(forces)
    peak = int(np.argmax(forces))
    assert 0 < peak < len(forces) - 1
    assert forces[-1] < 0.5 * forces[peak]
