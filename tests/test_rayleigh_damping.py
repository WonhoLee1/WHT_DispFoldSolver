# -*- coding: utf-8 -*-
"""Tests for Rayleigh damping in the HHT/Newmark 3D branch.

Setup: unit C3D8 steel cube, X=0 face held, X=1 face released from a
uniform 0.001 axial stretch. First axial mode dominates tip-x history.
"""
import numpy as np

from dispsolver.mesh3d import Mesh3D
from dispsolver.model.material import Material
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

E, NU, RHO = 200000.0, 0.3, 7.85e-9
DT = 2e-8
N_STEPS = 600


def _bar_solver():
    mesh = Mesh3D()
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    mat = Material(name="SteelBar")
    mat.Elastic(E=E, nu=NU)
    solver = DynamicSolver3D(mesh, materials={0: mat},
                             time_integrator="newmark", density=RHO)
    for nid in [1, 4, 5, 8]:
        solver.fix_dof(nid, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)
    solver.fix_dof(4, 2, 0.0)
    nid_map = mesh.node_id_to_index()
    for nid in [2, 3, 6, 7]:
        solver.u[3 * nid_map[nid] + 0] = 0.001
    return solver, nid_map


def _run(solver, n=N_STEPS):
    tip = []
    for _ in range(n):
        conv, _ = solver.solve_step(dt=DT)
        assert conv
        tip.append(solver.u[3 * solver.mesh.node_id_to_index()[2] + 0])
    return np.array(tip)


def _peaks(h):
    idx = []
    for i in range(1, len(h) - 1):
        if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0:
            idx.append(i)
    return np.array(idx)


def _first_mode(solver, tip_dof):
    # One zero step populates M_global; K at rest; free-DOF eigenpair.
    # The lowest modes are bending -- select the axial mode by max tip-x
    # amplitude (its effective mass dominates x-translation).
    solver.solve_step(dt=DT)
    from scipy.linalg import eigh
    K, _ = solver.assemble_system(solver.u, dt=DT, update_state=False)
    if hasattr(K, "toarray"):
        K = K.toarray()
    M = solver.M_global.toarray()
    fixed = sorted(solver.fixed_dofs.keys())
    free = [i for i in range(solver.num_dofs) if i not in set(fixed)]
    w2, vec = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    tip_local = free.index(tip_dof)
    axial = int(np.argmax(abs(vec[tip_local, :])))
    om = float(np.sqrt(w2[axial]))
    phi = np.zeros(solver.num_dofs)
    phi[free] = vec[:, axial]
    return om, phi


def _modal_decay(alpha, beta, xi):
    s0, _ = _bar_solver()
    tip = 3 * s0.mesh.node_id_to_index()[2] + 0
    om1, phi = _first_mode(s0, tip)
    assert abs(xi - (alpha / (2 * om1) + beta * om1 / 2)) < 1e-9
    s1, _ = _bar_solver()
    s1.solve_step(dt=DT)
    M = s1.M_global.toarray()
    s1.set_rayleigh_damping(alpha, beta)
    s1.u[:] = 0.001 * phi / max(abs(phi).max(), 1e-30)
    s1.v[:] = 0.0
    s1.a[:] = 0.0
    c = []
    for _ in range(N_STEPS):
        conv, _ = s1.solve_step(dt=DT)
        assert conv
        c.append(float(phi @ (M @ s1.u)))
    c = np.array(c)
    p = _peaks(c)
    assert len(p) >= 8
    rs = [c[p[i + 1]] / c[p[i]] for i in range(2, 7)]
    theory = np.exp(-2 * np.pi * xi / np.sqrt(1 - xi ** 2))
    return float(np.mean(rs)), theory


def _per_cycle_ratio(h):
    # Two-mode beating (uniform-stretch release excites higher axial modes)
    # averages out: mean of consecutive positive-peak ratios over the window
    # where mode 1 dominates (early contaminant dead, late floor not yet).
    p = _peaks(h[50:])
    assert len(p) >= 10
    rs = [h[50 + p[i + 1]] / h[50 + p[i]] for i in range(1, 9)]
    return float(np.mean(rs))


def _period(h):
    p = _peaks(h)
    assert len(p) >= 3
    return float(np.mean(np.diff(p))) * DT


def test_zero_damping_bit_identical():
    s1, _ = _bar_solver()
    s2, _ = _bar_solver()
    s2.set_rayleigh_damping(0.0, 0.0)
    for _ in range(10):
        s1.solve_step(dt=DT)
        s2.solve_step(dt=DT)
    assert np.array_equal(s1.u, s2.u)


def test_mass_proportional_decay():
    xi = 0.03
    s0, _ = _bar_solver()
    tip = 3 * s0.mesh.node_id_to_index()[2] + 0
    om1, _ = _first_mode(s0, tip)
    ratio, theory = _modal_decay(2 * xi * om1, 0.0, xi)
    np.testing.assert_allclose(ratio, theory, rtol=0.03)


def test_stiffness_proportional_decay():
    xi = 0.03
    s0, _ = _bar_solver()
    tip = 3 * s0.mesh.node_id_to_index()[2] + 0
    om1, _ = _first_mode(s0, tip)
    ratio, theory = _modal_decay(0.0, 2 * xi / om1, xi)
    np.testing.assert_allclose(ratio, theory, rtol=0.03)


def test_energy_dissipation_monotone():
    s0, _ = _bar_solver()
    T = _period(_run(s0))
    om1 = 2 * np.pi / T
    s1, _ = _bar_solver()
    s1.set_rayleigh_damping(2 * 0.05 * om1, 0.0)
    E = []
    for _ in range(N_STEPS):
        s1.solve_step(dt=DT)
        K, _ = s1.assemble_system(s1.u, dt=DT, update_state=False)
        M = s1.M_global
        E.append(0.5 * s1.u @ (K @ s1.u) + 0.5 * s1.v @ (M @ s1.v))
    E = np.array(E)
    assert np.all(E[1:] <= E[:-1] * (1 + 1e-6))
    assert E[-1] < 0.5 * E[0]
