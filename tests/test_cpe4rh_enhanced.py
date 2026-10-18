"""
tests/test_cpe4rh_enhanced.py
=============================
CPE4RH (dispsolver/element/q4_visco_hybrid_reduced_jax.py) with Abaqus
HOURGLASS=ENHANCED / STIFFNESS, beyond the contract suite rows (CPE4RH,
CPE4RH_STIFFNESS in tests/element_contract/).

All probes go through `DynamicSolver._assemble()` / `solve_step()` with the
dict form of `element_type` (the batch path production runs), except the
history test, which uses the uniform-string form on purpose (sequential
path) -- both are covered.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import jax
jax.config.update("jax_enable_x64", True)

from dispsolver.mesh import Mesh
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.material.arruda_boyce import ArrudaBoyce

_LM = 3.0
_BETA = (1.0 + 3.0 / (5.0 * _LM ** 2) + 99.0 / (175.0 * _LM ** 4)
         + 513.0 / (875.0 * _LM ** 6) + 42039.0 / (67375.0 * _LM ** 8))


def _ab_params(E, nu):
    """Arruda-Boyce (mu, K) whose initial moduli are (E, nu) -- `mu` is not
    the initial shear modulus (AGENTS.md 1.4, tests/test_cpe4rh_patch.py)."""
    G = E / (2.0 * (1.0 + nu))
    return {"mu": G / _BETA, "lambda_m": _LM, "K": E / (3.0 * (1.0 - 2.0 * nu))}


REF = _ab_params(4000.0, 0.3)
# ex12's PSA: mu = 0.16785, K = 8.3333 -> initial G0 = 0.181, nu = 0.489.
PSA = {"mu": 0.16785, "lambda_m": _LM, "K": 8.3333}


def _mat():
    return ViscoelasticMaterial(ArrudaBoyce(), g_i=[0.0], tau_i=[1.0])


def _solver(mesh, etype, params, uniform=False, **kw):
    s = DynamicSolver(mesh, {0: _mat()} if not uniform else _mat(), rho=1.0,
                      material_params=({0: dict(params)} if not uniform else dict(params)),
                      element_type=(etype if uniform else {0: etype}),
                      nlgeom=True, elem_jit="jax", mode="quasistatic", verbose=False, **kw)
    s.sta_status = False
    return s


def _single(coords):
    m = Mesh()
    for i, xy in enumerate(coords):
        m.add_node(i + 1, float(xy[0]), float(xy[1]))
    m.add_element(1, [1, 2, 3, 4], "Q4", 0)
    return m


def _rot8(th):
    c, s = np.cos(th), np.sin(th)
    T = np.zeros((8, 8))
    for a in range(4):
        T[2 * a:2 * a + 2, 2 * a:2 * a + 2] = [[c, -s], [s, c]]
    return T


DISTORTED = np.array([[0.0, 0.0], [1.1, 0.1], [0.9, 0.3], [-0.1, 0.25]])


@pytest.mark.parametrize("label,params", [("nu0.3", REF), ("PSA", PSA)])
def test_enhanced_small_strain_stiffness_equals_cpe4i(label, params):
    """K(u=0) of CPE4RH/ENHANCED == CPE4I's, on a distorted quad.

    The hybrid part is consistent with the mixed formulation: one constant
    pressure p = K (J - 1) at the single point is exactly the constant-
    pressure part of the u-p field, so the 1-point term reproduces Q1E4's
    constant-strain (incl. volumetric) energy, and the hourglass term built
    from Q1E4 with lam0 = K - 2 G0 / 3 supplies the rest -- at nu = 0.489
    (PSA) as well as 0.3. Measured 2026-10-11: nu 0.3 1.8e-10, PSA 2.4e-10
    (roundoff/AGENTS.md 4.3 noise level); STIFFNESS differs by 1.2e-01 /
    1.4e-02.
    """
    m = _single(DISTORTED)
    K = {}
    for name, et, kw in (("rh", "CPE4RH", {"hourglass": "ENHANCED"}),
                         ("i", "CPE4I", {}),
                         ("stf", "CPE4RH", {"hourglass": "STIFFNESS"})):
        K[name] = np.asarray(_solver(m, et, params, **kw)._assemble(np.zeros(8), 1.0)[1].todense())
    err = np.max(np.abs(K["rh"] - K["i"])) / np.max(np.abs(K["i"]))
    err_s = np.max(np.abs(K["stf"] - K["i"])) / np.max(np.abs(K["i"]))
    assert err < 1e-8, err
    assert err_s > 1e-3, err_s


def test_unset_hourglass_resolves_to_enhanced_for_cpe4rh():
    """Keywords Guide: ENHANCED is the Abaqus/Standard default for
    hyperelastic materials; CPE4R (J2) keeps STIFFNESS."""
    s = _solver(_single(DISTORTED), "CPE4RH", REF)
    assert s._hourglass_for(0) == "ENHANCED"
    assert s.element_large_deformation_report is not None
    from dispsolver.solver.dynamic import _ELEMENT_LARGE_DEF
    assert _ELEMENT_LARGE_DEF["CPE4RH"]["supports_ul"] is False


def test_hourglass_force_objective_under_superposed_rotation():
    """f(Q x) = T8(Q) f(x) with hourglass content on an AR-7 element.
    Measured 2026-10-11: see the assertion; CPE4R's twin test measured 5.5e-14."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.15], [0.0, 0.15]])
    h = np.array([1.0, -1.0, 1.0, -1.0])
    u0 = (coords @ (np.array([[1.03, 0.01], [0.0, 0.98]]) - np.eye(2)).T).reshape(-1)
    u0[0::2] += 4e-3 * h
    u0[1::2] += 1e-3 * h
    s = _solver(_single(coords), "CPE4RH", REF, hourglass="ENHANCED")
    x0 = coords + u0.reshape(4, 2)
    f0 = np.asarray(s._assemble(u0, 1.0)[0])
    worst = 0.0
    for th in (0.3, 1.0, np.pi / 2, 2.5):
        c, sn = np.cos(th), np.sin(th)
        u = (x0 @ np.array([[c, -sn], [sn, c]]).T - coords).reshape(-1)
        f = np.asarray(s._assemble(u, 1.0)[0])
        worst = max(worst, np.max(np.abs(f - _rot8(th) @ f0)) / np.max(np.abs(f0)))
    assert worst < 1e-10, worst


# ----------------------------------------------------------------------
# Bending mesh independence (the 2026-07-30 cantilever-stack measurement)
# ----------------------------------------------------------------------
def _cantilever(etype, params, dx, n_rows, L=4.0, h=0.2, **kw):
    nx = int(round(L / dx))
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, h, n_rows + 1)
    m = Mesh()
    nid = {}
    k = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            m.add_node(k, float(x), float(y))
            nid[(i, j)] = k
            k += 1
    e = 1
    for j in range(n_rows):
        for i in range(nx):
            m.add_element(e, [nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)],
                              nid[(i, j + 1)]], "Q4", 0)
            e += 1
    return m, nid, nx


def _tip_stiffness(etype, params, dx, n_rows, **kw):
    m, nid, nx = _cantilever(etype, params, dx, n_rows)
    s = _solver(m, etype, params, **kw)
    _, K, _ = s._assemble(np.zeros(s.n_dofs), 1.0)
    K = K.tocsr()
    idx = {n: i for i, n in enumerate(s.sorted_nids)}
    fixed = [2 * idx[nid[(0, j)]] + d for j in range(n_rows + 1) for d in (0, 1)]
    free = np.setdiff1d(np.arange(s.n_dofs), fixed)
    tip = [2 * idx[nid[(nx, j)]] + 1 for j in range(n_rows + 1)]
    f = np.zeros(s.n_dofs)
    f[tip] = 1.0 / len(tip)
    u = np.zeros(s.n_dofs)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    return 1.0 / float(np.mean(u[tip]))


@pytest.mark.parametrize("label,params,n_rows", [("nu0.3", REF, 1), ("nu0.3", REF, 4),
                                                 ("PSA", PSA, 1)])
def test_cantilever_mesh_independence(label, params, n_rows):
    """k(dx=0.125) / k(dx=0.5) on a 4.0 x 0.2 plane-strain cantilever.

    Measured 2026-10-11 (ratio; in brackets k(dx 0.5) / k_CPE4I(dx 0.5)):

        case        CPE4RH-ENH       CPE4I    CPE4RH-STF       CPE4H
        nu0.3 1row  0.9963 [1.000]   0.9963   0.1911 [0.152]   0.2273 [2.65]
        nu0.3 4row  0.9831 [1.000]   0.9831   0.8811 [1.053]   0.3495 [3.10]
        PSA   1row  0.9963 [1.000]   0.9963   0.1911 [0.111]   0.2273 [1.93]

    CPE4RH/ENHANCED equals CPE4I on every mesh (same small-strain stiffness).
    STIFFNESS (0.005 G0) is far too soft at one row and its stiffness moves
    5x with dx. CPE4H -- the ex12 PSA default -- shear-locks: 1.9-3.1x too
    stiff at dx 0.5 and 4.4x mesh dependence.
    """
    k = {}
    for name, et, kw in (("enh", "CPE4RH", {"hourglass": "ENHANCED"}),
                         ("i", "CPE4I", {}),
                         ("stf", "CPE4RH", {"hourglass": "STIFFNESS"}),
                         ("h", "CPE4H", {})):
        k[name] = (_tip_stiffness(et, params, 0.125, n_rows, **kw),
                   _tip_stiffness(et, params, 0.5, n_rows, **kw))
    r = {n: v[0] / v[1] for n, v in k.items()}
    assert abs(r["enh"] / r["i"] - 1.0) < 1e-6, r
    assert abs(k["enh"][1] / k["i"][1] - 1.0) < 1e-6, k
    assert abs(r["enh"] - 1.0) < 0.02, r
    assert abs(r["stf"] - 1.0) > 0.1, r      # characterization
    assert abs(r["h"] - 1.0) > 0.3, r        # characterization


# ----------------------------------------------------------------------
# Hourglass resistance must survive converged increments (TL dispatch)
# ----------------------------------------------------------------------
def _cantilever_solve(n_steps, uniform):
    m, nid, nx = _cantilever("CPE4RH", REF, 0.5, 1)
    s = _solver(m, "CPE4RH", REF, uniform=uniform, hourglass="ENHANCED",
                max_iter=30, tol=1e-10, atol=1e-12, rtol=1e-9)
    idx = {n: i for i, n in enumerate(s.sorted_nids)}
    clamp = [2 * idx[nid[(0, j)]] + d for j in range(2) for d in (0, 1)]
    tip = [2 * idx[nid[(nx, j)]] + 1 for j in range(2)]
    for k in range(1, n_steps + 1):
        d = 0.2 * k / n_steps
        s.set_prescribed_dofs(clamp + tip, [0.0] * 4 + [d, d])
        assert s.solve_step(dt=1.0 / n_steps) > 0
    return s.u.copy()


@pytest.mark.parametrize("uniform", [False, True], ids=["batch", "sequential"])
def test_bending_history_independent(uniform):
    """Elastic bending reached in 1 or 4 increments must be the same state.

    Under the former UL dispatch the hourglass force was rebuilt from the
    INCREMENTAL displacement on config n, so the accumulated hourglass
    resistance -- with ENHANCED, the element's bending stiffness -- reset at
    every converged increment. Measured 2026-10-11 (tip deflection 0.2 on a
    4.0 x 0.2 one-row cantilever): 1-vs-4-step max |du| / max |u| = 1.1e-07
    on both paths (the Newton tolerance); with supports_ul forced back to
    True on the batch path, 2.2e-02.
    """
    u1 = _cantilever_solve(1, uniform)
    u4 = _cantilever_solve(4, uniform)
    err = np.max(np.abs(u1 - u4)) / np.max(np.abs(u1))
    assert err < 1e-6, err
