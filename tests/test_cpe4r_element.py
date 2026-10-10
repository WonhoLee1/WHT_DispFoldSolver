"""
tests/test_cpe4r_element.py
===========================
Abaqus CPE4R (dispsolver/element/cpe4r_jax.py) beyond the element contract
suite (tests/element_contract/, where CPE4R and CPE4R_STIFFNESS run C1-C9).

Every probe goes through `DynamicSolver._assemble()` with the dict form of
`element_type`, i.e. the production dispatch path (see the contract
harness docstring for why).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import jax
jax.config.update("jax_enable_x64", True)

from dispsolver.mesh import Mesh
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.material.plastic import J2Plasticity
from dispsolver.material.neohookean import NeoHookean

E, NU = 4000.0, 0.3
E_PS = E / (1.0 - NU ** 2)


def _elastic():
    return J2Plasticity(E=E, nu=NU, sigma_y0=1e12, H=0.0)


def _solver(mesh, etype, mat=None, **kw):
    s = DynamicSolver(mesh, {0: mat or _elastic()}, rho=1.0, element_type={0: etype},
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


# ----------------------------------------------------------------------
# ENHANCED == CPE4I at small strain, on a distorted element
# ----------------------------------------------------------------------
def test_enhanced_small_strain_stiffness_equals_cpe4i():
    """The defining property (cpe4r_jax.py docstring, derivation 2).

    K(u=0) of CPE4R/ENHANCED vs Q4_EAS (EAS-4 = CPE4I, J2 elastic range) on
    a non-parallelogram quad. Measured 2026-10-10: 2.0e-10 -- the 1e-10
    symmetry-breaking noise `tangent_voigt_jax` adds to F (AGENTS.md 4.3),
    not a formulation difference. STIFFNESS differs by 1.2e-01.
    """
    m = _single(DISTORTED)
    _, K_r, _ = _solver(m, "CPE4R", hourglass="ENHANCED")._assemble(np.zeros(8), 1.0)
    _, K_i, _ = _solver(m, "Q4_EAS")._assemble(np.zeros(8), 1.0)
    _, K_s, _ = _solver(m, "CPE4R", hourglass="STIFFNESS")._assemble(np.zeros(8), 1.0)
    K_r, K_i, K_s = (np.asarray(k.todense()) for k in (K_r, K_i, K_s))
    err = np.max(np.abs(K_r - K_i)) / np.max(np.abs(K_i))
    err_s = np.max(np.abs(K_s - K_i)) / np.max(np.abs(K_i))
    assert err < 1e-8, err
    assert err_s > 1e-2, err_s


# ----------------------------------------------------------------------
# Hourglass force is objective (needs the polar-rotation pull-back)
# ----------------------------------------------------------------------
def test_hourglass_force_objective_under_superposed_rotation():
    """f(Q x) = T8(Q) f(x) for a state WITH hourglass content.

    Contract C2 rotates a homogeneous F0, whose hourglass coordinate q is
    zero, so it cannot see the hourglass frame. Here u0 = stretch +
    hourglass on an AR-7 element (anisotropic H). Measured 2026-10-10 at
    0.3/1.0/pi/2/2.5 rad: worst 5.5e-14 with the R^T q pull-back; with q
    left in the fixed frame (kernel-level A/B) 0.36 / 1.02 / 1.21 / 0.72.
    """
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.15], [0.0, 0.15]])
    h = np.array([1.0, -1.0, 1.0, -1.0])
    u0 = (coords @ (np.array([[1.03, 0.01], [0.0, 0.98]]) - np.eye(2)).T).reshape(-1)
    u0[0::2] += 4e-3 * h
    u0[1::2] += 1e-3 * h
    s = _solver(_single(coords), "CPE4R", hourglass="ENHANCED")
    x0 = coords + u0.reshape(4, 2)
    f0 = np.asarray(s._assemble(u0, 1.0)[0])
    worst = 0.0
    for th in (0.3, 1.0, np.pi / 2, 2.5):
        c, sn = np.cos(th), np.sin(th)
        Q = np.array([[c, -sn], [sn, c]])
        u = (x0 @ Q.T - coords).reshape(-1)
        f = np.asarray(s._assemble(u, 1.0)[0])
        worst = max(worst, np.max(np.abs(f - _rot8(th) @ f0)) / np.max(np.abs(f0)))
    assert worst < 1e-10, worst


# ----------------------------------------------------------------------
# 07-30 cantilever stack: bending stiffness must not depend on dx
# ----------------------------------------------------------------------
def _cantilever_tip_stiffness(etype, dx, n_rows, L=4.0, h=0.2, **kw):
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
    s = _solver(m, etype, **kw)
    _, K, _ = s._assemble(np.zeros(s.n_dofs), 1.0)
    K = K.tocsr()
    idx = {n: i for i, n in enumerate(s.sorted_nids)}
    fixed = [2 * idx[nid[(0, j)]] + d for j in range(n_rows + 1) for d in (0, 1)]
    free = np.setdiff1d(np.arange(s.n_dofs), fixed)
    f = np.zeros(s.n_dofs)
    tip = [2 * idx[nid[(nx, j)]] + 1 for j in range(n_rows + 1)]
    f[tip] = 1.0 / len(tip)                       # unit total transverse tip load
    u = np.zeros(s.n_dofs)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    return 1.0 / float(np.mean(u[tip]))


@pytest.mark.parametrize("n_rows", [1, 4])
def test_cantilever_stack_mesh_independence(n_rows):
    """k(dx=0.125) / k(dx=0.5) -- the 2026-07-30 measurement.

    That session's 1-point + FB element (q4_reduced_jax.py, 0.05*(lam+2mu))
    measured 10.19x, Q4_EAS 0.999x on its stack (dev_log/
    session_20260730_reduced_integration_failure.md; that script is not in
    the repo, so this is a re-built equivalent). Measured 2026-10-10 on a
    4.0 x 0.2 plane-strain cantilever, unit transverse tip load:

        rows  ENHANCED  Q4_EAS    STIFFNESS(0.005G)   k_EB/k_ENH (dx .125 / .5)
        1     0.9963    0.9963    0.1911  (5.2x)       1.0015 / 0.9979
        4     0.9831    0.9831    0.8811  (1.13x)      0.9969 / 0.9801

    ENHANCED equals Q4_EAS to 1e-8 on every mesh (same small-strain
    stiffness, see the test above); the residual 0.4% / 1.7% is the
    continuum (clamp/shear) convergence both share, not hourglass. Abaqus'
    default STIFFNESS is the mesh-dependent one: k = 0.005*G*V*sum(b^2)
    grows as 1/dy^2 per element, so at one row the coarse (AR-10) mesh is
    5.2x stiffer than the fine one.
    """
    k = {}
    for name, et, kw in (("enh", "CPE4R", {"hourglass": "ENHANCED"}),
                         ("eas", "Q4_EAS", {}),
                         ("stf", "CPE4R", {"hourglass": "STIFFNESS"})):
        k[name] = (_cantilever_tip_stiffness(et, 0.125, n_rows, **kw),
                   _cantilever_tip_stiffness(et, 0.5, n_rows, **kw))
    ratio = {n: v[0] / v[1] for n, v in k.items()}
    assert abs(ratio["enh"] - 1.0) < 0.02, ratio
    assert abs(k["enh"][0] / k["eas"][0] - 1.0) < 1e-6, k
    assert abs(k["enh"][1] / k["eas"][1] - 1.0) < 1e-6, k
    I = 0.2 ** 3 / 12.0
    k_eb = 3.0 * E_PS * I / 4.0 ** 3
    assert abs(k_eb / k["enh"][0] - 1.0) < 0.02, (k_eb, k["enh"][0])
    assert abs(ratio["stf"] - 1.0) > 0.1, ratio    # characterization


# ----------------------------------------------------------------------
# Plasticity: homogeneous J2 path identical to the existing J2 element
# ----------------------------------------------------------------------
def test_j2_homogeneous_path_matches_q4_eas():
    """Plane-strain stretch + shear into the plastic range, state committed
    each increment. Under homogeneous F the hourglass coordinate is zero
    and the centroid F equals every Gauss point's F, so CPE4R must
    reproduce Q4_EAS's (CPE4I's) J2 response exactly.

    sigma_y0 = 80 MPa, H = 400 MPa (AGENTS.md 1.4 PET), final F11 = 1.06.
    Measured 2026-10-10: force max relative difference over the path
    3.0e-14 (vs Q4_COROTATIONAL: 3.9e-13); final eqps 2.459e-02 in both.
    """
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.5], [0.0, 0.5]])
    mat = J2Plasticity(E=E, nu=NU, sigma_y0=80.0, H=400.0)
    sr = _solver(_single(coords), "CPE4R", mat=mat, hourglass="ENHANCED")
    se = _solver(_single(coords), "Q4_EAS", mat=mat)
    worst = 0.0
    for t in np.linspace(0.0, 1.0, 13)[1:]:
        F = np.array([[1.0 + 0.06 * t, 0.02 * t], [0.0, 1.0 - 0.01 * t]])
        u = (coords @ (F - np.eye(2)).T).reshape(-1)
        fr, _, st_r = sr._assemble(u, 1.0)
        fe, _, st_e = se._assemble(u, 1.0)
        sr.state, se.state = np.asarray(st_r), np.asarray(st_e)
        fr, fe = np.asarray(fr), np.asarray(fe)
        worst = max(worst, np.max(np.abs(fr - fe)) / np.max(np.abs(fe)))
    eqps_r = sr.state[0, 0, 4]
    eqps_e = se.state[0, 0, 4]
    assert eqps_e > 1e-2, "path never yielded -- test is not testing plasticity"
    assert abs(eqps_r - eqps_e) < 1e-10, (eqps_r, eqps_e)
    assert worst < 1e-8, worst


# ----------------------------------------------------------------------
# No silent fallthrough
# ----------------------------------------------------------------------
def test_non_j2_material_is_refused():
    with pytest.raises(ValueError, match="J2Plasticity only"):
        _solver(_single(DISTORTED), "CPE4R", mat=NeoHookean())


def test_unknown_hourglass_is_refused():
    with pytest.raises(ValueError, match="hourglass"):
        _solver(_single(DISTORTED), "CPE4R", hourglass="RELAX STIFFNESS")


def test_unset_hourglass_takes_abaqus_standard_default():
    """Keywords Guide: STIFFNESS is the Abaqus/Standard default except for
    hyperelastic/hyperfoam materials -- CPE4R here carries J2 only."""
    assert _solver(_single(DISTORTED), "CPE4R").hourglass == "STIFFNESS"
