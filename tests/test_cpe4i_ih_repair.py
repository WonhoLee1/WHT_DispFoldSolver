"""CPE4I / CPE4IH repair (2026-10-20).

Four defects, each pinned by a test whose expected value is independent of the code under test:

1. CPE4IH evaluated a nonlinear material on B u alone (no enhanced strain M alpha) and condensed once from
   alpha = 0: plastic bending of a strip did not converge at kappa = 0.04 and was 2.3 % off a fibre oracle.
2. A singular enhanced block (plastic collapse) raised inside the parallel assembly -> SystemError, the run died
   instead of cutting the increment back.
3. CPE4I had 4 internal degrees of freedom; Abaqus' non-hybrid incompatible-mode elements carry 5 for plane strain
   (Theory Guide 3.2.5: 4 for CPS4I, 5 for CPE4I; 13 = 9 + 4 for C3D8I) -- the extra one is a bilinear
   volumetric mode.
3b. That mode must keep the Irons patch test exact: it is orthogonal to every constant strain.
"""
import math
import os
import tempfile

import numpy as np
import pytest

from dispsolver.element2d.cpe4i_numba import (
    N_ENH, _GPS, _WTS, _shape_derivs_cpe4i, compute_cpe4i_element_numba, solve_small)
from dispsolver.element2d.cpe4ih_numba import (
    compute_cpe4ih_element_numba, compute_cpe4ih_nonlinear_numba)
from dispsolver.io import execute_inp
from dispsolver.material3d.numba_materials import MAT_LINEAR_ELASTIC

X_DISTORTED = np.array([[0.0, 0.0], [2.0, 0.1], [1.7, 1.3], [-0.1, 0.9]])


def _props(E=3000.0, nu=0.3):
    p = np.zeros(64)
    p[0] = E
    p[1] = nu
    return p


# ---- 1. nonlinear CPE4IH reduces to the closed-form linear element --------------------------------------------

@pytest.mark.parametrize("nu", [0.0, 0.3, 0.49, 0.4999])
def test_nonlinear_routine_reproduces_the_linear_closed_form(nu):
    rng = np.random.default_rng(3)
    u = 1e-3 * rng.standard_normal(8)
    p = _props(nu=nu)
    fe_l, Ke_l, e1 = compute_cpe4ih_element_numba(X_DISTORTED, u, MAT_LINEAR_ELASTIC, p, np.zeros((4, 7)), 0.0)
    fe_n, Ke_n, e2 = compute_cpe4ih_nonlinear_numba(X_DISTORTED, u, MAT_LINEAR_ELASTIC, p, np.zeros((4, 7)), 0.0)
    assert e1 == 0 and e2 == 0
    np.testing.assert_allclose(fe_n, fe_l, rtol=1e-9, atol=1e-9 * np.abs(fe_l).max())
    np.testing.assert_allclose(Ke_n, Ke_l, rtol=1e-9, atol=1e-9 * np.abs(Ke_l).max())


# ---- 1a. the condensed tangent is the derivative of the condensed force in a PLASTIC state -----------------------

def _fd_tangent_error(fn, u, mid, props):
    n = len(u)
    fe, Ke, err = fn(X_DISTORTED, u, mid, props, np.zeros((4, 7)), 1.0)
    h = 1e-7
    Kfd = np.zeros((n, n))
    for j in range(n):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        Kfd[:, j] = (fn(X_DISTORTED, up, mid, props, np.zeros((4, 7)), 1.0)[0]
                     - fn(X_DISTORTED, um, mid, props, np.zeros((4, 7)), 1.0)[0]) / (2 * h)
    return err, np.linalg.norm(Ke - Kfd) / np.linalg.norm(Kfd)


@pytest.mark.parametrize("fn", [compute_cpe4i_element_numba, compute_cpe4ih_element_numba])
def test_plastic_tangent_is_the_jacobian_of_the_internal_force(fn):
    """Measured 2026-10-20: CPE4I 1e-10; CPE4IH before the repair 28 % off a central finite difference at 2-4 times
    the yield strain, after the repair 1e-10 like CPE4I."""
    from dispsolver.material.resolve import resolve_material
    from dispsolver.model.material import Material
    m = Material(name="M")
    m.elastic = (3000.0, 0.3)
    m.Plastic([(5.0, 0.0), (15.0, 1.0)])                           # yield strain 5/3000
    mid, props = resolve_material(m, 2)
    checked = 0
    for seed in range(6):
        u = 2e-3 * np.random.default_rng(100 + seed).standard_normal(8)
        err, rel = _fd_tangent_error(fn, u, mid, props)
        if err == 0:
            checked += 1
            assert rel < 1e-7, (seed, rel)
    assert checked >= 4


# ---- 1b. plastic bending against the fibre oracle (CPE4IH now converges) ---------------------------------------

def test_cpe4ih_plastic_bending_matches_the_oracle(tmp_path):
    from tests import test_plastic_bending_2d as T
    kappa = 0.04
    moment, converged = T.fe_moment("CPE4IH", kappa, tmp_path)
    assert converged, "CPE4IH did not converge to kappa = 0.04 in plastic bending"
    assert moment == pytest.approx(T.oracle_moment(kappa), rel=0.02)


# ---- 2. a collapse load must end in a cut-back, never an exception -----------------------------------------------

def test_solve_small_flags_a_singular_matrix_instead_of_raising():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    X, ok = solve_small(A, np.eye(2))
    assert ok is False and np.all(np.isfinite(X))
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    X, ok = solve_small(A, np.array([[1.0], [2.0]]))
    assert ok and np.allclose(A @ X, [[1.0], [2.0]])


def _cylinder_deck(et, p_ratio, nr=4, nt=12, a=1.0, b=2.0, E=1000.0, nu=0.3, sy=1.0, H=0.01):
    """Quarter of a thick-walled cylinder under internal pressure (plane strain, elastic-plastic)."""
    lines = ["*HEADING", "thick cylinder", "*NODE"]
    nid, k = {}, 0
    for j in range(nt + 1):
        th = 0.5 * math.pi * j / nt
        for i in range(nr + 1):
            r = a + (b - a) * i / nr
            k += 1
            nid[(i, j)] = k
            lines.append(f"{k}, {r * math.cos(th):.12f}, {r * math.sin(th):.12f}")
    lines.append(f"*ELEMENT, TYPE={et}, ELSET=EL")
    e = 0
    for j in range(nt):
        for i in range(nr):
            e += 1
            lines.append(f"{e}, {nid[(i, j)]}, {nid[(i + 1, j)]}, {nid[(i + 1, j + 1)]}, {nid[(i, j + 1)]}")
    lines += ["*NSET, NSET=YSYM"] + [str(nid[(i, 0)]) for i in range(nr + 1)]
    lines += ["*NSET, NSET=XSYM"] + [str(nid[(i, nt)]) for i in range(nr + 1)]
    lines += ["*MATERIAL, NAME=M", "*ELASTIC", f"{E}, {nu}", "*PLASTIC", f"{sy}, 0.0", f"{sy + H}, 1.0",
              "*SOLID SECTION, ELSET=EL, MATERIAL=M", "*STEP, NAME=S, NLGEOM=NO", "*STATIC",
              "0.05, 1.0, 1e-5, 0.1", "*BOUNDARY", "YSYM, 2, 2, 0.0", "XSYM, 1, 1, 0.0", "*CLOAD"]
    p = p_ratio * (2.0 / math.sqrt(3.0)) * sy * math.log(b / a)        # plane-strain limit pressure x ratio
    dth = 0.5 * math.pi / nt
    for j in range(nt + 1):
        w = 0.5 if j in (0, nt) else 1.0
        th = 0.5 * math.pi * j / nt
        F = p * a * dth * w
        lines.append(f"{nid[(0, j)]}, 1, {F * math.cos(th):.12e}")
        lines.append(f"{nid[(0, j)]}, 2, {F * math.sin(th):.12e}")
    lines.append("*END STEP")
    return "\n".join(lines) + "\n"


def _run(et, ratio):
    path = os.path.join(tempfile.mkdtemp(), "c.inp")
    with open(path, "w") as f:
        f.write(_cylinder_deck(et, ratio))
    return execute_inp(path, dim=2, verbose=False)


@pytest.mark.parametrize("et", ["CPE4I", "CPE4IH"])
def test_pressure_below_the_limit_converges(et):
    assert _run(et, 0.90)["converged"]


@pytest.mark.parametrize("et", ["CPE4I", "CPE4IH"])
def test_pressure_above_the_limit_is_reported_not_raised(et):
    """Above p_lim = 2/sqrt(3) sy ln(b/a) no equilibrium exists: the answer is 'did not converge'.
    Before the repair CPE4I died with SystemError here. (Full-integration CPE4 locks and 'converges' --
    that is the locking being measured, so it is not asserted.)"""
    res = _run(et, 1.10)
    assert res["converged"] is False


# ---- 3. five internal degrees of freedom, patch test intact ----------------------------------------------------

def test_cpe4i_has_five_internal_degrees_of_freedom():
    assert N_ENH == 5


def test_the_volumetric_mode_is_orthogonal_to_constants():
    """Integral of (detJ0/detJ) * xi*eta * detJ over the element = detJ0 * sum_q w xi eta = 0."""
    s = sum(_WTS[q] * _GPS[q, 0] * _GPS[q, 1] for q in range(4))
    assert abs(s) < 1e-15


@pytest.mark.parametrize("nu", [0.3, 0.4999])
def test_constant_strain_patch_is_exact_on_a_distorted_quad(nu):
    """Irons patch test for the 5-mode element: the internal force of a constant-strain field is exactly B-integral
    (alpha stays 0), and the element reproduces K u for it."""
    E = 3000.0
    p = _props(E=E, nu=nu)
    eps = np.array([1e-3, -4e-4, 3e-4])
    u = np.zeros(8)
    for a in range(4):
        x, y = X_DISTORTED[a]
        u[2 * a] = eps[0] * x + 0.5 * eps[2] * y
        u[2 * a + 1] = eps[1] * y + 0.5 * eps[2] * x
    fe, Ke, err = compute_cpe4i_element_numba(X_DISTORTED, u, MAT_LINEAR_ELASTIC, p, np.zeros((4, 7)), 0.0)
    assert err == 0
    # constant stress field: sigma from the plane-strain moduli; nodal force = int B^T sigma dV, computed independently
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    sig = np.array([(lam + 2 * mu) * eps[0] + lam * eps[1], lam * eps[0] + (lam + 2 * mu) * eps[1], mu * eps[2]])
    f_ref = np.zeros(8)
    for q in range(4):
        dN = _shape_derivs_cpe4i(_GPS[q, 0], _GPS[q, 1])
        J = dN @ X_DISTORTED
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        dNdx = np.linalg.solve(J, dN)
        for a in range(4):
            f_ref[2 * a] += (dNdx[0, a] * sig[0] + dNdx[1, a] * sig[2]) * detJ * _WTS[q]
            f_ref[2 * a + 1] += (dNdx[1, a] * sig[1] + dNdx[0, a] * sig[2]) * detJ * _WTS[q]
    np.testing.assert_allclose(fe, f_ref, rtol=1e-8, atol=1e-8 * np.abs(f_ref).max())


def test_incompressible_limit_spectrum_of_cpe4i_and_cpe4ih_agree():
    """At nu -> 0.5 the two elements share ONE locked (mean-dilatation) mode and the same deformation modes.
    (The earlier test asserted CPE4I had two locked modes on a distorted quad; it never held.)"""
    p = _props(nu=0.4999)
    G = 3000.0 / (2 * 1.4999)
    _, Ki, _ = compute_cpe4i_element_numba(X_DISTORTED, np.zeros(8), 0, p, np.zeros((4, 7)), 0.0)
    _, Kh, _ = compute_cpe4ih_element_numba(X_DISTORTED, np.zeros(8), 0, p, np.zeros((4, 7)), 0.0)
    ei = np.linalg.eigvalsh(0.5 * (Ki + Ki.T)) / G
    eh = np.linalg.eigvalsh(0.5 * (Kh + Kh.T)) / G
    assert np.sum(ei > 50) == 1 and np.sum(eh > 50) == 1
    np.testing.assert_allclose(ei[3:7], eh[3:7], rtol=2e-3)
