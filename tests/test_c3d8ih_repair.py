"""C3D8IH repair (2026-10-20) -- the 3D twin of tests/test_cpe4i_ih_repair.py.

1. C3D8IH evaluated a nonlinear material on B u alone (no enhanced strain B~ alpha) and condensed once from
   alpha = 0, so the plastic state and the tangent were taken at the wrong strain.
2. A singular enhanced block raised inside the parallel assembly (SystemError) instead of cutting the increment back.

Expected values are independent of the kernel: a closed-form fibre oracle for plastic bending, the closed form
of the SAME element for a linear material, and the plane-strain limit pressure of a thick cylinder.
"""
import math
import os
import tempfile

import numpy as np
import pytest

from dispsolver.element3d.c3d8ih_numba import compute_c3d8ih_element_numba
from dispsolver.io import execute_inp
from dispsolver.material.resolve import resolve_material
from dispsolver.model.material import Material

CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
# a distorted hexahedron: perturb every node deterministically
HEX_DISTORTED = CUBE + np.array([[0.00, 0.00, 0.00], [0.10, 0.02, 0.00], [0.05, -0.08, 0.03], [-0.07, 0.04, 0.00],
                                 [0.02, 0.00, -0.05], [0.06, 0.03, 0.04], [0.00, 0.09, 0.07], [-0.04, 0.00, 0.06]])


def _mat_props(plastic=None, E=3000.0, nu=0.3):
    m = Material(name="M")
    m.elastic = (E, nu)
    if plastic is not None:
        m.Plastic(plastic)
    return resolve_material(m, 3)


# ---- 1. nonlinear path == linear closed form when the material never yields --------------------------------------

@pytest.mark.parametrize("nu", [0.3, 0.4999])
def test_nonlinear_path_reproduces_the_linear_closed_form(nu):
    """J2 with an unreachable yield stress is linear elastic; it takes the new iterating path, the linear
    elastic material takes the closed form. Same element, same answer."""
    mid_l, p_l = _mat_props(E=3000.0, nu=nu)
    mid_n, p_n = _mat_props(plastic=[(1e12, 0.0), (1e12 + 1.0, 1.0)], E=3000.0, nu=nu)
    assert mid_l != mid_n
    rng = np.random.default_rng(5)
    u = 1e-6 * rng.standard_normal(24)                 # small strain: Hencky -> GL conversion error ~ eps^2
    fe_l, Ke_l, e1 = compute_c3d8ih_element_numba(HEX_DISTORTED, u, mid_l, p_l, np.zeros((8, 7)))
    fe_n, Ke_n, e2 = compute_c3d8ih_element_numba(HEX_DISTORTED, u, mid_n, p_n, np.zeros((8, 7)))
    assert e1 == 0 and e2 == 0
    np.testing.assert_allclose(fe_n, fe_l, rtol=1e-5, atol=1e-5 * np.abs(fe_l).max())
    np.testing.assert_allclose(Ke_n, Ke_l, rtol=1e-5, atol=1e-5 * np.abs(Ke_l).max())


# ---- 1a. the condensed tangent is the derivative of the condensed force in a PLASTIC state -----------------------

def _fd_tangent_error(u, mid, props):
    n = len(u)
    fe, Ke, err = compute_c3d8ih_element_numba(HEX_DISTORTED, u, mid, props, np.zeros((8, 7)))
    h = 1e-7
    Kfd = np.zeros((n, n))
    for j in range(n):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        Kfd[:, j] = (compute_c3d8ih_element_numba(HEX_DISTORTED, up, mid, props, np.zeros((8, 7)))[0]
                     - compute_c3d8ih_element_numba(HEX_DISTORTED, um, mid, props, np.zeros((8, 7)))[0]) / (2 * h)
    return err, np.linalg.norm(Ke - Kfd) / np.linalg.norm(Kfd)


def test_plastic_tangent_is_the_jacobian_of_the_internal_force():
    """Measured 2026-10-20 on the unrepaired kernel: 9-15 % off a central finite difference at 2-4 times the
    yield strain (the material was evaluated on B u without the enhanced strain). Repaired: ~1e-3, which is the
    first-order-in-strain floor of the Hencky -> Green-Lagrange strain mapping every 3D small-strain-in-CR
    kernel uses (it scales with the strain: 7.6e-5 at eps = 1e-4). The bound sits between the two."""
    mid, props = _mat_props(plastic=[(5.0, 0.0), (15.0, 1.0)])      # yield strain 5/3000
    checked = 0
    for seed in range(6):
        u = 2e-3 * np.random.default_rng(100 + seed).standard_normal(24)
        err, rel = _fd_tangent_error(u, mid, props)
        if err == 0:
            checked += 1
            assert rel < 5e-3, (seed, rel)
    assert checked >= 4


# ---- 1b. plane-strain plastic bending against the fibre oracle -----------------------------------------------------

def _bending_deck(etype, kappa, L=10.0, h=1.0, nx=20, ny=4, nsteps=10):
    """Plane-strain strip as a one-element-thick 3D slab (u_z = 0 everywhere); end sections turned rigidly."""
    from tests import test_plastic_bending_2d as T
    nodes2, elems2 = T.build_mesh("CPE4", L, h, nx, ny)
    n2 = len(nodes2)
    lines = ["*HEADING", "plastic bending 3D", "*NODE"]
    for k in range(2):
        for i, p in enumerate(nodes2):
            lines.append(f"{k * n2 + i + 1}, {float(p[0])!r}, {float(p[1])!r}, {float(k)!r}")
    lines.append(f"*ELEMENT, TYPE={etype}, ELSET=EL")
    for e, el in enumerate(elems2):
        bot = [n for n in el]
        lines.append(f"{e + 1}, " + ", ".join(str(n) for n in bot) + ", " + ", ".join(str(n + n2) for n in bot))
    lines += ["*NSET, NSET=ALLN, GENERATE", f"1, {2 * n2}", "*MATERIAL, NAME=M", "*ELASTIC", f"{T.E}, {T.NU}", "*PLASTIC",
              f"{T.SY0}, 0.0", f"{T.SY0 + T.H * 0.5}, 0.5", "*SOLID SECTION, ELSET=EL, MATERIAL=M"]
    left = [i for i, p in enumerate(nodes2) if abs(p[0]) < 1e-12]
    right = [i for i, p in enumerate(nodes2) if abs(p[0] - L) < 1e-12]
    phi = kappa * L
    yc = h / 2
    R = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    for k in range(1, nsteps + 1):
        th = phi * k / nsteps / 2.0
        chord = 2.0 * (L / (phi * k / nsteps)) * np.sin(phi * k / nsteps / 2.0) - L
        lines += [f"*STEP, NAME=B{k}, NLGEOM=YES", "*STATIC", "1.0, 1.0, 1e-5, 1.0", "*BOUNDARY", "ALLN, 3, 3, 0.0"]
        for layer in range(2):
            for i in left:
                X = np.array(nodes2[i]); Xc = np.array([0.0, yc])
                u = R(-th) @ (X - Xc) + Xc - X
                lines += [f"{layer * n2 + i + 1}, 1, 1, {float(u[0])!r}", f"{layer * n2 + i + 1}, 2, 2, {float(u[1])!r}"]
            for i in right:
                X = np.array(nodes2[i]); Xc = np.array([L, yc])
                u = R(th) @ (X - Xc) + Xc - X + np.array([chord, 0.0])
                lines += [f"{layer * n2 + i + 1}, 1, 1, {float(u[0])!r}", f"{layer * n2 + i + 1}, 2, 2, {float(u[1])!r}"]
        lines.append("*END STEP")
    return "\n".join(lines) + "\n", nodes2, left, n2


def _moment(etype, kappa, tmp_path):
    text, nodes2, left, n2 = _bending_deck(etype, kappa)
    p = tmp_path / "b3.inp"
    p.write_text(text)
    r = execute_inp(str(p), dim=3, verbose=False)
    f = np.asarray(r["solver"].f_int).reshape(-1, 3)
    y = np.array([nodes2[i][1] for i in left])
    left_idx = np.asarray(left, dtype=int)
    m = sum(float((f[left_idx + k * n2, 0] * (y - 0.5)).sum()) for k in range(2))
    return abs(m), r["converged"]


@pytest.mark.parametrize("etype", ["C3D8IH"])
def test_plastic_bending_matches_the_oracle(tmp_path, etype):
    from tests import test_plastic_bending_2d as T
    kappa = 0.04                                       # outer fibres at twice the yield strain
    moment, converged = _moment(etype, kappa, tmp_path)
    assert converged, f"{etype} did not converge to kappa = {kappa} in plastic bending"
    assert moment == pytest.approx(T.oracle_moment(kappa), rel=0.03)


# ---- 2. a collapse load must end in a cut-back, never an exception ------------------------------------------------

def _cylinder_deck(etype, p_ratio, nr=4, nt=12, a=1.0, b=2.0, E=1000.0, nu=0.3, sy=1.0, H=0.01):
    """Quarter thick-walled cylinder, one element thick in z (u_z = 0: plane strain), internal pressure."""
    lines = ["*HEADING", "thick cylinder 3D", "*NODE"]
    nid, k = {}, 0
    for z in (0, 1):
        for j in range(nt + 1):
            th = 0.5 * math.pi * j / nt
            for i in range(nr + 1):
                r = a + (b - a) * i / nr
                k += 1
                nid[(i, j, z)] = k
                lines.append(f"{k}, {r * math.cos(th):.12f}, {r * math.sin(th):.12f}, {float(z)}")
    lines.append(f"*ELEMENT, TYPE={etype}, ELSET=EL")
    e = 0
    for j in range(nt):
        for i in range(nr):
            e += 1
            quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            ids = [nid[(x, y, 0)] for x, y in quad] + [nid[(x, y, 1)] for x, y in quad]
            lines.append(f"{e}, " + ", ".join(str(n) for n in ids))
    lines += ["*NSET, NSET=ALLN, GENERATE", f"1, {k}", "*NSET, NSET=YSYM"]
    lines += [str(nid[(i, 0, z)]) for z in (0, 1) for i in range(nr + 1)]
    lines += ["*NSET, NSET=XSYM"] + [str(nid[(i, nt, z)]) for z in (0, 1) for i in range(nr + 1)]
    lines += ["*MATERIAL, NAME=M", "*ELASTIC", f"{E}, {nu}", "*PLASTIC", f"{sy}, 0.0", f"{sy + H}, 1.0",
              "*SOLID SECTION, ELSET=EL, MATERIAL=M", "*STEP, NAME=S, NLGEOM=NO", "*STATIC", "0.05, 1.0, 1e-5, 0.1",
              "*BOUNDARY", "ALLN, 3, 3, 0.0", "YSYM, 2, 2, 0.0", "XSYM, 1, 1, 0.0", "*CLOAD"]
    p = p_ratio * (2.0 / math.sqrt(3.0)) * sy * math.log(b / a)
    dth = 0.5 * math.pi / nt
    for j in range(nt + 1):
        w = 0.5 if j in (0, nt) else 1.0
        th = 0.5 * math.pi * j / nt
        F = 0.5 * p * a * dth * w                       # unit thickness shared by the two layers
        for z in (0, 1):
            lines.append(f"{nid[(0, j, z)]}, 1, {F * math.cos(th):.12e}")
            lines.append(f"{nid[(0, j, z)]}, 2, {F * math.sin(th):.12e}")
    lines.append("*END STEP")
    return "\n".join(lines) + "\n"


def _run(etype, ratio):
    path = os.path.join(tempfile.mkdtemp(), "c3.inp")
    with open(path, "w") as f:
        f.write(_cylinder_deck(etype, ratio))
    return execute_inp(path, dim=3, verbose=False)


def test_pressure_below_the_limit_converges():
    assert _run("C3D8IH", 0.90)["converged"]


def test_pressure_above_the_limit_is_reported_not_raised():
    """No equilibrium exists above p_lim = 2/sqrt(3) sy ln(b/a): the answer is 'did not converge', never an exception."""
    assert _run("C3D8IH", 1.10)["converged"] is False
