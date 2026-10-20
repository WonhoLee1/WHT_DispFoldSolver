"""2D elements follow the material model (PRD: dev_log/prd_2d_material_wiring_20261018.md).

Three independent checks, none of which uses the code under test for its expected value:

* plastic memory: pull a plane-strain element past yield and return the end displacement to zero. A material
  that remembers its plastic strain is left in compression, one that does not retraces the loading path and
  ends at exactly zero stress.
* hyperelastic stiffness: at a tiny uniaxial strain a compressible Neo-Hooke solid has the tangent
  K + 4/3 mu  with  K = 2/D1, mu = 2 C10 (closed form), whatever finite-strain treatment an element uses.
* no silent wrong answer: an element whose kernel cannot follow a nonlinear material must refuse it.

Found 2026-10-18: CPE4 (NLGEOM), CPE4R, CPE4H, CPE3, CPE6, CPE6M, CPE8, CPE8R solved every material as linear
elastic from E and nu without saying so.
"""
import numpy as np
import pytest

from dispsolver.io import execute_inp


def _mid(a, b):
    return tuple((np.array(a) + np.array(b)) / 2.0)


_QUAD = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
_TRI = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
_QUAD8 = _QUAD + [_mid(_QUAD[a], _QUAD[b]) for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
_TRI6 = _TRI + [_mid(_TRI[a], _TRI[b]) for a, b in [(0, 1), (1, 2), (2, 0)]]
_SHAPE = {"CPE4": _QUAD, "CPE4I": _QUAD, "CPE4R": _QUAD, "CPE4H": _QUAD,
          "CPE3": _TRI, "CPE6": _TRI6, "CPE6M": _TRI6, "CPE8": _QUAD8, "CPE8R": _QUAD8}

_PLASTIC = ["*ELASTIC", "1000.0, 0.3", "*PLASTIC", "10.0, 0.0", "20.0, 0.1"]
_NEO_HOOKE = ["*HYPERELASTIC, NEO HOOKE", "100.0, 0.001"]          # C10 = 100, D1 = 0.001


def _deck(etype, nlgeom, material, targets):
    """Uniaxial strain (u_x = eps x, u_y = 0 on every node) through the listed (step name, eps) targets."""
    nodes = _SHAPE[etype]
    lines = ["*HEADING", "2D material probe", "*NODE"]
    lines += [f"{i + 1}, {float(p[0])!r}, {float(p[1])!r}" for i, p in enumerate(nodes)]
    lines += [f"*ELEMENT, TYPE={etype}, ELSET=EL", "1, " + ", ".join(str(i + 1) for i in range(len(nodes))),
              "*MATERIAL, NAME=M"] + material + ["*SOLID SECTION, ELSET=EL, MATERIAL=M"]
    for name, eps in targets:
        lines += [f"*STEP, NAME={name}, NLGEOM={nlgeom}", "*STATIC", "0.1, 1.0, 1e-4, 0.1", "*BOUNDARY"]
        for i, p in enumerate(nodes):
            lines += [f"{i + 1}, 1, 1, {float(eps * p[0])!r}", f"{i + 1}, 2, 2, 0.0"]
        lines += ["*END STEP"]
    return "\n".join(lines) + "\n"


def _area(etype):
    nodes = np.array(_SHAPE[etype])
    corners = nodes[:3] if etype in ("CPE3", "CPE6", "CPE6M") else nodes[:4]
    return 0.5 * abs(np.dot(corners[:, 0], np.roll(corners[:, 1], -1)) - np.dot(corners[:, 1], np.roll(corners[:, 0], -1)))


def _sigma_xx(res, etype):
    """Mean sigma_xx of the (homogeneous) field: sum_i f_x,i x_i = sigma_xx * area for any isoparametric element,
    so it needs no assumption about which nodes sit on a 'right-hand face' (a triangle has none)."""
    f = np.asarray(res["solver"].f_int).reshape(-1, 2)
    x = np.array(_SHAPE[etype])[:, 0]
    return float((f[:, 0] * x).sum()) / _area(etype)


def _run(tmp_path, deck):
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    return execute_inp(str(path), dim=2, verbose=False)


_SUPPORTED = [(e, n) for e in ("CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE3", "CPE6", "CPE6M", "CPE8", "CPE8R")
              for n in ("NO", "YES")]


@pytest.mark.parametrize("etype, nlgeom", _SUPPORTED)
def test_plastic_strain_is_remembered(tmp_path, etype, nlgeom):
    res = _run(tmp_path, _deck(etype, nlgeom, _PLASTIC, [("PULL", 0.05), ("RETURN", 0.0)]))
    assert res["converged"]
    assert _sigma_xx(res, etype) < -5.0          # residual compression; the peak stress was about +14


@pytest.mark.parametrize("etype, nlgeom", _SUPPORTED)
def test_hyperelastic_tangent_comes_from_the_material(tmp_path, etype, nlgeom):
    eps = 1.0e-3
    res = _run(tmp_path, _deck(etype, nlgeom, _NEO_HOOKE, [("PULL", eps)]))
    K, mu = 2.0 / 0.001, 2.0 * 100.0
    assert _sigma_xx(res, etype) == pytest.approx((K + 4.0 * mu / 3.0) * eps, rel=0.01)


def test_axisymmetric_kernel_still_refuses_a_nonlinear_material(tmp_path):
    # the axisymmetric kernels integrate over the ring and are not wired to the material model: refuse, never
    # solve as linear elastic
    deck = ("*HEADING\naxisymmetric probe\n*NODE\n1, 1.0, 0.0\n2, 2.0, 0.0\n3, 2.0, 1.0\n4, 1.0, 1.0\n"
            "*ELEMENT, TYPE=CAX4, ELSET=EL\n1, 1, 2, 3, 4\n*MATERIAL, NAME=M\n"
            + "\n".join(_PLASTIC) + "\n*SOLID SECTION, ELSET=EL, MATERIAL=M\n"
            "*STEP, NAME=S, NLGEOM=NO\n*STATIC\n1.0, 1.0\n*BOUNDARY\n1, 1, 2, 0.0\n*END STEP\n")
    with pytest.raises(NotImplementedError, match="linear-elastic-only|axisymmetric kernels"):
        _run(tmp_path, deck)


@pytest.mark.parametrize("etype", ["CPE3", "CPE6", "CPE6M", "CPE8", "CPE8R"])
def test_those_kernels_still_run_linear_elastic_materials(tmp_path, etype):
    res = _run(tmp_path, _deck(etype, "NO", ["*ELASTIC", "1000.0, 0.3"], [("PULL", 1.0e-3)]))
    assert res["converged"]
    lam, mu = 1000.0 * 0.3 / (1.3 * 0.4), 1000.0 / 2.6
    f = np.asarray(res["solver"].f_int).reshape(-1, 2)
    nodes = np.array(_SHAPE[etype])
    corners = nodes[:3] if etype in ("CPE3", "CPE6", "CPE6M") else nodes[:4]
    area = 0.5 * abs(np.dot(corners[:, 0], np.roll(corners[:, 1], -1)) - np.dot(corners[:, 1], np.roll(corners[:, 0], -1)))
    # sum_i f_x,i x_i = sigma_xx * area holds for any isoparametric element
    assert float((f[:, 0] * nodes[:, 0]).sum()) == pytest.approx((lam + 2.0 * mu) * 1.0e-3 * area, rel=1e-6)


def _closed_form_residual_stress(eps=0.05, E=1000.0, nu=0.3, sy0=10.0, H=100.0):
    """Residual sigma_xx after uniaxial STRAIN (plane strain, u_y = 0) to eps and back to zero, J2 with linear
    isotropic hardening (the *PLASTIC table 10 at 0 -> 20 at 0.1 is H = 100), small strain.

    Plastic strain is isochoric and symmetric: ep_t = ep diag(1, -1/2, -1/2), equivalent plastic strain = ep,
    stress difference q = sigma_xx - sigma_yy = 2 mu e - 3 mu ep.
      loading   q = sy0 + H ep            ->  ep  = (2 mu eps - sy0) / (3 mu + H)
      unloading the stress state at e = 0 would be q = 3 mu ep > current yield, so the flow REVERSES; the
                reversed increment a hardens isotropically (equivalent plastic strain ep + a):
                -3 mu (ep - a) = -(sy0 + H (ep + a))  ->  a = (3 mu ep - sy0 - H ep) / (3 mu + H)
      residual  sigma_xx(e=0) = -2 mu (ep - a)
    """
    mu = E / (2.0 * (1.0 + nu))
    ep = (2.0 * mu * eps - sy0) / (3.0 * mu + H)
    a = (3.0 * mu * ep - sy0 - H * ep) / (3.0 * mu + H)
    assert a > 0.0                                  # the unloading really does reverse the flow
    return -2.0 * mu * (ep - a)


def test_the_closed_form_matches_the_one_element_it_was_derived_for():
    assert _closed_form_residual_stress() == pytest.approx(-8.92, abs=0.01)


@pytest.mark.parametrize("etype, nlgeom", _SUPPORTED)
def test_plastic_residual_stress_matches_the_closed_form(tmp_path, etype, nlgeom):
    res = _run(tmp_path, _deck(etype, nlgeom, _PLASTIC, [("PULL", 0.05), ("RETURN", 0.0)]))
    expected = _closed_form_residual_stress()
    # CPE4R is Total Lagrangian (Green-Lagrange strain, 0.05 -> 0.05125), the other kernels evaluate the small
    # strain the closed form assumes, so CPE4R gets the looser bound its strain measure implies
    rel = 0.08 if etype == "CPE4R" else 1e-6
    assert _sigma_xx(res, etype) == pytest.approx(expected, rel=rel)


# ---- objectivity and consistent tangent, for every element that follows the material model --------------------

_ALL_ELEMENTS = ["CPE3", "CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE6", "CPE6M", "CPE8", "CPE8R"]


def _rotation_deck(etype, theta_deg, step_deg=2.0, eps=0.05):
    """Stretch plastically, then turn the stretched body rigidly in steps of at most step_deg degrees.

    Inside one *STEP the displacements are interpolated linearly, so a large increment moves the nodes along a
    chord (the element shrinks mid-step) and plastic strain accrues on that artificial path; small steps follow
    the rotation arc closely enough that this vanishes."""
    nodes = np.array(_SHAPE[etype])
    deck = _deck(etype, "YES", _PLASTIC, [("S1", eps)])
    n = max(1, int(round(theta_deg / step_deg)))
    F = np.diag([1.0 + eps, 1.0])
    lines = []
    for k in range(1, n + 1):
        t = np.radians(theta_deg * k / n)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        lines += [f"*STEP, NAME=R{k}, NLGEOM=YES", "*STATIC", "1.0, 1.0, 1e-5, 1.0", "*BOUNDARY"]
        for i, p in enumerate(nodes):
            u = R @ (F @ p) - p
            lines += [f"{i + 1}, 1, 1, {float(u[0])!r}", f"{i + 1}, 2, 2, {float(u[1])!r}"]
        lines += ["*END STEP"]
    return deck + "\n".join(lines) + "\n"


@pytest.mark.parametrize("etype", _ALL_ELEMENTS)
def test_plastically_stretched_element_rotates_without_changing_its_stress(tmp_path, etype):
    theta = 40.0
    f0 = np.asarray(_run(tmp_path, _deck(etype, "YES", _PLASTIC, [("S1", 0.05)]))["solver"].f_int).reshape(-1, 2)
    res = _run(tmp_path, _rotation_deck(etype, theta))
    assert res["converged"]
    f1 = np.asarray(res["solver"].f_int).reshape(-1, 2)
    t = np.radians(theta)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    assert np.abs(f1 - f0 @ R.T).max() / np.abs(f0).max() < 2e-3


def _tangent_vs_finite_difference(solver, u, h=1e-7):
    K, _ = solver.assemble_system(u)
    K = K.toarray() if hasattr(K, "toarray") else np.asarray(K)
    Kfd = np.zeros_like(K)
    for j in range(len(u)):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        solver.assemble_system(up)
        fp = np.array(solver.f_int, copy=True)
        solver.assemble_system(um)
        fm = np.array(solver.f_int, copy=True)
        Kfd[:, j] = (fp - fm) / (2.0 * h)
    return K, Kfd


# CPE4I used to fail this for the plastic state (1.9e-3): it condensed the enhanced strain parameters with one linear
# solve instead of iterating them to equilibrium. Fixed 2026-10-18; the case is an ordinary test again.
_TANGENT_CASES = [(e, tag) for e in _ALL_ELEMENTS for tag in ("plastic", "hyperelastic")]


@pytest.mark.parametrize("etype, tag", _TANGENT_CASES)
def test_tangent_is_the_jacobian_of_the_internal_force(tmp_path, etype, tag):
    material = _PLASTIC if tag == "plastic" else _NEO_HOOKE          # a plastic or a hyperelastic loaded state
    targets = [("S1", 0.05)]
    res = _run(tmp_path, _deck(etype, "NO", material, targets))       # leaves committed state variables in the solver
    solver = res["solver"]
    rng = np.random.default_rng(5)
    u = np.array(solver.u, copy=True) + rng.normal(0.0, 2e-4, len(solver.u))
    K, Kfd = _tangent_vs_finite_difference(solver, u)
    scale = np.abs(Kfd).max()
    assert np.abs(K - Kfd).max() / scale < 2e-4
