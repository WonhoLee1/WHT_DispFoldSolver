"""A4 -- StepPlan (dispsolver/model/step_plan.py).

1. Twin decks: a 2D plane-strain mesh and the same mesh extruded one element
   thick in 3D (unit thickness) give the same in-plane f_ext (3D summed over
   the two through-thickness node layers) and the same prescribed values.
2. Follower vs reference pressure differs only when NLGEOM is on.
3. Regression against the per-solver code A4 deleted: tests/fixtures/
   step_plan_old.npz holds the f_ext / prescribed vectors that
   `_update_step_bcs` / `_compute_step_external_loads` produced at commit
   140a571 (captured by `capture_old_fixture()` below, which only runs on
   that commit).
"""
import os
import warnings

import numpy as np
import pytest

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.model.regions import RegionError
from dispsolver.model.step_plan import lower_step

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "step_plan_old.npz")
TIMES = (0.3, 0.7, 1.0)
AMP = "*AMPLITUDE, NAME=A1\n0.,0., 0.5,0.2, 1.,1.5\n"

# ------------------------------------------------------------------ decks
NODES3 = """*NODE
1,0,0,0
2,1,0,0
3,2,0,0
4,0,1,0
5,1,1,0
6,2,1,0
7,0,0,1
8,1,0,1
9,2,0,1
10,0,1,1
11,1,1,1
12,2,1,1
"""
ELEMS3 = """*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1,2,5,4,7,8,11,10
2, 2,3,6,5,8,9,12,11
*ELSET, ELSET=E1
1
*ELSET, ELSET=E2
2
*NSET, NSET=LEFT
1,4,7,10
*NSET, NSET=RIGHT
3,6,9,12
"""
MATS = "*MATERIAL, NAME=STEEL\n*ELASTIC\n200000., 0.3\n*DENSITY\n7.8e-9\n"
NODES2 = """*NODE
1,0,0
2,1,0
3,2,0
4,0,1
5,1,1
6,2,1
"""
ELEMS2 = """*ELEMENT, TYPE=CPE4, ELSET=EALL
1, 1,2,5,4
2, 2,3,6,5
*ELSET, ELSET=E1
1
*ELSET, ELSET=E2
2
*NSET, NSET=LEFT
1,4
*NSET, NSET=RIGHT
3,6
"""
STEP = "*STEP\n*STATIC\n0.1, 1.0\n"


def _model(text, dim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model = _parse_source(text, dim=dim, strict=True)
        solver, sysm = model.create_solver(dim=dim)
    st = model.steps[[s for s in model.steps if s != "Initial"][-1]]
    st.parent_model = model
    if dim == 2:
        # The deleted 2D pressure code looked regions up in mesh.element_sets only,
        # which the deck reader does not fill: give it the sets it could see.
        for name, ids in (("E1", {1}), ("E2", {2}), ("EALL", {1, 2})):
            solver.mesh.element_sets.setdefault(name, ids)
    return solver, sysm, st


def deck3(body, amp=False, nlgeom=False):
    step = "*STEP, NLGEOM=YES\n*STATIC\n0.1, 1.0\n" if nlgeom else STEP
    return (NODES3 + ELEMS3 + MATS + "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n"
            + (AMP if amp else "") + step + body + "*END STEP\n")


def deck2(body, amp=False, nlgeom=False):
    step = "*STEP, NLGEOM=YES\n*STATIC\n0.1, 1.0\n" if nlgeom else STEP
    return (NODES2 + ELEMS2 + MATS + "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n1.\n"
            + (AMP if amp else "") + step + body + "*END STEP\n")


def _twin_bodies(amp):
    a = ", AMPLITUDE=A1" if amp else ""
    b2 = (f"*BOUNDARY{a}\nLEFT, 1, 1, 0.5\nLEFT, 2, 2, -0.25\n"
          f"*CLOAD{a}\nRIGHT, 1, 10.\nRIGHT, 2, 4.\n"
          f"*DLOAD{a}\nE1, P3, 7.\nE2, P2, 3.\n")
    # 3D: CLOAD is per node and there are two nodes per 2D node -> halve it
    b3 = (f"*BOUNDARY{a}\nLEFT, 1, 1, 0.5\nLEFT, 2, 2, -0.25\n"
          f"*CLOAD{a}\nRIGHT, 1, 5.\nRIGHT, 2, 2.\n"
          f"*DLOAD{a}\nE1, P5, 7.\nE2, P4, 3.\n")
    return b2, b3


def _plan(text, dim, nlgeom, u=None, procedure=None):
    solver, sysm, st = _model(text, dim)
    if procedure:
        st.procedure = procedure
    n = dim * len(solver.mesh.nodes)
    solver._u_step_start = np.zeros(n) if u is None else u.copy()
    plan = lower_step(st, sysm, solver.mesh, dim, nlgeom,
                      explicit_density=getattr(solver, "_explicit_density", None))
    return solver, plan


def _disp2(n_nodes=6):
    k = np.arange(2 * n_nodes)
    return 0.05 * np.sin(1.3 * k + 0.4) + 0.02 * k / n_nodes


def _extrude(u2):
    u3 = np.zeros(36)
    for lay in (0, 1):
        for i in range(6):
            u3[3 * (6 * lay + i): 3 * (6 * lay + i) + 2] = u2[2 * i: 2 * i + 2]
    return u3


def _fold3(f3):
    F = f3.reshape(12, 3)
    return (F[:6, :2] + F[6:, :2]).ravel()


# --------------------------------------------------------------- twin decks
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("procedure", ["STATIC", "DYNAMIC"])
@pytest.mark.parametrize("nlgeom", [False, True])
def test_twin_decks_give_identical_in_plane_vectors(amp, procedure, nlgeom):
    b2, b3 = _twin_bodies(amp)
    u2 = _disp2() if nlgeom else None
    u0_2 = 0.01 * np.cos(np.arange(12))
    u0_3 = _extrude(u0_2)
    _, p2 = _plan(deck2(b2, amp, nlgeom), 2, nlgeom, u0_2, procedure)
    _, p3 = _plan(deck3(b3, amp, nlgeom), 3, nlgeom, u0_3, procedure)
    for t in TIMES:
        f2 = p2.f_ext(t, u2 if u2 is not None else np.zeros(12))
        f3 = p3.f_ext(t, _extrude(u2) if u2 is not None else np.zeros(36))
        ref = np.abs(f2).max()
        assert ref > 0
        assert np.abs(_fold3(f3) - f2).max() <= 1e-12 * ref
        # the extruded z component carries nothing (no out-of-plane load)
        assert np.abs(f3.reshape(12, 3)[:, 2]).max() <= 1e-12 * ref

        d2, v2 = p2.prescribed(t, u0_2)
        d3, v3 = p3.prescribed(t, u0_3)
        m2 = {int(d): float(v) for d, v in zip(d2, v2)}          # (2 * n + c)
        m3 = {int(d): float(v) for d, v in zip(d3, v3)}          # (3 * n + c)
        assert m2 and len(m3) == 2 * len(m2)
        for dof, val in m2.items():
            node, comp = divmod(dof, 2)
            for lay in (0, 1):
                v = m3[3 * (6 * lay + node) + comp]
                assert abs(v - val) <= 1e-12 * max(abs(val), 1e-12)


def test_prescribed_ramps_from_step_start_and_scales_with_amplitude():
    b2, _ = _twin_bodies(False)
    u0 = 0.01 * np.cos(np.arange(12))
    _, plan = _plan(deck2(b2), 2, False, u0)
    d, v = plan.prescribed(0.5, u0)
    for dof, val in zip(d, v):
        tgt = {0: 0.5, 1: -0.25}[int(dof) % 2]
        assert val == pytest.approx(u0[dof] + (tgt - u0[dof]) * 0.5, rel=1e-14)
    b2a, _ = _twin_bodies(True)
    _, plan_a = _plan(deck2(b2a, True), 2, False, u0)
    d, v = plan_a.prescribed(1.0, u0)
    for dof, val in zip(d, v):
        tgt = {0: 0.5, 1: -0.25}[int(dof) % 2]
        assert val == pytest.approx(tgt * 1.5, rel=1e-14)


def test_unknown_amplitude_and_region_raise_with_location():
    with pytest.raises(ValueError, match="amplitude"):
        _plan(deck2("*BOUNDARY, AMPLITUDE=NOPE\nLEFT, 1, 1, 0.5\n"), 2, False)
    with pytest.raises(RegionError):
        _plan(deck2("*CLOAD\nNOPE, 1, 1.\n"), 2, False)
    with pytest.raises(RegionError):
        _plan(deck2("*DLOAD\nNOPE, P3, 1.\n"), 2, False)


def test_2d_body_load_still_refused():
    with pytest.raises(NotImplementedError):
        _plan(deck2("*DLOAD\nE1, BX, 1.\n"), 2, False)


# -------------------------------------------------- follower vs reference
def test_pressure_is_follower_only_under_nlgeom():
    body = "*BOUNDARY\nLEFT, 1, 2\n*DLOAD\nE1, P3, 7.\nE2, P2, 3.\n"
    u = _disp2()
    for dim, deck, ext in ((2, deck2, lambda x: x), (3, deck3, _extrude)):
        uu = ext(u)
        n = len(uu)
        _, ref = _plan(deck(body), dim, False)
        _, fol = _plan(deck(body, nlgeom=True), dim, True)
        f_ref0 = ref.f_ext(1.0, np.zeros(n))
        # reference: independent of u
        assert np.array_equal(ref.f_ext(1.0, uu), f_ref0)
        # follower at u = 0 is the reference load
        assert np.abs(fol.f_ext(1.0, np.zeros(n)) - f_ref0).max() <= 1e-12 * np.abs(f_ref0).max()
        # follower with a displaced geometry differs
        assert np.abs(fol.f_ext(1.0, uu) - f_ref0).max() > 1e-3 * np.abs(f_ref0).max()


# --------------------------------------------- regression vs deleted code
def _old_cases():
    """name -> (dim, text, u_for_f_ext or None). Reference-geometry (STATIC,
    NLGEOM off) 3D decks, and 2D decks at u = 0 and at a displaced u (where the
    old 2D code was a follower load whatever the step said)."""
    a = ", AMPLITUDE=A1"
    return {
        "3d_static": (3, deck3("*BOUNDARY\nLEFT, 1, 3\n*BOUNDARY\nRIGHT, 1, 1, 0.3\n*CLOAD\nRIGHT, 1, 10.\nRIGHT, 3, 2.\n"
                               "*DLOAD\nE1, P2, 3.\nE2, P5, 4.\n"), None),
        "3d_amp": (3, deck3(f"*BOUNDARY{a}\nLEFT, 1, 3, 0.2\n*CLOAD{a}\nRIGHT, 2, 10.\n*DLOAD{a}\nE1, P3, 3.\n",
                            amp=True), None),
        "3d_grav": (3, deck3("*BOUNDARY\nLEFT, 1, 3\n*DLOAD\nE1, GRAV, 9810., 0., 0., -1.\nEALL, BZ, -2.\nEALL, BX, 1.5\n"), None),
        "2d_cload": (2, deck2("*BOUNDARY\nLEFT, 1, 2\n*CLOAD\nRIGHT, 1, 10.\nRIGHT, 2, -3.\n"), None),
        "2d_cload_amp": (2, deck2(f"*BOUNDARY{a}\nLEFT, 1, 2, 0.4\n*CLOAD{a}\nRIGHT, 2, 10.\n", amp=True), None),
        # pressure only: the old 2D sign is opposite (see SIGN_FLIPPED)
        "2d_p_u0": (2, deck2("*BOUNDARY\nLEFT, 1, 2\n*DLOAD\nE1, P3, 7.\nE2, P2, 3.\nE2, P3, 1.\n"), None),
        "2d_p_u": (2, deck2("*BOUNDARY\nLEFT, 1, 2\n*DLOAD\nE1, P3, 7.\nE2, P2, 3.\nE2, P3, 1.\n"), "disp"),
        "2d_p_amp": (2, deck2(f"*BOUNDARY{a}\nLEFT, 1, 2, 0.4\n*DLOAD{a}\nE1, P1, 7.\n", amp=True), "disp"),
    }


def _u_for(dim, tag):
    if tag is None:
        return np.zeros(dim * 6 if dim == 2 else 36)
    return _disp2()


def _u0_for(dim):
    n = 12 if dim == 2 else 36
    return 0.01 * np.cos(np.arange(n))


def capture_old_fixture():     # pragma: no cover -- run once on 140a571
    out = {}
    for name, (dim, text, tag) in _old_cases().items():
        solver, sysm, st = _model(text, dim)
        if 2 == dim:
            st.procedure = "STATIC"
        u = _u_for(dim, tag)
        u0 = _u0_for(dim)
        for t in TIMES:
            solver._u_step_start = u0.copy()
            solver.u[:] = u
            solver._update_step_bcs(st, sysm, t, 1.0)
            out[f"{name}/{t}/f"] = solver._compute_step_external_loads(st, sysm, t, 1.0)
            items = sorted(solver.fixed_dofs.items())
            out[f"{name}/{t}/dofs"] = np.array([d for d, _ in items], dtype=np.int64)
            out[f"{name}/{t}/vals"] = np.array([v for _, v in items])
    np.savez(FIXTURE, **out)


# The deleted 2D pressure code (solver/traction.py) pushed a POSITIVE pressure
# OUTWARD (tension) -- opposite to Abaqus and to the 3D solver -- and split each
# face force between its two nodes by the swept area about the coordinate ORIGIN
# (W = -p x1 x x2), so the nodal distribution depended on where the edge sat. Its
# face resultant was right in magnitude and direction. StepPlan uses the loads
# catalog (Abaqus sign, consistent nodal split): compare resultants, opposite sign.
SIGN_FLIPPED = {"2d_p_u0", "2d_p_u", "2d_p_amp"}


@pytest.mark.parametrize("name", ["3d_static", "3d_amp", "3d_grav", "2d_cload", "2d_cload_amp",
                                  "2d_p_u0", "2d_p_u", "2d_p_amp"])
def test_matches_the_deleted_solver_code(name):
    dim, text, tag = _old_cases()[name]
    data = np.load(FIXTURE)
    u = _u_for(dim, tag)
    u0 = _u0_for(dim)
    # 3D decks are NLGEOM off (reference geometry, as the old 3D always was);
    # 2D decks compare the follower geometry the old 2D code always used.
    nlgeom = dim == 2
    _, plan = _plan(text, dim, nlgeom, u0)
    for t in TIMES:
        f_old = data[f"{name}/{t}/f"] * (-1.0 if name in SIGN_FLIPPED else 1.0)
        f_new = plan.f_ext(t, u)
        scale = max(np.abs(f_old).max(), 1e-300)
        if name in SIGN_FLIPPED:
            r_new = f_new.reshape(-1, 2).sum(axis=0)
            r_old = f_old.reshape(-1, 2).sum(axis=0)
            assert np.abs(r_new - r_old).max() <= 1e-12 * np.abs(r_old).max(), (name, t)
        else:
            assert np.abs(f_new - f_old).max() <= 1e-12 * scale, (name, t)
        d, v = plan.prescribed(t, u0)
        order = {int(k): float(x) for k, x in zip(d, v)}          # last wins, as fix_dof did
        assert sorted(order) == list(data[f"{name}/{t}/dofs"])
        np.testing.assert_allclose([order[k] for k in sorted(order)], data[f"{name}/{t}/vals"],
                                   rtol=1e-12, atol=1e-14)
