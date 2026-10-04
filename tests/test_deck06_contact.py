"""Deck #06 (bolted flange + gasket, 3D contact) regression gate.

Plan: dev_log/plan_deck06_contact_fix_20261004.md. Every check here is
measured on the ACCEPTED state re-assembled with the contact freeze cleared,
i.e. with the true one-sided contact law -- never on the solver's own
converged flag, its .sta file, or a residual computed while nodes were frozen
(AGENTS.md sec4.9).

`xfail(strict=True)` entries are a characterization baseline, as in
tests/element_contract: each carries the number measured on 2026-10-04 before
the fix pass. An entry that starts passing becomes an XPASS failure -- the
signal to delete the mark, never to widen it.
"""
import contextlib
import io
import os
import warnings

import numpy as np
import pytest

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.solver.step_executor import MultiStepExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, "benchmark_inp", "phase2_methodology", "06_bolt_preload_gasket_flange_3d.inp")

# Step 1 bolt-seat reaction R_z [N], each method's OWN limit on this mesh.
# There is no single reference: PENALTY is node-to-surface (35 gasket nodes
# on a non-matching flange mesh) and NITSCHE is surface-to-surface, so their
# discrete limits differ (1.5%). Mesh error is ~5% (plan P0-3), so neither is
# the continuum answer. Measured 2026-10-04 after the Pardiso symmetry fix
# (unsymmetric tangents had been solved symmetrized) and the real Nitsche:
#   PENALTY: x10 -6543.05, x30 -6625.60, x50 -6642.39, x100 -6655.36,
#            x300 -6663.55 -> limit ~ -6668
#   NITSCHE (consistent): x1 -6584.62, x10 -6568.99, x100 -6567.76 -- nearly
#            gamma-independent, as a consistent method should be
R_LIMIT = {"PENALTY": -6668.0, "NITSCHE": -6567.8}
R_REF = R_LIMIT["NITSCHE"]    # force scale for residual tolerances
RF_RTOL = {"PENALTY": 0.02,   # penalty regularization at Abaqus's default 10x k_ref measured 1.6%
           "NITSCHE": 0.01}
RESID_RTOL = 1e-3             # acceptance A2, relative to |R_REF|

# 1/4 model, pipe bore R=15, flange height 8.0-1.5=6.5, p=5 MPa:
# sum f_x = sum f_y = p*R*H, sum f_z = 0 (acceptance A5).
PIPE_P, PIPE_R, PIPE_H = 5.0, 15.0, 6.5


def _build(method, scale=1.0):
    _, model = _parse_source(INP, dim=3, strict=False)
    objs = list(model.interactions.values()) + list(getattr(model.root_assembly, "contact_pairs", []))
    for st in model.steps.values():
        objs += [getattr(e, "entity", e) for e in st.interactions.values()]
    for o in objs:
        if hasattr(o, "constraint_enforcement"):
            o.constraint_enforcement = method
        prop = getattr(o, "interaction_property", None)
        if prop is not None and not isinstance(prop, str):
            prop.stiffness_scale_factor = scale
    return model, MultiStepExecutor(model, verbose=False, dim=3)


_CACHE = {}


def _step1(method, scale=1.0):
    """Run BOLT_PRELOAD and measure the accepted state with the freeze cleared."""
    key = (method, scale)
    if key not in _CACHE:
        model, ex = _build(method, scale)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            ok = ex.execute_step(model.steps["BOLT_PRELOAD"], from_parent_checkpoint=True)
        s = ex.solver
        for c in s.constraints:
            if hasattr(c, "frozen_active_nodes"):
                c.frozen_active_nodes = None
        free = np.ones(s.num_dofs, dtype=bool)
        free[list(s.fixed_dofs.keys())] = False
        _, f_int = s.assemble_system(s.u, dt=1.0)
        bolt = ex.sys.global_nsets["FLANGE_1.FLANGE_BOLT_SEAT"]
        _CACHE[key] = {
            "ok": bool(ok),
            "rf": float(sum(f_int[3 * i + 2] for i in bolt)),
            "resid": float(np.linalg.norm(f_int[free])),  # f_ext = 0 in step 1
        }
    return _CACHE[key]


_xf = pytest.mark.xfail(strict=True)

RF_CASES = [
    pytest.param("PENALTY", 1.0, marks=_xf, id="penalty-default"),          # -5618.34 (-15.7%), 13.5 um penetration
    pytest.param("PENALTY", 10.0, id="penalty-x10"),                        # -6543.05 (-1.9%)
    pytest.param("PENALTY", 100.0, id="penalty-x100"),                      # -6655.36; stalled 40+ min before the Pardiso fix
    pytest.param("NITSCHE", 1.0, id="nitsche-default"),                     # -6584.62 (+0.26%); was -6412.48 as a penalty
    pytest.param("NITSCHE", 100.0, id="nitsche-x100"),                      # -6567.76
    # gamma0 = 5e4 (x1000): fails from the first increment (all attempts cut
    # back). Far beyond the stability threshold; not characterized further.
    pytest.param("NITSCHE", 1000.0, marks=_xf, id="nitsche-x1000"),
]


@pytest.mark.parametrize("method,scale", RF_CASES)
def test_step1_reaction_reaches_own_stiffness_limit(method, scale):
    r = _step1(method, scale)
    ref = R_LIMIT[method]
    assert r["ok"], "BOLT_PRELOAD did not converge"
    assert abs(r["rf"] - ref) <= RF_RTOL[method] * abs(ref), f"RF_bolt={r['rf']:.2f}, limit={ref}"


@pytest.mark.parametrize("method,scale", [
    ("PENALTY", 1.0), ("PENALTY", 10.0), ("NITSCHE", 1.0),
    # PENALTY x50 used to be FALSELY accepted (ok=True, exhaustion branch,
    # r = 90.2 N, RF_bolt=-6956.73). Now it genuinely converges (r ~ 3e-2 N).
    ("PENALTY", 50.0),
])
def test_step1_accepted_state_is_in_equilibrium_without_freeze(method, scale):
    r = _step1(method, scale)
    assert r["ok"]
    assert r["resid"] <= RESID_RTOL * abs(R_REF), f"unfrozen |r_free|={r['resid']:.3e}"


@pytest.mark.skip(reason="DUAL_MORTAR stalls at r~1e3 N for 6+ increment attempts and does "
                         "not return in useful time (plan P8); a hanging test is not a gate")
def test_step1_dual_mortar():
    r = _step1("DUAL_MORTAR")
    assert r["ok"] and abs(r["rf"] - R_REF) <= 0.02 * abs(R_REF)


# ---------------------------------------------------------------------------
# Contact patch test: the one configuration here with an EXACT answer.
# Two stacked unit blocks, frictionless HARD contact, top face pressed by
# DELTA, symmetry on x=0/y=0, bottom fixed in z. Each block is in uniaxial
# stress (the interface slides freely), so F = |DELTA| / (h/E_bot + h/E_top).
# Stiffnesses are large so the solver's absolute `r_norm < 1.0` acceptance
# (plan P3) is not what decides the result: at E~1e3 the same test was
# accepted with a 10% residual.
# ---------------------------------------------------------------------------
PATCH_DELTA, PATCH_E_BOT, PATCH_E_TOP = -1e-4, 1e8, 2e10
PATCH_F = abs(PATCH_DELTA) / (1.0 / PATCH_E_BOT + 1.0 / PATCH_E_TOP)
PATCH_RTOL = 5e-3


def _patch_block(name, n, z0, z1, mat):
    L = [f"*PART, NAME={name}", "*NODE"]
    g, nid = {}, 1
    xs, zs = np.linspace(0, 1, n + 1), np.linspace(z0, z1, n + 1)
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                g[(i, j, k)] = nid
                L.append(f"{nid}, {xs[i]:.9f}, {xs[j]:.9f}, {zs[k]:.9f}")
                nid += 1
    L.append(f"*ELEMENT, TYPE=C3D8, ELSET=E_{name}")
    eid, top, bot = 1, [], []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c = [g[(i, j, k)], g[(i + 1, j, k)], g[(i + 1, j + 1, k)], g[(i, j + 1, k)],
                     g[(i, j, k + 1)], g[(i + 1, j, k + 1)], g[(i + 1, j + 1, k + 1)], g[(i, j + 1, k + 1)]]
                L.append(f"{eid}, " + ", ".join(map(str, c)))
                if k == n - 1:
                    top.append(eid)
                if k == 0:
                    bot.append(eid)
                eid += 1
    for nm, ids in [("XSYM", [g[(0, j, k)] for k in range(n + 1) for j in range(n + 1)]),
                    ("YSYM", [g[(i, 0, k)] for k in range(n + 1) for i in range(n + 1)]),
                    ("ZBOT", [g[(i, j, 0)] for j in range(n + 1) for i in range(n + 1)]),
                    ("ZTOP", [g[(i, j, n)] for j in range(n + 1) for i in range(n + 1)])]:
        L.append(f"*NSET, NSET={nm}")
        L += [", ".join(map(str, ids[m:m + 16])) for m in range(0, len(ids), 16)]
    L.append(f"*SURFACE, NAME={name}_TOP, TYPE=ELEMENT")
    L += [f"{e}, S2" for e in top]
    L.append(f"*SURFACE, NAME={name}_BOT, TYPE=ELEMENT")
    L += [f"{e}, S1" for e in bot]
    L += [f"*SOLID SECTION, ELSET=E_{name}, MATERIAL={mat}", "1.0,", "*END PART"]
    return L


def _patch_deck(path, n_bot, n_top, overclosure=0.0, delta=PATCH_DELTA):
    L = ["*HEADING", "contact patch test"]
    L += _patch_block("BOT", n_bot, 0.0, 1.0, "M_BOT")
    L += _patch_block("TOP", n_top, 1.0 - overclosure, 2.0 - overclosure, "M_TOP")
    L += ["*ASSEMBLY, NAME=A", "*INSTANCE, NAME=B1, PART=BOT", "*END INSTANCE",
          "*INSTANCE, NAME=T1, PART=TOP", "*END INSTANCE",
          "*SURFACE, NAME=S_SLAVE, COMBINE=UNION", "B1.BOT_TOP",
          "*SURFACE, NAME=S_MASTER, COMBINE=UNION", "T1.TOP_BOT", "*END ASSEMBLY",
          "*MATERIAL, NAME=M_BOT", "*ELASTIC", f"{PATCH_E_BOT}, 0.38",
          "*MATERIAL, NAME=M_TOP", "*ELASTIC", f"{PATCH_E_TOP}, 0.30",
          "*SURFACE INTERACTION, NAME=INT", "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD",
          "*CONTACT PAIR, INTERACTION=INT", "S_SLAVE, S_MASTER",
          "*BOUNDARY", "B1.XSYM, 1, 1, 0.0", "B1.YSYM, 2, 2, 0.0", "B1.ZBOT, 3, 3, 0.0",
          "T1.XSYM, 1, 1, 0.0", "T1.YSYM, 2, 2, 0.0",
          "*STEP, NAME=PRESS, NLGEOM=YES", "*STATIC", "0.25, 1.0, 1e-5, 0.5",
          "*BOUNDARY", f"T1.ZTOP, 3, 3, {delta}", "*END STEP"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


PATCH_CASES = [
    # measured 2026-10-04, relative error vs PATCH_F (matching 2/2 | non-matching 3/2 slave/master)
    pytest.param("PENALTY", 1.0, marks=_xf, id="penalty-x1"),       # -2.7e-2 | -2.3e-2
    pytest.param("PENALTY", 10.0, id="penalty-x10"),                # -2.8e-3 | -2.4e-3
    # The former "Nitsche" (p = gamma*g only) erred exactly as 1/gamma:
    # -1.7e-1 / -2.0e-2 / -4.5e-4 at x1 / x10 / x1000 -- a penalty. With the
    # bulk-stress (consistency) term it reproduces the patch at default gamma.
    pytest.param("NITSCHE", 1.0, id="nitsche-x1"),
    pytest.param("NITSCHE", 10.0, id="nitsche-x10"),
]


@pytest.mark.parametrize("n_bot", [2, 3], ids=["matching", "nonmatching"])
@pytest.mark.parametrize("method,scale", PATCH_CASES)
def test_contact_patch_exact_force(tmp_path, method, scale, n_bot):
    path = str(tmp_path / f"patch_{n_bot}.inp")
    _patch_deck(path, n_bot, 2)
    _, model = _parse_source(path, dim=3, strict=True)
    for o in list(model.interactions.values()) + [getattr(e, "entity", e) for st in model.steps.values()
                                                   for e in st.interactions.values()]:
        if hasattr(o, "constraint_enforcement"):
            o.constraint_enforcement = method
        prop = getattr(o, "interaction_property", None)
        if prop is not None and not isinstance(prop, str):
            prop.stiffness_scale_factor = scale
    ex = MultiStepExecutor(model, verbose=False, dim=3)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        ok = ex.execute_step(model.steps["PRESS"], from_parent_checkpoint=True)
    s = ex.solver
    for c in s.constraints:
        if hasattr(c, "frozen_active_nodes"):
            c.frozen_active_nodes = None
    free = np.ones(s.num_dofs, dtype=bool)
    free[list(s.fixed_dofs.keys())] = False
    _, f = s.assemble_system(s.u, dt=1.0)
    top = sum(f[3 * i + 2] for i in ex.sys.global_nsets["T1.ZTOP"])
    assert ok
    assert np.linalg.norm(f[free]) <= 1e-3 * PATCH_F
    assert abs(abs(top) / PATCH_F - 1.0) <= PATCH_RTOL, f"RF_top={top:.6e}, exact={PATCH_F:.6e}"


# Before 2026-10-04 this gave (0, 0, -5.00) N at every t: C3D8 surface pressure
# fell back to splitting the magnitude over the nodes as a -z force, unramped.
def test_step2_pressure_load_vector():
    model, ex = _build("PENALTY")
    with contextlib.redirect_stdout(io.StringIO()):
        s, sys_ = ex.initialize_solver()
    step = model.steps["INTERNAL_PRESSURE"]
    step.parent_model = model   # as MultiStepExecutor.execute_step does
    expect = PIPE_P * PIPE_R * PIPE_H
    for t in (0.5, 1.0):
        f = s._compute_step_external_loads(step, sys_, t, 1.0)
        sums = np.array([f[0::3].sum(), f[1::3].sum(), f[2::3].sum()])
        np.testing.assert_allclose(sums, [t * expect, t * expect, 0.0], rtol=1e-6, atol=1e-6 * expect)


def test_increment_log_records_real_increments():
    """The .sta used to get one hard-coded INC=1/ATT=1 row per step. The
    solver now keeps one record per attempt; with *STATIC 0.25 the preload
    step needs at least four converged increments ending at step time 1.0."""
    model, ex = _build("PENALTY", 10.0)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        assert ex.execute_step(model.steps["BOLT_PRELOAD"], from_parent_checkpoint=True)
    log = ex.solver.increment_log
    done = [r for r in log if r["converged"]]
    assert len(done) >= 4
    assert [r["inc"] for r in done] == list(range(1, len(done) + 1))
    assert abs(done[-1]["t_step"] - 1.0) < 1e-12
    assert all(r["n_iter"] > 0 for r in log)


# Interference fit: top block placed 2e-3 INTO the bottom one (well past the
# 1e-5 flush guard), top face held in place. Exact force is the same uniaxial
# solution with the overclosure as the imposed shortening. Before the P6 fix
# the contact normal was oriented from the sign of the initial gap, so an
# initially overclosed pair got its normal flipped and read as OPEN.
OVERCLOSURE = 2e-3


# x10 for PENALTY (patch-test error 0.3%; x1000 is in the P7 stiff-penalty
# non-convergence regime), x1000 for NITSCHE (its x1 error is -17%, see above).
@pytest.mark.parametrize("method,scale", [("PENALTY", 10.0), ("NITSCHE", 1000.0)])
def test_contact_initial_overclosure_pushes_apart(tmp_path, method, scale):
    path = str(tmp_path / "interference.inp")
    _patch_deck(path, 2, 2, overclosure=OVERCLOSURE, delta=0.0)
    _, model = _parse_source(path, dim=3, strict=True)
    for o in list(model.interactions.values()) + [getattr(e, "entity", e) for st in model.steps.values()
                                                   for e in st.interactions.values()]:
        if hasattr(o, "constraint_enforcement"):
            o.constraint_enforcement = method
        prop = getattr(o, "interaction_property", None)
        if prop is not None and not isinstance(prop, str):
            prop.stiffness_scale_factor = scale
    ex = MultiStepExecutor(model, verbose=False, dim=3)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        ok = ex.execute_step(model.steps["PRESS"], from_parent_checkpoint=True)
    s = ex.solver
    _, f = s.assemble_system(s.u, dt=1.0)
    top = sum(f[3 * i + 2] for i in ex.sys.global_nsets["T1.ZTOP"])
    exact = OVERCLOSURE / (1.0 / PATCH_E_BOT + 1.0 / PATCH_E_TOP)
    assert ok
    # compression shows as a NEGATIVE z-reaction on the top face (as in the
    # pressed patch test above)
    assert top < 0.0, f"RF_top={top:.4e}: no compressive contact force from the interference"
    assert abs(-top / exact - 1.0) <= 0.02, f"RF_top={top:.6e}, exact={exact:.6e}"
