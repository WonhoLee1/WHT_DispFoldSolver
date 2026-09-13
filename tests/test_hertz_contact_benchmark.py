"""
test_hertz_contact_benchmark.py
=================================
Regression gate for benchmark_element/benchmark_3d_contact.py -- the
quantitative Hertz-contact comparison for Phase 1 contact
(dispsolver/model/interaction.py + dispsolver/constraint3d/surface_contact3d.py).

See dev_log/hertz_contact_benchmark_20260913.md for the full writeup. Two
honestly-characterized findings, both xfailed with the measured evidence
(not force-fixed or hidden):

1. A UNIFORM mesh coarser than the actual Hertz contact half-width
   (b~0.4mm at reachable loads vs dx=1mm) structurally cannot resolve a
   patch narrower than one element -- b_fem measures 0 even with a
   nonzero, physically reasonable reaction force. This is a resolution
   limit of the test setup, not a solver defect.
2. A GRADED mesh (fine enough to resolve the patch) reliably DIVERGES the
   moment growing contact requires a NEW node column to activate -- and
   this is NOT a step-size artifact: bisecting the load increment through
   7+ halvings (min_frac 1/128 -> 1/8192) still fails at the same
   physical point, just after a few more columns activate first. This is
   the "residual grows monotonically instead of decreasing" pathology
   tests/test_contact_phase1.py's own docstring already named as a known
   Phase 1 gap (design doc sec8: active-set-stability tracking, deferred to
   Phase 2) -- reproduced here quantitatively for the first time, on a
   different (multi-node-activation) geometry than that test's own
   single-node-activation case.

**Update, same day**: the active-set-stability mechanism (SDI handling,
`DynamicSolver3D._contact_active_set()`/`max_sdi_iters`) AND a genuine,
independent line-search bug (found while verifying SDI: the Armijo
backtracking search used to silently accept a residual-INCREASING full
step whenever none of its 6 candidates improved on it, rather than
correctly rejecting the increment) are now both fixed in `dynamic3d.py`
-- see the dev_log's "Update" section for the exact mechanism and
verification (full 3D regression suite unaffected). Measured effect: the
graded-mesh benchmark now genuinely progresses further (n_active 39->45,
final_uz -0.0004->-0.000425 at the same min_frac) before still hitting a
DIFFERENT, deeper wall -- once ~45+ nodes are simultaneously active, the
Newton direction itself stalls near a residual plateau (now correctly
REJECTED as non-convergent rather than silently accepted, which is
itself real progress, but the increment still can't be completed with
plain penalty + Newton at this active-set size). This is likely genuinely
Phase 2 scope (semi-smooth Newton / augmented Lagrange for a better-
conditioned many-simultaneous-constraint tangent), not a further Newton-
driver control-flow fix -- do not keep iterating on solve_step() itself
for this benchmark without new evidence pointing at a specific mechanism.

Do not widen these xfail reasons or force a pass by loosening solver
tolerances -- the fix is Phase 2's active-set-stability convergence
machinery, not a benchmark-side workaround. An unexpected pass (xpass)
here is the correct signal that Phase 2 has landed and this benchmark
should be re-run for real quantitative Hertz agreement.
"""

import pytest

from benchmark_element.benchmark_3d_contact import (
    hertz_line_contact_reference,
    run_hertz_contact_benchmark,
)


def test_hertz_reference_formula_self_consistent():
    """The two algebraically-equivalent closed forms for peak pressure
    (p0 = 2P/(pi*b) vs p0 = sqrt(P*E*/(pi*R))) must agree -- this is
    checked internally by hertz_line_contact_reference's own assert, this
    test just confirms it doesn't raise and returns sane values."""
    ref = hertz_line_contact_reference(P_per_length=100.0, R=254.0, E=206000.0, nu=0.3)
    assert ref["b"] > 0.0
    assert ref["p0"] > 0.0
    assert ref["E_star"] > 206000.0  # E* = E/(1-nu^2) > E always


def test_hertz_contact_uniform_mesh_cannot_resolve_patch():
    """XFAIL: uniform mesh (dx=1mm) is coarser than the Hertz half-width
    (~0.4mm) at reachable loads -- the contact patch can never grow past
    a single node column. Measured 2026-09-13 (dev_log/hertz_contact_benchmark_20260913.md):
    depending on exactly how much bisection/iteration budget is given,
    this either (a) "converges" per the solver's own loose acceptance
    criterion with b_fem=0 (patch narrower than one element -- a
    resolution limit, not a defect), or (b) DIVERGES outright even on the
    very first contact-activation increment once given enough Newton
    iterations/bisection depth to expose that the loose criterion's early
    "converged" call in case (a) wasn't really settled. Both outcomes are
    instances of the SAME underlying Phase 1 contact-activation fragility
    this whole test module documents -- xfail either way rather than
    requiring one specific exact number, since which exact outcome occurs
    is itself sensitive to convergence-tolerance tightness (consistent
    with a discontinuous-residual/line-search pathology, not a clean
    deterministic threshold). Explicit fast max_iters/min_frac here (not
    the exploratory defaults) so this regression test finishes in seconds,
    not the ~46 minutes a deep-bisection run of this same config took
    during investigation."""
    res = run_hertz_contact_benchmark(uz_target=-0.02, n_steps=40, nx=60, max_iters=60, min_frac=1.0 / 64)
    if res["status"] == "DIVERGED":
        pytest.xfail(
            f"DIVERGED at final_uz={res['final_uz']:.6f} (n_active={res['n_active']}) -- "
            "Phase 1 contact-activation Newton fragility, see docstring and dev_log"
        )
    if res["b_fem"] == 0.0 and res["n_active"] > 0:
        pytest.xfail(
            f"uniform dx={res['dx_mesh']:.3f}mm > Hertz b_ref={res['b_ref']:.3f}mm at this load "
            "-- patch narrower than one element, not a solver defect (see docstring)"
        )
    assert res["ratio_b_fem_to_ref"] > 0.5


def test_hertz_contact_graded_mesh_diverges_on_column_activation():
    """XFAIL: graded mesh resolves the patch (multiple columns activate),
    but Newton reliably diverges once growing contact requires a NEW
    column to activate -- independent of load-step size (verified via a
    bisection-depth sweep in dev_log/hertz_contact_benchmark_20260913.md,
    not re-swept here for test speed -- this test just confirms the
    characterized failure mode still reproduces at a fixed, fast
    bisection budget)."""
    res = run_hertz_contact_benchmark(
        uz_target=-0.01, n_steps=100, x_fine=3.0, dx_fine=0.1, n_coarse=15,
        max_iters=60, min_frac=1.0 / 64,
    )
    if res["status"] == "DIVERGED":
        pytest.xfail(
            f"DIVERGED at final_uz={res['final_uz']:.6f} (n_active={res['n_active']} active nodes) -- "
            "known Phase 1 active-set-activation Newton limitation, see docstring and dev_log"
        )
    assert res["status"] == "CONVERGED"
