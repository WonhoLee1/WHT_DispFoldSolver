# Hertz contact benchmark vs Phase 1 contact — findings (2026-09-13)

Follow-up to `dev_log/3d_contact_implementation_design_20260913.md`'s own
flagged gap: the design doc's Phase 1 acceptance-test note explicitly said
the closed-form Hertz half-width/peak-pressure formulas "still need to be
derived fresh and carefully... not attempted here" (in
`tests/test_contact_phase1.py`, which only checks qualitative sanity).
This session did that derivation and built the quantitative comparison
(`benchmark_element/benchmark_3d_contact.py`). Result: a real, reproducible
Newton-convergence limitation in Phase 1's plain-penalty contact was found
and characterized — not fixed (that's Phase 2 scope, already anticipated
by the design doc's own §8).

## Closed-form reference

`hertz_line_contact_reference()` derives, fresh (not from memory):

- Effective radius, cylinder (R) vs. flat rigid plane: `R_eff = R`
  (1/R_eff = 1/R + 1/inf).
- Effective modulus, rigid plane contributes no compliance:
  `E* = E / (1 - nu^2)`.
- Half-width: `b = sqrt(4*P*R_eff / (pi*E*))` (Johnson 1985 *Contact
  Mechanics* eq. 4.45; Timoshenko & Goodier 1951 *Theory of Elasticity*
  2nd ed. section 140 — same result).
- Peak pressure: `p0 = 2*P/(pi*b)`, algebraically cross-checked against
  the equivalent closed form `p0 = sqrt(P*E*/(pi*R_eff))` (verified
  identical to 1e-9 relative in code, not just asserted).

`P` is the FEM's own measured reaction line load (`total_normal_force`
from `SurfaceContactConstraint3D.assemble()`, divided by the block's unit
depth) — matching `reference_abaqus_docs/bmk_hertzcontact.txt`'s own
comparison methodology (Figure 2 compares FEM pressure to "the analytical
distribution" AT the FEM's converged load; 2D plane-strain contact has no
closed form linking absolute approach displacement to load, so comparing
at a chosen displacement is not meaningful — comparing at the FEM's own
resulting load is the only well-posed check).

## Geometry

Same "D-cross-section" curved block as
`tests/test_contact_phase1.py::test_hertz_inspired_curved_block_contact`
(flat top, single circular arc R=254mm on the bottom, E=206 GPa, nu=0.3 —
exactly the real benchmark's cylinder/material parameters), factored into
`make_hertz_curved_block_mesh()`. Added a graded x-grid option
(`_graded_x`: fine `dx_fine` spacing within `|x|<x_fine`, coarser linear
spacing out to `half_w`) because the original uniform mesh (dx=1mm) is an
order of magnitude coarser than the actual Hertz half-width at reachable
loads (measured b~0.4-0.5mm) — a uniform mesh fine enough to resolve that
directly everywhere would need hundreds of elements for no benefit far
from the contact patch.

## The finding: Newton diverges at contact-patch growth, independent of step size

With a UNIFORM mesh (dx=1mm), the contact patch never grows past the
single center-node column (predicted b~0.4mm < dx=1mm) — the FEM simply
can't resolve a patch narrower than one element, so `b_fem` measures 0
even while `total_normal_force` is nonzero and physically reasonable.
This is a resolution limit, not a solver problem, and is exactly why
`test_hertz_inspired_curved_block_contact` (uniform nx=16, dx=3.75mm)
converges cleanly all the way to its -0.5mm target: with that mesh, the
contact patch structurally can never need a second column to activate in
the range tested, so no discontinuous-residual event is ever triggered.

With a GRADED mesh (`x_fine=3.0, dx_fine=0.1, n_coarse=15`), the patch
starts resolving properly (n_active grows 3 -> 39 -> 51 nodes, i.e. 1 -> 13
-> 17 x-columns, as displacement ramps), but the solve reliably DIVERGES
the moment growth requires a genuinely NEW column to activate — and this
is **not a step-size artifact**: bisecting the load increment down through
7+ halvings (`min_frac` swept 1/128 -> 1/8192, i.e. the smallest
attempted increment shrank from ~1e-4mm to ~1.2e-8mm) still fails at the
same physical point. Two independent runs:

| min_frac | final_uz reached | n_active | n_calls before giving up |
|---|---|---|---|
| 1/128  | -0.0004mm    | 39 (13 columns) | 5  |
| 1/8192 | -0.000436mm  | 51 (17 columns) | 15 |

The finer bisection budget let a FEW more columns activate (13 -> 17)
before hitting the same wall again, but arbitrarily small further
increments do not get past it — this is the "residual grows
monotonically every Newton iteration instead of decreasing" failure mode
`tests/test_contact_phase1.py`'s own docstring already predicted and
named (design doc §8 point 4: "line search interacting badly with a
discontinuous residual"), now reproduced quantitatively and independent
of load-step size, on a DIFFERENT geometry (curved, multi-node-activation
case) than that test's own simple-block case (single-node activation,
never re-triggers this).

**This is a genuine, already-anticipated Phase 1 gap, not something to
force a workaround for in this session**: the design doc's own Phase 2
scope (§8: "active-set-stability convergence check and contact damping")
exists specifically because Phase 1's plain penalty + plain backtracking
line search was never expected to robustly handle sequential multi-node
contact-patch growth. This benchmark is the first concrete demonstration
of exactly that failure mode with real numbers, which sharpens Phase 2's
acceptance criteria (it should specifically be re-run against this same
benchmark once built).

## What the benchmark DOES show, with the caveat above

At the last-converged state before divergence (min_frac=1/8192 run):
`P_per_length=93.8 N/mm`, `b_fem=0.8mm` (17 columns x dx_fine=0.1mm/2, i.e.
the raw active-node half-extent) vs. Hertz `b_ref=0.366mm` at that same
load — ratio ~2.2. This is reported as a **preliminary, not validated**
data point: it comes from an intermediate, not-yet-fully-ramped state
reached only because the solve got stuck, not a deliberately-targeted
converged equilibrium at a chosen load. Plausible (not confirmed)
contributors to the 2.2x gap: (a) discrete node-based one-sided contact
activation inherently overshoots the true continuum patch edge by up to
one element width in the direction of growth, comparable in size to the
gap itself here (dx_fine=0.1mm vs b_ref=0.366mm, i.e. ~30% of the whole
half-width is one element -- non-negligible discretization bias); (b) the
finite block height/width (H=half_w=30mm) is only ~80x the patch, not the
Hertz theory's implicit "infinite half-space," though this is normally a
weak effect until the patch approaches ~R/3. Not chased further — getting
a genuinely well-converged, mesh-independent patch measurement needs
Phase 2's convergence robustness first.

## Update: two real Newton-driver defects found and fixed (same day, following external research)

A dispatched research agent (document-specialist) confirmed this exact
failure signature -- "residual grows/doesn't decrease exactly when a
contact constraint's active-set status flips" -- is the textbook
Abaqus/Standard "severe discontinuity iteration" (SDI) problem, and that
the standard production fix is to exempt such an iteration from the
normal convergence/line-search check rather than trying to force
convergence through the resulting discontinuity
(https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/gss/ch11s03.html).
Implemented in `DynamicSolver3D.solve_step()` (both this method and
`_solve_step_with_hooks()`):

1. **SDI handling** (`_contact_active_set()`, `max_sdi_iters` param):
   track whether any contact constraint's active node set changed since
   the previous iterate; if so, exempt that iteration from the
   convergence check and give it its own separate iteration budget
   (`max_sdi_iters`, default 200) instead of eating into `max_iters`
   (which now counts only iterations where the active set was already
   stable). Zero behavior change for any non-contact solve (verified:
   14/14 unrelated 3D element tests pass unchanged).
2. **A real, independent line-search bug found while verifying #1**:
   the existing 6-step backtracking Armijo line search set
   `found_valid = True` unconditionally on its very FIRST trial
   (`r_t_norm < best_r_norm or not found_valid`, and `not found_valid`
   is always true on trial 0) -- so a full Newton step (s=1.0) that made
   the residual WORSE, with none of the 5 remaining smaller trials
   improving on it either, was silently ACCEPTED anyway rather than
   triggering the increment cutback. This is not contact-specific (it's
   a base Newton-loop defect present for every 3D solve), but rarely
   triggers because most of this solver's problems have a full step
   that naturally decreases residual -- multiple simultaneous contact
   activations exposed it directly: measured residual creeping
   1.112 -> 1.143 over 50 "accepted" iterations, never once rejected,
   burning the entire iteration budget on a plateau instead of cutting
   back after a few iterations. Fixed: `found_valid` is now only set on
   a genuine improvement over the running best or on meeting the
   acceptance threshold. Verified via the same instrumented trace: the
   identical failing increment now correctly rejects at iteration 12
   (`converged=False`) instead of grinding through 62.

**Both fixes are real and independently verified** (full 3D regression
suite: 46 passed, 5 xfailed, 1 failed -- the single failure is the same
pre-existing, unrelated `test_c3d4_anp_tangent_consistency` kernel-level
tangent test confirmed via `git diff` to predate and be untouched by
either change).

**Net effect on the Hertz benchmark, measured**: genuine progress, not a
full resolution. With the SAME graded mesh and `min_frac=1/1024`, the
ramp now reaches `n_active=45` (up from 39) and `final_uz=-0.000425` (up
from -0.0004) before exhausting its bisection budget -- i.e. the fixes
let it push one more activation transition (39->45 nodes) further than
before, but it still cannot complete the full ramp to `uz_target=-0.01`.
The remaining blocker is a DIFFERENT, deeper issue than either fix
addressed: once enough nodes are simultaneously active (45+), the
underlying Newton *direction* itself appears to stall near a plateau
(now correctly detected and rejected rather than silently accepted, but
not actually resolved) -- likely needs either a genuinely better-
conditioned tangent for many simultaneous unilateral constraints
(semi-smooth Newton / Alart-Curnier, the research agent's #2
recommendation) or Phase 2's augmented Lagrangian, rather than a further
Newton-driver control-flow tweak. Not chased further this session --
tracked as Phase 2's concrete next target (design doc updated).

## Update: augmented Lagrangian attempted (Phase 2), found to DIVERGE, not fixed

Following the SDI+line-search work above, implemented augmented
Lagrangian enforcement per design doc sec4 (`SurfaceContactConstraint3D
augmented_lagrange=True`, `DynamicSolver3D.solve_step_augmented()`,
`ContactPair(constraint_enforcement="AUGMENTED_LAGRANGE")`): standard
Uzawa-style outer loop, `p = max(0, lam + k*(-gap))` frozen during inner
Newton iterations, `lam` updated to the converged `p` between outer
cycles.

**Result: reproducibly DIVERGES, root cause not found.** Measured on the
simplest possible test (single C3D8 element, uniform 4-node contact, no
multi-node active-set complexity at all): `max_penetration` grows
GEOMETRICALLY across outer cycles, ratio ~1.422 essentially exactly
repeated for 7 consecutive cycles (0.138 -> 0.200 -> 0.285 -> 0.405 ->
0.577 -> 0.820 -> 1.166), not a decaying overshoot. Swept the
under-relaxation factor omega in {0.1, 0.3, 0.5, 1.0} -- ALL diverge
(just at different rates), ruling out "the augmentation step is simply
too aggressive" as the explanation. A from-first-principles 1D spring
re-derivation of the expected fixed point for this exact test's
parameters (effective axial stiffness in series with the contact
penalty) predicts the OPPOSITE, stable behavior: increasing `lam` should
monotonically DECREASE equilibrium penetration toward zero. The
consistent, exact geometric ratio across cycles strongly suggests a
clean linear-recursion sign or bookkeeping defect in
`update_augmented_multipliers()`'s interaction with `assemble()`'s
augmented branch, not a physics/tuning subtlety -- but the specific bug
was not found despite this derivation and the relaxation sweep.

**Also confirmed NOT a regression**: `augmented_lagrange` defaults False
on every constraint (opt-in only), and `tests/test_augmented_lagrangian_contact.py::test_plain_penalty_contact_still_unaffected`
plus the full `tests/test_contact_phase1.py` suite confirm plain-penalty
Phase 1 contact is completely unaffected by this scaffolding.

**Left in the codebase, marked clearly broken** (both in-code docstrings
and `tests/test_augmented_lagrangian_contact.py`'s honest xfail), per
this project's characterize-honestly-don't-hide discipline -- not
reverted, since the API shape (CAE `constraint_enforcement=
"AUGMENTED_LAGRANGE"`, the per-node `_lam` state, the
`solve_step_augmented()` outer-loop structure) is still believed sound
and matches the design doc's own description; only the multiplier
UPDATE itself has an unidentified defect. **Do not build further Phase 2
work (deformable-vs-deformable) on top of this until the divergence is
actually root-caused and fixed** -- it would compound an already-broken
foundation.

## Where this is tracked

- `benchmark_element/benchmark_3d_contact.py` — the benchmark itself
  (formula + graded mesh + FEM comparison), usable as-is for re-running
  once Phase 2 lands.
- `tests/test_hertz_contact_benchmark.py` — regression test, honestly
  marked xfail with this exact finding (not force-passed).
- `dev_log/3d_contact_implementation_design_20260913.md` — Phase 2's
  acceptance criteria section updated to reference this benchmark as its
  concrete re-run target.
