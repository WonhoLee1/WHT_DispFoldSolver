# Deformable-vs-deformable: nonlinear penalty / soft laws / AL wiring (2026-09-16)

Priority item #2 of `dev_log/contact_development_report_20260915.md`'s
"다음 단계" list ("변형체-변형체에 nonlinear penalty/soft contact 연결"),
picked up after Stage A surface-to-surface contact
(`dev_log/contact_surface_to_surface_implementation_20260916.md`) per
user instruction "우선순위 2번 진행".

## Finding: this was a resolver-level restriction, not a missing capability

`DeformableSurfaceContactConstraint3D.assemble()` already routed every
non-AL evaluation through the same generic `self.law.evaluate(penetration)
-> (f_mag, k_diag)` seam `SurfaceContactConstraint3D` uses -- its own
module docstring already (accurately) claimed "HardLaw,
NonlinearPenaltyLaw, and the three soft laws all plug in unchanged."
What actually blocked this was `ContactPair.build_runtime_constraint()`
in `dispsolver/model/interaction.py`, which raised `NotImplementedError`
for ANY non-HARD/LINEAR combination whenever `master` was not an
`AnalyticalRigidSurface` -- confirmed via `git log -S` that this
restriction's own commit comment said "the law-resolution logic below
is already generic... extending this is a smaller follow-on than the
initial mechanism, not a rewrite" -- i.e. explicit, stated caution, not
a report of a known defect. Relaxing it was the fix; verifying it
actually produces correct results (not just "no longer raises") was the
real work.

## Changes

- `dispsolver/model/interaction.py`: removed the
  `normal_behavior != "HARD" or penalty_form != "LINEAR"` blanket
  `NotImplementedError` for non-rigid masters. `constraint_enforcement`
  restriction relaxed from "PENALTY only" to "PENALTY or
  AUGMENTED_LAGRANGE" (the pre-existing, separate "AL only applies to
  HARD contact" check a few lines below already covers the
  AL+soft-law case correctly, unconditionally of master rigidity --
  confirmed by test, not just re-used on faith). `augmented_lagrange`
  is now actually passed through to the
  `DeformableSurfaceContactConstraint3D(...)` constructor call (it was
  silently dropped before, even though nothing else blocked AL from
  reaching that class -- a second, adjacent gap found while fixing the
  first).
- `dispsolver/constraint3d/surface_contact3d_deformable.py`:
  - Added the same `augmented_lagrange=True` + non-`HardLaw`
    `ValueError` guard `SurfaceContactConstraint3D.__init__` already
    has. Without it, `assemble()`'s `elif self.augmented_lagrange`
    branch always wins over `self.law.evaluate(...)`, so a caller
    combining AL with a soft/nonlinear law would silently get plain-
    linear AL behavior with the law object completely ignored --
    exactly AGENTS.md's `sec4.8`/`sec4.16` silent-wrong-answer defect
    class. Refused explicitly instead, mirroring the sibling class.
  - Fixed a stale module docstring line ("Augmented Lagrangian is NOT
    supported here yet") that was already false as of this session's
    earlier PDASS work (`dev_log/contact_pdass_precise_design_
    20260915.md` sec3 added real `augmented_lagrange` support to this
    class directly, and section 6 of `dev_log/contact_development_
    report_20260915.md` already recorded it as verified -- only the
    CAE resolver above had never been updated to expose it).

## Verification (`tests/test_deformable_soft_nonlinear_contact.py`, 8 tests, all passing)

Built a second deformable-vs-deformable two-cube fixture (mirrors
`test_augmented_lagrangian_contact.py`'s rigid-plane one, but with the
master resolved as a genuine deformable cube's top face via
`ContactPair(master=[(n1,n2,n3,n4)], ...)`'s raw-face-tuple-list
acceptance path):

1. `test_hard_linear_still_works_unaffected` -- the original path is
   bit-for-bit unaffected by the relaxation.
2. `test_nonlinear_penalty_no_longer_raises_and_converges` -- confirms
   the resolved `law` is a real `NonlinearPenaltyLaw` (not a silent
   HardLaw fallback) and the solve converges.
3. `test_soft_laws_no_longer_raise_and_converge` (parametrized
   SOFT_LINEAR/SOFT_EXPONENTIAL/SOFT_TABULAR) -- same check for all
   three soft laws.
4. `test_augmented_lagrangian_now_wired_for_deformable_hard` -- AL+HARD
   now builds and converges (penetration collapses to the
   floating-point floor immediately, PDASS-in-solve_step() already
   doing most of the work automatically, same pattern already observed
   for `SurfaceToSurfaceContactConstraint3D`).
5. `test_augmented_lagrangian_with_soft_law_still_rejected` -- confirms
   the Abaqus AL-only-HARD restriction still correctly fires for a
   deformable master (this was never meant to be relaxed).
6. `test_class_level_guard_rejects_al_with_nonhard_law` -- the new
   class-level `ValueError` guard fires directly (not only through the
   CAE path).

Full contact regression family (`test_contact_chatter_stabilization.py`,
`test_augmented_lagrangian_contact.py`, `test_contact_phase1.py`,
`test_surface_to_surface_contact.py`, this file): **31 passed**, zero
regressions.

## What is still open

- `interaction.py`'s deformable branch still only supports the
  point-collocation `DeformableSurfaceContactConstraint3D` -- routing to
  `SurfaceToSurfaceContactConstraint3D` (a `discretization` selector,
  per `dev_log/contact_surface_to_surface_implementation_20260916.md`'s
  own "what is deliberately NOT done" section) is a separate, unstarted
  task.
- `SurfaceToSurfaceContactConstraint3D` itself is still restricted to
  `HardLaw` only (its own `__init__` guard) -- extending it to
  nonlinear/soft laws was not requested as part of this priority item
  and was not attempted here.
