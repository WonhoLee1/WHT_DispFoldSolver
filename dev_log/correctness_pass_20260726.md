# Correctness-first backlog pass — 2026-07-26

Closes the plan in `witty-juggling-moore.md`. Summary of what changed
and why, for future reference (the plan file itself is ephemeral).

## 1. False-positive divergence abort (the real bug)

`res_norm_0 = res_norm + 1e-16` anchored the divergence-check reference
on ~1e-16 whenever the first Newton iteration's residual was
legitimately ~0 (a purely displacement-driven step: load enters via BC
row elimination, so nothing is out of balance on iteration 1). The next
iteration's genuine residual then divided out past the `1e15` divergence
threshold and a healthy, converging step was aborted as "diverged."

This was originally hypothesized (previous plan draft) to be a
`Q4_UP`/mixed-material-batch-assembly defect, because the only failing
reproducer used `element_type={0:"Q4",1:"Q4_UP"}` in a multi-pid
config. Reproducing with the new reason-tagging (below) showed the
actual reason was `"divergence"`, not anything `Q4_UP`-specific — it's a
general false-positive for **any** model whose first iteration has
near-zero residual, which is why it also silently explained all 5
`test_solver.py` failures (single-element stretch tests, same pattern).

Fix: leave `res_norm_0` unset (and `res_ratio` a placeholder `1.0`)
until a residual with real magnitude appears; guard the stall-detection
counter the same way so an unset reference can't be counted as
"no progress" either.

## 2. Solver return-code overload (the enabler)

`solve_step` had 7 distinct failure sites — element inversion, 2x
NaN/Inf, line-search collapse, divergence, stall, max-iter exhaustion —
all collapsing to the same `-(max_iter)` or `-(n_iter+1)`. A failing
test reporting `-40` gave no way to tell which fired; this is what let
bug 1 hide as an apparent `Q4_UP` defect instead of the actual general
issue. Added `self.last_failure_reason` + `_fail(reason, code=None)`,
surfaced in the `.sta` cutback line. Integer contract unchanged
(additive only).

## 3. `test_rbe2.py` — stale test, not a bug

3 failures asserted `len(result.constraints) == 1` /
`isinstance(c, RBE2HingeConstraint)`. `*RIGID BODY` deliberately
condenses to `result.rbe2_constraints` / `KinematicRBE2Constraint`
(exact kinematic condensation, see AGENTS.md §4.10) — `result.constraints`
is now only for `*MPC`. Retargeted the three asserts.

## Result

`pytest tests/`: **135 passed / 10 failed -> 145 passed / 0 failed / 1
xfailed.**

## Verification (gating, not skipped)

- `verification.run_all`: 9/9 PASS (re-run after the fix, since it
  touches shared convergence logic broadly and AGENTS.md §4.7 already
  records one prior convergence-logic change backfiring).
- Full `ex12_abaqus_inp_plate_fold.py` 101-step run: still reaches
  90°/side, **81.16 µm tip staircase, PSA carrying 99.5%** (identical
  to pre-fix), 207.6s (no regression from the ~204s post-perf-fix
  baseline).
- **Warped-element question formally closed via this same run**: the
  `.sta` line's SEVERE DISCON column is literally
  `self._check_mesh_quality(self.u).get('n_inverted', 0)`
  (`dynamic.py:1306`) evaluated every converged step — parsed all 101
  rows, **max 0, zero nonzero entries**. This is the solver's own
  `det(J)` metric (not the offline area-ratio proxy used earlier), so
  the earlier "44 elements warped" notice from the pre-fix Arruda-Boyce
  run is confirmed gone under the current code, not just approximated.

## Also closed (cheap items)

- `AGENTS.md` §1.4: was stale ("multi-material layup not implemented"),
  contradicted by the 14-layer PET-PSA stack that's existed for several
  commits. Updated to document it while keeping ex03/ex11 accurate as
  single-material.
- `AGENTS.md` §3.1: before/after PNG capture scope settled by the user
  — `ex11`/`ex12`/`ex13` family only, not a repo-wide sweep.
- `.sta` status line no longer pollutes `verification.run_all` output
  (`solver.sta_status = False` in `verification/element_backends.py`) —
  a side effect of adding that status line earlier this session.
- `verification/` itself was never tracked in git despite being
  referenced as mandatory throughout AGENTS.md §3 — committed.

## Deliberately not done this round (unchanged from the plan)

- λ_m = 3.0 assumed for PSA Arruda-Boyce — blocked on real material
  data.
- Sequential-fallback slowness for standalone Yeoh/ArrudaBoyce — no
  numpy batch primitives to compose, not hit by any current model.
- Qt viewer file loading — needs per-element state/stress recorded as
  `cell_data` in the solve loop first; not done.
