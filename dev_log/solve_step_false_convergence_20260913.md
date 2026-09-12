# [CRITICAL] `DynamicSolver3D.solve_step()` reports `converged=True` even when Newton never converged

**2026-09-13, found while independently verifying a `benchmark_element/` fix
for `C3D8_FBAR`.**

## The bug

`dispsolver/solver3d/dynamic3d.py`, `solve_step()` (around line 440-523):

```python
for iter_count in range(1, max_iters + 1):
    ...
    if iter_count > 1 and bc_err < 1e-5 and (rel_r < 5e-3 or r_norm < 1e-3):
        ...
        return True, iter_count
    ...
    if iter_count > 1 and du_norm < 1e-4 and bc_err_next < 1e-5:
        ...
        return True, iter_count
    ... # line search, u_k = best_u
self.u = u_k
return True, max_iters   # <-- unconditional success
```

If the loop runs all `max_iters` iterations without either convergence
branch firing, execution falls off the bottom of the `for` loop and hits
`return True, max_iters` **regardless of how large the residual still
is**. There is no `else: return False, max_iters`. This is a stronger,
more literal version of AGENTS.md §4.9's "convergence success is not
proof of physical correctness" — here, "convergence success" isn't even
evidence Newton did anything at all in the last iteration; it's the
function's unconditional exit value.

## How this was found

A `benchmark_element/` fix agent working on `C3D8_FBAR` (see
`dev_log/benchmark_element_folder_setup_20260913.md` and the fixes
described below) reported that `run_volumetric_locking_test('C3D8_FBAR',
nu_val=0.49999)` returns `verdict='VOL-LOCKED'` at exactly `iters=25` (the
`max_iters` cap), and that re-solving the identical kernel with a
hand-rolled tight-tolerance Newton loop converges the same problem to
`6.46e-16` residual in 14 iterations with a very different (non-locked)
answer. Verified independently in this session:

```
C3D8_FBAR vol : w_tip=0.000838, ratio_to_ref=0.0194, verdict=VOL-LOCKED, iters=25
C3D8I     vol : w_tip=0.0158,   ratio_to_ref=0.366,  verdict=STIFFENED,  iters=25
```

Both hit the `max_iters=25` cap exactly. Per the code above, both of
these are `solve_step` claiming success without having actually reached
a converged state — the reported "VOL-LOCKED"/"STIFFENED" verdicts for
these two cases are unverified, not necessarily real physics.

## Why this isn't a quick fix to just land here

`dynamic3d.py` has a very large uncommitted diff right now (447
insertions / 145 deletions vs. `HEAD` as of this writing) from what
appears to be a different, concurrently-active session's ongoing work —
same pattern as `dispsolver/element3d/c3d4_anp_numba.py`, which was
independently found mid-edit and briefly broken (`NameError:
'self_weight' is not defined`) during this same investigation. Editing
`solve_step()` right now risks colliding with that in-flight work.
**Deliberately not touched in this session** — flagging here instead so
whichever session lands next on `dynamic3d.py` sees it before assuming
the current 3D benchmark verdicts are reliable.

## Recommended fix (for whoever picks this up)

Add an explicit `return False, max_iters` (or raise, matching this
project's established "don't silently succeed" convention — see AGENTS.md
§4.8's tie/RBE2 lesson and §4.9's `sanity_report()` philosophy) after the
loop, replacing the current fallthrough. Once fixed:

1. Re-run every `benchmark_element/benchmark_3d_mechanics.py` test whose
   `iters` field is at or very near `max_iters` (25) — at minimum
   `C3D8_FBAR`'s and `C3D8I`'s volumetric-locking results above — since
   those verdicts may change once real non-convergence is reported as
   `DIVERGED` instead of a fake `PASS`.
2. Check whether raising `max_iters` (rather than accepting `DIVERGED`)
   resolves some of these — a near-incompressible (ν=0.49999) F-bar/EAS
   problem converging slowly is plausible and not necessarily a separate
   element defect; the point is `solve_step` must actually report which
   case it is instead of masking the difference.
3. This may also explain some of the "VOL-LOCKED" verdicts currently
   attributed to genuine element locking in
   `dev_log/benchmark_3d_mechanics_20260913.md` for elements whose
   `iters` field is at or near the cap — re-check the report's raw
   `iters` values, not just the pass/fail verdict, before trusting any
   single one of them as final.

## What was fixed instead, in the same investigation (real, verified kernel bugs)

Not related to the above, but found by the same set of parallel
`benchmark_element/` fix agents this session, each independently
verified against the real `benchmark_element/benchmark_3d_mechanics.py`
functions before being trusted (per this project's established
"don't trust an agent's self-report" discipline):

- **`C3D10` / `C3D10M`**: root cause was in `benchmark_element/mechanics_patches.py`'s
  own mesh generator, not the element kernel — the "interior" nodes it
  labeled for the patch test were geometrically on the cube's own outer
  faces (face diagonals of a 5-tet cube split), so they were never
  actually interior and never got a Dirichlet BC, producing a spurious
  equilibrium mismatch matching the observed ~1e-4 error scale exactly.
  Fixed by rebuilding the mesh on the same 3x3x3-grid topology already
  proven correct for `C3D4`. Official Abaqus patch test:
  `C3D10` 6.7e-4 FAIL -> 3.16e-19 PASS, `C3D10M` 6.7e-4 FAIL -> 4.06e-19
  PASS. Independently re-verified in this session.
- **`C3D8I`**: the 9-mode EAS enhancement was computed but never added to
  `E_voigt` before the material call (`f_int`/stress were from the
  compatible strain alone -- bit-identical to plain `C3D8`, which is
  exactly why its pre-fix bending ratio 0.46 matched `C3D8`'s 0.46
  exactly). Fixed with a per-call local Newton loop condensing the 9
  alpha DOFs before assembly. Official Abaqus patch test: 1.8e-6 FAIL ->
  1.76e-11 PASS; bending ratio: 0.46 LOCKED -> 0.959 LOCKING-FREE.
  Independently re-verified in this session. **Known remaining gap,
  correctly flagged by the fix agent, not fixed**: only the 9 principal
  alpha_i modes are implemented; real `C3D8I` has 4 additional beta_j
  volumetric-relief modes (13 total internal DOFs per
  `benchmark_element/reference_abaqus_docs/incompatible.txt`) needed for
  volumetric-locking relief specifically -- the post-fix volumetric
  verdict (STIFFENED, ratio 0.366) may partly be the missing beta_j modes
  and partly the `solve_step` bug above (this test also hit `iters=25`,
  see above) -- don't assume either cause alone without re-testing after
  the `solve_step` fix.
- **`C3D8H`**: two bugs, both in `dispsolver/element3d/c3d8_hybrid_numba.py`.
  (1) The pressure-term modulus decode only handled two material-encoding
  conventions and silently misread `MAT_CUSTOM_ELASTIC`'s flattened 6x6
  matrix (what every `benchmark_element` call actually uses) as if it
  were `[E,nu]` or `[mu,K]` directly. (2) The deviatoric Gauss-point
  tangent used the raw, un-projected material tangent while the force
  used the correctly-projected deviatoric stress, double-counting bulk
  stiffness in the tangent only (not the residual) -- FD-Jacobian error
  39% -> 8.1e-5 after fixing. Official Abaqus patch test: 6.7e-6 FAIL ->
  2.2e-8 PASS. Bending stays LOCKED (~0.50) -- correctly identified by the
  fix agent as expected, not a defect: `C3D8H`'s real Abaqus counterpart
  cures volumetric locking only, not shear locking (confirmed against
  `reference_abaqus_docs/hybridincompress.txt` -- no shear-locking claim
  in the actual theory). Independently re-verified in this session.
- **`C3D8_FBAR`**: `B_L` (the force/tangent virtual-strain operator) was
  built from `F_bar` instead of the real `F` -- the F-bar method must use
  `F_bar` in the constitutive call only, with the virtual strain built
  from the real kinematics (de Souza Neto et al. 1996; same "which F does
  this operator use" class as this project's own §4.17 `Q4_VISCO_SIMO`
  fix). Fixed (24 call sites). This has **no effect on the patch test**
  (F_bar=F exactly for a homogeneous deformation, verified: 2.5357e-6 ->
  2.5328e-6, unchanged) but is the physically correct form for problems
  where F genuinely varies inside the element (bending, volumetric).
  **The patch-test/volumetric-locking FAILs for this element are the
  `solve_step` bug above, not this kernel bug** -- re-test once that's
  fixed before concluding anything further about `C3D8_FBAR` itself.
- **`C3D4_ANP`**: real defect confirmed (Bonet & Burton nodal-pressure
  averaging computed in Pass 1 but never actually used -- a `B_bar`
  construction that subtracted then immediately re-added the identical
  term, making it a no-op), but **not fixed**: the file was found to be
  under active concurrent edit by a different session mid-investigation
  (`dispsolver/element3d/c3d4_anp_numba.py`, currently broken with a
  `NameError: 'self_weight' is not defined` as of this writing -- verify
  it's fixed before relying on this element for anything). Left alone
  deliberately to avoid a write collision; whoever picks this up next
  should re-diagnose against whatever state that session lands, not
  resume from this session's aborted fix attempt.

All of the fixes above were independently re-run against the live
`benchmark_element/benchmark_3d_mechanics.py` functions in this session
before being written down here -- not taken on the fixing agent's word
alone, per this project's established verification discipline.

## Update, later the same day: the `solve_step` bug fixed, `C3D4_ANP` landed

**`solve_step()` fixed.** Both occurrences of the unconditional
`return True, max_iters` fallthrough in `dispsolver/solver3d/dynamic3d.py`
(`solve_step()` and the sensor/hooks variant, `_solve_step_with_hooks()`)
now re-check the loop's own convergence criteria one last time and return
`False` if they aren't actually met, instead of silently claiming success.
Effect, measured directly: `run_volumetric_locking_test('C3D8_FBAR',
nu_val=0.49999)`, which previously (falsely) reported `VOL-LOCKED` at
`iters=25` (the cap), now correctly reports `DIVERGED` -- genuine
non-convergence, not real locking data. `C3D8H`'s volumetric result was
unaffected by this fix (`iters=24`, i.e. it was already a real, honestly-
earned convergence) -- confirms the fix doesn't just break everything
that happens to run long.

**`C3D8_FBAR` patch-test failure re-diagnosed, NOT the `solve_step` bug.**
Directly tested with `max_iters=80`: the patch test still stops at
`iter=19` with the IDENTICAL `max_error=2.5327970679902922e-06` as with
`max_iters=25` -- it satisfies the loop's own loose accept criterion
(`rel_r < 5e-3`) well before exhausting any iteration budget, so this is
not the false-convergence bug. Compare `C3D8`/`C3D8I` (after this
session's EAS fix), which reach `1e-11`..`1e-19` error under the exact
same accept criterion -- they converge quadratically and blow straight
through the loose bar to near-exact; `C3D8_FBAR` needs 19 iterations to
even reach the loose bar and then sits there, ~2.5e-6 from exact. That
signature (slow/near-linear convergence, not a bad residual) points at an
incomplete/inconsistent TANGENT, not the fixed `B_L` residual bug --
matches `dev_log/3d_element_defect_audit_20260912.md` §2.5's original
"F-bar C3D8 omits its volumetric-consistency tangent term" finding, which
this update's earlier section had provisionally (and incorrectly)
attributed entirely to the `solve_step` bug. **Dispatched to a follow-up
fork agent to implement the missing tangent term** (de Souza Neto et al.
1996 F-bar tangent, the extra term from `Jbar` depending on every node in
the element, not just the local Gauss point) -- still in progress as of
this writing, see the session transcript for its outcome, not repeated
here to avoid this file going stale the way `AGENTS.md §4.13` did.

**`C3D4_ANP` fixed and independently verified** (the concurrent-edit
collision noted above turned out to be this same session's own earlier
fork agent, not a different session -- `ListAgents` confirmed no other
agent was live on this file by the time it was revisited). Root cause was
exactly as diagnosed: `c_e = self_weight` was an incomplete edit (a
placeholder name, never assigned) left mid-implementation of the
self-consistent single-operator fix direction. Completed it: `c_e =
J_bar_e / detF` (this element's own dilatation relative to the
volume-weighted nodal average from Pass 1) -- `c_e = 1` exactly when
this element's own `detF` already matches its neighbors' average
(homogeneous/patch-test case, and the isolated-element case), pulling
toward the nodal average otherwise. Verified genuinely non-trivial: fed
random nodal displacement noise directly into
`compute_nodal_dilatations_pass1`, confirmed `c_e` values meaningfully
different from 1 (0.968-1.049 range) across neighboring elements -- this
is NOT another no-op.

Before -> after: official Abaqus patch test, `C3D4_ANP`:
`numba.TypingError` (crash) -> `PASS, 2.3e-8`. Plain `C3D4` unaffected
(`PASS, 1.2e-19`), confirming no regression to the untouched control.

## Update, later still: `C3D8_FBAR`'s xfail reason was itself wrong -- narrowed with a nu sweep

**The "genuine near-incompressible conditioning (K/G~1e5)" reason given
immediately above was an overstatement, corrected after a direct nu
sweep at the user's request.**

`run_volumetric_locking_test('C3D8_FBAR', nu_val=X)` for
`X = 0.30, 0.40, 0.45, 0.48, 0.49, 0.495, 0.499`: **all PASS**, `iters`
3->28 (smoothly increasing, as expected), `ratio_to_ref` stable in
`0.554-0.565` throughout. This element works correctly across the entire
practically-used near-incompressible range -- there is no general
conditioning problem here.

The actual failure is a **narrow band**: `nu` in roughly `[0.4999,
0.49992]` diverges even at `max_iters=300` (confirmed directly, not
inferred), while both `0.49985` (converges, 28 iters) and `0.49995`
(converges, 47 iters) on either side of that band are fine. Diagnosed
the initial tangent's conditioning at a few points:

```
nu=0.495    condition number ~1.1e6
nu=0.499    condition number ~5.2e6
nu=0.4999   condition number ~5.2e7
nu=0.49995  condition number ~1.0e8
```

Condition number does grow steadily (expected, `K_bulk` diverges as
`nu->0.5`) but doesn't itself explain a narrow non-convergence island --
a smoothly growing condition number should mean smoothly harder (not
"fails only in this specific window") convergence. Instrumenting the
Newton loop directly at `nu=0.4999` showed the residual norm genuinely
**oscillating and occasionally increasing** step-to-step (not just slow
monotonic decrease), meaning the Newton *direction* itself is sometimes
bad there, not just poorly scaled -- backtracking line search (which
only shortens the same direction) cannot fix a bad direction.

**More importantly**: at the benchmark's own default test point,
`nu=0.49999` (just past the bad band), pushing `max_iters` up to 250 does
make `solve_step` report `converged=True` at `iters=222` -- but the
resulting answer is **wrong**, `ratio_to_ref = -16.1` (sign-flipped and
16x too large in magnitude), not a slowly-converging correct one. This is
a real, separate finding: at this exact extreme, Newton is landing on a
spurious/unphysical equilibrium point that happens to satisfy
`solve_step`'s loose relative-residual acceptance criterion. Simply
raising `max_iters` is not a fix and must not be treated as one.

**Not chased further, by design**: per the user's explicit direction,
`nu` beyond `0.499` is understood to be far outside what's practically
used for this project, so this narrow, extreme-limit instability is
recorded precisely (band location, condition numbers, the spurious-
convergence finding) rather than investigated to a root cause that would
cost more effort than its practical relevance justifies right now. If
ever revisited: the FD-tangent's per-column adaptive step
(`c3d8_fbar_tl_numba.py`, Dennis & Schnabel convention, already correct
for the mesh-refinement axis) is not the suspected cause here (verified
present and correctly implemented) -- the more likely next thing to check
is catastrophic cancellation inside the residual/stress evaluation itself
at extreme bulk modulus (`K ~ E/(3(1-2*nu))` reaching `~1e8-1e9` for
`E=2e5`), not the FD step size, or a genuine non-uniqueness/bifurcation
in the F-bar formulation's equilibrium near this specific loading and
mesh combination.

**Honest remaining gap, not a fake pass**: on the actual cantilever
bending / volumetric-locking benchmark (structured tet mesh, smooth
bending load, nu=0.49999), `C3D4_ANP`'s result is STILL bit-identical to
plain `C3D4` (`ratio_to_ref=0.0605` both). Instrumented directly: for the
real converged solution of this specific problem, every element's
`detF` differs from its 4-node nodal average by at most `~8.7e-8` across
the WHOLE mesh -- i.e. `c_e ≈ 1` everywhere in practice for this
particular smooth loading, so nodal averaging has nothing to correct.
This is a property of the benchmark's geometry/loading (a structured
mesh under smooth bending doesn't develop the element-to-element
pressure checkerboard Bonet-Burton ANP specifically targets), not
necessarily evidence the fix is incomplete -- but it also hasn't been
proven complete: a proper Bonet-Burton verification would look at
per-element PRESSURE variance directly (as the original paper does)
rather than tip deflection, or use an unstructured/irregular mesh where
checkerboarding is known to appear. Left as an open, explicitly-flagged
item (`tests/test_3d_mechanics_benchmarks.py`'s `_VOL_XFAIL["C3D4_ANP"]`)
rather than claimed fixed.
