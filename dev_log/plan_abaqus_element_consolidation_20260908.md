> **CORRECTION (2026-09-08, end of session -- read this first)**: the
> user pushed back on everything below section 0 as still being
> "maintain/patch the existing historical architecture" rather than what
> was actually asked: a genuine REDEVELOPMENT targeting the verified
> Abaqus/OptiStruct element framework as the spec, not this codebase's
> own ad hoc inventions. The clearest concrete instance: this document's
> "Family B: do not attempt, keep SRI as a permanent exception forever"
> is exactly the wrong shape of answer. `Q4_COROTATIONAL_SRI`'s
> shear-only-at-centroid device is **not a real Abaqus element** (see
> `q4_visco_eas_jax.py`'s own docstring: "Abaqus doesn't ship CPE4S") --
> it is this project's own invention, kept alive because it happened to
> work once co-rotational framing was bolted on. A true redevelopment
> means PET/GLASS should move to a REAL Abaqus element formulation
> (most likely `CPE4I`, the same incompatible-modes approach already
> built, verified, and proven for PSA this session -- not a from-scratch
> "faithful CPE4R" reimplementation, since AGENTS.md's own dev_log
> already found real 1-point-reduced+hourglass fails for a
> one-element-through-thickness layer, a genuine physical limitation
> Abaqus's own manual documents too, not an implementation shortcut of
> ours) -- not kept on its own invented device with a permanent
> co-rotational crutch just because F3 showed the crutch is load-bearing
> for THAT device. F3 is still a true, measured fact about the CURRENT
> device; it is not a reason to keep the device. **Next session: replan
> Family B as "migrate PET/GLASS off the invented SRI device onto a real
> Abaqus element (CPE4I first candidate), verified the same way CPE4RH
> was this session (patch test + objectivity + JAX/Numba agreement),"
> not as "leave SRI alone forever."** Everything in section 0's F1/F2/F4/F5
> findings and the T1-T5 verification methodology remains factually valid
> and load-bearing for that redevelopment -- only the Family B
> *recommendation* (section 2) needs to be re-derived under the
> redevelopment framing before work resumes.
>
> **Follow-up, same close-out (verbatim principle, do not soften this
> next session): "영구 예외 없음. 전면 재검토할 것임. abaqus 요소로 1:1
> 개발할 것임." (No permanent exceptions. Full comprehensive re-review.
> Every element developed 1:1 as a real Abaqus element.)** This is
> broader than just fixing Family B -- it means EVERY element currently
> in `dispsolver/element/` (not only the ones opus's review happened to
> touch) needs to be re-audited against a genuine Abaqus 1:1 counterpart:
> does a real Abaqus element of that formulation exist (CPE4/CPE4R/
> CPE4H/CPE4I/CPE4IH/CPE4RH), and does this codebase's implementation
> actually match that element's real integration/kinematics scheme --
> not "is it close enough and does it happen to work for our current
> mesh." Anything that is this project's own invented device with no
> real Abaqus counterpart (the SRI shear-only-at-centroid trick is the
> first confirmed instance, per its own docstring admission "Abaqus
> doesn't ship CPE4S" -- there may be others, e.g. the F-bar-based
> `Q4_VISCO_SIMO`/`Q4_UP` mean-dilatation device also has no distinct
> Abaqus element name) is a redevelopment target, not a thing to wrap a
> permanent exception around and move on from. Start next session's
> planning from a full inventory pass (which files/element-type strings
> have zero real Abaqus counterpart today) before re-deriving section 2's
> per-family recommendations -- do not just patch Family B and declare
> the review satisfied.

# Plan: Abaqus-standard element/material unification (2026-09-08, revised after opus Phase B review)

**Decision history**: scoped first as "naming only" (aliases added), then
escalated by explicit user request to full unification following real
Abaqus/OptiStruct architecture (element = kinematics, material =
constitutive, generic interface -- see
`C:\Users\GOODMAN\.claude\projects\...\memory\project_element_dev_standard_abaqus_optistruct.md`).
Phase A (material response interface) was implemented and verified with
zero behavior change. Before starting Phase B (retire co-rotational frame
extraction, migrate J2 elements to UL), an opus architecture review was
commissioned given the risk to the validated 90 deg/side production
result. **The review overturned the plan's premise with measured,
reproducible findings** -- this document is rewritten around those
findings rather than the original phase order.

## 0. What the opus review actually found (authoritative -- read this before anything else in this file)

Five findings, all backed by throwaway numerical probes (not just
argument), listed by the reviewing agent as F1-F5. Confidence: F1-F4
high (measured, two of them additionally provable analytically), F5
moderate-high (read from code, not reproduced live -- 10 minutes of
confirmation via T3 below closes this).

- **F1 -- plain `Q4_COROTATIONAL` is already, provably, exactly a
  Total-Lagrangian element.** `u_local = coords_curr @ R - coords` gives
  `F_local = Rᵀ F` exactly; PK2 is invariant under superposed rotation;
  `f(Rᵀ F) = Rᵀ f(F)`, which `T8 @ f_local` undoes exactly. Measured:
  force differs from a plain TL evaluation by ~1e-13 relative, tangent by
  ~1e-9 relative, at a 1.1 rad rotation + random distortion. **The frame
  extraction in `q4_corotational_jax.py` is dead weight -- a no-op.**
- **F2 -- finite-strain multiplicative J2 plasticity here is already
  exactly rotation-equivariant, INCLUDING its history state.** `F -> QF`
  implies `b_e_tr -> Q b_e_tr Qᵀ` (same eigenvalues -> same `eps_log`,
  `q_tr`, `dgamma`, `eqps`), and `F_p_inv_new = F⁻¹ F_e_new` is literally
  unchanged under `F -> QF`. **No state migration is needed for any of
  this work, in either direction.**
- **F3 -- `Q4_COROTATIONAL_SRI` (and `_HYBRID_SRI`) is the opposite
  case: the co-rotational frame is LOAD-BEARING, not dead weight.** SRI's
  mixed sampling (normal strain at Gauss points, shear at the centroid)
  is NOT a frame-equivariant operation for an inhomogeneous field.
  Measured objectivity error (strain-then-rotate, real free-span AR
  1.0x0.15): co-rotational SRI is objective to ~1e-14 at every angle;
  the SAME kinematics evaluated in a fixed (TL/UL) frame gives **4.4%
  error at 5.7 deg, growing to 38% at 90 deg**. Removing the frame from
  this family breaks it outright. This is PET + GLASS -- most of the
  production mesh.
- **F4 -- CRITICAL, currently shipping: the existing UL viscoelastic
  implementation is missing a stress push-forward (work-conjugacy
  error).** `_internal_force` (`q4_visco_simo_fs_jax.py:211`) computes
  `S_v = _simo_pk2(F_total)` (PK2 referred to the ORIGINAL config) and
  contracts it with `BL(F_inc)`/`detJ` built on the STEP-N config. The
  correct conjugate stress is `S_n = F_n S₀ F_nᵀ / det(F_n)`; that
  push-forward is absent. Measured on a homogeneous deformation (so
  F-bar and quadrature effects are identically zero, isolating the
  conjugacy gap): **24.9% relative force error** at `F_n =
  diag(1.20, 0.90)` + a 3% shear increment; **7.2%** at `F_n` = a 1.2 rad
  rotation + 5% stretch (first-order in stretch, not second-order).
  Newton still converges (K is autodiffed from f, so the linearization
  is self-consistent) -- **to a slightly wrong equilibrium**, invisible
  to any shape-level pass/fail gate. This affects every element sharing
  this code path: `Q4_VISCO_SIMO`, `Q4_UP`, `CPE4I`, `CPE4H`, `CPE4RH`
  under UL.
- **F5 -- `_ul_F_n` compounds within a Newton step (masked today, would
  be active for the co-rotational family).** `dynamic.py` writes
  `self._ul_F_n[elem_indices] = F_n_new` on EVERY assembly call (i.e.
  every Newton iteration, several call sites: ~3523, 3587, 3651, 3751,
  3926, 3964), while `_ul_u_ref` only advances on convergence. Within one
  step: iteration 2 evaluates `F_inc @ (F_inc @ F_n_conv)` (squared),
  iteration 3 cubes it. `save_state`/`restore_state` exist but are never
  called from inside `_solve_step_impl` (grep-confirmed no call sites).
  Bounded today by the commit-time resync (`dynamic.py:2625-2632`, which
  recomputes `_ul_F_n` for all elements from `u_total` against the
  initial-config gradients) -- so the COMMITTED state is correct, this
  is a convergence-robustness defect, not silent state corruption. It is
  masked because the current production path converges in ~1 iteration
  per step; the co-rotational family runs 4-7 iterations per step
  (AGENTS.md §1.2), which would convert this into an active bug the
  moment that family is migrated onto the same UL machinery.

### Why the rigid-rotation canary used throughout this session cannot catch F4 or F5

Measured directly: at 90 deg PURE rigid rotation with ZERO strain,
`|f|` is 3.4e-12 for the (objective) co-rotational SRI kernel and
**3.7e-13 for the (non-objective) fixed-frame SRI evaluation** -- the
WRONG formulation passes MORE cleanly than the correct one, because a
homogeneous deformation makes Gauss-point and centroid gradients
coincide, making SRI's mixed sampling exactly inert in that one test.
The canary this session relied on (including for CPE4RH, CPE4I, and
every corotational-family check) is **necessary but not sufficient** --
keep it, but its passing must never again be read as evidence of
large-rotation correctness on its own.

## 1. Corrected verification methodology (T1-T5) -- add these as permanent tests

All five are single-element, sub-minute, and each names the specific
defect it catches. **T1 and T2 do not exist in this codebase today and
are the two that would have caught F3/F4 before they shipped.**

- **T1 -- strain-then-rotate objectivity** (catches F3): apply a fixed
  small inhomogeneous nodal perturbation (~0.4% strain), THEN rigidly
  rotate the whole configuration by theta in {5.7, 28.6, 57.3, 90} deg.
  Assert `||f(theta) - T8(theta)@f(0)|| / ||f(0)|| < 1e-12`. Run per
  element type, per material, at the real free-span aspect ratio. The
  existing zero-strain rotation canary is KEPT but DEMOTED to "necessary,
  not sufficient."
- **T2 -- UL == TL consistency** (catches F4): reach one total
  deformation two ways on a homogeneous field -- (a) TL: `u=u_total`,
  `F_n=I`; (b) UL: pre-deform coords by `F_n`, pass `u=u_inc`, supply
  `F_n`. Assert `max|f_TL - f_UL| / max|f_TL| < 1e-10`. **Fails today at
  24.9%.** The single most important test to add -- it fails on code
  that ships today.
- **T3 -- mid-step `F_n` idempotence** (catches F5): call `_assemble(u,
  dt)` twice with the same `u`, no `_ul_u_ref` advance between calls;
  assert bitwise-identical residuals. Cheaper element-level proxy:
  `F_n_new` for `u_inc=0` must equal the `F_n` passed in.
- **T4 -- plasticity patch test in a rotated reference** (confirms F2
  holds through any future UL migration of Family A): constant-stress
  patch driven past yield, evaluated with an already-rotated `F_n`.
  Must recover exact stress and match `eqps`/`F_p_inv` to ~1e-12 against
  the TL path -- F2 guarantees this passes; a failure means broken state
  plumbing, not broken theory. A clean diagnostic signal either way.
- **T5 -- small-increment UL-vs-corotational agreement**: real
  hinge-zone element, ~1 deg increment. Non-SRI variants MUST agree to
  ~1e-10 (F1 makes this mandatory). SRI variants will NOT agree -- that
  disagreement is the expected diagnostic result, not a tolerance to
  relax away.

Run T1-T3 on every relevant commit going forward. Only after T1-T3 pass
does a full ex12/ex13 production re-run become a meaningful additional
gate (not a substitute for T1-T3 -- a smooth, uninverted U-shape can and
does hide a 25% force error or a 38% orientation-dependent stiffness;
§1.0's shape-level success criteria alone would not have caught either).

## 2. Per-family verdict and recommended order of work

**Family C -- viscoelastic UL (`Q4_VISCO_SIMO`, `Q4_UP`, `CPE4I`,
`CPE4H`, `CPE4RH`): fix NOW. This is not a Phase B/consolidation
question -- it is a correctness bug in currently-shipping production
code**, independent of everything else in this plan.
- **C1**: push the stress forward before contracting --
  `S_n = F_n @ S0 @ F_n.T / det(F_n)` -- in `q4_visco_simo_fs_jax.py`'s
  `_internal_force` (line ~211) before it is used with `BL(F_inc)`. Port
  the same fix to every Numba sibling that shares this UL pattern
  (`q4_visco_hybrid_simo_numba.py`, `q4_visco_eas_numba.py`,
  `q4_visco_hybrid_up_numba.py`, `q4_visco_hybrid_reduced_numba.py` --
  audit each for the identical missing push-forward, do not assume only
  the JAX file has it).
- **C2**: write `_ul_F_n` only on Newton convergence, not on every
  assembly call, at every site in `dynamic.py` currently doing the
  latter (~3523, 3587, 3651, 3751, 3926, 3964) -- the commit-time resync
  (`dynamic.py:2625-2632`) already computes the authoritative value from
  `u_total`, so this is a matter of not overwriting it mid-step, not
  computing something new.
- Gate: T2, T3, then a full ex12/ex13 production re-run to confirm C1/C2
  change the QUANTITATIVE result (they should -- up to 25% force error
  is being removed) without breaking convergence or the qualitative
  U-shape/no-inversion criteria.

**Family A -- `Q4_COROTATIONAL`, `Q4_COROTATIONAL_EAS`, non-SRI
`_HYBRID`: proceed, but delete the frame outright rather than migrate to
UL.** F1 makes deletion a bitwise-verifiable no-op; F2 means no state
migration is needed either way. This achieves Phase B's original stated
goal (retire the co-rotational mechanism) for these elements WITHOUT
adopting UL's still-being-fixed machinery. A later switch to UL for
these specific elements is a legitimate follow-on, justified by
numerical conditioning at that point, not by any physics requirement --
do not treat it as required by this plan. Gate: bitwise comparison
against the current (frame-extracting) kernel at several states, plus T1
and T5.

**Family B -- `Q4_COROTATIONAL_SRI`, `Q4_COROTATIONAL_HYBRID_SRI`: DO
NOT ATTEMPT this migration.** F3 is the hard blocker: SRI's mixed
sampling is not frame-equivariant, so removing the co-rotational frame
removes the element's objectivity (38% error at 90 deg, measured). This
is PET + GLASS, the majority of the production mesh -- leave it exactly
as-is. Record the objectivity table (F3's numbers) in AGENTS.md so this
is not re-litigated in a future session that hasn't read this file. If
uniform UL treatment must eventually reach this family, that requires
rederiving SRI as a frame-invariant projection (e.g. a B-bar-style shear
formulation) -- a real formulation project, scoped and verified
separately, not something Phase B/C's naming collapse can absorb for
free.

### Binding order constraint

**C2 (the `_ul_F_n` per-iteration compounding fix) must land before
Family A's migration**, if Family A ever does move to UL as a follow-on.
Family A runs 4-7 Newton iterations per step (AGENTS.md §1.2) versus the
~1 iteration that currently masks F5 -- migrating first means debugging
F5 through a multi-minute production run instead of through a
10-minute T3 check.

## 3. Revised phase plan

- **Phase A -- material response interface: DONE, unaffected.**
  `dispsolver/material/response_interface.py`, verified bit-identical
  against existing functions at 12 states (J2 + viscoelastic), zero
  element kernel touched. Stands as completed -- orthogonal to every
  finding above.
- **Phase B0 (NEW, highest priority) -- Family C bug fixes (C1, C2)**.
  Not started. See section 2. Gate: T2, T3, production re-run.
- **Add T1, T2 as permanent tests** (alongside B0, before any further
  migration work) -- not started.
- **Phase B1 (was: "Phase B") -- Family A frame deletion to TL**. Not
  started. Gate: bitwise + T1 + T5. Must follow B0's C2 fix if this
  family later adopts UL as a follow-on (see binding order constraint).
- **Phase B2 -- Family B: explicitly SKIPPED, not migrated.** Carry this
  exclusion through Phase C/D's dispatch-simplification work (SRI stays
  a distinct, frame-extracting code path in `dynamic.py` and
  `model_builder.py`'s name mapping, permanently, not as a stopgap).
- **Phase C -- collapse element kinematics to one-per-Abaqus-name**:
  proceeds as originally planned, WITH Family B's SRI variants excluded
  from the collapse (they keep their own dispatch, not folded into a
  material-agnostic generic kinematics).
- **Phase D -- dynamic.py dispatch simplification**: proceeds as
  planned, same SRI exclusion carried through.
- **Phase E -- full production re-verification**: proceeds as planned,
  now also re-verifying that B0's C1/C2 fixes changed the quantitative
  (not just qualitative) production result as expected.

## 4. Honest scope note (updated)

The original plan underestimated the risk in exactly the way the opus
review's canary-adequacy critique warns about generically: the natural
first instinct (`git-stash`-style A/B, or a smooth production re-run)
would not have caught F3 or F4 -- both are real, and F4 in particular is
shipping in production right now. Treat T1/T2 as mandatory additions to
this repo's permanent test suite regardless of how the rest of this plan
proceeds, independent of user priority on the remaining phases.

## 5. Phase B0 (F4/F5 bug fixes) -- DONE, next session (2026-09-08 continued)

Per section 2/3's priority ("Family C: fix now, not a Phase B question"),
this was done FIRST, before touching anything else in this plan.

### F4 fix: work-conjugacy push-forward

Applied `S_n = F_n @ S @ F_n.T / det(F_n)` (push the material's PK2,
referred to the ORIGINAL config, forward to configuration n) before
contracting with `BL(F_inc)`/`w=detJ` (step-n quantities), to:
- `q4_visco_simo_fs_jax.py::_internal_force` (Q4_VISCO_SIMO/Q4_UP)
- `q4_visco_eas_jax.py::_residuals` (CPE4I) and `_residuals_h` (CPE4H/CPE4IH)
- `q4_visco_hybrid_reduced_jax.py::_reduced_hybrid_force` (CPE4RH -- this
  session's own new element had inherited the identical bug at
  construction time; its own patch tests all used `F_n=identity`, which
  makes the missing push-forward an exact no-op and invisible to them)
- The 3 Numba counterparts (`q4_visco_eas_numba.py`,
  `q4_visco_hybrid_up_numba.py`, `q4_visco_hybrid_reduced_numba.py`).
  `q4_visco_hybrid_simo_numba.py::_internal_force_numba` needed no fix --
  it has no `F_n` parameter at all; `dynamic.py`'s call site for it
  always passes the untouched reference `coords`/total `u` (unconditionally
  TL for this one Numba path, confirmed by reading the call site).

The EAS-mode residual `f_a` (both JAX and Numba, in the CPE4I/CPE4H
branches) deliberately keeps the un-pushed stress -- its own coupling
operator is already built from the TOTAL enhanced F, not the incremental
one, so it doesn't share this specific mismatch; revisit only if a future
audit finds otherwise.

Verified: `F_n=identity` (TL) reproduces the pre-fix value bit-for-bit
(confirmed by every pre-existing patch test in `tests/test_cpe4rh_patch.py`
still passing unchanged), and a genuine UL state now matches the
equivalent TL evaluation to ~1e-16 relative (was 24.9%/7.2% before the
fix, matching the opus review's measurements exactly).

### F5 fix: `_ul_F_n` mid-step compounding

Removed the 5 per-Newton-iteration writes to `self._ul_F_n[elem_indices]`
in `dynamic.py` (previously at lines ~3523, 3587, 3651, 3751, 3964) --
the commit-time resync (`dynamic.py` ~2614-2632, inside `if
self.ul_mode:`) already recomputes `_ul_F_n` for every element from the
converged `u_total` once per step, and is the sole authoritative write.

### New permanent regression tests (both passing at machine precision)

- `tests/test_ul_tl_consistency.py` (T2, 8/8): CPE4RH/CPE4I/CPE4H/
  Q4_VISCO_SIMO all agree between a directly-reached total deformation
  (TL) and the same deformation reached incrementally through UL, to
  ~1e-16 relative.
- `tests/test_ul_fn_idempotence.py` (T3, 1/1): confirms
  `DynamicSolver._assemble` no longer mutates `_ul_F_n` and gives
  bitwise-identical residuals across repeated calls with the same `u`.

### Full verification after the fix

`pytest tests/` (251 items): 234 passed, 4 skipped, 1 deselected (the
slow real-fold integration test, see below), 4 xfailed, 5 failed -- every
one of the 5 confirmed via direct `git stash` A/B as pre-existing/
unrelated to F4/F5, not a new regression:
- `test_eas_jax_verify.py::test_elastic_force_and_tangent` /
  `test_plastic_force`: byte-identical failure before and after (plain
  J2 `q4_eas.py`/`q4_eas_jax.py`, untouched by F4/F5).
- `test_abaqus_config.py::test_config_presets`: fails both before (a
  different `AttributeError`) and after (a stale `dt_max` expectation) --
  config preset drift, unrelated to element physics.
- `test_prescribed_skip.py::test_prescribed_skip_solution_equivalence`:
  didn't even converge before (`assert -20 >= 0`); converges now but
  skip/no-skip differ by 2.17e-9 against a 1e-10 tolerance -- unrelated
  to F4/F5 (plain NeoHookean/Q4, no viscoelastic UL path), and if
  anything the fix made the underlying solve healthier.
- `test_rbe2_condensation_reconstruction.py`: genuinely a stale hardcoded
  expectation (`master_id in (10000, 20000)`) left over from THIS
  session's earlier node-id-overflow fix (10000 -> 100000 default);
  fixed (`(100000, 100306)` -- the two plates share one sequential node
  counter, not two round offsets), same class of stale test as
  `tests/test_rigid_plate_tie.py` (also fixed earlier this session).

`verification.run_all`: 12/14, same 2 pre-existing/unrelated failures as
documented throughout this session (2-Point Bending Gulati Elastica,
Convergence Cantilever Mesh Q4bbar) -- unaffected.

**Real production integration test**
(`tests/test_tie_gap_verification.py::test_surface_tie_large_rotation_gap_quantitative`,
a genuine 50-step 90-degree fold using `examples/ex12_rigid_plate_display_fold.inp`
with the real CPE4I element carrying the F4 fix): confirmed via a direct
timing probe (monkeypatched `DynamicSolver._assemble`) that the FIRST
assembly call after the fix takes a long time (one-time JAX JIT compile
of the now-larger computational graph) but the 2nd and 3rd calls
complete in ~0.4s each -- the fix has NO per-call performance
regression, only the normal one-time JAX compile cost any fresh process
pays for a changed kernel. Deselected from the main pytest run for this
reason (would have added many extra minutes) but confirmed alive and
actively computing (rising CPU time in `Get-Process`) rather than hung
at every check. Not run to full completion given session time
constraints -- if picked up again, expect the same multi-minute-per-run
patience the rest of this session's ex12 runs needed; nothing about the
fix should change the qualitative 90-degree fold outcome (F4 corrects a
stress reference-frame mismatch, which changes quantitative force
magnitude under UL, not the element's fundamental stability).

## 6. Next step (still pending, per section 0's correction)

F4/F5 (Family C, "fix now") are done. Family B's redevelopment (migrate
PET/GLASS off the invented SRI device onto a real Abaqus element, most
likely CPE4I) has NOT been started. Also still pending per the user's
"영구 예외 없음, 전면 재검토" directive: the full element-file inventory
pass (which `dispsolver/element/*.py` files/element-type strings have no
real Abaqus 1:1 counterpart today) -- partially done in conversation
(see the file-by-file classification already produced), not yet written
into this document as a formal table for the next session to work from.

## 7. Full element-file inventory (Abaqus 1:1 audit)

Every file in `dispsolver/element/`, classified against a real Abaqus
element name. This is the formal table the previous entry said was still
owed -- built directly from the file-by-file audit already done in
conversation, not re-derived.

| File | Element-type string(s) | Abaqus 1:1? | Verdict |
|---|---|---|---|
| `q4_eas.py` / `q4_eas_jax.py` / `q4_eas_numba.py` | `Q4_EAS` | **CPE4I** | Real match, correct. |
| `q4_visco_eas_jax.py` / `q4_visco_eas_numba.py` | `CPE4I`, `CPE4H`, `CPE4IH` | **CPE4I / CPE4H / CPE4IH** | Real match, correct (already named right). |
| `q4_visco_hybrid_up_numba.py` | `CPE4H` | **CPE4H** | Real match, correct. |
| `q4_visco_hybrid_reduced_jax.py` / `_numba.py` | `CPE4RH` | **CPE4RH** | Real match, correct (built this session). |
| `q4_corotational_jax.py` | `Q4_COROTATIONAL` | **CPE4** (per F1: provably == plain TL, frame is dead weight) | Real match once the frame is dropped (Phase B1, not started). |
| `q4_corotational_eas.py` / `q4_corotational_eas_jax.py` | `Q4_COROTATIONAL_EAS` | **CPE4I** (should just BE the plain CPE4I kernel, per F1/F2 -- no reason for a separate co-rotational-frame variant) | Redundant with `q4_eas_jax.py` once Family A migrates -- redevelopment target (merge away, not keep as its own file). |
| `q4_hybrid_jax.py` / `q4_hybrid_numba.py` | `Q4_HYBRID`, `Q4_COROTATIONAL_HYBRID`, `Q4_COROTATIONAL_HYBRID_EAS` | Claims **CPE4H/CPE4RH/CPE4IH** in its own docstring | **Unverified claim** -- an earlier dev_log (2026-07-30) found the co-rotational-hybrid-EAS kernel here was "a functional duplicate of CR-EAS... zero actual pressure/hybrid content." `tests/test_cpe4h_hybrid.py` exists and passed in this session's full suite run, which is evidence FOR real content, but the specific function flagged in 2026-07-30 needs a direct re-check before trusting the docstring. **Re-verify before folding into Phase C.** |
| `q4_sri_jax.py` / `q4_sri_numba.py` / `q4_corotational_sri_j2_numba.py` | `Q4_SRI`, `Q4_COROTATIONAL_SRI` | **No real Abaqus name** (own docstring: "Abaqus doesn't ship CPE4S") | Confirmed invented device. F3: frame is load-bearing for this one specifically. **Redevelopment target -- migrate PET/GLASS onto CPE4I instead** (per the corrected Family B plan, section 0's correction block). |
| `q4_sri_hybrid_jax.py` / `q4_sri_hybrid_numba.py` | `Q4_HYBRID_SRI`, `Q4_COROTATIONAL_HYBRID_SRI` | **No real Abaqus name** (same SRI device, + hybrid pressure) | Same redevelopment target as above -- should become CPE4IH once Family B migrates. |
| `q4_reduced_jax.py` | `Q4_COROTATIONAL_REDUCED` (never wired into any dispatch tuple) | **CPE4R** (genuinely faithful 1-point + Flanagan-Belytschko hourglass, for J2 plasticity) | The one file that IS an honest CPE4R -- just not suited to this repo's one-element-through-thickness PET/GLASS mesh (AGENTS.md's own 2026-07-30 finding, a real physical limitation Abaqus shares, not a bug). Keep as-is; do not delete, do not force it into use here. |
| `q4_up_jax.py` | `Q4_UP` (plain, non-viscoelastic) | **CPE4H** (Green-Lagrange TL, no `F_n`/UL) | Confirmed unused anywhere in `fold_model_config.py`. Redundant with the real `CPE4H` (`q4_visco_hybrid_up_numba.py` + its JAX equivalent inside `q4_visco_eas_jax.py`) once a non-viscoelastic material needs CPE4H -- candidate for deletion rather than a UL retrofit, pending confirmation nothing else references it. |
| `q4_visco_hybrid_fs_jax.py` | `Q4_UP` + `LinearViscoelastic` material | **CPE4H** (Green-Lagrange + F-bar TL, no `F_n`/UL, no Numba port) | Confirmed used only by old `ex03`/`ex06`-`ex08` scripts, `LinearViscoelastic` material class itself superseded by `ViscoelasticMaterial`. Legacy stepping-stone, superseded by `q4_visco_eas_jax.py`'s real CPE4H branch -- candidate for deletion. |
| `q4_visco_hybrid.py` / `q4_visco_hybrid_jax.py` | (no distinct element-type string found in `dynamic.py`'s dispatch -- appears dead) | **CPE4H**, small-strain only | Confirmed legacy (small-strain predecessor of `q4_visco_hybrid_fs_jax.py`, itself superseded). Candidate for deletion. |
| `q4_visco_simo_fs_jax.py` / `q4_visco_hybrid_simo_numba.py` | `Q4_VISCO_SIMO`, `Q4_UP` (+ `ViscoelasticMaterial`) | **No distinct Abaqus name** (F-bar mean-dilatation is a locking-control DEVICE, not an Abaqus element in its own right -- real Abaqus locking control for this case is CPE4H/CPE4I) | Currently the DEFAULT/production PSA path's plain fallback and where F4 was first found and fixed. Once Family C's fix is confirmed solid and CPE4I/CPE4H are trusted for every current use of this path, this becomes a redevelopment target too (redundant with CPE4H) -- lower priority, not blocking. |
| `q4.py` / `q4_jax.py` / `q4_numba.py` | `Q4` (default) | **CPE4** (plain full-integration, small-strain by design) | Correct as-is -- used only for rigid, Dirichlet-BC-driven plate elements that don't need their own large-deformation capability. Not a redevelopment target. |
| `t3.py` | `T3`/`TRIA3` | **CPE3** | Explicitly small-strain only by its own docstring; confirmed unused in any production mesh (`fold_model_config.py` never emits triangles) -- benchmark/reference-only. Not a redevelopment target unless triangles are ever needed for production. |
| `q4_plastic_numba.py` | (shared material kernel, not an element in its own right) | n/a | Infrastructure (J2 stress+tangent), used by several of the files above. Not itself classified. |
| `shell_2d_jax.py` | (own beam/shell dispatch, not part of the plane-strain quad family) | Would map to **B21/B22** (beam) territory, not CPE4-anything | Out of scope for this consolidation -- different element category entirely. |
| `rbe2.py` / `rbe2_jax.py` | RBE2 constraint (not a continuum element) | Matches Abaqus `*RIGID BODY`/RBE2 concept already | Out of scope -- a constraint, not a continuum element; also already documented (AGENTS.md 4.10) as the abandoned mechanism for driving the plate (superseded by `RigidBodyPart`). Not touched by this consolidation. |

### Reading this table for next steps

- **Real redevelopment targets** (invented devices masquerading as or
  standing in for real Abaqus elements, actually used somewhere that
  matters): the SRI family (`q4_sri_jax.py`/`q4_sri_numba.py`/
  `q4_corotational_sri_j2_numba.py`/`q4_sri_hybrid_*.py`) for PET/GLASS --
  this is Family B, the next big item.
- **Needs re-verification, not yet a confirmed problem**:
  `q4_hybrid_jax.py`/`q4_hybrid_numba.py`'s claimed CPE4H/CPE4RH/CPE4IH
  content -- do this before relying on it for anything in Phase C.
- **Deletion candidates** (confirmed superseded/unused, not a
  redevelopment target because there is nothing to redevelop toward --
  just remove): `q4_up_jax.py`, `q4_visco_hybrid_fs_jax.py`,
  `q4_visco_hybrid.py`, `q4_visco_hybrid_jax.py`. Confirm zero remaining
  references (including in tests) before deleting any of them.
- **Correct as-is, no action needed**: `q4_eas_jax.py`, `q4_visco_eas_jax.py`,
  `q4_visco_hybrid_up_numba.py`, `q4_visco_hybrid_reduced_jax.py`,
  `q4_reduced_jax.py` (kept for its honest-but-unsuited-here CPE4R), plain
  `q4.py`, `t3.py`, `shell_2d_jax.py`, `rbe2*.py`.

### 7b. Inventory corrections (2026-09-08, from the user's own follow-up questions)

Two rows in the section 7 table were too generous about "1:1 Abaqus" and
are corrected here (per the user's "영구 예외 없음 / 1:1" standard --
these are exactly the kind of "close enough" claims that standard rules
out):

- **`q4.py` / `q4_jax.py` / `q4_numba.py` (`Q4`) is NOT literally
  Abaqus CPE4.** Abaqus CPE4 is pure full 2x2 integration with no
  volumetric treatment (which is precisely why it volumetrically locks
  as nu -> 0.5, and precisely why CPE4H exists as its hybrid
  counterpart). This repo's `Q4` applies B-bar (selective reduced
  integration of the volumetric term only) on top -- an extra locking-
  control device Abaqus's CPE4 does not have. Accuracy-wise it is fine
  (and it measured perfectly global-axis isotropic, 1e-16, in this
  session's isotropy probe, unlike EAS/SRI) -- the mismatch is purely
  one of NAMING/1:1 fidelity. If the "1:1 with Abaqus" standard is
  applied strictly, either (a) rename it to something honest that is not
  a bare Abaqus element name, or (b) add a genuine no-B-bar full-
  integration CPE4 alongside it. Note in practice a pure CPE4 is rarely
  what anyone wants (it locks), so (a) is the cheaper honest answer.
- **`q4_reduced_jax.py` IS a faithful CPE4R** (1-point reduced
  integration + Flanagan-Belytschko hourglass control, FD-verified) --
  the table already said this, but to be explicit about WHY it is
  unwired: not a defect in the element, but the documented physical
  mismatch with THIS mesh (one-to-three elements through each layer's
  thickness, where the pure-bending nodal pattern coincides with the
  hourglass pattern; measured 10.19x mesh dependence, worse than plain
  corotational -- `dev_log/session_20260730_reduced_integration_failure.md`).
  Abaqus's own manual recommends >= 4 elements through the thickness for
  CPE4R for the same reason, so this is a limitation Abaqus shares, not
  one of ours. Keep the file, keep it unwired for this mesh, and do NOT
  treat "CPE4R is unused here" as evidence the implementation is wrong.

### 7c. RETRACTION of 7b's CPE4 claim + the real CPE4 built (2026-09-08)

**7b was wrong and is retracted.** It claimed `q4.py`'s B-bar made it
"not literally Abaqus CPE4". In fact Abaqus's fully-integrated first-order
continuum elements (CPE4, C3D8) apply exactly this selectively-reduced
volumetric treatment by default -- the dilatation is evaluated at the
element centroid. That is why CPE4 resists volumetric locking yet still
SHEAR-locks in bending, and why CPE4I (incompatible modes) exists as the
cure for the shear locking. A "pure full integration, no volumetric
treatment" element would have been OUR invention, not Abaqus's. The user
caught this directly ("B-bar는 abaqus에는 없는 기술?").

So the formulation in `q4.py` was right all along. What was genuinely
missing was everything AROUND it: `q4.py`'s family is small-strain,
linear-elastic-only, no Updated-Lagrangian support, hard-wired to one
material.

**Built: `dispsolver/element/cpe4_jax.py`** -- the same Abaqus CPE4
formulation, rebuilt to the target architecture:
- full 2x2 Gauss integration with the centroid-sampled volumetric
  treatment in its finite-strain form (F-bar, de Souza Neto et al. 1996:
  `F_bar = F sqrt(J0/J)`, the finite-strain generalisation of B-bar);
- finite strain, Updated-Lagrangian capable (`F = F_inc @ F_n`), with the
  F4 push-forward (`S_n = F_n S F_n^T / det F_n`) built in from the start;
- **material-agnostic**: the constitutive response arrives as a callable
  (`response_fn(F, state, dt, params)`, the Phase A interface). The
  element never branches on material type. This is the first element in
  this codebase written to the Abaqus separation from scratch rather than
  retrofitted;
- two entry points: `compute_cpe4` (autodiff consistent tangent, for
  materials safe to differentiate through) and
  `compute_cpe4_material_tangent` (material-supplied `C`, assembling
  `K = sum B^T C B` -- the established modified-Newton path for J2, whose
  spectral return map carries the AGENTS.md 4.3 autodiff NaN risk).

Also added `make_visco_response(base)` to `response_interface.py`: binds
the static `base` model selector into the callable so a material-agnostic
element only ever traces arrays/floats (a raw string inside a traced
pytree is a `jax.jit` error), with per-`base` caching so the
`static_argnames` contract doesn't retrigger tracing on every call.

**Verification** (`tests/test_cpe4_element.py`, 24/24 passing): rigid
rotation canary (both materials x 5 angles), **global-axis isotropy
(the gate SRI fails at 1-5% and the EAS mode set fails at 5-9%, and CPE4
passes)**, UL == TL consistency, constant-strain patch test on 4
irregular quads, and material-agnosticism (identical element code driven
by J2Plasticity and by ViscoelasticMaterial, plus force agreement between
the two tangent entry points).

## 8. F6 -- EAS enhanced-mode transpose bug (found + FIXED, 2026-09-08)

**The most severe defect found in this whole effort so far, and it was
live in production.** Found by the second opus review (commissioned to
adjudicate a global-axis-isotropy anomaly I measured), independently
re-derived and numerically confirmed here before fixing.

### The bug

`_enhanced_grad_modes` built the EAS-4 enhanced deformation-gradient
modes as `(detJ0/detJ) * D_k @ J0^{-1}` where it must be
`(detJ0/detJ) * D_k @ J0^{-T}`.

It is an index-TYPE mismatch, not a sign or scaling slip. `_jac` builds
`J[i][j] = dX_j/dxi_i` (rows natural, columns spatial), so
`inv(J0)[a][b] = dxi_b/dX_a` -- rows SPATIAL, columns NATURAL. `D_k`'s
second index is NATURAL (the modes are written in natural coordinates),
so the pull-back `d(u~)_i/dX_m = D_ij * dxi_j/dX_m = D_ij * inv(J0)[m][j]`
requires the transpose. Written with `J0^{-1}` the contraction pairs a
natural index against a spatial one.

Note the opus review also corrected my own initial hypothesis twice: my
proposed "fix" (`J0 D_k J0^{-1}`) would have been a NO-OP (left-multiplying
every mode by a constant invertible matrix does not change the span, and
covariance here is a span condition on the four modes, not a per-mode
condition -- the alphas are condensed internal DOFs), and my claimed
transformation direction was itself off by a transpose.

### Why nobody caught it

For an axis-aligned rectangular element `J0` is DIAGONAL, so
`J0^{-1} == J0^{-T}` and the bug is **exactly invisible**. Every patch
test, every unit test, and AGENTS.md 4.1's aspect-ratio sweep use
axis-aligned reference elements. A 90-degree rotation also permutes the
axes cleanly, so even a 90-degree check passes -- the error only appears
at intermediate angles.

### Why it mattered anyway (it was ACTIVE in production)

`ul_large_rotation_mode` defaults True (`fold_model_config.py`), so the
Updated-Lagrangian reference configuration IS the last converged
configuration -- `J0` carries the accumulated fold rotation and goes
off-diagonal within a few increments of a 90-degree fold. Measured
bending-energy ratio (1.000 = exact) vs reference rotation angle, at the
production free-span geometry, independently reproduced here:

    ref rotation:      0deg   10deg   30deg   45deg   60deg   90deg
    before (J0^-1):   1.000   9.384   3.377  10.279   9.804   1.000
    after  (J0^-T):   1.000   1.000   1.000   1.000   1.000   1.000

The opus review's own sweep across the three real `gen_ex12_inp.py`
column widths is worse still (1.000 = exact):

    dx=0.25mm, AR 7 (hinge/tip):
      angle:   0     10      20      30      45      60      75     90
      before: 1.000  9.738   0.749   3.936  11.419  10.744   3.910  1.000
    dx=0.50mm, AR 14 (free hinge span):
      before: 1.000  9.105   5.326  25.774  48.194  40.089  13.908  1.000
    dx=1.00mm, AR 28 (plate body):
      before: 1.000  1.438  46.230 125.287 195.218 153.864  52.284  1.000
    (after the fix: exactly 1.000 in every cell of all three rows)

Peak error **195x** at AR 28 / 45 deg. Note it is NOT a simple
stiffening: 0.749 at AR 7 / 20 deg is 25% too SOFT. The error is erratic
in both angle and aspect ratio, which is why no single "it runs a bit
stiff" symptom ever pointed at it. Exactly 1.000 at 0 and 90 degrees
because those keep the error operator diagonal.

An aspect-ratio sweep at a fixed 30-degree reference rotation (the UL
case) shows the same thing from the other direction: before the fix the
ratio runs 0.250 (AR 1) -> 30.430 (AR 15); after, 1.000 everywhere.

Global-axis isotropy over the same rotations went from ~6e-2..1e-1 to
~1e-12. So the CPE4I family -- which is what production PSA runs on
(`fold_model_config.py` sets `psa_element_type = "CPE4I"`, routed via
`q4_visco_eas_jax._enh_modes_all` into the same broken function) -- has
been running with wildly wrong bending stiffness for every element whose
reference frame had rotated off-axis, i.e. essentially all of them
mid-fold, ever since UL was enabled.

### Fix applied (all five copies)

- `q4_eas_jax.py::_enhanced_grad_modes` -- `einsum('kij,jm->kim')` ->
  `einsum('kij,mj->kim')`. This one also fixes `q4_visco_eas_jax.py`
  (CPE4I/CPE4H/CPE4IH JAX), which imports it.
- `q4_eas.py::_enhanced_grad_modes` (numpy) -- `D @ J0inv` -> `D @ J0inv.T`.
- `q4_eas_numba.py::_enhanced_grad_modes` -- same.
- `q4_visco_eas_numba.py::_enh_modes_numba` (hand-expanded) -- swap
  `i12 <-> i21` in the four mode expansions, which is exactly the
  transpose in expanded form.
- `_jit_cache.py`: `DISPFOLD_CACHE_VERSION` v2 -> v3, so the on-disk
  Numba cache cannot serve the pre-fix kernels (the same stale-cache trap
  this session already hit once with the `_simo_pk2_numba` fix).

### Verification

- Independent numerical reproduction of both the defect and the fix
  (`scratch/verify_eas_transpose.py`): isotropy 1e-1 -> 1e-12, rotated-
  reference bending 10.3x -> 1.000.
- Both Numba ports now match the fixed JAX reference on a 37-degree
  rotated element: `q4_visco_eas_numba` 2.7e-15, `q4_eas_numba` 0.0.
- Full element test sweep after the fix: 68 passed, 2 failed -- the 2 are
  the already-documented pre-existing `test_eas_jax_verify.py` failures,
  with byte-identical numbers to before the fix (as expected: those tests
  use axis-aligned elements, where this fix is a mathematical no-op).

### Consequence for the migration plan

This strengthens the case for moving PET/GLASS onto CPE4I rather than
keeping the home-grown SRI device: the CPE4I anisotropy I measured
earlier (5-9%) was this bug, not an inherent property of incompatible
modes, and it is now gone. The SRI device's own 1-5% anisotropy is a
separate question still awaiting the remainder of the opus report.

## 9. Design requirement: TL/UL is NOT a solver-level switch (2026-09-08)

Raised by the user after F6: "아, 그럼 아무거나 TL을 사용하거나 UL을
사용하면 안되는거네. 요소에 따라서 대변형이라고 하더라도 방식이
다른거겠네?" -- correct, and this session produced three independent
proofs of it:

1. **SRI (`Q4_COROTATIONAL_SRI`)**: its sampling rule is written in
   Cartesian x/y components, so it is not frame-invariant. It is only
   correct on a TL path with an axis-aligned mesh (diagonal `J0`).
   Routing it to UL -- where the reference configuration itself rotates
   -- degrades it silently (measured 3.6e-2 .. 1.6e-1 isotropy error).
   Its correctness in production today rests entirely on the accident
   that `dynamic.py` dispatches it TL with the original axis-aligned
   coords.
2. **EAS (`CPE4I`)**: before F6 it was exact under TL (axis-aligned, so
   the transpose bug cancelled) and catastrophically wrong under UL (up
   to 195x). Same element, same code -- only the reference frame's
   rotation decided whether it was right.
3. **Viscoelastic UL kernels**: F4 showed UL imposes an ADDITIONAL
   correctness requirement TL does not -- the PK2 push-forward
   `S_n = F_n S F_n^T / det F_n`. Under TL (`F_n = I`) that term is an
   exact no-op, so its absence was invisible for as long as the
   viscoelastic elements stayed TL-only.

### The structural problem

`ul_large_rotation_mode` is a single global flag (`fold_model_config.py`)
applied to every element the model contains. Elements that cannot
support a rotated reference get silently dragged onto that path anyway.
There is no error, no convergence failure, and no diagnostic -- the
model just produces wrong stiffness. That is precisely how F6 stayed
live in production.

Abaqus does not expose this choice: the user sets NLGEOM and each
element handles its own large-deformation treatment internally and
consistently.

### Requirement for the ongoing Abaqus 1:1 redevelopment

Each element must **declare its own capabilities**, and the solver must
check them at dispatch instead of assuming:

- `supports_ul` (does a rotated reference configuration keep the
  formulation valid?)
- `frame_invariant` (is every sampling/enhancement rule stated in a
  frame-covariant form -- natural-frame or invariant quantities, not
  fixed Cartesian components?)
- `requires_axis_aligned_mesh` (the SRI case today -- and this must be a
  hard, checked precondition, not a comment)

Dispatching an element with `supports_ul = False` into a UL model should
be a loud failure, not a silent 195x error. The new `cpe4_jax.py` is
already written to be safe in both regimes (frame-covariant kinematics +
push-forward built in) and should carry these declarations first, as the
template for the rest of the migration.

### 9b. NLGEOM switch implemented (2026-09-08)

User decision after the section 9 discussion: "그럼 옵션에서 ul_mode =
true 방식이 아니라 그럼 우리도 nlgeom을 on-off 하는 방식으로 전격 변경!"
-- adopt Abaqus's own switch. Done:

- **`DynamicSolver(nlgeom=...)`** is now the user-facing geometric-
  nonlinearity switch. `ul_mode` / `ul_large_rotation_mode` still work as
  deprecated aliases (existing configs and examples are unaffected).
  `FoldModelConfig.solver.nlgeom` added, and `to_solver_kwargs()` now
  emits `nlgeom` (any of the three fields being on turns it on).
- **`_ELEMENT_LARGE_DEF`** (dynamic.py): explicit per-element declaration
  of `supports_ul` / `frame_invariant` / a one-line reason, covering
  every element type the dispatch recognises. An element type missing
  from the table is treated as NOT UL-capable -- conservative by design,
  so a new element can never be silently opted into a formulation it
  hasn't been shown to support.
- **`DynamicSolver._use_ul_for(pid)`**: the per-element decision. All six
  per-pid dispatch sites in `_assemble_multi_material_batch` and the
  sequential fallback now ask this instead of reading the global
  `self.ul_mode`. The one remaining global use (the commit-time `_ul_F_n`
  resync in `solve_step`) is correct as-is: it recomputes from `u_total`
  for all elements and is valid regardless of per-element treatment.
- **`DynamicSolver.element_large_deformation_report()`**: prints what
  NLGEOM actually resolves to, per pid. This is the intended answer to
  "is this model really running Updated Lagrangian?" -- which is now a
  per-element question, not a flag lookup.

Verified: with `nlgeom=True`, `CPE4I` resolves to "UL (rotated
reference)" while `Q4_COROTATIONAL_SRI` correctly stays "TL" with the
reason attached. Regression sweep after the change: 44 passed, 1 failed
(`test_abaqus_config.py::test_config_presets`, the pre-existing `dt_max`
preset drift already confirmed unrelated by git-stash A/B).

The immediate practical effect: the SRI family can no longer be dragged
onto a rotated reference by a global flag. That hazard was live and
silent until today.

---

## §10 Alpha-clamp saturation resolved: `_ALPHA_MAX` 0.05 -> 0.5

Open item from the handoff doc, resolved same day. Measurement (25-step
ex12 fold, `scratch/alpha_sat2.log`, old clamp 0.05): `max|eas_alpha|`
grows smoothly (9.99e-7 at step 1) and hits the clamp exactly at step
~18 (theta ~3.5deg), pinning at 0.05 with the saturated-element fraction
climbing to 8.8% for the rest of the run. Clean monotonic growth into
the clamp, no oscillation -- genuine physical demand, not instability.

`git log -S` traced `_ALPHA_MAX` 5.0 -> 0.05 to commit `f65f90d`
(2026-07-30), before F6 was found. Working hypothesis: 0.05 was a
band-aid for the pre-fix transpose bug's nonlinear blow-up under UL
rotation (up to 195x, sec 8), not an independently-derived physical
bound.

Raised to 0.5 (10x) in `q4_eas_jax.py`, `q4_visco_eas_jax.py`,
`q4_visco_eas_numba.py` together; `DISPFOLD_CACHE_VERSION` v3 -> v4.

Re-verification:
- `tests/test_cpe4_element.py` + `tests/test_ul_tl_consistency.py`:
  32/32.
- `scratch/tier1_numba_jax_tilted.py`, A/B'd 0.05 vs 0.5 directly: at the
  scale that used to saturate, numba/JAX force error on the
  TILTED+DISTORTED UL case dropped **52% -> 4.35e-04** (axis-aligned
  5.2e-3 -> 8.2e-12; TL 4.1e-6 -> 8.8e-15). `K_err` (tangent-only) stays
  large at every scale including well below either clamp -- pre-existing,
  clamp-independent: the numba viscoelastic path is FD-tangent only (its
  own docstring), JAX is exact autodiff. Per AGENTS.md sec 4.4 a
  mismatched tangent costs Newton iterations, not correctness; force
  agreement is the actual gate and that is what improved 3 orders of
  magnitude.
- `scratch/alpha_sat3.log` (40-step re-run, new clamp) and
  `scratch/pytest_full_alpha05.log` (full suite): launched, check before
  relying on this further.

Conclusion: closed for the range measured. Family B (migrate PET/GLASS
off `Q4_COROTATIONAL_SRI` onto CPE4I) no longer blocked by this.

---

## §11 `_ALPHA_MAX` clamp REMOVED — §10's conclusion is superseded (2026-09-09)

Full write-up: **`dev_log/eas_stabilization_modernization_20260909.md`**.

§10 raised `_ALPHA_MAX` 0.05 -> 0.5 and closed the item on the reading that
alpha's growth into the clamp was "genuine physical demand, not instability."
**That reading was wrong and is retracted here.** Two measurements settle it:

1. Re-reading §10's own 40-step probe (`scratch/alpha_sat3.log`, clamp 0.5):
   `max|alpha|` grows smoothly to 1.4e-1 at theta = 4.71 deg and then **jumps to
   0.49999 in a single increment**, pinning at the clamp for the remaining 11
   attempts (all cutbacks) and stalling the run at theta = 5.39 deg. A
   step-function jump to the bound is a blow-up signature, not smooth demand.
   The "smooth monotonic growth" seen at the old 0.05 clamp was simply the
   pre-blow-up part of the same curve.
2. Driving the CPE4I element-local Newton directly on the tilted+distorted UL
   probe of `scratch/tier1_numba_jax_tilted.py`: **undamped, it does not
   converge in 25 iterations** (|f_a| oscillating 2e-4..1.8e-1, |alpha| wandering
   to 0.34), while `K_aa` there is still positive definite -- so this is NOT
   the Wriggers & Reese (CMAME 135:201-209, 1996) compressive rank deficiency,
   it is plain Newton overshoot. Adding a **backtracking line search**
   converges the same state in 10 iterations to |f_a| = 5.6e-13 and
   |alpha| = 1.8e-2, i.e. an order of magnitude SMALLER than the undamped
   iterate was wandering through.

The clamp was bounding a divergent local iteration.

**What replaced it** (all five EAS files + `dynamic.py`):
- **Primary**: line-search damped, tolerance-terminated element-local Newton
  (step set {1, 0.5, 0.25, 0.1}, tol 1e-8 on ||d(alpha)||_inf, cap 12). This is
  one of the two remedies Pfefferkorn, Bieber, Oesterle, Bischoff & Betsch
  (IJNME 2021, DOI 10.1002/nme.6605) report for EAS's documented "lack of
  robustness in the Newton-Raphson scheme", and it was already this repo's own
  precedent -- `q4_eas.py` has always had it and is the one EAS kernel that
  never needed a clamp. A line search cannot move the converged root.
- **Fallback**: kernels return a `status` (last accepted ||d(alpha)||_inf, inf if
  non-finite); `DynamicSolver` abandons the increment when any element exceeds
  `eas_local_tol`, reusing the existing `_fail(...)`/cutback path. This mirrors
  the Abaqus Analysis User's Guide's documented behaviour for element-level
  calculation trouble: "the increment will be attempted again with a time
  increment of [factor] times the current time increment."
- **Rejected**: SPD projection / eigenvalue floor on `K_aa`. No surveyed source
  recommends it (checked directly against Bieber, Auricchio, Reali & Bischoff,
  IJNME 124(11):2638-2675, 2023); and since `K_aa` also enters the condensed
  FORCE, perturbing it reintroduces exactly the force/tangent inconsistency the
  clamp had.
- **Researched, deliberately NOT adopted**: Q1/E4T transposed Wilson modes
  (Glaser & Armero, Eng. Comput. 14(7):759-791, 1997) -- the literature's real
  cure for the compressive hourglass instability, but Abaqus CPE4I is the
  standard (untransposed) incompatible-modes element, so adopting it would move
  `CPE4I` AWAY from the 1:1 target. If genuine compressive hourglassing is ever
  observed, add Q1/E4T as a separately named opt-in element, not as a
  redefinition of CPE4I. Same for MIP (residual-preserving tangent
  modification) as a further robustness step.

`DISPFOLD_CACHE_VERSION` bumped v4 -> v5.

Verification summary (details in the dedicated dev_log): element subset 51/51;
`scratch/tier1_numba_jax_tilted.py` numba-vs-JAX force error on the
tilted+distorted UL case **4.35e-04 -> 2.71e-15**; full `pytest tests/`
**2 failed / 265 passed** against a same-machine pre-change baseline of
**3 failed / 264 passed** (the fixed one is
`test_eas_jax_verify.py::test_elastic_force_and_tangent`, which compares the
already-line-searched NumPy kernel against the JAX one); `verification.run_all`
**12/14**, the same two long-documented unrelated failures.
