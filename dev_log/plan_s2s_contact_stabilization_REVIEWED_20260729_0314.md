# PLAN (reviewed + enhanced) — Surface-to-Surface penalty contact & contact stabilization

**Status**: review of `plan_surface_to_surface_contact_stabilization.md`
(authored by the Antigravity CLI agent), enhanced against the actual
repository state and this project's documented failure history.
**Not implemented.**

The original plan's three core requirements (S2S formulation,
penalty-first approach, contact-convergence stabilization) and its four
recommendations are **kept**. What follows adds what the original was
missing: it proposed building on an empty field, but this repository
already has a contact subsystem, a stabilization engine, and — most
importantly — hard-won documented failures that bear directly on two of
its four recommendations.

---

## §0. CRITICAL — the original plan proposes rebuilding what exists

The original plan lists `dispsolver/constraint/surface_to_surface_contact.py`
as **[NEW]** with no reference to any existing code. In fact this repo
already contains **three** contact/stabilization implementations:

| Existing file | What it is | Status |
|---|---|---|
| `dispsolver/contact/contact_solver.py` (361 lines) | **Penalty NTS (node-to-segment) contact** — `SpatialHashGrid` broad-phase (O(N) avg), JAX-compatible closest-point projection, `log(cosh)` regularized contact energy, `ContactPair.apply_penalty(u, f_int, K_T)` | Fully wired through the `.inp` path (`*CONTACT PAIR` parsing in `abaqus_parser.py`, `ContactPair` construction in `model_builder.py`, exposed as `ModelBuilderResult.contact_pair_objects`) — **but zero usage in `examples/` or `tests/`.** Parsed and built, never exercised. |
| `dispsolver/contact/contact_surface.py` (174 lines) | `ContactSurface` + `auto_detect_exterior()` — exterior edge extraction with outward normals | Same: wired into the `.inp` path, unexercised |
| `dispsolver/constraint/contact_jax.py` (262 lines) | Penalty **self-contact**, node-to-node, spatial hash + JAX autodiff tangent (`f = dΠ/dx`, `K_t = d²Π/dx²`) | Separate implementation, separate purpose |
| `dispsolver/solver/stabilization.py` (46 lines) | `AbaqusViscousStabilization` — `F_stab = c·M_diag·v` with **automatic damping scaling to keep `E_stab/E_strain < 0.5%`** (Abaqus `*STABILIZE` equivalent) | Exists, referenced in AGENTS.md §2 |

### Required plan changes

1. **Locate the new code in `dispsolver/contact/`, not `dispsolver/constraint/`.**
   Contact lives in its own package here. Adding
   `constraint/surface_to_surface_contact.py` splits the contact
   subsystem across two packages for no reason. Proposed:
   `dispsolver/contact/s2s_contact.py`.

2. **Decide explicitly what happens to the existing `ContactPair` (NTS).**
   Three defensible options — pick one and state it, do not leave a
   third contact implementation lying around:
   - (a) **Extend**: add an S2S integration mode to `ContactPair`
     alongside its existing NTS mode (shares the broad-phase, surface
     handling, and `.inp` wiring for free).
   - (b) **Replace**: S2S supersedes NTS; deprecate `ContactPair`'s NTS
     path and re-point `model_builder.py`'s `contact_pair_objects`
     construction at the new class.
   - (c) **Parallel**: keep both, selectable — only justified if NTS is
     still wanted for some cases. Needs a stated reason.
   Recommended: **(a)**, because the reusable parts below are the
   expensive parts.

3. **Reuse rather than re-derive** — these already work and are the
   bulk of a contact implementation:
   - `SpatialHashGrid` (broad-phase segment query)
   - `ContactSurface` / `auto_detect_exterior()` (surface + normal extraction)
   - `_closest_point_on_segment_jax()` (projection)
   - The `.inp` `*CONTACT PAIR` / `*SURFACE` parse→build path
   - The `penalty_constraints` protocol: any object exposing
     `apply_penalty(u, f_int, K_T)` is assembled by `DynamicSolver`
     (`dynamic.py:3553`) — matching this interface means **zero solver
     changes are needed**, contradicting the original plan's item 3
     ("`dispsolver/solver/dynamic.py`: integrate S2S contact residual/
     stiffness into Newton loop"). Confirm before writing solver code.

4. **`log(cosh)` regularization already exists and is smoother than the
   proposed C².** `contact_solver.py`'s `_nts_contact_energy` uses
   `log(cosh)` — that is C^∞, strictly smoother than the original
   plan's piecewise-quadratic C² transition. Either reuse it or state
   why a *less* smooth formulation is preferred. (If reusing: `log(cosh)`
   overflows for large arguments; check the existing implementation
   guards this, e.g. via the `x + log1p(exp(-2x)) - log(2)` identity.)

---

## §1. CRITICAL — documented failure history that constrains two of the four recommendations

`AGENTS.md` records specific, expensive failures on **this exact model**.
Two of the original plan's four recommendations walk directly into them.
Neither is necessarily wrong — but the plan must address them explicitly,
or a future agent will either repeat the failure or wrongly revert the
work on sight of AGENTS.md.

### 1a. Recommendation 2 (Augmented Lagrangian) vs AGENTS.md §4.10

AGENTS.md §4.10 is titled **"`RBE2HingeElement` was already tried and
abandoned for driving the plate — don't reintroduce it"**, and the
mechanism it abandoned is *penalty + Augmented Lagrangian*. The recorded
symptom: the tie's gap-to-displacement ratio saturated at **~40%
regardless of penalty stiffness across 8 orders of magnitude
(1e2 → 1e10)**, with Newton iteration count flat at 3 — ruling out
ill-conditioning. Root cause: the compliance was **not in the tie** but
in the AL-constrained rigid-body link, so the tie was chasing a
never-quite-rigid target.

**Why S2S AL may still be legitimate** (this is the argument the plan
must make explicitly, not assume): the §4.10 failure was *architectural*
— a compliant plate constraint upstream of a stiff tie. The current
architecture no longer has that: the plate is made **exactly** rigid by
kinematic condensation (`dispsolver/constraint/rbe2_condensed.py`,
`KinematicRBE2Constraint` — slave DOFs eliminated, zero compliance by
construction, see AGENTS.md §4.10's fix and §2's table). AL applied to
*contact between an exactly-rigid plate and the display* is a different
problem from AL applied to *making the plate rigid*.

**Action required in the plan**: add an explicit note stating the above,
so that (i) the implementer doesn't reintroduce AL on the rigid-body
constraint, and (ii) a reviewer reading AGENTS.md §4.10 doesn't revert
the contact AL by mistake. Also update AGENTS.md §4.10 with a
back-reference once implemented.

### 1b. Recommendation 1 (Adaptive Penalty Scaling) vs AGENTS.md §4.12

AGENTS.md §4.12 documents tip-element inversion (`det(F) ≈ -0.10`) at
47.9°/side, and its explicit, counterintuitive conclusion:

> **Counterintuitive but important**: stiffening the tie (or adding
> Augmented Lagrangian to make it exactly rigid too) makes this *worse*,
> not better — it forces the display bottom row even more tightly onto
> the rigid plate line, concentrating *more* shear into the tip element.
> [...] **try mesh grading toward that boundary before reaching for
> tie-stiffness changes** — this session's evidence is that stiffness
> changes there are counterproductive.

The proposed auto-scaling law is `k_n = α_scale · E · L_segment / A_element`.
`A_element` appears in the **denominator**, so `k_n` goes **up** where
elements are small — and the current mesh is deliberately graded *fine*
(0.25 mm) exactly at the free tips and the plate/hinge boundary
(`_graded_display_x()` in `examples/gen_ex12_inp.py`), which is
precisely where §4.12 says extra stiffness causes inversion.

**Action required in the plan**:
- Add a **cap** on `k_n` (absolute, or relative to the current working
  `k_tie = 1e4` — note `run_folding_from_result()` currently *overrides*
  every tie's stiffness to `1e4` at solve time, so whatever the config
  says is not what runs; see `examples/ex12_abaqus_inp_plate_fold.py`).
- Add an explicit acceptance gate: **the full 90°/side fold must still
  reach closure with `n_inverted == 0`** after adaptive scaling is
  enabled. This is the exact failure mode §4.12 warns about, and it does
  not show up in a 5-step smoke run.

---

## §2. CRITICAL — the verification plan would pass while contact does nothing

This is the single most important gap. The original plan's verification is:

```bash
pytest tests/test_s2s_contact.py -q
python -u examples/DispFoldApp.py --mode build --contact s2s_penalty --max-steps 5
python -u -m verification.run_all --elem_jit jax
```

**All three of these can pass with contact silently inert.** This is not
hypothetical — it is exactly AGENTS.md §4.8, marked `[CRITICAL, silent]`:

> `DynamicSolver._assemble()` has an if/elif chain [...] the tie-penalty
> application (`self.penalty_constraints`) [...] lives in one shared
> block at the very end [...] But the `use_j2_batch` and
> `use_multi_material_batch` branches used `return` [...] Any model using
> a multi-pid / multi-material setup [...] got **zero tie force and zero
> tie stiffness, silently**, for the entire run.
> **Symptom**: Newton converged perfectly (small residuals, no cutbacks,
> "Increment Converged" every step) [...] but the *display*'s
> displacement was **exactly `0.0` at every node, every step**.

The folding model routes through `_assemble_multi_material_batch` — the
same branch. A new S2S contact object added to `penalty_constraints`
sits in the identical risk position. The guard comment is still in place
(`dynamic.py:3450-3451`, "must NOT `return` directly here"), but any new
assembly branch, or adding contact to a different list, silently
reintroduces it.

AGENTS.md §4.9 states the rule directly: **"Convergence success is not
proof of physical correctness — use `diagnostics.py`."**

### Required verification additions

1. **Non-inertness assertion (unit level).** Construct a configuration
   with a *known* penetration and assert the contact force is nonzero
   and of the expected sign/magnitude. A test that only checks "solve
   converged" is worthless here.

2. **Contact patch test** (the standard contact verification, analogous
   to the element patch test already in `verification/benchmarks.py`):
   two blocks pressed together with a uniform pressure must transmit
   that pressure **exactly**, with no spurious oscillation across the
   interface — including with **non-matching meshes**, which is the
   whole point of S2S over NTS. This is the single most diagnostic test
   for an S2S implementation and it is missing from the original plan.
   Suggest registering it in `verification/benchmarks.py` (see §4).

3. **Runtime physical-correctness check.** Enable
   `dispsolver/solver/diagnostics.py`'s `sanity_report()` /
   `check_region_tracking()` during the full-fold run (the pattern
   `examples/ex12_rigid_plate_display_fold_corotational.py` already
   uses — it calls `sanity_report()` every step and `break`s on
   failure). Add the contact's driving/driven DOF groups to it.

4. **Full 90°/side closure, not `--max-steps 5`.** Both §4.12's tip
   inversion (47.9°) and any contact chattering appear late in the fold.
   A 5-step run reaches ~2° and proves nothing about contact behavior.
   The gate is: `reached_target=True`, zero cutbacks, `n_inverted == 0`.

5. **A/B against the current tie.** The model presently uses
   `SurfaceTieConstraint` and reaches full closure. Any S2S contact
   result must be compared against that baseline (tip slip, hinge
   curvature, final shape) — a contact model that converges to a
   *different* answer than the working tie needs that difference
   explained, not assumed to be the improvement.

---

## §3. Stabilization — reuse the existing energy-fraction controller

The proposed proximity-gated damping is genuinely new and worth having:

$$\mathbf{F}_{stab} = -c_{stab}\left[1 - \frac{g_n}{d_{stab}}\right]_+ \mathbf{v}_{rel}$$

It differs from the existing `AbaqusViscousStabilization`
(`F_stab = c·M_diag·v`) in two useful ways: it is **gated by proximity**
(active only within `d_stab`) and uses **relative** velocity across the
interface rather than absolute nodal velocity. Keep it.

**But reuse the existing damping-magnitude controller rather than
inventing a second one.** `AbaqusViscousStabilization` already
auto-scales `c` to hold `E_stab / E_strain < 0.5%` — the Abaqus
convention, and the correct way to pick `c_stab`. Setting
`stabilization_damping: 1e-3` as a fixed config number (as the original
plan does) means the artificial energy introduced is unknown and
unbounded. Requirements:

- Track and **report** accumulated stabilization energy as a fraction of
  strain energy every step, and warn/fail past the threshold. Without
  this, stabilization silently converts a divergence into a *wrong
  converged answer* — the worst failure mode, and one that looks like
  success.
- Recommendation 3 (ramp-down `c_stab(t) = c_0·max(0, 1 - t/Δt_ramp)`)
  is good and should be kept, but note that a **time-based** ramp is
  awkward for a quasi-static solve driven by an adaptive `dt`
  controller (`AdaptiveDtController`); the fold's `t ∈ [0,1]` is
  pseudo-time, and contact engagement does not necessarily happen early.
  Consider ramping on **contact-state stability** (e.g. decay once the
  active set stops changing between iterations) rather than on `t`.

---

## §4. Integration with the verification-suite extension (already planned)

A separate plan already exists and is approved:
`dev_log/plan_verification_suite_extension_20260728_2248.md` — it adds
convergence-order studies and a JAX-vs-NumPy speed benchmark to
`verification/`. The contact work should land in that same framework
rather than inventing a parallel one:

- Add **`contact_patch_test`** to `verification/benchmarks.py`
  (§2.2 above) with a tight tolerance — this is a correctness gate, and
  it belongs in the mandatory `verification.run_all` suite per
  `verification/RULES.md`.
- Contact adds cost per Newton iteration; the speed benchmark from that
  plan should gain a contact-enabled variant so the cost is measured,
  not guessed.
- `verification/RULES.md` must list contact under the mandatory
  post-change gate — it already covers `dispsolver/constraint/`, but
  **`dispsolver/contact/` is not in its five-directory list**
  (`solver/`, `element/`, `material/`, `constraint/`, `mesh/`). Add it,
  or the new code is outside the project's own change-gate policy.

**Stale doc found while reviewing** (unrelated to this feature, flag
separately): `verification/RULES.md:86-88` states "The dispsolver
codebase does **not** use Numba. Only NumPy and JAX backends are
available." — but `dispsolver/element/q4_numba.py` now exists and
`examples/DispFoldApp.py` exposes `--elem_jit numba`. RULES.md's own
text anticipates this ("If a Numba backend is added, register it in..."),
so it just needs updating.

---

## §5. Recommended sequencing (the original plan is one big step)

Given §2's silent-failure history, land this in gated milestones rather
than all at once. Each milestone must pass its gate before the next
starts.

| Milestone | Scope | Gate |
|---|---|---|
| **M0** | Determine the fate of the existing `ContactPair` (§0.2). Write one test that exercises it as-is, to find out whether the existing NTS path works or is broken. | Known-good or known-broken — documented either way. No new code yet. |
| **M1** | S2S Gauss-point penalty contact only. No AL, no adaptive scaling, no stabilization. Fixed `k_n`. | `contact_patch_test` passes with **non-matching meshes**; non-inertness assertion passes; full 90° fold reaches closure with `n_inverted == 0`; A/B vs. the `SurfaceTieConstraint` baseline explained. |
| **M2** | Proximity stabilization damping + energy-fraction reporting (§3). | Stabilization energy < 0.5% of strain energy, reported every run; fold still reaches closure. |
| **M3** | Augmented Lagrangian option (§1a). | Residual penetration < 1e-6 mm; AGENTS.md §4.10 note added; fold still closes. |
| **M4** | Adaptive penalty scaling (§1b). | `n_inverted == 0` at full closure **with** scaling on — the §4.12 regression gate. `k_n` cap in place. |

Recommendation 4 (initial overclosure / gap smoothing at `t=0`) is
low-risk and can fold into M1.

---

## §6. Open questions for the author of the original plan

1. **What problem is S2S contact solving that the current
   `SurfaceTieConstraint` does not?** The tie currently reaches full
   90°/side closure. Contact allows separation and sliding, which a tie
   forbids — is that the goal (i.e. the display should be able to lift
   off the plate), or is this about the multi-plate ("Plate C" in the
   plan's diagram) configuration that doesn't exist in the model yet?
   The plan's ASCII diagram shows **three** plates (L/C/R) but the
   current model has **two**. If a center plate is being added, that is
   a separate model change with its own geometry/config work
   (`create_folding_plate_parts()` builds exactly left+right), and it
   should be stated as a prerequisite.
2. **Friction**: `friction_coeff: 0.1` appears in the proposed config
   but no friction formulation appears anywhere in the plan's math
   (no tangential return-mapping / stick-slip). Either drop the
   parameter or add the formulation — a config knob that nothing reads
   is worse than nothing.
3. **`--max-steps 5`** in the verification section: see §2.4. Was this
   intended as a smoke test only, with a full run implied elsewhere?

---

## Summary of what this review adds

- Contact/stabilization subsystems **already exist**; locate the work in
  `dispsolver/contact/`, reuse the broad-phase/surface/`.inp` plumbing,
  and decide the fate of the existing `ContactPair` (§0).
- Two of the four recommendations collide with documented failures
  (AL vs AGENTS.md §4.10; adaptive stiffness vs §4.12's "stiffening
  makes inversion worse") — both need explicit justification and
  regression gates, not silent adoption (§1).
- The proposed verification would pass while contact is **completely
  inert** — the exact §4.8 failure signature. Needs non-inertness
  assertions, a non-matching-mesh contact patch test, runtime
  `sanity_report()`, and full-closure (not 5-step) runs (§2).
- Reuse the existing `E_stab/E_strain < 0.5%` auto-scaling instead of a
  fixed damping constant; reconsider time-based ramp-down under an
  adaptive-`dt` quasi-static solve (§3).
- Land it in gated milestones; add `dispsolver/contact/` to
  `verification/RULES.md`'s mandatory-gate directory list (§4, §5).
