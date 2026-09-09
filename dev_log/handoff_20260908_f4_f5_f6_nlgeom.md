# Handoff — 2026-09-08 session (F4/F5/F6 fixes, CPE4, NLGEOM)

Resume point for the Abaqus/OptiStruct 1:1 element redevelopment. The
governing plan is `dev_log/plan_abaqus_element_consolidation_20260908.md`
— **read its top CORRECTION block first**, it carries the user's standing
directive ("영구 예외 없음, 전면 재검토, abaqus 요소로 1:1 개발").

Nothing is committed. The user's instruction was to commit once, after the
redesign lands — that has not happened yet.

---

## 1. What was fixed (all three were live in production)

### F4 — missing work-conjugacy push-forward
PK2 computed from the TOTAL deformation gradient was contracted directly
with `BL(F_inc)` / `w=detJ` built on the step-n configuration. The stress
must be pushed forward first: `S_n = F_n S F_n^T / det(F_n)` (no-op when
`F_n = I`, so TL was always fine and the bug was invisible there).
Measured up to **24.9%** internal-force error on a homogeneous state.

Fixed in 7 places: `q4_visco_simo_fs_jax.py::_internal_force`,
`q4_visco_eas_jax.py::_residuals` and `::_residuals_h`,
`q4_visco_hybrid_reduced_jax.py::_reduced_hybrid_force`, plus the numba
mirrors in `q4_visco_eas_numba.py`, `q4_visco_hybrid_up_numba.py`,
`q4_visco_hybrid_reduced_numba.py`. (`q4_visco_hybrid_simo_numba.py`
needed none — it has no `F_n` parameter and its dispatch is
unconditionally TL.)

### F5 — `_ul_F_n` compounding inside a Newton step
`self._ul_F_n[elem_indices] = F_n_new` was written on EVERY assembly call,
not just on convergence, so iteration 2 evaluated `F_inc @ (F_inc @ F_n)`.
Removed all 5 write sites in `dynamic.py`; the commit-time resync
(`dynamic.py` ~2614-2632) is now the sole authoritative write.

### F6 — EAS enhanced-mode transpose (the big one)
`(detJ0/detJ) · D_k · J0^{-1}` must be `· J0^{-T}`. Index-TYPE mismatch:
`inv(J0)` has rows spatial / cols natural, `D_k`'s second index is
natural. **Invisible on axis-aligned elements** (diagonal `J0` makes the
two identical) — which is every existing test — but the UL reference
configuration rotates during the fold, and then:

    ref rotation:   0°     10°     20°     30°     45°     60°     75°   90°
    AR 7   before: 1.000  9.738   0.749   3.936  11.419  10.744  3.910  1.000
    AR 14  before: 1.000  9.105   5.326  25.774  48.194  40.089 13.908  1.000
    AR 28  before: 1.000  1.438  46.230 125.287 195.218 153.864 52.284  1.000
    (after the fix: exactly 1.000 in every cell)

Peak **195×** bending-stiffness error, erratic in both angle and aspect
ratio (0.749 = 25% too SOFT at one point). Fixed in all 5 copies:
`q4_eas_jax.py` (which `q4_visco_eas_jax.py` imports, so CPE4I/CPE4H/
CPE4IH are covered), `q4_eas.py`, `q4_eas_numba.py`, and the
hand-expanded `q4_visco_eas_numba.py` (`i12`↔`i21` swap). Cache version
bumped to `v3` in `_jit_cache.py` so no stale numba kernels survive.

Note: my own first hypothesis for the fix (`J0 D_k J0^{-1}`) was WRONG —
it is numerically identical to the buggy code (left-multiplying every
mode by a constant invertible matrix does not change the span the
condensation searches). The opus review caught that.

---

## 2. Evidence the F6 fix is real, not a degradation

`§4.8` of AGENTS.md warns that clean convergence can mean "the physics
quietly stopped happening". Two independent discriminators were run:

- **Interlayer slip (opus's Tier 4, the adjudicating measurement)**:
  post-fix ex12 gives **68.04 µm staircase at 4.08°**, 99.3% carried by
  PSA. Pre-fix reference (`dev_log/interlayer_shear_verification_20260726.md`)
  was 81 µm at a FULL 90° fold. Slip grew enormously at a fraction of the
  angle — the direction that only "PSA is no longer artificially stiff"
  explains. A degraded (plain-Q4) element would have slipped LESS.
- **Fold progress**: pre-fix runs collapsed in line search around step
  6-8 (~0.07°); post-fix reached 19 steps / **4.08°** with 10 cutbacks
  (`scratch/ex12_after_f6.log`). Encouraging, but per opus this is NOT by
  itself validation — see the open item below.

---

## 3. New: real CPE4 + NLGEOM architecture

### `dispsolver/element/cpe4_jax.py` (new)
The real Abaqus CPE4, rebuilt to the target architecture: full 2×2
integration + centroid-sampled volumetric treatment (F-bar, the
finite-strain form of the B-bar that Abaqus's fully-integrated first-order
solids already use), finite strain, UL-capable, F4 push-forward built in,
and **material-agnostic** — the constitutive response arrives as a
callable (`response_fn(F, state, dt, params)`, the Phase A interface).
Two entry points: autodiff tangent, and a material-supplied-tangent
variant for J2 (whose spectral return map is unsafe to autodiff through).
`tests/test_cpe4_element.py`: **24/24**, including the global-axis
isotropy gate that SRI and pre-fix EAS both fail.

Note an earlier claim in the plan doc that `q4.py`'s B-bar made it "not
really CPE4" was **retracted** — Abaqus CPE4 does apply exactly that
volumetric treatment. The user caught this.

### NLGEOM replaces the global `ul_mode` flag
`DynamicSolver(nlgeom=...)` is now the switch; `ul_mode` /
`ul_large_rotation_mode` remain as deprecated aliases. Whether a given
element actually runs UL is decided per pid by `_use_ul_for(pid)` against
the new `_ELEMENT_LARGE_DEF` table, which declares `supports_ul` /
`frame_invariant` / a reason for every element type. Unknown element
types are refused UL (conservative by design).
`solver.element_large_deformation_report()` prints what each pid resolves
to. Verified: with `nlgeom=True`, `CPE4I` → UL, `Q4_COROTATIONAL_SRI` →
TL (correctly, since its Cartesian shear sampling is not frame-invariant).

This closes the structural hazard that let F6 stay live: a single global
flag was dragging elements onto a formulation they cannot support.

---

## 4. Open items — pick up here

0. **[SUPERSEDED 2026-09-09 -- read this before item 1 below]** Item 1's
   conclusion ("genuine physical demand, not instability -- the growth was
   clean, no oscillation") **was wrong and has been retracted**. The clamp was
   bounding a DIVERGENT element-local Newton, not a physical demand: on the
   tilted+distorted UL probe the undamped local Newton fails to converge in 25
   iterations while `K_aa` stays positive definite, and a backtracking line
   search converges the same state in 10 iterations to an `|alpha|` an order of
   magnitude SMALLER. `_ALPHA_MAX` has been removed entirely and replaced by
   (a) a line-search damped, tolerance-terminated local Newton and (b)
   element-local non-convergence surfaced as a global increment cutback. See
   **`dev_log/eas_stabilization_modernization_20260909.md`** and
   `dev_log/plan_abaqus_element_consolidation_20260908.md` §11. Also note the
   full-suite baseline quoted in section 5 below is stale: the current
   same-machine reference is 3 failed / 264 passed pre-change, 2 failed / 265
   passed after.

1. **[RESOLVED 2026-09-08, same session -- but see item 0 above, this
   resolution was overturned] alpha clamp saturation measurement.**
   A 25-step ex12 fold (`scratch/alpha_sat2.log`, `_ALPHA_MAX=0.05`)
   showed `max|eas_alpha|` growing smoothly and monotonically
   (9.99e-7 at step 1 to 4.41e-2 at step ~15), then **hitting the clamp
   exactly at step ~18** (theta ~3.5deg) and pinning at `5.0000e-02` for
   the rest of the run, with the saturated-element fraction climbing to
   **8.8%**. This confirmed real, physically-driven demand (not
   instability — the growth was clean, no oscillation) exceeds the old
   clamp at very early fold angles.

   Git-blame showed `_ALPHA_MAX` was tightened `5.0 -> 0.05` in commit
   `f65f90d` (2026-07-30), most likely to survive the pre-F6-fix
   transpose bug's nonlinear blow-up under UL rotation (up to 195x
   bending error, AGENTS.md sec 4.14) — i.e. the tight clamp was very
   likely a band-aid for a bug that is now actually fixed.

   **Fix applied**: raised `_ALPHA_MAX` 0.05 -> 0.5 (10x headroom) in
   `q4_eas_jax.py`, `q4_visco_eas_jax.py`, `q4_visco_eas_numba.py`
   (kept consistent across all three, per the plan). Bumped
   `DISPFOLD_CACHE_VERSION` v3 -> v4 in `_jit_cache.py` (numba on-disk
   cache would otherwise keep serving the old-constant kernel).

   **Re-verification**:
   - `tests/test_cpe4_element.py` + `tests/test_ul_tl_consistency.py`:
     32/32 passed.
   - `scratch/tier1_numba_jax_tilted.py`, A/B'd against the old 0.05
     value directly (temporarily reverted, re-tested, restored): at the
     "large" perturbation scale that used to saturate, numba/JAX force
     error on the TILTED+DISTORTED UL case dropped from **52%
     (5.202e-01) to 4.35e-04** (`max|alpha|` now 0.107-0.205, comfortably
     under the new 0.5 clamp instead of pinned at the old 0.05). Axis-
     aligned case: 5.2e-3 -> 8.2e-12. TL case: 4.1e-6 -> 8.8e-15. The
     tangent-only `K_err` metric stays large (order 1-1000) in the UL
     case at every scale tested, including well below either clamp —
     this is a pre-existing, clamp-independent property of the numba
     path's FD-built tangent (its own docstring: "FD-tangent throughout,
     no analytic tangent") vs JAX's exact autodiff tangent; per AGENTS.md
     sec 4.4 (modified-Newton precedent) a mismatched *tangent* only
     costs Newton iterations, not correctness — the force is the
     correctness gate, and that is the number that improved by 3 orders
     of magnitude.
   - `scratch/alpha_sat3.log` (PID 1545, 40-step re-run with the new
     clamp): launched, not yet inspected — check this before Family B if
     picking this up fresh; confirm `max|a|` clears 3.5deg without
     re-saturating at the new 0.5 bound.
   - `scratch/pytest_full_alpha05.log` (PID 1557): full `pytest tests/`
     re-run launched after the raise, not yet inspected.

   **Conclusion**: alpha-clamp concern closed for the range tested (up
   to ~4deg fold with old data, re-confirming further with alpha_sat3).
   Safe to proceed to Family B once alpha_sat3/pytest results are in.
2. **Family B — migrate PET/GLASS off the invented SRI device onto
   CPE4I.** Now clearly supported: CPE4I is exact at every aspect ratio
   (1.000) vs SRI's uniform 1.225×, and F6 (which had made CPE4I look
   worse) is fixed. SRI is also only safe TL + axis-aligned; the warning
   is now documented in `q4_sri_jax.py::_F_sri`.
3. **opus's remaining re-verification tiers** (from the
   `opus-eas-isotropy` review, partially relayed into the plan doc):
   Tier 0 — land the isotropy / rotated-reference-AR-sweep / patch-test
   probes as permanent tests; Tier 2 — full `pytest tests/` + `python -m
   verification.run_all` (mandatory per `verification/RULES.md`);
   Tier 3 — `sanity_report()` every step per AGENTS.md §4.9.
   Discriminators 4-5 of his list were never received (agent
   `opus-eas-isotropy` still had them queued).
4. **AGENTS.md edits opus recommended** (not yet made): annotate §4.1's
   AR sweep as *axis-aligned reference* and record the rotated-reference
   sweep beside it; fix the stale claim that `pet_element_type = "Q4_EAS"`
   is canonical (config says `Q4_COROTATIONAL_SRI`); add a §4.x entry for
   the generalizable lesson — **element verification performed only on
   axis-aligned rectangles cannot detect a frame-covariance defect**,
   which belongs next to §4.9.

## 5. Test/verification state at session end

- `tests/test_cpe4_element.py` 24/24, `tests/test_cpe4rh_patch.py` 19/19,
  `tests/test_ul_tl_consistency.py` 8/8, `tests/test_ul_fn_idempotence.py`
  1/1, `tests/test_response_interface_phase_a.py` 12/12.
- Full suite (before the NLGEOM change): 234 passed / 5 failed, every
  failure confirmed pre-existing by `git stash` A/B — `test_eas_jax_verify.py`
  ×2, `test_abaqus_config.py::test_config_presets` (stale `dt_max`
  expectation), `test_prescribed_skip.py` (2e-9 vs a 1e-10 tolerance),
  and `test_rbe2_condensation_reconstruction.py` (stale node-id
  expectation — fixed this session).
- `verification.run_all` 12/14, the 2 failures long-documented as
  unrelated (2-Point Bending Gulati Elastica, Convergence Cantilever
  Mesh Q4bbar).
- **Not yet re-run after the NLGEOM change**: full `pytest tests/` and
  `verification.run_all`. Do both before committing.
