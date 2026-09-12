# 3D Solver — Missing/Unimplemented Capabilities (consolidated, 2026-09-13)

Everything below was found and directly verified during this session's
`benchmark_element/` work (not carried over from stale claims — each item
lists what was actually checked). Grouped by kind, roughly priority-ordered
within each group by how many benchmarks/use-cases it blocks.

## A. Missing material models (3D solid path, `dispsolver/material3d/`)

1. **No real hyperelasticity.** `MAT_HYPERELASTIC_NEOHOOKEAN` in
   `numba_materials.py` (lines ~90-116) computes `S = C_mat @ E_voigt` —
   bit-identical to `MAT_CUSTOM_ELASTIC`/`MAT_LINEAR_ELASTIC` at the same
   (E, nu), verified directly this session (diff = 0.0 exactly). No
   `ln(J)` or `(J-1)` volumetric term, no deviatoric-invariant-based
   stored-energy function. Blocks: `bmk_rubberdisk`, `bmk_elasticsheet`
   (also 2D), any large-strain rubber/soft-tissue problem. Also relevant
   to why `C3D8_FBAR`'s ill-conditioning at extreme nu isn't mitigated by
   switching "material type" — there's no actual hyperelastic law to
   switch to yet, a genuine `W_vol(J)` form would likely condition much
   better near nu->0.5 than a linear `C` matrix driven through finite-
   strain kinematics.
2. **No composite/layered/orthotropic material.** `numba_materials.py`'s
   dispatch table is isotropic-only (`MAT_LINEAR_ELASTIC`,
   `MAT_HYPERELASTIC_NEOHOOKEAN` (fake, see above), `MAT_J2_PLASTICITY`,
   `MAT_VISCOELASTIC_PRONY`, `MAT_CUSTOM_ELASTIC`) — no per-layer
   orthotropic stiffness, no layup stacking. Blocks: `bmk_thickcompcyl`
   (the single best-instrumented candidate in the static-analysis list —
   exact tabulated reference numbers already in hand, just needs this),
   `bmk_compositeshells`.
3. **No Gurson (porous-plasticity) material.** Blocks: `bmk_neckingtensilebar`
   (though that one also has no clean closed-form number even with it —
   lower priority than the two above).
4. **`MAT_VISCOELASTIC_PRONY` exists but is undispatched/error-flagged**
   in at least one path per the earlier 3D-audit doc
   (`dev_log/3d_element_defect_audit_20260912.md`) — not re-verified this
   session, flagging as carried-forward/unconfirmed rather than re-stating
   as fact.

## B. Missing constraint/contact capability (`dispsolver/constraint3d/`)

5. **No contact module at all.** The directory has only
   `rbe3_distributing3d.py` and `surface_tie3d.py` (a bonded tie, not a
   frictional/frictionless contact pair) — confirmed by directory listing
   this session. Blocks: `bmk_hertzcontact`, `bmk_pipecrushing`, and any
   future problem needing separable/normal-only contact (this project's
   2D side has real contact history — see AGENTS.md §4.10's RBE2-vs-
   RigidBodyPart lesson — 3D has nothing equivalent yet).

## C. Missing solver capability (`dispsolver/solver3d/dynamic3d.py`)

6. **No Riks/arc-length method.** `grep -i "riks\|arc.length"` returns
   nothing in `dynamic3d.py`. Only plain load/displacement-controlled
   Newton with a fixed-direction line search exists. Blocks:
   `bmk_roofsnapthrough` (the actual snap-through instability point
   fundamentally cannot be traced without it — load-controlled Newton
   diverges exactly at the limit point by construction), and any future
   buckling/post-buckling/snap-through problem in general.
7. **`solve_step()` / `_solve_step_with_hooks()` false-convergence bug —
   FIXED this session, noted here for completeness/traceability.** Both
   used to unconditionally `return True, max_iters` when the iteration
   budget ran out regardless of residual size, masking real non-
   convergence as false success. Fixed
   (`dev_log/solve_step_false_convergence_20260913.md`). Worth a
   follow-up sweep: check whether any OTHER Newton loop in this codebase
   (2D `dynamic.py`, any other 3D entry point) has the same
   `return True` fallthrough pattern — not checked this session outside
   these two functions.
8. **`nlgeom=False` (linear-geometry) assembly path was removed
   entirely** from `dynamic3d.py` (raises `NotImplementedError` — see
   `dev_log/3d_element_defects_and_kernel_renovation_plan.md` for when/why).
   Every `benchmark_element/` test runs with `nlgeom=True` even for
   nominally-linear problems, which works at these small strains but
   means there is currently no fast small-deformation-only 3D path at
   all. Not necessarily worth reinstating (a correctly-behaving nlgeom=True
   path is more general), but worth knowing this option doesn't exist if
   a future benchmark wants a cheap linear reference solve.
9. **No per-Gauss-point Cauchy-stress-recovery API.** Confirmed absent in
   both `dynamic3d.py` and `dispsolver/element3d/*.py` (checked when
   writing `run_abaqus_official_patch_test`, re-confirmed for the
   `rad_stretch` design). Every benchmark in this suite can only check
   DISPLACEMENT fields, never stress, even when the real Abaqus reference
   (e.g. the patch test's own Step 1, or `rad_stretch`'s exact
   `sigma_rr(r)`/`sigma_thetatheta(r)`) is stated in terms of stress. A
   general per-element stress-recovery helper (extend
   `mechanics_patches.compute_per_element_dilatation`, added this session
   for the ANP checkerboard test, into a full Cauchy-stress version) would
   benefit essentially every benchmark in this folder at once — the
   single highest-leverage piece of new benchmark infrastructure
   identified this session.

## D. Known element-specific gaps (not blocking, but real)

10. **`C3D8I` is missing 4 of its 13 real internal DOFs.** Real Abaqus
    `C3D8I` has 9 principal alpha_i modes + 4 beta_j volumetric-locking-
    relief modes (`reference_abaqus_docs/incompatible.txt`). This
    session's fix implemented only the 9 alpha_i modes (fixed the dead-
    EAS/patch-test/bending-locking failures) — the missing beta_j modes
    are the likely reason its volumetric-locking verdict is `STIFFENED`
    rather than `INCOMPRESSIBLE-OK` post-fix.
11. **`C3D8H`'s volumetric ratio (0.55) sits just under this benchmark
    suite's own 0.60 bar** — flagged as possibly a benchmark-design
    limitation (the cantilever-bending metric conflates shear-locking-
    limited stiffness with volumetric relief for an element that only
    cures the latter) rather than a kernel defect; not resolved either
    way.
12. **`C3D4_ANP`'s fix is verified non-trivial but unverifiable by the
    current benchmark geometry** (smooth cantilever bending doesn't
    develop element-to-element pressure disagreement for ANP to relieve).
    A proper verification needs either an irregular/unstructured mesh or
    a direct per-element pressure-variance measurement under confined
    compression — a mesh generator (`make_confined_compression_mesh`) and
    a generic per-element dilatation post-processor
    (`compute_per_element_dilatation`) were added to
    `benchmark_element/mechanics_patches.py` this session for exactly
    this, but the actual test function
    (`run_anp_checkerboard_test`, in `benchmark_3d_mechanics.py`) was
    **not finished** — this is the most immediately-actionable unfinished
    item from this session's own work, separate from any `dispsolver`
    capability gap.
13. **`C3D8_FBAR` has a narrow (nu~0.4999-0.49992) Newton-instability
    band**, and at the benchmark's own extreme test point (nu=0.49999,
    just past that band) it eventually reports `converged=True` but to a
    wrong, unphysical answer (ratio -16.1) rather than a slow-but-correct
    one. Root cause not identified (ruled out: FD-tangent step size,
    which is already correct; suspected but not confirmed: catastrophic
    cancellation in the residual/stress evaluation at extreme bulk
    modulus, or a genuine equilibrium non-uniqueness for this specific
    mesh/loading). Not pursued further per explicit user direction (nu
    beyond 0.499 is understood to be outside this project's practical
    range).
14. **Unexplained: `C3D8_CR` and `C3D8H` now measure bit-identical
    (1e-12) on bending and volumetric tests** despite being different
    formulations (corotational vs. hybrid-pressure). Flagged in the
    auto-generated report's dispatch-anomaly section, not yet
    investigated — could be a coincidence at this specific small-strain
    linear-elastic test point (both reduce to the same answer when
    neither's special mechanism is engaged), or a real dispatch/kernel
    issue worth checking per this project's own precedent (AGENTS.md
    §4.2/§4.8/§4.16 — identical numbers between different elements is
    usually not a coincidence).

## E. Longer-standing, pre-existing (not new this session, listed for completeness)

15. **2D vs 3D material-interface convergence** — 2D's JAX-closure
    `response_fn(F, state, dt, params)` vs. 3D's Numba int-tag + flat-
    array convention remain architecturally separate (see
    `dev_log/plan_2d_3d_interface_alignment_20260912.md` and
    `[[project_numba_production_focus]]` memory) — unrelated to this
    session's benchmark work but the two would need to converge before
    e.g. a shared material library (real hyperelasticity, once built)
    could serve both 2D and 3D without being written twice.

## Update: items 12, 9, 1 executed (2026-09-13, later same day)

Via `.omc/plans/3d_capability_buildout_20260913.md` (see that file for
full results, independently re-verified):
- **Item 1 (real hyperelasticity): DONE.** `MAT_HYPERELASTIC_NEOHOOKEAN`
  is now a genuine finite-strain Neo-Hookean law, no longer bit-identical
  to linear elastic. `tests/test_neohookean_3d.py`.
- **Item 9 (stress recovery API): DONE.** `dispsolver/postprocess3d/field_output.py`,
  covering C3D4/C3D4_ANP/C3D8/C3D8R/C3D8_CR (not yet the other 6 element
  types — extending coverage is a smaller follow-up, not a new capability
  gap). `tests/test_field_output.py`. Also surfaced a subtle, real
  finding: this solver's patch test always runs through finite-strain
  (`nlgeom=True`) kinematics, so a naive stress comparison against
  Abaqus's small-strain reference numbers picks up a genuine O(eps)
  correction — not a bug, but anyone extending stress checks to other
  benchmarks needs to account for this (see the plan doc's T2 entry).
- **Item 12 (ANP checkerboard verification): attempted, still open.**
  `run_anp_checkerboard_test`/`run_anp_checkerboard_comparison` added,
  but the confined-compression mesh tried ALSO fails to trigger real
  checkerboarding (plain C3D4's own baseline variance is ~zero on it) —
  same root issue as the original cantilever-bending attempt. Honestly
  xfailed, not chased further this round.

## Suggested priority if picked up next

**User direction (2026-09-13): composite/orthotropic material (item 2)
and Gurson porous plasticity (item 3) are explicitly deprioritized** —
don't pick these up without being asked again, even though
`bmk_thickcompcyl` has exact numbers waiting on the composite-material
one. This overrides the "item 3, unlocks thickcompcyl" ranking below.

Remaining items, highest-leverage-per-effort:
1. Finish `run_anp_checkerboard_test` (item 12) — infrastructure already
   built this session, just needs wiring + a test run. Smallest effort,
   closes out this session's own loose end.
2. Per-Gauss-point Cauchy stress recovery (item 9) — unlocks stress-level
   checking for every existing and future benchmark at once, including
   the already-designed `rad_stretch` benchmark.
3. Real hyperelasticity (item 1) — unlocks `bmk_rubberdisk`-class
   problems and is independently useful for `C3D8_FBAR`'s conditioning
   question (item 13) if ever revisited.
4. Contact (item 5) and Riks/arc-length (item 6) are the largest
   individual efforts and each unlock exactly one benchmark from the
   13 reviewed — lower priority unless a specific problem needs them.

De-prioritized, per direction above (don't start without being asked):
- Composite/layered/orthotropic material (item 2) — `bmk_thickcompcyl`
  stays blocked.
- Gurson porous plasticity (item 3) — `bmk_neckingtensilebar` stays
  blocked (low value anyway, no clean closed-form number even with it).
