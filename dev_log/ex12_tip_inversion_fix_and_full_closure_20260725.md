# Session log: tip-element inversion fix → full 90°/side U-shape closure achieved

**Date**: 2026-07-25 (late session, continuation of the same day's earlier
`walkthrough_20260725.md` / `AGENTS.md` work)
**Status**: Part A of the plan below is **DONE**. Part B is **NOT STARTED** —
this is the handoff point for the next agent/session.

---

## 1. What to read first

- **`AGENTS.md`** (repo root) — canonical cross-tool project rule file.
  Sections most relevant to this log: **§1.0** (U-shape success criteria),
  **§4.10** (RBE2HingeElement regression — already fixed, don't reintroduce),
  **§4.12** (this session's tip-inversion fix, just added).
- **The approved plan file** (Claude Code plan-mode artifact, still on disk):
  `C:\Users\GOODMAN\.claude\plans\async-frolicking-unicorn.md` — has the
  full diagnosis, options-evaluated writeup, and Part B (`ex13`) design
  that has not been implemented yet. If that file isn't visible to you
  (different agent / different machine), this log + `AGENTS.md` §4.12
  contain the same information restated.

## 2. Which files to actually look at for this work

- **`examples/ex12_abaqus_inp_plate_fold.py`** — the primary reference
  example (see `AGENTS.md` §1.0 "Achieved" note). Reads a model from an
  Abaqus `.inp` deck and runs the 90°/side fold. **This is the file to run
  to reproduce the full-closure result.**
- **`examples/gen_ex12_inp.py`** — generates
  `examples/ex12_rigid_plate_display_fold.inp` (the deck the above script
  reads). `_graded_display_x()` in this file is the actual fix (see §3
  below). Run this first if you change display-mesh parameters, then
  re-run `ex12_abaqus_inp_plate_fold.py`.
- `examples/ex12_rigid_plate_display_fold_corotational.py` — a **parallel,
  still-unfixed** model-building path (builds the model directly in
  Python, no `.inp` round trip). Still stalls around 7.9°/side (dt-decay,
  not a hard failure — see `AGENTS.md` §1.2/§4.11). Not the one to build
  on for further physics work, but relevant for Part B (see §5).
- `dispsolver/solver/diagnostics.py` — runtime sanity-check helpers
  (`sanity_report`, `fold_success_verdict`, `check_smooth_curvature`,
  `check_tie_gap`, `check_plate_closure`). **Known limitation found this
  session, not yet fixed**: `check_smooth_curvature`'s discrete
  second-difference assumes uniform node spacing along x; it produces
  false-positive "kink detected" results on a *graded* (non-uniform)
  mesh — e.g. it flagged `x=10.0` (a mesh-spacing transition point, 0.25mm
  → 1mm) as a sharp crease even though the actual `uy` values there are
  smooth and regular (see §4 below for the raw numbers that disprove it).
  If you rely on this function for a graded mesh, either fix it to use the
  actual (non-uniform) x-spacing in the second-difference formula, or
  verify its verdict manually against raw displacement data first.

## 3. What was fixed (Part A of the plan)

**Symptom**: `ex12_abaqus_inp_plate_fold.py` reached 47.9°/side with clean
convergence, then the display's two outermost free-edge elements
(x∈[-40,-39] and x∈[39,40]) inverted (`det(F)≈-0.10`, genuine flip).

**Root cause**: the plate is driven by *exact* RBE2 kinematic condensation
(`dispsolver/constraint/rbe2_condensed.py`, zero compliance) while the
display is only coupled to it via a penalty-only `SurfaceTieConstraint`.
That rigid-vs-compliant stiffness mismatch is absorbed as shear strain,
worst at the two tip elements because — unlike interior elements — they
have no neighbor on their outward side to redistribute it.

**Important, counterintuitive finding**: stiffening the tie (or adding
Augmented Lagrangian to make it exactly rigid) makes this *worse*, not
better — see `AGENTS.md` §4.12 for why. Don't try that first.

**Fix actually applied** (mesh discretization only, no solver/constraint
code touched):
- `examples/gen_ex12_inp.py`: added `_graded_display_x()`, replacing the
  uniform `xs_disp = np.linspace(-40,40,81)` with a clustered x-grid —
  0.25mm spacing near the four transition locations (`x=±40` tips,
  `x=±10` hinge/plate boundary), 0.5mm through the free hinge span
  (`x∈[-8,8]`), 1mm elsewhere. ~120 columns instead of 80.
- `examples/ex12_abaqus_inp_plate_fold.py`: changed the cutback-abort
  guard from `if dt < dt_ctrl.dt_min:` to `<=` — `AdaptiveDtController`
  clips `dt` to exactly `dt_min` via `np.clip`, so the strict `<` guard
  could never fire, meaning a still-failing run would cutback forever
  instead of aborting. (This was a real observed bug in the *previous*
  attempt, before the mesh fix — 335+ cutback loops with no termination.)

## 4. Result (verified, not just claimed)

Ran `python -u examples/ex12_abaqus_inp_plate_fold.py`:

```
STEP 101 | t=1.0000s | dt=0.0075s | Iters=1 | θ=[-89.99997659°, 89.99997659°] | max|u|=37.50mm
SUCCESS: 90-degree display folding simulation completed in 85.28s!
```

- **101 steps, zero cutbacks, 1 Newton iteration per step throughout.**
- `pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q`
  → 6 passed (no regression from the two small changes above).
- Follow-up quantitative verification (separate standalone script, not
  saved as a file — rerun the example itself to reproduce):
  - `solver._check_mesh_quality(solver.u)`:
    `n_inverted=0, n_warped=0, min_detJ=0.96, min_aspect_ratio=1.0`.
  - Tie gap (`check_tie_gap`): mean ≈0.0017mm, max ≈0.0026mm at
    `max|u|=37.5mm` — effectively fully tight.
  - Plate rotation error (`check_plate_closure`): both sides ≈5e-6° off
    target — essentially exact.
  - **Caveat**: `check_smooth_curvature` reported a false-positive "kink"
    at the mesh-spacing transition point `x=10.0` (see §2's diagnostics.py
    note). Manually inspecting the raw `(x_current, y_current)` profile
    for the display's bottom-row hinge-zone nodes confirms a genuinely
    smooth, monotonic teardrop/U curve (no real discontinuity) — matches
    the final PNG (`examples/ex12_final_folding_shape.png`) visually: two
    plates ending parallel and facing each other, connected by a rounded
    U loop. This satisfies all three of `AGENTS.md` §1.0's success
    criteria.

`AGENTS.md` has been updated: §1.0 has an "Achieved" note, new §4.12
documents the fix in full (read that section for the complete root-cause
writeup with file:line citations), and the §5 file map now marks
`ex12_abaqus_inp_plate_fold.py` + `gen_ex12_inp.py` as the primary
reference.

## 5. What's NOT done yet (Part B of the plan — pick up here)

`examples/ex13_unified_model_io.py` (not created): a unified example,
based on `ex12_abaqus_inp_plate_fold.py` (post-fix), supporting three
interchangeable model sources:
1. **`build`**: construct the model directly in Python — reuse
   `create_folding_plate_parts`/`RigidBodyPart`/`SurfaceTieConstraint`
   (as `ex12_rigid_plate_display_fold_corotational.py` does) or
   `gen_ex12_inp.py`'s model-building logic, whichever is closer to the
   `.inp` deck's actual content, so `build` and `read` produce equivalent
   models.
2. **`read`**: read `.inp` via `dispsolver.io.read_abaqus_input` — as
   `ex12_abaqus_inp_plate_fold.py` currently does.
3. **`roundtrip`**: build in Python, **export to `.inp`** (new
   capability — check `dispsolver/io/` for any existing writer to
   reuse/extend before writing one from scratch; `abaqus_parser.py` /
   `model_builder.py` are the read-side to mirror), re-read that exported
   `.inp`, solve, and confirm it matches the `build` model (node/element/
   constraint counts).

Verification plan for Part B is written out in full in the plan file
(§ "Verification" at the end) — summary: `build` and `read` should reach
comparable fold angles; `roundtrip`'s re-read model should exactly match
`build`'s topology; final state should pass the same checks as §4 above
(mesh quality, tie gap, plate closure, visual/manual curvature check —
remember the `check_smooth_curvature` caveat if the graded mesh is
reused).

## 6. Minor known follow-ups (not blocking, not started)

- `dispsolver/solver/diagnostics.py::check_smooth_curvature` should be
  fixed to use actual (non-uniform) x-spacing rather than an index-based
  second difference, or documented more prominently as unsafe on graded
  meshes (see §2).
- `ex12_rigid_plate_display_fold_corotational.py`'s dt-decay stall
  (~7.9°/side) is still unfixed (`AGENTS.md` §1.2/§4.11) — separate from
  everything in this log, which is entirely about the `.inp`-based path.
- The known plate-element-reassembly performance waste (`AGENTS.md`
  §4.11, "Known remaining performance issue") is also still unfixed.
