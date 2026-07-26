# ex13_unified_model_io.py: read + roundtrip modes (2026-07-26)

## Context

Earlier session plan (`async-frolicking-unicorn.md`, Part B) proposed
`ex13_unified_model_io.py` with three interchangeable model sources
(`build`, `read`, `roundtrip`) sharing one solve loop, to consolidate the
parallel `.inp`-based and pure-Python modeling paths. Part A (tip-element
inversion fix) was completed in that session; Part B was never started
(`examples/ex13_*` did not exist). User asked to check status and
continue -- confirmed not implemented, then asked to proceed with
`read` + `roundtrip` first, deferring `build` (would require
re-implementing the current 14-layer PET-PSA/Q4_VISCO_SIMO layup a
second time as raw Python objects -- significant duplication, out of
scope for this pass).

## Implementation

- `examples/ex12_abaqus_inp_plate_fold.py`: `run_abaqus_inp_folding()`
  now takes optional `inp_path`, `before_png_name`, `after_png_name`
  params (defaults unchanged, so direct `python ex12_abaqus_inp_plate_fold.py`
  behavior is identical) and returns a summary dict (`solver`, `result`,
  `n_nodes`, `n_elements`, `n_rbe2_constraints`, `n_penalty_constraints`,
  `reached_target`) instead of nothing -- reused as-is, no solve-loop
  duplication.
- `examples/ex13_unified_model_io.py` (new): `--mode {read,roundtrip}`.
  - `read`: reads the standard on-disk `ex12_rigid_plate_display_fold.inp`.
  - `roundtrip`: calls `gen_ex12_inp.generate()` in-memory, writes the
    text to a temp `.inp` file, then runs the exact same
    `run_abaqus_inp_folding()` path -- proves the generator's text output
    is self-consistent through the reader without needing a separate
    `.inp` writer (none exists in `dispsolver/io/` -- confirmed via
    grep before implementing).
  - `build` mode intentionally NOT implemented (see Context).

## Verification

Both modes run end to end to full closure:
- `read`:      nodes=2003, elements=1800, rbe2_constraints=2,
  penalty_constraints=2, reached_target=True, 292.32s.
- `roundtrip`: nodes=2003, elements=1800, rbe2_constraints=2,
  penalty_constraints=2, reached_target=True, 295.32s.

Identical mesh/constraint counts and both reach the full 90deg/side
target -- confirms `generate()` -> temp file -> `read_abaqus_input()`
round-trips consistently. `pytest tests/test_convergence_fixes.py
tests/test_rigid_plate_tie.py` still 6 passed (ex12 signature change is
backward compatible, no existing caller broke).

## `build` mode added (same session, follow-up)

Implemented `run_build()` in `ex13_unified_model_io.py`: constructs the
identical 14-layer PET-PSA mesh (reusing `gen_ex12_inp._graded_display_x()`
and `create_folding_plate_parts`), materials (`J2Plasticity`,
`ViscoelasticMaterial(ArrudaBoyce(), ...)`, `NeoHookean`), RBE2 hinge
constraints (`KinematicRBE2Constraint`, same as the .inp `*RIGID BODY`
path -- not `RigidBodyPart`/Dirichlet, so it plugs into the same
`theta_targets`-driven solve loop), and `SurfaceTieConstraint` ties --
all as plain Python objects, no `.inp` text anywhere. Refactored
`ex12_abaqus_inp_plate_fold.py`'s solve loop out into
`run_folding_from_result(result, ...)`, taking any object with the same
fields as `ModelBuilderResult` (`mesh`/`materials`/`material_params`/
`solver_config`/`rbe2_constraints`/`penalty_constraints`/`rbe2_elements`/
`boundaries`) -- `build` mode passes a `types.SimpleNamespace` built
directly, `read`/`roundtrip` pass the real `ModelBuilderResult`.

### Critical bugs found and fixed while verifying `build` mode

Comparing `build`'s freshly-constructed `ArrudaBoyce`/`ViscoelasticMaterial`
params against what `read_abaqus_input` actually produced from the same
`.inp` text turned up **two silent parser/builder bugs that had been
active since the PSA-Arruda-Boyce material was introduced**:

1. `abaqus_parser.py::_parse_hyperelastic` detected the hyperelastic
   model type via `params.get("neo hooke", params.get("type", "NEO HOOKE"))`
   -- but Abaqus writes the model name as a bare flag
   (`*HYPERELASTIC, ARRUDA-BOYCE"), which the lexer stores as
   `params["arruda-boyce"] = "yes"`, never checked. Every `ARRUDA-BOYCE`
   block was silently misread as the `NEO HOOKE` default, reinterpreting
   the `mu, lambda_m, D` data row as `C10, D1` -- i.e. the PSA layer's
   base material was actually **NeoHookean(mu=0.3357, lambda=0.4429)**,
   not Arruda-Boyce(mu=0.16785, lambda_m=3.0), this whole time. Fixed by
   checking for each known model's bare-flag key
   (`"arruda-boyce"`/`"arruda boyce"`, `"yeoh"`, `"neo hooke"`) before
   falling back to `type=`/default.
2. `model_builder.py`'s two ARRUDA-BOYCE branches (composite path and
   legacy-flat path) read `hyper.get("K", 100.0)` directly, but the
   parser stores the compressibility parameter as `"D"` (matching the
   `mu, lambda_m, D` Abaqus data-row convention), never `"K"` -- so `K`
   silently defaulted to 100.0 regardless of the actual `D` value.
   Fixed to convert `K = 2.0/D if D > 0 else 100.0`, mirroring the
   NEO HOOKE branch's existing `D1`->`K` conversion.
3. `viscoelastic.py::ViscoelasticMaterial.simo_fs_args()` computed
   `kappa` only via `_extract_lam_mu(params)`, which requires either
   `('mu','lambda')` or `('E','nu')` -- Arruda-Boyce/Yeoh's own params
   (`mu, lambda_m, K`) have neither, so once bug #1/#2 were fixed and the
   base material was genuinely `ArrudaBoyce`, this raised
   `KeyError: 'E'` the first time `Q4_VISCO_SIMO` actually evaluated it.
   Fixed by using `params['K']` directly as kappa when present, before
   falling back to the Lame-based derivation for `NeoHookean`.

After all three fixes: PSA is now genuinely Arruda-Boyce
(mu=0.16785 MPa, lambda_m=3.0, **K=8.3333 MPa** -- previously silently
100.0), `read`/`roundtrip`/`build` all re-verified end to end (full
90deg/side closure, zero cutbacks, ~290-294s each), and
`pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py
tests/test_hyperelastic.py` 22 passed.

**Takeaway**: this bug was invisible from convergence output alone (the
run "succeeded" the whole time) -- it only surfaced by cross-checking the
*same* material spec built two independent ways (`.inp` text vs. raw
Python objects) and comparing the actual resolved `material_params` dict.
This is exactly the kind of silent-wrong-material class of bug the
`build`/`read` parity check was meant to catch.
