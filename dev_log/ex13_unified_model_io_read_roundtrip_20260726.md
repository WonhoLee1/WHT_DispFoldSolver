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

## Remaining (deferred)

`build` mode: construct the model directly as Python objects (mirroring
`ex12_rigid_plate_display_fold_corotational.py`'s `create_folding_plate_parts`
/ `RigidBodyPart` / `SurfaceTieConstraint` usage) with the *same* 14-layer
PET-PSA/Q4_VISCO_SIMO layup as `gen_ex12_inp.py`, so all three modes are
truly comparable. Not started -- flagged as the next step if full
three-way parity is wanted.
