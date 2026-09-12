# benchmark_element/ folder setup & first evidence-backed run (2026-09-13)

Picked up `dev_log/3d_elements_mechanics_patch_and_benchmark_plan_20260913.md`
(written by a different session) at the user's request: consolidate 3D
solid-element benchmarking into a dedicated `benchmark_element/` folder so
any agent can read the code + this log and know what to fix next, backed
by numbers rather than claims.

## What was already done before this session picked it up

The other session had already implemented steps 1-2 of the plan:
`verification/mechanics_patches.py` (mesh generators) and
`verification/benchmark_3d_mechanics.py` (the 4-suite runner + markdown
report generator) existed, untracked, uncommitted. Step 3 (pytest
regression) and step 4 (AGENTS.md sync) were not done.

## What this session did

1. **Fixed a real blocker before the suite could even run**: both
   `run_cantilever_bending_test` and `run_volumetric_locking_test`
   constructed `DynamicSolver3D(..., nlgeom=False)`. `dynamic3d.py`'s
   linear-geometry assembly path was removed in an earlier renovation pass
   (raises `NotImplementedError`, see
   `dev_log/3d_element_defects_and_kernel_renovation_plan.md`), so every
   run crashed on the second test. Switched both to `nlgeom=True` — at the
   tiny prescribed strains these tests use (`1e-3`), this reaches the same
   linear-elastic answer through the NL Newton path rather than changing
   the test's physics.
2. **Moved the three benchmark files into a new `benchmark_element/`
   package**: `mechanics_patches.py`, `benchmark_3d_mechanics.py`
   (mechanics/locking), and `benchmark_3d_elements.py` (rank/tangent/perf,
   moved alongside per the plan's own "keep them paired but separate"
   note). Fixed the one cross-import (`verification.mechanics_patches` ->
   `benchmark_element.mechanics_patches`). Verified both scripts still
   import and run correctly from the new location (repo root is already
   on `sys.path` via the editable install, same as every other
   `verification/*.py` script).
3. **Fixed a real bug in the report generator, not just moved it**:
   `save_mechanics_report()`'s "§2 analysis" section was a hand-written,
   fixed narrative ("all elements pass to < 1e-10", "C3D8I/C3D10M fully
   overcome shear locking") that had nothing to do with the actual
   `patch_res`/`bending_res`/`vol_res` dicts passed in — it would print
   the same claims even when the table right above it showed failures.
   Replaced with a data-driven §2 (pass/fail element lists computed from
   the real thresholds) and a new §3: an automatic dispatch-anomaly
   detector that flags any two elements whose bending/volumetric `w_tip`
   match to 1e-12 relative — this codebase's historical signature of a
   silent dispatch bug (AGENTS.md §4.2/§4.8/§4.16), not a coincidence.
4. **Wrote `tests/test_3d_mechanics_benchmarks.py`** (plan step 3): patch
   test for all 11 elements, bending-locking-relief for
   `C3D8I`/`C3D8R`/`C3D10M`, volumetric-locking-relief for
   `C3D8_FBAR`/`C3D8H`/`C3D4_ANP`/`C3D10M`. Known-failing combinations are
   `pytest.xfail`, not skipped, each with the measured number and a
   pointer to this file — same discipline as
   `tests/element_contract/test_contract.py`'s `_XFAIL` table. A future
   fix shows up as `xpass`, which is the actual signal to remove the
   marker.
5. **Synced AGENTS.md** (plan step 4): §2.2 now points at both benchmark
   reports and states the current known-failing set inline (so the doc
   can't silently go stale the way §4.13 did — see
   `dev_log/project_skip_fully_prescribed_deferred` equivalent lesson);
   §3 command list and §5 file tree updated to the new folder.
6. Wrote `benchmark_element/README.md` — durable usage/extension guide
   independent of this dated log.

## What the first real (bug-fixed) run found

Full table: `dev_log/benchmark_3d_mechanics_20260913.md`. Headline,
**not fixed in this session** (out of scope — this session built the
infrastructure, it did not attempt the 11-element renovation itself,
which is `dev_log/3d_element_defects_and_kernel_renovation_plan.md`'s job):

- **5 of 11 elements fail the basic Irons patch test** (error >= 1e-6,
  should be < 1e-6, ideally machine precision per the plan's own
  criterion): `C3D8I`, `C3D8_FBAR`, `C3D8H`, `C3D10`, `C3D10M`. A patch
  test failure means the element can't even reproduce a linear
  displacement field on a distorted mesh — a more basic property than any
  locking behavior, so these elements' bending/volumetric numbers below
  should be read with that caveat.
- **Only `C3D8R` actually measured LOCKING-FREE** at L/h=10 (ratio 0.87).
  `C3D8I` (0.46), `C3D8_FBAR` (0.50), `C3D8H` (0.38) were all expected by
  the plan to be locking-free and measured LOCKED instead — their
  enhancement/reduced-integration mechanism is not relieving shear locking
  in this test, independent of (and in addition to) their patch-test
  failure above. `C3D10`/`C3D10M` did measure LOCKING-FREE (1.02 / 1.07 —
  over 100%, likely because the coarse beam mesh under-integrates
  displacement here, worth a closer look) despite failing the patch test.
- **`C3D4_ANP` produced bit-identical `w_tip` to plain `C3D4`** on both the
  bending test (differs only at ~1e-15, pure roundoff) and the volumetric
  test (identical to full double precision). This is not a dispatch
  routing bug — `dynamic3d.py` does route `C3D4_ANP` to a distinct kernel
  (`assemble_mesh_c3d4_anp_numba`, kernel group `k=9` vs. plain `C3D4`'s
  `k=6`) — it corroborates, with a fresh quantitative measurement, the
  qualitative finding already on record in
  `dev_log/3d_element_defect_audit_20260912.md` §6: `C3D4_ANP` "implements
  none of" the Bonet & Burton average-nodal-pressure mechanism it's named
  for; the kernel is a plain linear tet under a different name.

None of the above are new mysteries — they match/confirm what
`dev_log/3d_element_defect_audit_20260912.md` and
`dev_log/3d_element_defects_and_kernel_renovation_plan.md` already
characterized. What's new is that they're now backed by a number any
agent can reproduce with one command, and gated by a pytest file so a
regression (or a real fix) shows up automatically.

## Explicitly not done this session

- Did not attempt to fix any of the 11 elements' underlying formulations
  — that's the renovation plan's Tier B/C work, tracked separately.
- Did not re-run `verification/run_all.py` or the full `pytest tests/`
  suite for this change (only the two new/changed benchmark scripts and
  the new test file were exercised) — do that before relying on this as a
  "nothing else broke" signal.
