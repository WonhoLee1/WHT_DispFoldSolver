# Result persistence, interlayer-slip table, animation — 2026-07-26

## Problem

A finished solve left nothing behind that could be re-examined. The Qt
viewer (`PostprocessViewer(solver)` / `launch_from_solver(solver)`) takes
a **live solver object**, not a file -- grepping `viewer.py` for
load/open/read finds nothing. The VTKHDF exporters
(`export_vtkhdf`, `TransientVTKHDFExporter`) are write-only for ParaView,
have no reader, and were not wired into the ex12/ex13 reference scripts
at all. So every new question about a completed run meant re-solving
(~5 min), and the earlier interlayer-shear work resorted to an ad-hoc
`np.save(u)` of the final displacement only -- no mesh, no history, no
metadata.

## What was added

### 1. `dispsolver/postprocess/result_io.py`

`ResultWriter` (accumulate) / `Result` + `load_result` (read back) /
`to_vtkhdf` (project to ParaView).

**Layout is modeled on VTKHDF, per the user's direction** -- static
topology (Connectivity/Offsets/Types), per-step data stored
*concatenated* with offset tables, named PointData/CellData groups,
Steps/Values. Consequence: `to_vtkhdf()` is a slab copy, not a
translation, which is what makes the planned VTKHDF export cheap.

Added beyond VTKHDF, in a `model` block: NodeIds, ElementIds,
ElementPid, ElementTypes, Materials, Constraints, Meta. VTKHDF is a
rendering format with 0-based indices and no place for original ids or
material identity, but analysis needs exactly those ("which rows are
PSA?").

**Backends by extension: `.pkl` (default) and `.h5`.** The pickled
object is a **plain dict of numpy arrays and builtins, never a class
instance**, so unpickling does not depend on this module's class
definitions -- refactoring `Result` cannot rot previously saved runs,
which is the usual failure mode of pickle-based result files.

Nothing is written until `save()`: holding an HDF5 handle open across a
multi-minute solve locks the path on Windows, the failure
`TransientVTKHDFExporter` already documents.

### 2. `dispsolver/postprocess/interlayer.py`

`LayerSlipTracker` -- the interlayer-slip math (previously inline in
`examples/check_interlayer_shear.py`) lifted into a reusable class that
resolves column geometry once and records per-layer tip slip cheaply
every increment. `format_table()` prints slip vs fold angle.

Validated against the independently verified numbers from
`dev_log/interlayer_shear_verification_20260726.md`: reproduces
81.16 um at both tips (PSA 99.5%) and 202.80 um at x=-10 (PSA 97.3%)
exactly.

### 3. `dispsolver/postprocess/animate.py`

`animate` (GIF/MP4), `plot_step`, `save_frame`, `plot_slip_history` --
all driven by a loaded `Result`, no solver involved. Re-animating the
101-step fold takes ~14 s versus ~300 s to re-solve.

### 4. Wired into the solve loop

`ex12_abaqus_inp_plate_fold.py` now writes `examples/ex12_result.pkl`
and prints the slip table on **both** exit paths -- success and
non-convergence abort. A run that stalled at 40 deg is precisely the one
whose history is worth keeping. `ex13` writes one result per mode.
Layer material labels are derived from the mesh + pid, not hard-coded,
so the table stays correct if the layup changes.

Supporting change: `ModelBuilderResult` gained `material_names`
(pid -> *MATERIAL name). The builder already computed this internally
(`_material_name_to_pid`) but discarded it, so nothing downstream could
tell PET rows from PSA rows by name.

### 5. `postprocess/__init__.py` imports Qt lazily

It previously did `from .viewer import ...` at module import, so merely
loading a saved result pulled in PySide6. Now behind a PEP-562
`__getattr__`.

## Verification

- Round trip through both backends, VTKHDF projection shapes correct.
- Abort path exercised deliberately (short `t_total`): result saved with
  6 steps and table printed despite `reached_target=False`.
- Full 101-step run: SUCCESS in 301.25 s, result 5.1 MB, table shows
  monotonic slip growth to **81.16 um at 90 deg, PSA carrying 99.5%** --
  matching the independent verification exactly.
- Reload 0.02 s; animation 13.8 s; slip-history plot regenerated.
- `pytest` 22 passed; `verification.run_all` **9/9 PASS**.

## Note on the slip-history plot

First version was misleading: PET rows contribute ~0 slip, so the
boundary above a PET row sits on top of the boundary above the PSA row
beneath it, collapsing 14 curves into ~7 coincident pairs and hiding
every red (PSA) curve under a blue one. Fixed by drawing PSA boundaries
last, solid and thicker, with the pairing explained in the subtitle.
