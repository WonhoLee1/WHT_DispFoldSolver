# benchmark_element/ — evidence-based element benchmarking

This folder is the single home for **mechanics-accuracy evidence** on
`dispsolver`'s finite elements — currently the 11 3D solid elements
(`dispsolver/element3d/`), moved here from `verification/` on 2026-09-13
per the plan in `dev_log/3d_elements_mechanics_patch_and_benchmark_plan_20260913.md`.
2D elements have their own equivalent (`tests/element_contract/`, see
AGENTS.md §4.16) — don't duplicate that suite here; this folder is for the
element families that don't already have one.

**Why this folder exists**: every previous "this element works" claim in
this codebase that wasn't backed by a number like the ones these scripts
produce turned out to be wrong at least once (AGENTS.md §4.14-§4.18 — F4,
F6, B1-B4, the CPE4IH condensation-symmetry bug, three Numba/JAX
mismatches). The rule going forward: **before claiming an element is fixed
or good, produce a number from this folder's scripts that says so** — the
same discipline `tests/element_contract/` established for 2D.

## What's here

| File | Role | Runtime |
|---|---|---|
| `IMPROVEMENT_GUIDE.md` | **Agent Renovation Handbook**: Root causes, theoretical mechanisms, source file locations, and step-by-step recipes to update elements based on benchmark findings. | — |
| `benchmark_3d_elements.py` | **Math soundness & performance**: stiffness-matrix spectral rank (rigid-body-mode count), analytic-vs-FD tangent consistency, Numba assembly speed, literature/bibliography catalog. | seconds |
| `mechanics_patches.py` | Parametric mesh + BC generators shared by the mechanics suite: distorted multi-element patches (hex/tet/wedge/quadratic-tet), parametric cantilever beam meshes. No solving here — pure geometry. | — |
| `benchmark_3d_mechanics.py` | **Mechanical accuracy & locking**: Irons distorted patch test (tension/compression/in-plane shear/out-of-plane shear), MacNeal-Harder cantilever bending vs. Timoshenko theory (shear-locking ratio), volumetric-locking sweep toward the incompressible limit (ν→0.49999). Writes `dev_log/benchmark_3d_mechanics_YYYYMMDD.md` and generates actionable renovation guides. | ~1-2 min |

Deliberately kept as **two separate suites** (per the plan's own review
note): rank/tangent/performance is a different question from
mechanical-accuracy/locking, and the second is far more expensive
(mesh + Newton solve per element per mode vs. a single stiffness build).
Don't merge them back into one file.

`tests/test_3d_mechanics_benchmarks.py` is the pytest regression wrapper —
CI-checkable pass/fail thresholds derived from these same functions, so a
future change that silently breaks an element's locking-relief property
fails a test, not just a report nobody reread.

## How to run

```bash
python -u benchmark_element/benchmark_3d_elements.py     # rank/tangent/perf, seconds
python -u benchmark_element/benchmark_3d_mechanics.py    # mechanics/locking, ~1-2 min
pytest tests/test_3d_mechanics_benchmarks.py -v          # regression gate
```

Both scripts must be run with the repo root as the working directory (they
write to `dev_log/` with a relative path, and rely on the repo root already
being on `sys.path` — same convention every other `verification/*.py`
script in this repo already uses).

## How to read the output / how to decide what to fix next

`benchmark_3d_mechanics.py`'s markdown report has one row per element and
one column per test. Read it like this:

- **Patch columns (Tension/Compression/Shear XY/Shear YZ)**: these must be
  ✅ (error `< 1e-6`) for an element to be trusted at all — a patch-test
  failure means the element cannot even reproduce a *linear* displacement
  field exactly, which is a stronger and more basic claim than any locking
  or accuracy property. Fix patch-test failures before touching anything
  else for that element (same principle as AGENTS.md §4.16's contract
  suite: a defect here masks/invalidates every downstream measurement).
- **Bending ratio / locking verdict**: `> 0.85` = "LOCKING-FREE" is the
  claim an enhanced/reduced-integration element (`C3D8I`, `C3D8R`,
  `C3D10M`) exists to deliver. If it isn't hitting that, the improvement
  the element was supposed to provide isn't actually wired in — check
  whether the kernel's enhancement/reduced-integration term is really
  feeding back into the assembled stiffness (this is the exact failure
  mode AGENTS.md §4.16 found for `C3D8_EAS`'s 9-mode mechanism: computed,
  never fed back).
- **Volumetric verdict**: elements advertised as incompressibility-safe
  (`C3D8_FBAR`, `C3D8H`, `C3D4_ANP`, `C3D10M`) should read
  `INCOMPRESSIBLE-OK` at ν=0.49999. `VOL-LOCKED` there is a real defect,
  not a tolerance issue — the whole reason that element variant exists is
  to not do this.

**When two element rows print identical numbers to full precision** (seen
2026-09-13: `C3D4` and `C3D4_ANP` matched on every column of one run),
treat that as a dispatch-integrity finding first, not a physics one — this
codebase's history (AGENTS.md §4.2/§4.8/§4.16) is that identical numbers
between two supposedly-different elements usually means one of them is
silently running the other's code, not that both formulations happen to
agree. Check the dispatch table before concluding the "advanced" variant
provides no benefit.

**Known constraint this suite works around**: `dynamic3d.py`'s
linear-geometry (`nlgeom=False`) assembly path was removed (raises
`NotImplementedError`, see `dev_log/3d_element_defects_and_kernel_renovation_plan.md`).
All mechanics tests here run with `nlgeom=True`; at the small prescribed
strains these tests use (`1e-3`), this is mathematically equivalent to a
linear solve reached via the nonlinear Newton path, not a formulation
change — but it does mean a genuinely broken NL Newton path could produce
a false "FAIL" here that a true linear solve wouldn't. If a result looks
wrong, check convergence (`iters`, `status`) before concluding the element
itself is at fault.

## How to extend this when a new element/kernel needs benchmarking

1. Add the element's connectivity pattern to `mechanics_patches.py`'s
   `make_distorted_patch_mesh()` / `make_cantilever_beam_mesh()` if it
   isn't one of the 11 already covered (new element family — new node
   topology).
2. Add its name to `ALL_3D_ELEMENTS` in `benchmark_3d_mechanics.py` (and
   the equivalent list in `benchmark_3d_elements.py` if the rank/tangent
   suite doesn't already cover it).
3. Re-run both scripts, and re-run `pytest tests/test_3d_mechanics_benchmarks.py`.
   A newly-passing threshold is real evidence; don't hand-edit the
   markdown report or the pytest thresholds to make something look fixed.
4. If a fix changes an element's numbers, note the before/after in a dated
   `dev_log/` entry (not by editing history in this README) — the report
   file itself is regenerated每 run and is not the permanent record; the
   permanent record is `dev_log/benchmark_3d_mechanics_YYYYMMDD.md` plus
   whatever `dev_log/*.md` walkthrough explains *why* the numbers moved.
