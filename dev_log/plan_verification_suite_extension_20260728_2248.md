# PLAN — Verification suite extension: nonlinear convergence study + speed benchmark

**Status**: approved & fully implemented (2026-07-28/29). All 12/12 verification benchmarks PASS, speed benchmark verified (15x-53x speedup), pytest 145 passed / 1 xfailed.
Original working copy: `C:\Users\GOODMAN\.claude\plans\x-y-plot-eager-puppy.md`.

---

## Context

The solver has two parallel validation layers today:

- `tests/` — pytest, CI-gated tight-tolerance unit checks (baseline: 145
  passed, 1 xfailed). Includes Q4 **and T3** element patch tests
  (`tests/test_kinematics.py`, tolerance 1e-12).
- `verification/` — a separate script-driven suite (`python -m
  verification.run_all`, 9 benchmarks, ~75s) with its own tolerance
  table, PASS/FAIL gating, and markdown report generation.
  `verification/RULES.md` is the canonical source of the project's
  "**mandatory** after any change to `dispsolver/solver/`, `/element/`,
  `/material/`, `/constraint/`, `/mesh/`" policy (AGENTS.md restates it).

The 9 existing benchmarks already cover: element- and solver-level patch
tests, cantilever / 3-point / 4-point bending (with Timoshenko and
Euler-Bernoulli closed forms already in `verification/theory.py`),
uniaxial tension/compression, and volumetric tension/compression. All
run through 2 solver backends (`jax` vs `numpy_sequential`) and 7
element-level backends.

**Two real gaps** relative to what a rigorous FEM verification suite
should have:

1. **No nonlinear large-deformation convergence study.** Every existing
   benchmark deliberately runs in the small-strain linear regime
   (`eps=1e-3`) so NeoHookean reduces exactly to linear elasticity — a
   stated design choice. Nothing exercises the corotational
   large-rotation element, and nothing measures a *convergence order*
   (error vs. mesh size fitted against theory).
2. **No timing/performance data at all.** `run_all.py` times only total
   suite wall-clock; `results.csv` has no timing column. A JAX
   vs. CPython/NumPy speed comparison is entirely new capability.

**Decisions already made with the user (settled, do not revisit):**
extend `verification/` in place (not a new folder); the convergence
study should be a rigorous mesh/load-step refinement order study (not
just a pass/fail regression); Mojo is **out of scope** — documented as a
future placeholder only (zero Mojo references exist anywhere in the repo).

## Two solver findings that shape the implementation

Both verified by reading `dispsolver/solver/dynamic.py` directly. These
are load-bearing — get them wrong and the convergence study silently
validates the wrong element.

**Finding 1 — `Q4_COROTATIONAL` requires dict-form dispatch.**
`DynamicSolver.__init__` picks an assembly strategy in priority order
(`_assemble()`, dynamic.py:3381-3408). The corotational vmap kernel
(`_coro_jax_vmap_by_pid`, built dynamic.py:961-977) is **only** reachable
via `_assemble_multi_material_batch`, which requires
`use_multi_material_batch=True`, which is only set when
`element_type_by_pid is not None` — i.e. when `element_type` was passed
as a `{pid: ...}` **dict**. Passing `element_type='Q4_COROTATIONAL'` as a
plain string with a `J2Plasticity` material instead sets
`use_j2_batch=True`, routing to `_assemble_j2_batch`, which builds
`F = I + grad_u` from global coordinates with **no rotation extraction**
— the plain Q4 B-bar kernel. Separately, `_pid_elem_indices`/`_pid_K_rows`
/`_pid_K_cols` are only built when `isinstance(material, dict)`. So
**both** `material` and `element_type` must be dicts, even for a
single-pid mesh.

**Finding 2 — no working `numpy_sequential` path exists for the
corotational element** (pre-existing bug, out of scope). The sequential
fallback (dynamic.py:3458-3462) calls
`compute_corotational_internal_force(coords, u_elem)` but the real
signature (`q4_corotational_jax.py:66-70`) needs a third
`material_stress_fn` arg — would raise `TypeError` if reached. It's
currently unreachable dead code (per Finding 1's routing). Design the
convergence study to not depend on a numpy-sequential leg for this
element; surface the bug separately rather than silently patching it here.

## A. Convergence-order study

### A.1 Test case: cantilever under a pure end moment

Chosen over the classic tip-point-load elastica because it has an
**exact closed form with no elliptic integrals** (constant curvature
`κ = M/(E*·I)` along the beam), and no point-load stress singularity to
conflate with discretization error. Still genuinely large-rotation.
Applied as a self-equilibrated, incrementally re-oriented (follower)
nodal force couple at the tip — uses only existing solver methods
(`set_prescribed_dofs`, `apply_load`, `solve_step`), no new constraint
machinery, no unverified RBE2 prescribed-rotation behavior.

New in `verification/theory.py`:
```python
def elastica_pure_moment_tip_state(M, L, E_star, I) -> dict:
    """Exact large-rotation tip state under a pure end moment.
        kappa = M / (E_star * I)
        theta = kappa * L                  (tip rotation, exact)
        x_tip = sin(theta) / kappa
        y_tip = (1 - cos(theta)) / kappa
    """
```

**Geometry/strain caveat**: the corotational element factors out large
*rotation*, but the material is still NeoHookean, which equals linear
elasticity only at small *local strain* (`ε_max ≈ θ·H/(2L)`). Start from
`L=20, H=1 (L/H=20), theta_target≈30-40°, nx ≈ 5×ny`. Tune empirically
at implementation time: if the finest mesh level's error plateaus
instead of shrinking, that's the NeoHookean-vs-linear floor — trim the
finest level(s) from the fit and/or raise `L/H` and/or lower
`theta_target_deg`.

### A.2 Three registered benchmarks

1. `convergence_cantilever_mesh_q4bbar` — small-strain sanity check
   reusing the existing `cantilever` physics + `theory.py`'s Timoshenko
   formula. Validates the order-fitting *machinery* against a case
   already trusted. Backend: `jax` only.
2. `convergence_elastica_mesh_corotational` — large-rotation, **mesh**
   refinement (sweep `ny`, fix `n_steps`). Exercises
   `dispsolver/element/q4_corotational_jax.py`.
3. `convergence_elastica_loadstep_corotational` — same case, **load-step**
   refinement (sweep `n_steps`, fix `ny`).

Element/material for 2-3: `element_type={0: 'Q4_COROTATIONAL'}`,
`material={0: J2Plasticity(E=E, nu=nu, sigma_y0=1e12, H=0.0)}` (dict
form per Finding 1; high-yield-J2-as-linear-elastic mirrors the existing
`Q4_EAS` convention in `make_solver`).

### A.3 Error metric + order fitting

Metric: tip position error vs. exact (matches the suite's existing
point-quantity convention; a true energy-norm error would need a new
stored-energy output from `dispsolver/element/`, out of scope).

```python
def fit_convergence_order(h_values, error_values) -> dict:
    """error ~= C*h^p  =>  log(error) = p*log(h) + log(C).
    Returns {'order', 'log_C', 'r_squared', 'h_values', 'error_values'}
    via np.polyfit on the log-log data."""
```

Pass criterion combines the fitted slope **and** fit quality, so a
non-power-law result can't slip through on a lucky slope:
```python
passed = (abs(fit['order'] - expected_order) < order_tol) and (fit['r_squared'] >= 0.95)
```
Expected orders (starting points, confirm empirically): mesh refinement
`2.0 ± 0.4`; load-step refinement `1.0 ± 0.4` (the follower-force scheme
freezes load direction per step from the previous step's converged
orientation — first-order by construction; a predictor-corrector could
reach 2nd order later, report order≈1 honestly for now).

### A.4 New/changed files for A

| File | Change |
|---|---|
| `verification/convergence.py` | **New.** `fit_convergence_order`, `_consistent_axial_nodal_forces` (work-equivalent linear-traction nodal forces, exact for linear `t(y)`), `_tip_axial_direction` (deformed tip-section orientation from tip node positions — mirrors the corotational element's own edge-vector rotation extraction), `_run_pure_moment_case`, `_convergence_result`, 3 driver functions, `ALL_CONVERGENCE_BENCHMARKS`. |
| `verification/theory.py` | Add `elastica_pure_moment_tip_state`. |
| `verification/element_backends.py` | Extend `make_solver` with a `Q4_COROTATIONAL` branch wrapping material/element_type in dicts (Finding 1), plus a comment documenting that `numpy_sequential` is not a real second backend for this element (Finding 2). |
| `verification/mesh_utils.py` | **No change** — `build_beam_mesh(nx, ny, ...)` already parameterizes the sweep. |

### A.5 Pipeline integration — additive only, zero risk to the existing 9

Result dicts reuse the existing
`backends: {name: {value, error_pct, passed}}` shape with one synthetic
key `'order_fit'` instead of `'jax'`/`'numpy_sequential'`. That means the
existing summary table, backend-comparison table, and CSV writer render
the new benchmarks **with no changes at all**.

- `verification/benchmarks.py`: `from .convergence import
  ALL_CONVERGENCE_BENCHMARKS` + `ALL_BENCHMARKS += ...` at the bottom.
  `run_benchmark()`/`run_all_benchmarks()` need no changes (they iterate
  the registry generically).
- `verification/run_all.py`: in `_generate_markdown_report()`'s existing
  per-benchmark loop, add one conditional block
  `if 'refinement_levels' in r:` rendering an h/error sub-table — guarded
  on a key the 9 existing benchmarks never have. Extend `_warmup_jax()`
  (or add a sibling) to pre-compile one throwaway `Q4_COROTATIONAL`
  solve before the timed bracket. `_save_results()` needs no change
  (JSON serializes the nested dict verbatim; CSV iterates `backends`).

## B. Speed benchmark (opt-in, not pass/fail gated)

`verification/speed_bench.py` (new):

```python
def _time_solver_repeated(mesh, E, nu, backend, element_type, bc_dofs, bc_vals,
                          load_dofs, load_vals, n_repeats=5, n_warmup=2) -> dict
```
Builds **one** solver (so JAX's shape-keyed jit cache is populated once),
then per repeat resets `u/v/a/time` to zero and times a single
`solve_step` with `time.perf_counter()`. Discards `n_warmup` repeats so
the first-call-per-mesh-shape JIT trace isn't misattributed as "JAX is
slow" (note: `run_all.py`'s existing `_warmup_jax()` only warms *one*
shape — each new mesh size retraces). Returns
mean/median/std/min/max/n_iter. Only valid for path-independent
materials (NeoHookean); a J2/viscoelastic case would also need
`solver.state` reset.

```python
def speed_benchmark_cantilever(..., mesh_sizes=((10,4), (40,4), (160,16)),
                               backends=('jax','numpy_sequential'),
                               n_repeats=5, n_warmup=2) -> dict
```
Reuses the existing `cantilever` geometry/BCs (not a new physical case —
this is purely timing) across 3 mesh densities (~30 / ~117 / ~2385
elements) so the "JAX's vectorization advantage grows with element
count" trend is visible. `(40,4)` matches the existing cantilever
benchmark's own default for continuity. Reports
`speedup_numpy_over_jax` per mesh size.

**Reporting**: a separate `verification/results/speed_report.md` via
`generate_speed_report()` — a clearly-labeled PERFORMANCE table per mesh
size, no PASS/FAIL column. Keeps `_generate_markdown_report()`'s
pass/fail tables completely untouched.

**Invocation**: new `--include-speed` flag in `run_all.py` (matches the
existing `--no-jit-warmup`/`--quiet` `action='store_true'` pattern).
Opt-in so the mandatory post-change gate's ~75s runtime is unaffected.

## C. Documentation

**`verification/RULES.md`**:
- New "Convergence-order studies" subsection listing the 3 benchmarks,
  stating they **are** part of the mandatory gate (correctness checks,
  just fitted-slope ones).
- New "Performance (opt-in)" subsection documenting `--include-speed`
  and stating explicitly it is **not** gated (informational only).
- "Out of scope / future" note: a Mojo backend has been discussed but
  there are zero Mojo references anywhere in this repo (no toolchain, no
  FFI) — documented placeholder only, not planned.
- Extend the pass-criteria table with the fitted-order tolerance rows.
- **Drive-by fix, flag separately in the commit message**: the table says
  "Patch test (element) | 0.01%" but `benchmarks.py:163` actually uses
  `tolerance=0.1`. Update the RULES.md text to 0.1% to match the code
  (safer than tightening a currently-passing test with no other
  justification). Unrelated pre-existing inconsistency, not part of this
  feature work.

**`verification/README.md`**: 3 new catalog table rows, updated
directory-structure listing (`convergence.py`, `speed_bench.py`),
`--include-speed` in Quick Start, and a short "Large-Rotation /
Corotational Verification" design note capturing Finding 1 so a future
maintainer doesn't rediscover the dict-dispatch requirement the hard way.

## Verification

1. `python -m verification.run_all` — all 12 benchmarks (9 existing + 3
   new) PASS; confirm the 9 existing ones' reported values/errors are
   **unchanged** from the current `verification/results/results.json`
   (the integration is meant to be purely additive).
2. Confirm the new convergence benchmarks actually exercise the intended
   element: assert inside `_run_pure_moment_case` (or check once
   manually) that the constructed solver has
   `use_multi_material_batch=True` and `element_type_by_pid is not None`
   — Finding 1's failure mode is silent, so verify it explicitly rather
   than trusting the constructor arg.
3. Inspect the fitted orders and `r_squared` per new benchmark; if the
   finest mesh level plateaus, apply the A.1 caveat's tuning
   (trim level / raise L/H / lower theta_target_deg) rather than
   widening `order_tol` to force a pass.
4. `python -m verification.run_all --include-speed` — confirm
   `results/speed_report.md` is generated, shows all 3 mesh sizes ×
   2 backends, and that the speedup ratio grows with mesh size.
5. `pytest tests/` — must stay at the 145 passed / 1 xfailed baseline
   (this work touches `verification/` only, so any change here is a
   regression to investigate).
6. Re-run `python -m verification.run_all` with no flags and confirm
   `runtime_seconds` hasn't grown disproportionately (the corotational
   warmup should keep the 3 new benchmarks' JIT cost out of the timed
   bracket).
