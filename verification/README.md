# FEM Verification Suite

Verification benchmarks for the **dispsolver** JAX-based 2D plane-strain FEM solver.

## Quick Start

```bash
# Run all benchmarks (generates results/verification_report.md)
python -m verification.run_all

# Run all benchmarks with performance data (opt-in)
python -m verification.run_all --include-speed

# Run a single benchmark
python -m verification.run_all --benchmark cantilever
```

## Benchmark Catalog

| # | Benchmark | Type | Theory | Backends |
|---|-----------|------|--------|----------|
| 1 | Patch test (element) | Element | Constant strain: ε = B·u | NumPy B-bar, NumPy EAS, JAX B-bar, JAX EAS, JAX Q1P0, JAX Visco FS, JAX Simo FS |
| 2 | Patch test (solver) | Solver | Internal node position | JAX, NumPy sequential |
| 3 | 3-point bending | Solver | δ = PL³/(48·E*·I) + Timoshenko | JAX, NumPy sequential |
| 4 | 4-point bending | Solver | δ = Pa(3L²-4a²)/(24·E*·I) | JAX, NumPy sequential |
| 5 | Cantilever | Solver | δ = PL³/(3·E*·I) + Timoshenko | JAX, NumPy sequential |
| 6 | Uniaxial tension | Solver | σ_xx = E(1-ν)/((1+ν)(1-2ν)) · ε_xx | JAX, NumPy sequential |
| 7 | Uniaxial compression | Solver | Same as tension (negative) | JAX, NumPy sequential |
| 8 | Volumetric compression | Solver | σ = K_ps · ε_vol | JAX, NumPy sequential |
| 9 | Volumetric tension | Solver | Same as compression (positive) | JAX, NumPy sequential |
| 10 | Convergence cantilever mesh (Q4 B-bar) | Solver | Mesh refinement order ≈ 2.0 | Fitted order only |
| 11 | Convergence elastica mesh (corotational) | Solver | Mesh refinement order ≈ 2.0 | Fitted order only |
| 12 | Convergence elastica load-step (corotational) | Solver | Load-step refinement order ≈ 1.0 | Fitted order only |

## Directory Structure

```
verification/
├── __init__.py
├── README.md              ← this file
├── RULES.md               ← post-change verification rules (READ THIS)
├── theory.py              ← analytical solutions (plane strain)
├── mesh_utils.py          ← mesh builders (block, beam, patch, single-element)
├── element_backends.py    ← unified NumPy/JAX element API
├── convergence.py         ← convergence-order studies (3 benchmarks)
├── speed_bench.py         ← performance benchmarks (opt-in)
├── benchmarks.py          ← 12 benchmark implementations
├── run_all.py             ← orchestrator + report generator
└── results/               ← output (auto-created)
    ├── verification_report.md
    ├── results.json
    ├── results.csv
    └── speed_report.md
```

## Key Design Decisions

### Plane Strain Throughout
All formulas use the plane-strain modulus E* = E/(1-ν²) for beam bending,
and the plane-strain constitutive matrix D. This matches the solver's
F_33 = 1 kinematics.

### NeoHookean as Linear Elastic Proxy
At small strains (ε < 1e-3), the compressible Neo-Hookean model reduces
exactly to linear elasticity. This allows using the JAX autodiff material
path for linear-elastic verification without implementing a separate
linear elastic material class.

### JAX vs NumPy Comparison
The solver has two assembly paths:
- **JAX vmap** (`fast_assembly=True`): JIT-compiled, vectorized — production path
- **NumPy sequential** (forced via `use_jax_vmap=False`): per-element loop — reference

Both must agree with theory. Disagreement indicates a vectorization bug.

### Timoshenko Beam Theory
For beam benchmarks, Timoshenko shear correction (κ_s = 5/6) is included
because the L/H ratios used (10:1) are not large enough for Euler-Bernoulli
to be sufficient. The verification uses L/H = 10 with 40×4 mesh.

### Large-Rotation / Corotational Verification

Benchmarks 11-12 exercise the ``Q4_COROTATIONAL`` element, which requires
the solver's ``element_type`` and ``material`` arguments to be passed as
``{pid: ...}`` dicts (not strings) to reach the corotational kernel
(see ``AGENTS.md`` Finding 1). The ``make_solver`` function in
``element_backends.py`` handles this wrapping automatically — do not pass
``element_type='Q4_COROTATIONAL'`` as a plain string, as this silently uses
the standard Q4 B-bar kernel with no rotation extraction.

There is no working ``numpy_sequential`` path for the corotational element
(Finding 2 — a pre-existing bug in ``dynamic.py``). The convergence study
uses only the JAX backend for corotational cases.

## See Also

- [`RULES.md`](RULES.md) — **mandatory** post-change verification rules
- [`results/verification_report.md`](results/verification_report.md) — latest report
