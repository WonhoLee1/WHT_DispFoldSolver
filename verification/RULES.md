# Verification Rules

## Mandatory Post-Change Verification

**After ANY change to the following code, you MUST run the verification suite
and ensure all benchmarks PASS before committing:**

- `dispsolver/solver/` — DynamicSolver, assembly, linear solver, time integration
- `dispsolver/element/` — Q4, Q4_EAS, Q4_UP, Q4_jax, T3, any element kernel
- `dispsolver/material/` — MaterialModel, NeoHookean, J2Plasticity, LinearViscoelastic
- `dispsolver/constraint/` — RBE2, contact, tie, penalty hinge
- `dispsolver/mesh/` — Mesh, connectivity, node/element data structures

## How to Run

```bash
# Full suite (all 9 benchmarks, all backends)
python -m verification.run_all

# Skip JAX JIT warm-up (faster start, first benchmark is slower)
python -m verification.run_all --no-jit-warmup

# Run a single benchmark
python -m verification.run_all --benchmark patch_test_element

# Quiet mode (no per-benchmark progress, just summary)
python -m verification.run_all --quiet
```

## Pass Criteria

| Benchmark | Tolerance | Rationale |
|-----------|-----------|-----------|
| Patch test (element) | 0.01% | Constant strain must be reproduced to near-machine precision |
| Patch test (solver) | 0.001% | Internal node must match analytical position |
| 3-pt / 4-pt bending | 5% | Timoshenko correction is approximate for coarse meshes |
| Cantilever | 5% | Same as above |
| Uniaxial tension/compression | 1% | Uniform stress field — should be very accurate |
| Volumetric compression/tension | 1% | Uniform hydrostatic field — should be very accurate |

If a benchmark fails after a code change:
1. **DO NOT commit the change.**
2. Investigate the root cause — the failure indicates a regression.
3. If the failure is due to an intentional design change (e.g., new element
   formulation with different convergence order), update the tolerance in
   `verification/benchmarks.py` and document the reason in the commit message.
4. Re-run until all benchmarks pass.

## What the Suite Tests

### Element-Level (direct stiffness/strain computation)
- **Patch test**: Single Q4 element under constant strain — checks that
  the B-matrix, constitutive law, and Gauss quadrature produce the exact
  analytical stiffness and strain.
- **Backend comparison**: NumPy B-bar vs NumPy EAS vs JAX B-bar vs JAX Q1P0
  — all should agree to <0.01% on the stiffness matrix.

### Solver-Level (full FEM solve via DynamicSolver)
- **Patch test (irregular mesh)**: 4-element patch with internal node —
  checks that the solver assembly + linear solver reproduce constant strain
  on non-rectangular geometry.
- **Beam bending (3-pt, 4-pt, cantilever)**: Compares tip/mid-span deflection
  against Euler-Bernoulli + Timoshenko beam theory with plane-strain modulus
  correction (E* = E/(1-ν²)).
- **Uniaxial tension/compression**: Block under prescribed displacement —
  checks stress recovery against the plane-strain constitutive relation.
- **Volumetric tension/compression**: Uniform hydrostatic loading — checks
  that the element handles volumetric deformation without locking.

### Backend Coverage
Each solver-level benchmark runs on two backends:
- **JAX** (`fast_assembly=True`): Uses JAX vmap vectorized assembly with
  autodiff-computed tangents. This is the production fast path.
- **NumPy sequential** (`fast_assembly=False` + forced sequential): Uses the
  per-element NumPy loop fallback. This is the reference slow path.

If the two backends disagree, it indicates a bug in the JAX vectorization
or the NumPy fallback — both must agree with theory.

### Numba
The dispsolver codebase does **not** use Numba. Only NumPy and JAX backends
are available. If a Numba backend is added, register it in
`verification/element_backends.py` with the same interface
(`compute_K(coords, E, nu) -> (8,8) ndarray`).

## Adding New Benchmarks

1. Add the benchmark function to `verification/benchmarks.py`.
2. Register it in the `ALL_BENCHMARKS` list.
3. Add the analytical solution to `verification/theory.py` if needed.
4. Add mesh builders to `verification/mesh_utils.py` if needed.
5. Run `python -m verification.run_all` to verify.
6. Update the pass criteria table above if a new tolerance is needed.

## Output Files

All results are written to `verification/results/`:
- `verification_report.md` — human-readable markdown summary
- `results.json` — structured data for CI/CD integration
- `results.csv` — flat table for spreadsheet analysis

The report includes:
- Summary table (pass/fail per benchmark)
- Backend comparison matrix
- Detailed per-benchmark results with theory values and error percentages
- Theory reference formulas
- Post-change verification rules (this document)
