# FEM Verification Report

**Generated**: 2026-07-26 22:17:40
**Runtime**: 76.6s
**Results**: 9 passed, 0 failed
**Codebase**: dispsolver (JAX-based 2D Plane Strain FEM)

## Summary

| # | Benchmark | Category | Theory | Tolerance | Status |
|---|-----------|----------|--------|-----------|--------|
| 1 | Patch Test (Element Level) | element | 1.0989e-03 | 0.10% | PASS |
| 2 | Patch Test (Solver Level, Irregular Mesh) | solver | 0.0000e+00 | 0.00% | PASS |
| 3 | 3-Point Bending | solver | 2.3530e-04 | 5.00% | PASS |
| 4 | 4-Point Bending | solver | 3.1671e-04 | 5.00% | PASS |
| 5 | Cantilever Bending | solver | 3.6712e-03 | 5.00% | PASS |
| 6 | Uniaxial Tension (Plane Strain) | solver | 1.0989e+00 | 1.00% | PASS |
| 7 | Uniaxial Compression (Plane Strain) | solver | -1.0989e+00 | 1.00% | PASS |
| 8 | Volumetric Compression (Plane Strain) | solver | -9.6154e-01 | 1.00% | PASS |
| 9 | Volumetric Tension (Plane Strain) | solver | 9.6154e-01 | 1.00% | PASS |

## Backend Comparison

| Benchmark | jax | jax_q4_bbar | jax_q4_eas | jax_q4_simo_fs | jax_q4_up | jax_q4_visco_fs | numpy_q4_bbar | numpy_q4_bbar_strain | numpy_q4_eas | numpy_sequential |
|---|---|---|---|---|---|---|---|---|---|---|
| Patch Test (Element Level) | — | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 0.0000e+00 (0.000%) | 1.0989e-03 (0.000%) | — |
| Patch Test (Solver Level, Irregular Mesh) | 1.0483e-07 (0.000%) | — | — | — | — | — | — | — | — | 9.5439e-08 (0.000%) |
| 3-Point Bending | 2.3698e-04 (0.714%) | — | — | — | — | — | — | — | — | 2.4667e-04 (4.834%) |
| 4-Point Bending | 3.2214e-04 (1.715%) | — | — | — | — | — | — | — | — | 3.2853e-04 (3.731%) |
| Cantilever Bending | 3.7327e-03 (1.676%) | — | — | — | — | — | — | — | — | 3.8420e-03 (4.652%) |
| Uniaxial Tension (Plane Strain) | 1.0989e+00 (0.000%) | — | — | — | — | — | — | — | — | 1.0989e+00 (0.000%) |
| Uniaxial Compression (Plane Strain) | -1.0989e+00 (0.000%) | — | — | — | — | — | — | — | — | -1.0989e+00 (0.000%) |
| Volumetric Compression (Plane Strain) | -9.6154e-01 (0.000%) | — | — | — | — | — | — | — | — | -9.6154e-01 (0.000%) |
| Volumetric Tension (Plane Strain) | 9.6154e-01 (0.000%) | — | — | — | — | — | — | — | — | 9.6154e-01 (0.000%) |

## Detailed Results

### Patch Test (Element Level) — PASS

- **Category**: element
- **Theory**: 1.098901e-03 N·mm
- **Tolerance**: 0.10%
- **Details**: Single Q4 element (2.0×1.0), α=1.0e-03. Strain: ε_xx=1.000000e-03, ε_yy=-4.285714e-04. Stress σ_xx=1.0989. U_theory=1.098901e-03. NumPy strain error: 0.00e+00 (machine precision).

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| numpy_q4_bbar | 1.098901e-03 | 0.0000 | — | PASS |
| numpy_q4_eas | 1.098901e-03 | 0.0000 | — | PASS |
| jax_q4_bbar | 1.098901e-03 | 0.0000 | — | PASS |
| jax_q4_eas | 1.098901e-03 | 0.0000 | — | PASS |
| jax_q4_up | 1.098901e-03 | 0.0000 | — | PASS |
| jax_q4_visco_fs | 1.098901e-03 | 0.0000 | — | PASS |
| jax_q4_simo_fs | 1.098901e-03 | 0.0000 | — | PASS |
| numpy_q4_bbar_strain | 0.000000e+00 | 0.0000 | — | PASS |

### Patch Test (Solver Level, Irregular Mesh) — PASS

- **Category**: solver
- **Theory**: 0.000000e+00 %
- **Tolerance**: 0.00%
- **Details**: Irregular 4-element patch, α=1.0e-03. Internal node (5) must settle at ux=5.000000e-04, uy=-2.142857e-04.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 1.048267e-07 | 0.0000 | 1 | PASS |
| numpy_sequential | 9.543853e-08 | 0.0000 | 1 | PASS |

### 3-Point Bending — PASS

- **Category**: solver
- **Theory**: 2.353000e-04 mm
- **Tolerance**: 5.00%
- **Details**: Beam 10.0×1.0 mm, mesh 40×4, P=0.001 N at center top. Timoshenko δ=2.353000e-04 mm (EB: 2.275000e-04). E*=1098.90, I=0.083333.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 2.369794e-04 | 0.7137 | 2 | PASS |
| numpy_sequential | 2.466733e-04 | 4.8335 | 1 | PASS |

### 4-Point Bending — PASS

- **Category**: solver
- **Theory**: 3.167125e-04 mm
- **Tolerance**: 5.00%
- **Details**: Beam 10.0×1.0, mesh 80×8, P=0.001 at L/4 and 3L/4 (top surface). Timoshenko δ=3.167125e-04 mm (EB: 3.128125e-04).

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 3.221427e-04 | 1.7146 | 2 | PASS |
| numpy_sequential | 3.285293e-04 | 3.7311 | 2 | PASS |

### Cantilever Bending — PASS

- **Category**: solver
- **Theory**: 3.671200e-03 mm
- **Tolerance**: 5.00%
- **Details**: Beam 10.0×1.0, mesh 40×4, P=0.001 at top-right tip. Timoshenko δ=3.671200e-03 mm (EB: 3.640000e-03).

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 3.732733e-03 | 1.6761 | 2 | PASS |
| numpy_sequential | 3.841985e-03 | 4.6520 | 2 | PASS |

### Uniaxial Tension (Plane Strain) — PASS

- **Category**: solver
- **Theory**: 1.098901e+00 MPa
- **Tolerance**: 1.00%
- **Details**: Block 2.0×1.0, mesh 10×5, ε_xx=1.0000e-03. Theory: σ_xx=1.0989, ε_yy=-4.285714e-04. Plane strain modulus (constrained): E(1-ν)/((1+ν)(1-2ν))=1346.15.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 1.098901e+00 | 0.0000 | 1 | PASS |
| numpy_sequential | 1.098901e+00 | 0.0000 | 1 | PASS |

### Uniaxial Compression (Plane Strain) — PASS

- **Category**: solver
- **Theory**: -1.098901e+00 MPa
- **Tolerance**: 1.00%
- **Details**: Block 2.0×1.0, mesh 10×5, ε_xx=-1.0000e-03. Theory: σ_xx=-1.0989, ε_yy=4.285714e-04. Plane strain modulus (constrained): E(1-ν)/((1+ν)(1-2ν))=1346.15.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | -1.098901e+00 | 0.0000 | 1 | PASS |
| numpy_sequential | -1.098901e+00 | 0.0000 | 1 | PASS |

### Volumetric Compression (Plane Strain) — PASS

- **Category**: solver
- **Theory**: -9.615385e-01 MPa
- **Tolerance**: 1.00%
- **Details**: Block 2.0×2.0, mesh 10×10, ε_vol=-1.0000e-03 (ε_xx=ε_yy=-5.0000e-04). c=E/((1+ν)(1-2ν))=1923.08. σ_hydro=-0.9615.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | -9.615385e-01 | 0.0000 | 1 | PASS |
| numpy_sequential | -9.615385e-01 | 0.0000 | 1 | PASS |

### Volumetric Tension (Plane Strain) — PASS

- **Category**: solver
- **Theory**: 9.615385e-01 MPa
- **Tolerance**: 1.00%
- **Details**: Block 2.0×2.0, mesh 10×10, ε_vol=1.0000e-03 (ε_xx=ε_yy=5.0000e-04). c=E/((1+ν)(1-2ν))=1923.08. σ_hydro=0.9615.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 9.615385e-01 | 0.0000 | 1 | PASS |
| numpy_sequential | 9.615385e-01 | 0.0000 | 1 | PASS |

## Backend Notes

### Available Backends

- **NumPy Q4 B-bar**: Pure NumPy 2×2 Gauss quadrature with B-bar SRI (dispsolver.element.q4). Baseline element formulation.
- **NumPy Q4 EAS-4**: Enhanced Assumed Strain with 4 internal modes, statically condensed (dispsolver.element.q4_eas). Eliminates bending locking.
- **JAX Q4 B-bar**: JAX autodiff via NeoHookean material, vmap-vectorized (dispsolver.solver.dynamic_jax). At small strain NeoHookean ≡ linear elastic.
- **JAX Q4 EAS-4**: JAX EAS with J2 plasticity (dispsolver.element.q4_eas_jax). J2 yield set to 1e12 (never yields) so response is linear elastic. Production EAS path for PET layers.
- **JAX Q1P0 Hybrid**: Q1P0 mixed formulation (u-p) with static condensation, energy-based JAX autodiff (dispsolver.element.q4_up_jax). For near-incompressible materials.
- **JAX Visco Hybrid F-bar**: Finite-strain Green-Lagrange F-bar viscoelastic element (dispsolver.element.q4_visco_hybrid_fs_jax). Prony g_i set to [0] for pure elastic verification. This is the ex03 PSA element path.
- **JAX Simo Visco F-bar**: Finite-strain Flory-split Simo viscoelastic element with pluggable hyperelastic base (dispsolver.element.q4_visco_simo_fs_jax). Prony g_i set to [0] for pure NeoHookean verification. The more rigorous formulation for large-rotation viscoelastic analysis.

### Numba Note

The dispsolver codebase does **not** use Numba. Only NumPy and JAX backends are available. If a Numba backend is added in the future, register it in `verification/element_backends.py` with the same interface.

## Theory Reference

All formulas are for **plane strain** (ε_zz = 0):

| Quantity | Formula |
|----------|---------|
| Plane strain D | E/((1+ν)(1-2ν)) · [[1-ν, ν, 0], [ν, 1-ν, 0], [0, 0, (1-2ν)/2]] |
| Plane strain modulus | E* = E/(1-ν²) |
| Beam I (per unit depth) | I = H³/12 |
| Cantilever (EB) | δ = PL³/(3·E*·I) |
| 3-pt bending (EB) | δ = PL³/(48·E*·I) |
| 4-pt bending (EB) | δ = Pa(3L²-4a²)/(24·E*·I) |
| Uniaxial (plane strain) | σ_xx = E/(1-ν²) · ε_xx (free lateral) |
| Hydrostatic (plane strain) | σ = c · ε_vol/2, c = E/((1+ν)(1-2ν)) |

Timoshenko shear correction (κ_s = 5/6 for rectangular cross-section) is included for beam benchmarks.

## Post-Change Verification Rules

See `verification/RULES.md` for the full rules. Key rule:

> **After any change to the solver (`dispsolver/solver/`) or element (`dispsolver/element/`) code, run `python -m verification.run_all` and ensure all benchmarks PASS before committing.**
