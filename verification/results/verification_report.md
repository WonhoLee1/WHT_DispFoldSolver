# FEM Verification Report

**Generated**: 2026-09-10 10:27:16
**Runtime**: 152.8s
**Results**: 12 passed, 2 failed
**Codebase**: dispsolver (JAX-based 2D Plane Strain FEM)

## Summary

| # | Benchmark | Category | Theory | Tolerance | Status |
|---|-----------|----------|--------|-----------|--------|
| 1 | Patch Test (Element Level) | element | 1.0989e-03 | 0.10% | PASS |
| 2 | Patch Test (Solver Level, Irregular Mesh) | solver | 0.0000e+00 | 0.00% | PASS |
| 3 | 3-Point Bending | solver | 2.3530e-04 | 5.00% | PASS |
| 4 | 4-Point Bending | solver | 3.1671e-04 | 5.00% | PASS |
| 5 | 2-Point Bending (Gulati Elastica) | solver | 4.5744e+02 | 5.00% | FAIL |
| 6 | 1-Layer Teardrop Folding Verification | solver | 1.4200e+02 | 1.00% | PASS |
| 7 | Cantilever Bending | solver | 3.6712e-03 | 5.00% | PASS |
| 8 | Uniaxial Tension (Plane Strain) | solver | 1.0989e+00 | 1.00% | PASS |
| 9 | Uniaxial Compression (Plane Strain) | solver | -1.0989e+00 | 1.00% | PASS |
| 10 | Volumetric Compression (Plane Strain) | solver | -9.6154e-01 | 1.00% | PASS |
| 11 | Volumetric Tension (Plane Strain) | solver | 9.6154e-01 | 1.00% | PASS |
| 12 | Convergence Cantilever Mesh (Q4 B-bar) | solver | 2.5000e+00 | 0.80% | FAIL |
| 13 | Convergence Elastica Mesh (Corotational) | solver | 2.7000e+00 | 0.50% | PASS |
| 14 | Convergence Elastica Load-Step (Corotational) | solver | 2.4000e+00 | 0.50% | PASS |

## Backend Comparison

| Benchmark | jax | jax_q4_bbar | jax_q4_corotational_eas | jax_q4_eas | jax_q4_simo_fs | jax_q4_up | jax_q4_visco_fs | jax_t3 | numba_q4_bbar | numba_q4_corotational | numba_q4_corotational_eas | numba_q4_eas | numba_q4_up | numba_t3 | numpy_q4_bbar | numpy_q4_bbar_strain | numpy_q4_corotational_eas | numpy_q4_eas | numpy_sequential | numpy_t3 | order_fit | teardrop_1layer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Patch Test (Element Level) | — | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 5.4945e-04 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | 5.4945e-04 (0.000%) | 1.0989e-03 (0.000%) | 0.0000e+00 (0.000%) | 1.0989e-03 (0.000%) | 1.0989e-03 (0.000%) | — | 5.4945e-04 (0.000%) | — | — |
| Patch Test (Solver Level, Irregular Mesh) | 3.7399e-11 (0.000%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.3978e-11 (0.000%) | — | — | — |
| 3-Point Bending | 2.3698e-04 (0.714%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.4670e-04 (4.843%) | — | — | — |
| 4-Point Bending | 3.2214e-04 (1.715%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.2853e-04 (3.731%) | — | — | — |
| 2-Point Bending (Gulati Elastica) | 2.5008e+02 (45.331%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0000e+00 (100.000%) | — | — | — |
| 1-Layer Teardrop Folding Verification | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.4200e+02 (0.000%) |
| Cantilever Bending | 3.7327e-03 (1.676%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.8420e-03 (4.652%) | — | — | — |
| Uniaxial Tension (Plane Strain) | 1.0990e+00 (0.011%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0990e+00 (0.011%) | — | — | — |
| Uniaxial Compression (Plane Strain) | -1.0988e+00 (0.011%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | -1.0988e+00 (0.011%) | — | — | — |
| Volumetric Compression (Plane Strain) | -9.6154e-01 (0.000%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | -9.6154e-01 (0.000%) | — | — | — |
| Volumetric Tension (Plane Strain) | 9.6154e-01 (0.000%) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.6154e-01 (0.000%) | — | — | — |
| Convergence Cantilever Mesh (Q4 B-bar) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.4716e+00 (78.862%) | — |
| Convergence Elastica Mesh (Corotational) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.7424e+00 (1.569%) | — |
| Convergence Elastica Load-Step (Corotational) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.3939e+00 (0.256%) | — |

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
| numpy_q4_corotational_eas | 1.098897e-03 | 0.0003 | — | PASS |
| numpy_t3 | 5.494505e-04 | 0.0000 | — | PASS |
| numba_q4_bbar | 1.098901e-03 | 0.0000 | — | PASS |
| numba_q4_eas | 1.098901e-03 | 0.0000 | — | PASS |
| numba_q4_corotational_eas | 1.098901e-03 | 0.0000 | — | PASS |
| numba_q4_up | 1.098901e-03 | 0.0000 | — | PASS |
| numba_q4_corotational | 1.098901e-03 | 0.0000 | — | PASS |
| numba_t3 | 5.494505e-04 | 0.0000 | — | PASS |
| jax_t3 | 5.494505e-04 | 0.0000 | — | PASS |
| jax_q4_bbar | 1.098901e-03 | 0.0000 | — | PASS |
| jax_q4_eas | 1.098901e-03 | 0.0000 | — | PASS |
| jax_q4_corotational_eas | 1.098901e-03 | 0.0000 | — | PASS |
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
| jax | 3.739930e-11 | 0.0000 | 1 | PASS |
| numpy_sequential | 2.397843e-11 | 0.0000 | 1 | PASS |
### 3-Point Bending — PASS

- **Category**: solver
- **Theory**: 2.353000e-04 mm
- **Tolerance**: 5.00%
- **Details**: Beam 10.0×1.0 mm, mesh 40×4, P=0.001 N at center top. Timoshenko δ=2.353000e-04 mm (EB: 2.275000e-04). E*=1098.90, I=0.083333.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 2.369794e-04 | 0.7137 | 1 | PASS |
| numpy_sequential | 2.466965e-04 | 4.8434 | 1 | PASS |
### 4-Point Bending — PASS

- **Category**: solver
- **Theory**: 3.167125e-04 mm
- **Tolerance**: 5.00%
- **Details**: Beam 10.0×1.0, mesh 80×8, P=0.001 at L/4 and 3L/4 (top surface). Timoshenko δ=3.167125e-04 mm (EB: 3.128125e-04).

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 3.221427e-04 | 1.7146 | 1 | PASS |
| numpy_sequential | 3.285293e-04 | 3.7311 | 1 | PASS |
### 2-Point Bending (Gulati Elastica) — FAIL

- **Category**: solver
- **Theory**: 4.574445e+02 MPa
- **Tolerance**: 5.00%
- **Details**: Substrate t=0.4mm, plate gap D=80.0mm, E=72300.0MPa. Gulati 2-point bending peak stress σ_max=457.44 MPa.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 2.500819e+02 | 45.3307 | 26 | FAIL |
| numpy_sequential | 0.000000e+00 | 100.0000 | -1 | FAIL |
### 1-Layer Teardrop Folding Verification — PASS

- **Category**: solver
- **Theory**: 1.420000e+02 elems
- **Tolerance**: 1.00%
- **Details**: 1-layer monolithic teardrop grid: 5828 nodes, 142 elements. Teardrop config (pivot=0.8mm, gap=7.5mm, cutout=active) verified.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| teardrop_1layer | 1.420000e+02 | 0.0000 | — | PASS |
### Cantilever Bending — PASS

- **Category**: solver
- **Theory**: 3.671200e-03 mm
- **Tolerance**: 5.00%
- **Details**: Beam 10.0×1.0, mesh 40×4, P=0.001 at top-right tip. Timoshenko δ=3.671200e-03 mm (EB: 3.640000e-03).

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 3.732733e-03 | 1.6761 | 1 | PASS |
| numpy_sequential | 3.841985e-03 | 4.6520 | 1 | PASS |
### Uniaxial Tension (Plane Strain) — PASS

- **Category**: solver
- **Theory**: 1.098901e+00 MPa
- **Tolerance**: 1.00%
- **Details**: Block 2.0×1.0, mesh 10×5, ε_xx=1.0000e-03. Theory: σ_xx=1.0989, ε_yy=-4.285714e-04. Plane strain modulus (constrained): E(1-ν)/((1+ν)(1-2ν))=1346.15.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | 1.099017e+00 | 0.0106 | 1 | PASS |
| numpy_sequential | 1.099017e+00 | 0.0106 | 1 | PASS |
### Uniaxial Compression (Plane Strain) — PASS

- **Category**: solver
- **Theory**: -1.098901e+00 MPa
- **Tolerance**: 1.00%
- **Details**: Block 2.0×1.0, mesh 10×5, ε_xx=-1.0000e-03. Theory: σ_xx=-1.0989, ε_yy=4.285714e-04. Plane strain modulus (constrained): E(1-ν)/((1+ν)(1-2ν))=1346.15.

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| jax | -1.098785e+00 | 0.0106 | 1 | PASS |
| numpy_sequential | -1.098785e+00 | 0.0106 | 1 | PASS |
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
### Convergence Cantilever Mesh (Q4 B-bar) — FAIL

- **Category**: solver
- **Theory**: 2.500000e+00 
- **Tolerance**: 0.80%
- **Details**: ny= 4 nx= 40: δ=3.732733e-03 err(vs ny=12)=2.41e-02
ny= 6 nx= 60: δ=3.660667e-03 err(vs ny=12)=4.37e-03
ny= 8 nx= 80: δ=3.648666e-03 err(vs ny=12)=1.07e-03

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| order_fit | 4.471560e+00 | 78.8624 | — | FAIL |

**Convergence Refinement Levels**

| h | Error |
|---|-------|
| 2.5000e-01 | 2.41e-02 |
| 1.6667e-01 | 4.37e-03 |
| 1.2500e-01 | 1.07e-03 |

**Fitted order**: 4.4716  (expected 2.5000)  **r²**: 0.9983

### Convergence Elastica Mesh (Corotational) — PASS

- **Category**: solver
- **Theory**: 2.700000e+00 
- **Tolerance**: 0.50%
- **Details**: ny= 4 nx= 40: tip_err=1.96e-02 iters=40
ny= 8 nx= 80: tip_err=3.79e-03 iters=40
ny=12 nx=120: tip_err=1.17e-03 iters=40
ny=16 nx=160: tip_err=4.15e-04 iters=40

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| order_fit | 2.742376e+00 | 1.5695 | — | PASS |

**Convergence Refinement Levels**

| h | Error |
|---|-------|
| 2.5000e-01 | 1.96e-02 |
| 1.2500e-01 | 3.79e-03 |
| 8.3333e-02 | 1.17e-03 |
| 6.2500e-02 | 4.15e-04 |

**Fitted order**: 2.7424  (expected 2.7000)  **r²**: 0.9918

### Convergence Elastica Load-Step (Corotational) — PASS

- **Category**: solver
- **Theory**: 2.400000e+00 
- **Tolerance**: 0.50%
- **Details**: n_steps= 2: tip_err(vs ref=16)=6.67e-03 iters=8
n_steps= 4: tip_err(vs ref=16)=1.47e-03 iters=12
n_steps= 8: tip_err(vs ref=16)=2.41e-04 iters=19

| Backend | Value | Error (%) | Iter | Passed |
|---------|-------|-----------|------|--------|
| order_fit | 2.393850e+00 | 0.2562 | — | PASS |

**Convergence Refinement Levels**

| h | Error |
|---|-------|
| 5.0000e-01 | 6.67e-03 |
| 2.5000e-01 | 1.47e-03 |
| 1.2500e-01 | 2.41e-04 |

**Fitted order**: 2.3939  (expected 2.4000)  **r²**: 0.9973

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
