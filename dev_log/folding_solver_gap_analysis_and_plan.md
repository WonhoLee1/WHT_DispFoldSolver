# WHT_DispFoldSolver — Folding Solver Gap Analysis & Development Plan

**Date**: 2026-07-24  
**Status**: Analysis complete — ready for Phase 1 implementation  
**Context**: Display panel folding simulation (7-layer PET/PSA, 90° hinge rotation, large deformation, multi-physics)

---

## 1. Problem Statement

The solver fails to converge on the display folding problem (`ex03_display_fold.py`, `ex03_optimized_v3.py`):

| Symptom | Observation |
|---------|-------------|
| **Divergence** | Residual force grows 10⁶–10¹⁶× within 10–18 NR iterations |
| **Cutback cascade** | dt: 1e-2 → 5e-3 → 2.5e-3 → 1.25e-3 → 6.25e-4 → ... |
| **Stall** | Displacement ratio plateaus at ~0.01–0.02 (never reaches tol=1e-3) |
| **NaN in JAX vmap** | "EAS JAX vmap NaN in N/300 elems" → fallback to sequential |
| **Riks failure** | "Zero tangent norm" — singular KKT from penalty RBE2 |

**Root cause**: Not a single bug — **systemic integration gaps** across materials, elements, constraints, and solver.

---

## 2. Technical Gap Analysis

### A. Material Models — Missing/Inconsistent Tangents

| Material | Status | Gap |
|----------|--------|-----|
| **NeoHookean/Yeoh/ArrudaBoyce** | ✅ JAX autodiff | Consistent tangent via `jax.hessian` |
| **J2Plasticity (NumPy)** | ✅ Analytical | Consistent algorithmic tangent (Simo 1992) |
| **J2Plasticity (JAX)** | ❌ **FD only** | `tangent_voigt_jax` uses 3× central FD — fails at repeated eigenvalues (F=I) |
| **Viscoelastic (Prony+WLF)** | ❌ Approximate | `g_eff` scaling only — no differentiation of `h_i` update |
| **LinearViscoelastic** | ❌ Not verified | Used in PSA layers but no finite-strain validation |

### B. Elements — Broken/Incomplete JAX Paths

| Element | NumPy Path | JAX Path | Status |
|---------|------------|----------|--------|
| **Q4 B-bar** | ✅ | ✅ `q4_jax.py` | Working |
| **Q4 EAS** | ✅ `q4_eas.py` | ❌ `q4_eas_jax.py` | **Alpha condensation: fixed 12 iter, no convergence check, FD tangent** |
| **Q4 UP (Q1P0)** | ✅ `q4_visco_hybrid.py` | ❌ `q4_visco_hybrid_fs_jax.py` | **`_internal_force` has undefined variables — DOES NOT RUN** |
| **T3 F-bar** | ✅ | ❌ | No JAX path |

### C. Constraints — Inconsistent Formulations

| Constraint | NumPy Path | JAX Path | Problem |
|------------|------------|----------|---------|
| **RBE2 Hinge** | True Lagrange multipliers → saddle-point KKT `[K C^T; C 0]` | Penalty + Augmented Lagrange → condensed `K_e = K_uu - K_uθ K_θθ⁻¹ K_uθ` | **Different math!** Penalty `k=1e6` arbitrary; no constraint satisfaction check |
| **Tie** | Lagrange (reuses RBE2) | ❌ | Not implemented in JAX |
| **Contact** | Penalty only | ❌ | No Lagrange multiplier contact |

### D. Solver & Nonlinear Strategy Defects

| Component | Issue | Impact |
|-----------|-------|--------|
| **Line search** | NaN/Inf backtracking only — **no Armijo/Wolfe** | Steps into bifurcation not rejected; no merit function for saddle-point |
| **Convergence criteria** | `du_norm/u_norm` unreliable for KKT; `rtol=5e-3` loose; `atol=1e-7` too strict for flat amplitude | False non-convergence or premature acceptance |
| **KKT regularization** | Ad-hoc: `1e-12` on θ diagonal, `1e-8*mean(diag)` on full system | Doesn't address 5-order block scaling (K~10⁵ vs C~1) |
| **PARDISO equilibration** | Only in `_solve_linear_system` (post-assembly) | PARDISO sees ill-scaled blocks during factorization |
| **Arc-length (Riks)** | "Zero tangent norm" — singular KKT from penalty RBE2 | Cannot pass limit points (hinge buckling) |
| **UL state management** | `_ul_F_n` synced only on convergence | Diverged elements keep stale `F_n` → inconsistent reference |

### E. Physics Scale Mismatches

| Parameter | Value | Problem |
|-----------|-------|---------|
| PET layer thickness | 0.05mm / 3 = **0.0167mm** | Aspect ratio > 30:1 (0.5mm / 0.0167mm) → bending locking |
| Hinge rotation | **90°** (1.57 rad) | Elements at hinge: extreme distortion, det(J) → 0 |
| PSA Poisson's ratio | **ν = 0.49** | Near-incompressible → volumetric locking without exact Q1P0 |
| Contact threshold | **d₀ = 0.2mm** | >> layer thickness → penetration or ill-conditioning |

---

## 3. Specific Code Bugs (Must Fix First)

### `dispsolver/element/q4_visco_hybrid_fs_jax.py` — **BROKEN**
```python
def _internal_force(...):
    ...
    Fbar = ...          # UNDEFINED
    s_dev = ...         # UNDEFINED  
    p = ...             # UNDEFINED
    e_dev = ...         # UNDEFINED
    q_new_all = ...     # UNDEFINED
    M = ...             # UNDEFINED
    gp = ...            # UNDEFINED (loop variable)
    ...
```

### `dispsolver/element/q4_eas_jax.py` — **Incomplete**
- Fixed 12 Newton iterations for α condensation — no convergence check
- Uses FD tangent (`tangent_voigt_jax`) — fails at F=I (repeated eigenvalues)
- No element inversion guard in UL mode

### `dispsolver/element/rbe2_jax.py` — **Wrong Formulation**
- Penalty method (`k=1e6`) instead of true Lagrange multipliers
- No constraint satisfaction verification (`g ≈ 0`)

---

## 4. Development Plan (Phased)

### Phase 1: Fix Broken Code (Days 1-2) 🔴 **CRITICAL**

| Task | File | Description |
|------|------|-------------|
| 1.1 | `q4_visco_hybrid_fs_jax.py` | Complete `_internal_force` — implement F-bar, Green-Lagrange, Prony update, consistent state |
| 1.2 | `q4_eas_jax.py` | Add α convergence check (residual norm < 1e-10); increase max iter to 20 |
| 1.3 | `rbe2_jax.py` | Implement true Lagrange multiplier formulation matching NumPy `RBE2HingeConstraint` |
| 1.4 | `q4_eas_jax.py` | Replace FD tangent with `jax.jacobian` on condensed force (autodiff-safe) |

**Verification**: `python -m verification.run_all --benchmark patch_test_element` passes

---

### Phase 2: Consistent Tangents (Days 3-5) 🟠 **HIGH**

| Task | File | Description |
|------|------|-------------|
| 2.1 | `plastic_jax.py` | ✅ Replaced FD tangent with `jax.jacfwd` autodiff + 1e-10 noise to break eigenvalue degeneracy at F=I. 9/9 benchmarks pass. |
| 2.2 | `q4_eas_jax.py` | Autodiff tangent: `K_e = jax.jacobian(lambda u: f_condensed(u))` |
| 2.3 | `viscoelastic.py` | Differentiate Prony `h_i` update → consistent viscoelastic tangent |
| 2.4 | `q4_visco_hybrid_fs_jax.py` | Verify autodiff tangent works (no eigendecomposition → NaN-safe) |

**Verification**: All 9 benchmarks pass with `fast_assembly=True` (JAX path) — **VERIFIED**

---

### Phase 3: Robust Nonlinear Solver (Days 6-8) 🟡 **MEDIUM**

| Task | File | Description |
|------|------|-------------|
| 3.1 | `dynamic.py` | **KKT equilibration**: Scale rows/cols so `\|J_ii\| ≈ 1` before PARDISO (pre-assembly) |
| 3.2 | `dynamic.py` | **Merit-function line search**: `φ = 0.5 * \|R\|²` or `φ = u^T R` for saddle-point |
| 3.3 | `dynamic.py` | **Adaptive convergence**: Combine `\|R\|/\|R₀\| < rtol`, `\|du\| < atol`, energy error `< 1e-12` |
| 3.4 | `dynamic.py` | **Working Riks arc-length**: Spherical constraint with consistent tangent; handle singular KKT |
| 3.5 | `dynamic.py` | **Element inversion detection**: Check `det(F) > 0` at all GPs; cutback dt if violated |

**Verification**: `ex03_display_fold.py` converges to 90° fold without cutback cascade

---

### Phase 4: Large Deformation Robustness (Days 9-11) 🟢 **MEDIUM**

| Task | File | Description |
|------|------|-------------|
| 4.1 | `dynamic.py` | **UL state sync for ALL elements** (converged + cutback) |
| 4.2 | `dynamic.py` | **Mesh distortion monitor**: Track min Jacobian, aspect ratio; warn/remesh |
| 4.3 | `constraint/` | **Lagrange multiplier contact** (replace penalty) |
| 4.4 | `material/` | **WLF at finite strain**: Verify `aT(T)` works with `F` not just small strain |

---

### Phase 5: Physics Fidelity & Validation (Days 12-14) 🔵 **LOW**

| Task | Description |
|------|-------------|
| 5.1 | Q1P0 patch test at J≠1 (finite strain volumetric locking) |
| 5.2 | Thermal-mechanical coupling if temperature varies during fold |
| 5.3 | Benchmark against Abaqus/ANSYS reference for 3-point bend + fold |
| 5.4 | Performance profiling: JAX vmap vs NumPy sequential |

---

## 5. Quick Wins (Test Immediately)

```python
# In ex03_display_fold.py or ex03_optimized_v3.py:

# 1. Disable JAX vmap for EAS (force sequential NumPy EAS)
solver = DynamicSolver(..., fast_assembly=False, element_type="Q4_EAS")

# 2. Use pure NumPy path for ALL materials
#    (set use_jax_vmap=False in DynamicSolver.__init__)

# 3. Use NumPy RBE2 (true Lagrange) not JAX RBE2
#    (remove rbe2_elements from JAX path)

# 4. Reduce dt_initial to 1e-4, increase max_cutbacks to 50

# 5. Use "quasistatic" mode (no inertia) — already in optimized_v3
```

---

## 6. Verification Checklist (Gate Criteria)

| Gate | Command | Pass Criteria |
|------|---------|---------------|
| **G1: Element Verification** | `python -m verification.run_all --benchmark patch_test_element` | Error < 0.01% (all backends) |
| **G2: Solver Verification** | `python -m verification.run_all --benchmark patch_test_solver` | Error < 0.001% (JAX & NumPy) |
| **G3: Structural Benchmarks** | `python -m verification.run_all` | All 9 benchmarks PASS |
| **G4: Material Tests** | `pytest tests/test_hyperelastic.py tests/test_plasticity.py tests/test_viscoelastic.py -v` | All tests PASS |
| **G5: Constraint Tests** | `pytest tests/test_rbe2.py tests/test_tie.py -v` | All tests PASS |
| **G6: Folding Simulation** | `python examples/ex03_display_fold.py` | Converges to 90° fold (t=1.0) without divergence |

---

## 7. File Index for Implementation

```
dispsolver/
├── element/
│   ├── q4_visco_hybrid_fs_jax.py     # Phase 1.1 — FIX FIRST
│   ├── q4_eas_jax.py                  # Phase 1.2, 1.4, 2.2
│   ├── rbe2_jax.py                    # Phase 1.3
│   ├── q4_jax.py                      # Reference (working)
│   └── q4_up_jax.py                   # Reference
├── material/
│   ├── plastic_jax.py                 # Phase 2.1
│   ├── viscoelastic.py                # Phase 2.3
│   └── linear_viscoelastic.py         # Phase 2.3
├── solver/
│   └── dynamic.py                     # Phase 3.1-3.5, 4.1-4.2
├── constraint/
│   ├── rbe2.py                        # Reference (NumPy Lagrange)
│   ├── tie.py                         # Reference
│   └── contact_jax.py                 # Phase 4.3 (new)
└── state/
    └── field.py                       # UL state management
```

---

## 8. Notes & References

### Key References
- **Simo & Hughes (1998)** — Computational Inelasticity (J2 return map, consistent tangent)
- **Simo & Rifai (1990)** — EAS method (IJNME 29:1595)
- **de Souza Neto et al. (1996)** — F-bar method (IJSS 33:3277)
- **Bonet & Wood (2008)** — Nonlinear Continuum Mechanics for FEM
- **Wriggers (2006)** — Computational Contact Mechanics
- **Crisfield (1997)** — Non-linear Finite Element Analysis (arc-length)

### Critical Formulas
- **J2 Algorithmic Tangent** (Simo 1992): `C_ep = C_el - (C_el : n ⊗ n : C_el) / (n : C_el : n + H)`
- **F-bar**: `F̄ = F * √(J₀/J)` where `J₀ = det(F)` at element center
- **EAS condensation**: `K_e = K_uu - K_ua K_aa⁻¹ K_ua^T`, `f_e = f_u - K_ua K_aa⁻¹ f_a`
- **KKT equilibration**: `D = diag(1/√|J_ii|)`, `J̃ = D J D`, `b̃ = D b`

---

## 9. Next Action

**Start Phase 1.1**: Fix `q4_visco_hybrid_fs_jax.py::_internal_force`

```bash
# After fix, verify:
python -c "from dispsolver.element.q4_visco_hybrid_fs_jax import compute_single; print('OK')"
python -m verification.run_all --benchmark patch_test_element
```

---

*Generated by Atlas (Master Orchestrator) — OhMyOpenCode*