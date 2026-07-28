# Implementation Plan — Complete Numba LLVM JIT Backend Expansion for All Elements

## Goal Description
Expand the Numba LLVM JIT backend (`dispsolver/element/q4_numba.py`) to cover **all remaining element formulations** in the `dispsolver` codebase. Every newly implemented Numba element will be registered in `verification/element_backends.py` and included in the verification suite (`python -u -m verification.run_all`) for 100% precision and speed validation.

---

## User Review Required

> [!IMPORTANT]
> **Verification Inclusion Requirement**:
> Each Numba element will be added to `verification/element_backends.py` and benchmarked directly in `verification/benchmarks.py` to ensure exact match against theory and JAX/NumPy backends.
> 
> - **Element Types to Implement in Numba & Verify**:
>   1. `Q4_UP` (Q1P0 Mixed u-p Pressure Condensation) → `numba_q4_up`
>   2. `Q4_COROTATIONAL` (Co-rotational Large-Rotation) → `numba_q4_corotational`
>   3. `T3` (3-Node Triangular Constant Strain) → `numba_t3`
>   4. `RBE2` (Kinematic Master-Slave Condensation) → `numba_rbe2`
>   5. `Q4_VISCO_FS` & `Q4_SIMO_FS` (Finite-Strain Viscoelastic F-bar) → `numba_q4_visco_fs`, `numba_q4_simo_fs`

---

## Proposed Changes

### 1. `dispsolver/element/q4_numba.py` [MODIFY]
Add `@numba.njit` kernels:

#### [ADD] `compute_q4_up_element`
- Q1P0 mixed u-p element with 1-point pressure condensation $P = K_b \cdot \text{div}(u)$.

#### [ADD] `compute_q4_corotational_element`
- Extract element rigid rotation angle $\theta_e$ from deformed element diagonal/edge vectors.
- Transform local stiffness: $\mathbf{K}_{\text{global}} = \mathbf{T}_e^T \mathbf{K}_{\text{local}} \mathbf{T}_e + \mathbf{K}_{\text{geo}}$.

#### [ADD] `compute_t3_element`
- Constant Strain Triangle (CST) 3x3 strain-displacement operator $B$.

#### [ADD] `compute_rbe2_numba`
- Master-slave transformation matrix $\mathbf{T}_e$ for rigid body nodes.

#### [ADD] `compute_q4_visco_fs_numba` & `compute_q4_simo_fs_numba`
- Finite-strain viscoelastic F-bar element stiffness and internal force.

---

### 2. `verification/element_backends.py` [MODIFY]
Register all new Numba backends in `BACKENDS`:
- `'numba_q4_up'`
- `'numba_q4_corotational'`
- `'numba_t3'`
- `'numba_rbe2'`
- `'numba_q4_visco_fs'`
- `'numba_q4_simo_fs'`

---

### 3. `dispsolver/solver/dynamic.py` [MODIFY]
Add `backend='numba'` solver dispatch for full-simulation runs.

---

## Verification Plan

### Automated Tests
1. Run full verification suite with all Numba backends included:
   ```bash
   python -u -m verification.run_all
   ```
2. Run speed benchmark to measure performance across all backends:
   ```bash
   python -u -m verification.run_all --include-speed
   ```
3. Update `verification/results/verification_report_20260729.md`.
