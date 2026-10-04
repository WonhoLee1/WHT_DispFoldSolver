# Master Implementation Plan: 14 Abaqus Benchmark Failure Resolution

## Executive Summary
This master implementation plan addresses the 14 failing Abaqus `.inp` benchmark decks out of the 20-benchmark evaluation suite for **WHTSolver**. By fixing initial stress broadcasting, reference node kinematic coupling, SDV state variable array sizing, and field export tensor formatting, WHTSolver will achieve a **100% (20/20) pass rate** with 100-character aligned CAE-grade `.sta` and `.msg` logs.

---

## Technical Failure Categorization & Fix Strategy

### Group 1: Predefined Initial Stress Broadcast & Component Mismatches (6 Benchmarks)
- **Affected Benchmarks**: `cook_2d`, `nlgeocantilever_c3d8i_coarse`, `roofsnapthrough_s4r`, `hertzcontact_2d_rsurf`, `pipecrushing_cpe4i`, `viscocylinder_cax4i_dyn`
- **Root Cause**: In `dispsolver/model/model.py` (lines 1025-1058 and 1098-1125), initial stress assignments (`*INITIAL CONDITIONS, TYPE=STRESS`) assume fixed 3-component (2D) or 6-component (3D) arrays, raising `ValueError: could not broadcast input array` when `state.stress_initial` dimension differs or `pf.values` is empty/6-comp.
- **Resolution Plan**:
  1. In `model.create_solver2d` and `model.create_solver3d`, dynamically check target array shape `n_comp = state.stress_initial.shape[0]`.
  2. Slice or zero-pad `pf.values` to fit `n_comp` exactly.
  3. Handle empty `pf.values` gracefully with zero-vector fallback.
  4. Update `vtkhdf_exporter.py` and `vtu_exporter.py` stress component broadcasting for arbitrary `(N_elems, n_comp)`.

---

### Group 2: Unconstrained Rigid Body & Reference Node Kinematic Coupling (4 Benchmarks)
- **Affected Benchmarks**: `neckingtensilebar_cax4r`, `rubberdisk_cax4ih_poly`, `ex03_display_fold`, `ex12_rigid_plate_display_fold`
- **Root Cause**: Reference nodes defined via `*RIGID BODY`, `*COUPLING`, or `*DISTRIBUTING` remain uncoupled in global stiffness matrix $K$, leaving zero diagonal rows that cause Pardiso linear solver failures (`Matrix A is singular, because it contains empty row(s)`).
- **Resolution Plan**:
  1. Integrate kinematic slave condensation in `model.create_solver` / `step_executor.py` for reference node DOFs.
  2. Bind reference point translational/rotational DOFs to target node set surfaces via `RBE2CondensationManager` or master-slave constraint matrices.

---

### Group 3: Integration Point SDV & 8-Node Element Array Length Mismatches (2 Benchmarks)
- **Affected Benchmarks**: `thickcylcreep1`, `elasticsheet_cps8r_biderman`
- **Root Cause**: `ValueError: all index and data arrays must have the same length` during scipy CSR matrix assembly or creep state variable initialization for quadratic 8-node elements (`CPS8R`, `CPE8R`).
- **Resolution Plan**:
  1. Fix `elem_sdvs` allocation in `dispsolver/material/creep.py` and `dispsolver/element2d/` to match integration point count $N_{\text{gp}}$ rather than node count.
  2. Ensure boundary node set expansion accounts for 8-node quadratic element topology in `assembly.py`.

---

### Group 4: CAE Field Exporters & Load Stepping Validation (2 Benchmarks)
- **Affected Benchmarks**: `thick_cylinder_2d`, `nlgeocantilever_cps4i_mload`
- **Root Cause**: Exporter tensor shape mismatch when `export_format="both"` (`vtu` + `vtkhdf`) is active during multi-step static runs.
- **Resolution Plan**:
  1. Standardize stress and strain tensor exports in `vtu_exporter.py` and `vtkhdf_exporter.py` for 2D plane strain/stress and 3D continuum elements.
  2. Verify execution under both `elem_jit="numba"` and `elem_jit="jax"`.

---

## Verification & Deliverables Plan

1. **Step 1: Execute Targeted Code Edits**
   - Apply fixes to `dispsolver/model/model.py`, `dispsolver/export/vtkhdf_exporter.py`, `dispsolver/export/vtu_exporter.py`, and `dispsolver/solver/step_executor.py`.

2. **Step 2: Run 20-Benchmark Evaluation Suite**
   - Run `python benchmark_abaqus_compatibility/run_all_20_benchmarks.py`.
   - Verify 20/20 (100%) PASS rate with zero cutbacks / exceptions.

3. **Step 3: Update CAE Log Artifacts & Reports**
   - Save updated `benchmark_results.json` and `README.md` in `benchmark_abaqus_compatibility/`.
   - Confirm 100-character line width alignment in all generated `.sta` and `.msg` files.
