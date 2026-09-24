# Abaqus .inp Solver Runner (CLI & Automation Engine) Implementation Plan

> **Date**: 2026-09-24  
> **Target**: Abaqus `.inp` File Interactive/Automated Execution Engine & Visualizer (VTKHDF + VTU Export)  
> **Location**: `dispsolver/cli/run_inp.py`, `dispsolver/export/vtkhdf_exporter.py`, `examples/run_abaqus_inp.py`, and `tests/test_cli_run_inp.py`  
> **Dev Log Document**: `dev_log/inp_solver_runner_plan_20260924.md`

---

## 1. Executive Summary & Goal

The goal of this task is to provide a user-friendly, production-grade command-line interface (CLI) and script entry point that takes any Abaqus `.inp` deck file path, automatically parses the model, constructs the finite element assembly (2D or 3D), executes the solution using `DynamicSolver2D` / `DynamicSolver3D` with adaptive Newton-Raphson time stepping, streams live iteration progress, and generates high-performance visualization & field data exports (**VTKHDF `.vtkhdf`** HDF5 format for ParaView 9.1+, as well as `.vtu` VTK XML fallback, `.png` plots, and `.json` summary reports).

```
                      Abaqus .inp Deck File
                               │
                               ▼
                    [dispsolver.cli.run_inp]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    [AbaqusInputReader]               [Dimension Detector]
   Keyword Parsing & Model            Auto-detects 2D vs 3D
               │                               │
               └───────────────┬───────────────┘
                               ▼
                     [MultiStepExecutor]
              Numba / JAX Element Kernel Assembly
              Adaptive Newton-Raphson & SDI Loop
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
 [Live Terminal Log]   [VTKHDF Field (.vtkhdf)] [Deformation Plots (.png)]
 Step/Inc/Iter/Energy   High-Perf ParaView 9.1+  Matplotlib / PyVista
                        HDF5 Binary Format
```

---

## 2. Theoretical Basis & Architecture

### 2.1 Keyword Parsing & Pre-Flight Diagnostics
- **Input Deck Parser**: Wraps `dispsolver.io.abaqus_reader.AbaqusInputReader` and `dispsolver.io.abaqus_lexer.tokenize`.
- **Pre-Flight Table**: Before running the solver, prints a diagnostic pre-flight summary:
  - Mesh statistics: node count, element count, element type distribution (e.g. `CPE4I`, `C3D8H`, `CPE4_CR`).
  - Physical features: Materials, Sections, Step sequence, Surface Ties, Contact Pairs, Nodal Transforms (`*TRANSFORM`).
  - Keyword Compatibility Audit: Lists supported keywords vs any unsupported keywords with graceful fallback/warnings.

### 2.2 Dimension Auto-Detection
- Inspects element types defined in `Part` and `Element` blocks.
- 2D element types (`CPE4`, `CPS4`, `CPE4I`, `CPE4H`, `CPE4R`, `CPE6M`, `CPE3`, `CPE8`, etc.) $\to$ `dim = 2`.
- 3D solid element types (`C3D8`, `C3D8I`, `C3D8H`, `C3D8R`, `C3D8_FBAR`, `C3D4`, `C3D10`, `C3D6`, etc.) $\to$ `dim = 3`.
- User can override explicitly via `--dim 2` or `--dim 3`.

### 2.3 Solver Execution Engine
- Uses `MultiStepExecutor` (`dispsolver.solver.step_executor.MultiStepExecutor`).
- Supports `--elem-jit {numba, jax}` backend selection (Numba JIT for production speed, JAX for AutoDiff/R&D).
- Real-time logging streaming step time $t$, increment $\Delta t$, Newton iterations $N_{iter}$, residual norm $\|r\|$, energy balance $\Delta E$, and active contact pair count (SDI).

### 2.4 Field Export: VTKHDF (`.vtkhdf`) Format Specification
- **Primary Export Format**: VTKHDF (`.vtkhdf`) powered by `h5py` (v3.15.1 installed).
- **ParaView / VTK 9.1+ Compatibility**:
  - Group `/VTKHDF`:
    - `Version`: `[1, 0]`
    - `Type`: `"UnstructuredGrid"`
    - `NumberOfPoints`: `[N_nodes]`
    - `NumberOfCells`: `[N_elements]`
    - `NumberOfConnectivityIds`: `[N_conn_total]`
    - `Points`: `(N_nodes, 3)` float64 coordinates (2D coordinates padded with $z=0$)
    - `Types`: `(N_elements,)` uint8 VTK cell codes:
      - CPE4/CPS4/CPE4I/CPE4H $\to$ `VTK_QUAD = 9`
      - CPE3/CPS3 $\to$ `VTK_TRIANGLE = 5`
      - CPE6/CPE6M $\to$ `VTK_QUADRATIC_TRIANGLE = 22`
      - CPE8/CPE8R $\to$ `VTK_QUADRATIC_QUAD = 23`
      - C3D8/C3D8I/C3D8H/C3D8R $\to$ `VTK_HEXAHEDRON = 12`
      - C3D4 $\to$ `VTK_TETRA = 10`
      - C3D10/C3D10M $\to$ `VTK_QUADRATIC_TETRA = 24`
      - C3D6 $\to$ `VTK_WEDGE = 13`
    - `Offsets`: `(N_elements + 1,)` int64 cell connectivity offset indices.
    - `Connectivity`: `(N_conn_total,)` int64 node index array.
    - `PointData/Displacement`: `(N_nodes, 3)` float64 nodal displacement vectors.
    - `PointData/ReactionForce`: `(N_nodes, 3)` float64 nodal reaction forces.
    - `CellData/Stress`: `(N_elements, 6)` float64 Cauchy stress tensors ($\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{yz}, \sigma_{zx}$).
    - `CellData/VonMises`: `(N_elements,)` float64 von Mises equivalent stress scalar.
    - `CellData/PlasticStrain`: `(N_elements,)` float64 equivalent plastic strain scalar.

### 2.5 Output & Visualization Pipeline
- **Console Report**: Table of step execution results, total wall time, convergence status.
- **Summary Report (`summary.json`)**: Detailed JSON report containing solver state, step time history, reaction forces, displacement norms, and warning logs.
- **Field Data Export (`.vtkhdf` & `.vtu`)**:
  - Default: Single/Multi-step **VTKHDF (`.vtkhdf`)** binary file for ultra-fast, zero-overhead ParaView loading.
  - Fallback: Legacy VTK XML Unstructured Grid (`.vtu`) export.
- **Deformable Mesh Plots (`.png`)**:
  - 2D: Automatic `koreanize-matplotlib` 9pt plots showing initial vs deformed mesh and von Mises stress / equivalent plastic strain contours.
  - 3D: PyVista 3D rendering with dark theme, coordinate axes, and right-click interactive projection views.

---

## 3. Implementation Plan & Action Items

### Step 1: Implement VTKHDF Exporter (`dispsolver/export/vtkhdf_exporter.py`)
- Standardized VTKHDF writer using `h5py` for 2D plane strain and 3D solid elements.
- Maps Abaqus element types to VTK cell type codes.
- Supports single time-frame export `export_vtkhdf()` and multi-step time series dataset aggregation.

### Step 2: Implement `dispsolver/cli/run_inp.py` CLI Runner
- Add `argparse` CLI entry point with options:
  - `inp_path` (positional, required): Path to Abaqus `.inp` deck file.
  - `--dim {auto,2,3}`: Spatial dimension (default `auto`).
  - `--elem-jit {numba,jax}`: Element assembly backend (default `numba`).
  - `--export-format {vtkhdf,vtu,both}`: Primary field export format (default `vtkhdf`).
  - `--out-dir OUT_DIR`: Output directory (default `./output_<inp_basename>/`).
  - `--step STEP_NAME`: Execute specific step only.
  - `--no-plot`: Suppress automatic figure plotting.
  - `--quiet`: Suppress verbose terminal progress logging.
- Add `main()` and module execution capability (`python -m dispsolver.cli.run_inp`).

### Step 3: Implement Standalone Example Script `examples/run_abaqus_inp.py`
- Create a clean example script demonstrating programmatic usage of `execute_inp()`, `export_vtkhdf()`, and `AbaqusInputReader`.
- Accepts an `.inp` path as command line argument or falls back to a sample benchmark deck.

### Step 4: Verification & Automated Tests (`tests/test_cli_run_inp.py` & `tests/test_vtkhdf_exporter.py`)
- Test VTKHDF exporter output validation (`h5py` dataset structure, `/VTKHDF/Type`, `/VTKHDF/Points`, `/VTKHDF/Types`, `/VTKHDF/PointData/Displacement`).
- Test CLI runner on 2D plane strain benchmark (`viscocylinder_cpe4i.inp`).
- Test CLI runner on 3D solid element benchmark.
- Verify exit code 0, correct creation of `.vtkhdf`, `summary.json`, `.png`, and accurate numerical results.

---

## 4. Work Log & Living Document

This plan is stored at `./dev_log/inp_solver_runner_plan_20260924.md` and will be updated as implementation progresses.
