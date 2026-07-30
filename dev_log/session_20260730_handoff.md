# Session Handoff Document (2026-07-30)

## 1. Accomplishments Overview

### 1.1 Qt Viewer & Environment Setup
- Integrated `.pkl` loading in `DispFoldApp.py --mode view` and `--view [file.pkl]`.
- Resolved Matplotlib 3.8+ compatibility in `viewer.py` (`mpl.cm.get_cmap` -> `plt.get_cmap`).
- Created project `.venv` using `uv venv .venv` and installed dependencies (`pypardiso`, `mkl`, `intel-openmp`, `jax`, `numba`, `PySide6`, `h5py`, `pytest`).
- Verified PyPardiso MKL solver instantiation inside `.venv`.
- Set `numba` as the default `--elem_jit` backend in `DispFoldApp.py` (3.15x faster element assembly).

### 1.2 Hinge Gap 1.0 mm Folding Benchmark
- Configured `hinge_half_gap = 1.0 mm` and updated `MeshGradingConfig` (`hinge_edge_cluster_width = 0.5`, `hinge_span_half_width = 0.25`).
- Solved full 90°/side (180° total) fold in 266.79s (16 steps, 0 cutbacks). Verified PSA layer carries 99.8% (+70.48 µm) of interlayer shear.

### 1.3 Abaqus-Grade CPE4H / CPE4RH / CPE4IH Hybrid Element Development
- Implemented **Q1P0 Hydrostatic Pressure (u-p Hybrid)** formulation in JAX (`dispsolver/element/q4_hybrid_jax.py`) and Numba C-JIT (`dispsolver/element/q4_hybrid_numba.py`).
- **Static Condensation at Element Level**: Pressure $p$ is statically condensed out at element level ($p = \frac{K}{V_e} \int (J-1) dV$), eliminating Volumetric Locking in nearly incompressible materials ($K \gg \mu$) with **zero global DOF overhead** (100% SPD system matrix, preserving PyPardiso factorisation speed).
- Registered `Q4_HYBRID`, `Q4_HYBRID_RH`, `Q4_COROTATIONAL_HYBRID`, `Q4_COROTATIONAL_HYBRID_RH`, `Q4_HYBRID_EAS`, `Q4_COROTATIONAL_HYBRID_EAS` in `dispsolver/solver/dynamic.py` and `dispsolver/io/model_builder.py`.
- Added `--elem_type` CLI option to `DispFoldApp.py`.

### 1.4 Abaqus Parser Warning Suppression
- Added `_ignored_keywords` set in `dispsolver/io/abaqus_parser.py` for harmless administrative and output control keywords (`*HEADING`, `*PREPRINT`, `*RESTART`, `*OUTPUT`, `*CONTROLS`, `*MONITOR`, etc.).
- Parsing `.inp` files now executes with zero `UserWarning: Unsupported keyword: *HEADING` noise.

### 1.5 Uniform Display Mesh Sizing
- Added `uniform: bool = True` option in `MeshGradingConfig` (`fold_model_config.py`), `display_builder.py`, and `DispFoldApp.py`.
- Enforces equal element width ($dx = \text{constant}$) across the entire panel length $x \in [-40, +40]\text{ mm}$.
- Verified that all 320 element columns have the exact same width $dx = 0.25\text{ mm}$ (`Unique element widths (dx): [0.25]`).

---

## 2. Abaqus Element Mapping Reference Table

| Abaqus Element Tag | Formulation Details | Solver Element Name | Co-rotational Large-Deformation Name |
|---|---|---|---|
| `CPE4` | Standard 4-node plane strain (2x2 Gauss) | `Q4` | `Q4_COROTATIONAL` |
| `CPE4H` | Full 2x2 + Q1P0 u-p Hybrid | `Q4_HYBRID` | `Q4_COROTATIONAL_HYBRID` |
| `CPE4R` | 1-point Reduced Integration + Hourglass Control | `Q4_REDUCED` | `Q4_COROTATIONAL_REDUCED` |
| `CPE4RH` | 1-point Reduced Integration + Hourglass + u-p Hybrid | `Q4_HYBRID_RH` | `Q4_COROTATIONAL_HYBRID_RH` |
| `CPE4I` | 4-mode Incompatible Modes (EAS-4) | `Q4_EAS` | `Q4_COROTATIONAL_EAS` |
| `CPE4IH` | 4-mode Incompatible Modes + u-p Hybrid | `Q4_HYBRID_EAS` | `Q4_COROTATIONAL_HYBRID_EAS` |

---

## 3. Verification & Test Suite Results

1. **Hybrid Element Unit Tests (`tests/test_cpe4h_hybrid.py`)**:
   - `test_q4_hybrid_jax_energy`: PASSED
   - `test_q4_hybrid_numba_kernel_matches`: PASSED
   - `test_volumetric_locking_relief`: PASSED
   - Total: 3 passed in 22.33s.

2. **Full Solver Regression Suite (`python -m verification.run_all`)**:
   - 12 / 12 passed in 221.1s (0 failures, 0 regressions).

---

## 4. Key Files Summary

- [q4_hybrid_jax.py](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element/q4_hybrid_jax.py): JAX element implementation for Q4_HYBRID & Q4_COROTATIONAL_HYBRID_EAS.
- [q4_hybrid_numba.py](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element/q4_hybrid_numba.py): Numba C-JIT multi-threaded kernel for Q4_HYBRID.
- [test_cpe4h_hybrid.py](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/tests/test_cpe4h_hybrid.py): Unit tests for hybrid elements and volumetric locking relief.
- [abaqus_parser.py](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/io/abaqus_parser.py): Suppressed benign keyword warnings.
- [display_builder.py](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/mesh/display_builder.py): Uniform mesh sizing option.
- [DispFoldApp.py](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/examples/DispFoldApp.py): CLI interface with `--mode`, `--elem_jit`, `--elem_type`, and `--mesh_grading`.
