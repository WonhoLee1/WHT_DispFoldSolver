# Abaqus-Style FEA Solver Configuration Architecture Reference (2026-09-06)

## 1. Overview
This document serves as the authoritative reference for the **Abaqus-style FEA Solver Configuration Engine** implemented in `dispsolver/solver/abaqus_config.py` and `dispsolver/fold_model_config.py`.

It maps nonlinear solver parameters directly to standard Abaqus/Standard keyword specifications (`*STEP`, `*DYNAMIC`, `*CONTROLS`, `*SECTION CONTROLS`, `*STABILIZE`), enabling bidirectional `.inp` text generation/parsing, JSON/YAML serialization, dynamic key-path overrides, preset loading, and direct `DynamicSolver.from_config(...)` instantiation.

---

## 2. Abaqus Keyword Mapping Table

| Abaqus FEA Keyword | Python Dataclass / Attribute | Default Value | Description |
|---|---|---|---|
| `*STEP, NLGEOM=YES` | `AbaqusStepConfig.nlgeom` | `True` | Geometric nonlinearity flag |
| `*STEP` (Step Time) | `AbaqusStepConfig.t_total` | `1.0 s` | Total analysis step time |
| `*STEP` (Time Increment) | `dt_init, dt_min, dt_max` | `0.0025, 1e-7, 0.1` | Time increment bounds |
| `*DYNAMIC, APPLICATION=...` | `AbaqusStepConfig.application` | `"MODERATE DISSIPATION"` | Application preset |
| `*DYNAMIC, ALPHA=...` | `AbaqusDynamicControls.alpha` | `-0.22` | HHT-$\alpha$ numerical damping |
| `*CONTROLS, PARAMETERS=TIME INCREMENTATION` | `I_0` | `10` | Max equilibrium iterations before cutback |
| | `I_R` | `16` | SD iteration count threshold |
| | `I_P` | `9` | Max severe discontinuity iterations |
| | `I_C` | `20` | Max cutback attempts allowed per step |
| | `I_L` | `4` | Consecutive low-iteration count to grow $dt$ |
| | `I_G` | `10` | High-iteration count threshold to shrink $dt$ |
| | `dt_growth_factor` | `1.5` | $dt$ growth multiplier |
| | `dt_cutback_factor` | `0.25` | $dt$ cutback reduction factor |
| `*CONTROLS, PARAMETERS=FIELD` | `R_n` | `0.005` (0.5%) | Residual force tolerance ratio |
| | `C_n` | `0.01` (1.0%) | Displacement correction tolerance ratio |
| | `rtol` | `1e-4` | Relative energy / KKT tolerance |
| | `atol` | `1e-6` | Absolute correction tolerance |
| | `max_disp_corr` | `0.03 mm` | Absolute trial step correction cap |
| `*SECTION CONTROLS` | `distortion_control` | `True` | Distortion control for viscoelastic PSA |
| | `distortion_j_crit` | `0.20` | Critical volume ratio $J_{\text{crit}}$ |
| `*STABILIZE` | `enabled`, `factor` | `False`, `0.0002` | Automatic stabilization damping |

---

## 3. Python API & Usage Examples

### 3.1 Basic Initialization & Preset Loading
```python
from dispsolver.fold_model_config import FoldModelConfig
from dispsolver.solver.abaqus_config import AbaqusSolverConfig

# 1. Load predefined preset
config = FoldModelConfig.get_preset("optistruct_galpha")

# 2. Modify via dot-notation key path
config.set_path("solver.max_cutbacks", 20)
config.set_path("drive.dt_max", 0.1)

# 3. Export to Abaqus .inp keyword snippet
abq_cfg = config.to_abaqus_config()
print(abq_cfg.to_inp_snippet())
```

### 3.2 DynamicSolver Instantiation
```python
from dispsolver.solver.dynamic import DynamicSolver

solver = DynamicSolver.from_config(
    mesh=mesh,
    material=materials,
    material_params=material_params,
    config=config,
    elem_jit="numba",
    verbose=True,
)
```

### 3.3 Command-Line Interface (CLI) Usage
Any script using `add_config_cli_args` and `apply_cli_overrides` supports:
```powershell
# Run with external JSON config file
python -u examples/ex13_unified_model_io.py --mode roundtrip --config my_config.json --elem_jit numba

# Run with preset and command-line key overrides
python -u examples/ex13_unified_model_io.py --mode roundtrip --preset teardrop --set solver.max_cutbacks=20 --set solver.integration_mode=generalized-0.8
```

---

## 4. Verification & Testing
- Unit Test Suite: `tests/test_abaqus_config.py` (6/6 PASS).
- System Verification: `python -m verification.run_all` (12/12 PASS).
