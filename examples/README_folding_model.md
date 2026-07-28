Primary entry point: **`examples/DispFoldApp.py`**. All model parameters
(geometry, layer structure, materials, mesh grading, drive/solver
tuning) live in one place: `dispsolver/fold_model_config.py`
(`FoldModelConfig`, default instance `DEFAULT_CONFIG`).

## Quick start

```bash
# 1. Main Unified Application Entry Point
python examples/DispFoldApp.py --mode inp --elem_jit numba       # read Abaqus .inp deck, solve with Numba
python examples/DispFoldApp.py --mode build --elem_jit numba     # build model directly in-memory, solve
python examples/DispFoldApp.py --mode roundtrip --elem_jit jax   # generate .inp in-memory, parse & solve

# 2. Legacy / Mode-specific Entry Points
python examples/ex13_unified_model_io.py --mode read --elem_jit numba
python examples/ex13_unified_model_io.py --mode build --viewer   # solve, then open the Qt viewer on the result
python examples/ex13_unified_model_io.py --open examples/ex12_result.pkl  # skip solving, open saved result
```

All three modes solve the same physical model and print an identical
summary line (`nodes=... elements=... reached_target=...`) when run
with default config — this is how `build`/`read` parity is verified.

## How the pieces connect

```
FoldModelConfig (dispsolver/fold_model_config.py)
        |
        +-- read/roundtrip: gen_ex12_inp.generate(config) -> .inp text
        |                         -> read_abaqus_input() -> ModelBuilderResult
        |
        +-- build: ex13_unified_model_io.run_build(config) -> Python objects directly
                         (same result shape as ModelBuilderResult)
        |
        v
  ex12_abaqus_inp_plate_fold.run_folding_from_result(result, config=...)
        (Newton solver setup, element-type dispatch, time-stepping,
         ResultWriter output — reads config.solver / config.drive)
```

**Important asymmetry**: for `read` mode, the `.inp` file on disk is
the actual input — editing `DEFAULT_CONFIG` and running `--mode read`
on a stale `.inp` changes nothing. You must re-run
`python examples/gen_ex12_inp.py` (or use `--mode roundtrip`, which
regenerates the `.inp` text in-memory every run) to make a
geometry/material config change take effect. `--mode build` has no
intermediate file, so a config edit takes effect on the very next run.
`config.solver`/`config.drive` (Newton tuning, element type, time-
stepping) are **not** representable in Abaqus `.inp` syntax at all —
they take effect immediately in every mode regardless of `.inp` state.

## Config field reference

| Field | Meaning | Default | Affects |
|---|---|---|---|
| `geometry.display_half_length` | Display half-length (mm), \|x\| at the free tip | 40.0 | .inp geometry, build geometry |
| `geometry.hinge_half_gap` | Plate inner edge / free-hinge-span boundary, \|x\| | 10.0 | .inp geometry, build geometry, tie node selection |
| `geometry.hinge_pivot_x` | Hinge pivot offset from center | 3.0 | plate rigid-body pivot |
| `geometry.layer_pattern` | List of `LayerSpec(material_name, thickness_mm, n_rows)`, one repeating unit -- `material_name` must be a key in `materials.definitions` | `[PSA 0.03mm/1 row, PET 0.05mm/3 rows]` (layer 1 = PSA, layer 2 = PET, ...) | display layer stack |
| `geometry.n_layer_pairs` | Times `layer_pattern` repeats | 7 | total layer count / thickness |
| `geometry.plate_thickness` | Rigid plate thickness (mm) | 0.5 | plate mesh |
| `geometry.plate_mesh_nx` / `plate_mesh_ny` | Plate mesh density | 30 / 2 | plate mesh |
| `grading.tip_dx` / `hinge_edge_dx` / `hinge_span_dx` / `plate_body_dx` | Element size (mm) per x-zone | 0.25 / 0.25 / 0.5 / 1.0 | display mesh element count |
| `grading.tip_cluster_width` / `hinge_edge_cluster_width` / `hinge_span_half_width` | Zone widths (mm) | 2.0 / 2.0 / 8.0 | display mesh zone boundaries |
| `materials.definitions["PET"]` | `MaterialDef(id, name, type, params={E, nu, sigma_y0, H})` | E=4000, nu=0.3, sigma_y0=80, H=400 | PET layers |
| `materials.definitions["PSA"]` | `MaterialDef(..., params={mu, lambda_m, K, prony_g, prony_tau, wlf_T_ref, wlf_C1, wlf_C2})`. `mu`/`K` derived from a target small-strain modulus `E` via `mu=E/(2*(1+nu))`, `K=E/(3*(1-2*nu))` at `nu=0.49` | `E=0.05 MPa` -> `mu=0.016779, K=0.83333` | PSA layers |
| `materials.definitions["STEEL"]` | `MaterialDef(..., params={E, nu})` | E=20000, nu=0.3 | plate material (E forced ~0 at solve time regardless, see `run_folding_from_result`) |
| `drive.theta_max_deg` | Fold angle per plate (deg) | 90.0 | hinge rotation BC |
| `drive.t_total` / `dt_init` / `dt_max` / `dt_min` | Adaptive-dt time bounds | 1.0 / 0.005 / 0.01 / 1e-5 | time stepping |
| `solver.tol` / `rtol` / `atol` / `max_iter` | Newton convergence | 1e-3 / 1e-4 / 1e-6 / 25 | Newton solve |
| `solver.ul_mode` / `alpha` | Updated-Lagrangian mode, HHT-alpha damping | True / -0.15 | Newton solve |
| `solver.theta_penalty_k` | Rotation-BC penalty stiffness | 1e8 | hinge BC enforcement |
| `solver.target_iters` | Adaptive-dt controller target Newton iteration count | 5 | time stepping |
| `solver.pet_element_type` / `psa_element_type` | Element formulation per layer family | `Q4_COROTATIONAL` / `Q4_VISCO_SIMO` | element dispatch |

## Task-oriented recipes

**Change the layer structure** (add/remove/reorder layers, change per-layer
thickness or row count): edit `geometry.layer_pattern` and/or
`geometry.n_layer_pairs`. Then re-run `gen_ex12_inp.py` (for `read`) or
just re-run `--mode build`/`--mode roundtrip`.

**Change display length**: edit `geometry.display_half_length`. The
plate x-range moves with it automatically (plate spans
`hinge_half_gap` to `display_half_length`, mirrored — not an
independent field, so it can't disagree with the tip/hinge geometry).

**Change hinge pivot position**: edit `geometry.hinge_pivot_x`.

**Change plate length**: not directly settable — plate length is
derived from `display_half_length - hinge_half_gap` by design (see
above). Change one of those two instead.

**Change plate thickness / mesh density**: `geometry.plate_thickness`,
`geometry.plate_mesh_nx`/`plate_mesh_ny`.

**Change element size**: `grading.tip_dx` / `hinge_edge_dx` /
`hinge_span_dx` / `plate_body_dx` (mm). Segment point counts are
derived from `length/dx`, so this is a real element-size knob
regardless of geometry changes.

**Change element type**: `solver.pet_element_type` /
`solver.psa_element_type` (any type string `DynamicSolver` recognizes,
e.g. `"Q4"`, `"Q4_EAS"`).

**Change solver time-stepping**: `drive.t_total` / `dt_init` /
`dt_max` / `dt_min`, `solver.target_iters`.

**Change Newton tolerances**: `solver.tol` / `rtol` / `atol` /
`max_iter`.

## Postprocess viewer

`--viewer` (open the just-solved result) / `--open PKL_PATH` (skip
solving, open an existing saved result) both launch
`dispsolver.postprocess.viewer.launch_from_result()`. In the viewer:

- Mouse scroll = zoom in/out anchored at the cursor; middle-button drag
  = pan (in addition to the standard matplotlib toolbar's pan/zoom-
  rectangle/home buttons). "Fit View" button resets to the full current
  shape. The view no longer resets when you change field/step/layer
  visibility.
- The Part/Layer panel labels rigid-plate parts as "Plate Left"/"Plate
  Right" (they aren't display layers); ordinary display layers keep
  "Layer N (material)". Scrolls independently if the layer list is
  taller than the panel.
- A console "MODEL REVIEW" (distinct materials + representative
  properties, per-part element counts) prints once at solver startup
  and once when a saved result is opened in the viewer.

## Building a custom config

```python
from dispsolver.fold_model_config import FoldModelConfig, DEFAULT_CONFIG
import dataclasses

my_config = dataclasses.replace(
    DEFAULT_CONFIG,
    geometry=dataclasses.replace(DEFAULT_CONFIG.geometry, display_half_length=50.0),
)

# read/roundtrip: regenerate the .inp with the new geometry
from gen_ex12_inp import generate
with open("examples/ex12_rigid_plate_display_fold.inp", "w") as f:
    f.write(generate(my_config))

# build: pass config straight through, no .inp involved
from ex13_unified_model_io import run_build
run_build(my_config)
```

There is no CLI flag surface for individual fields — `ex13_unified_model_io.py`
keeps its `--mode` flag only. To change model parameters, construct/edit
a `FoldModelConfig` in a script as above.
