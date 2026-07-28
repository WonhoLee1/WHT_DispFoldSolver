"""
fold_model_config.py
=====================
Single source of truth for every tunable number in the ex12/ex13
display-folding model -- geometry, mesh grading (element size), layer
structure, materials, drive/time-stepping, and Newton-solver tuning.

Why this exists
---------------
Before this module, the same numbers (hinge pivot at x=+-3, hinge/plate
boundary at x=+-10, the 14-layer PET/PSA thickness pattern, ...) were
hard-coded independently in `gen_ex12_inp.py` (writes the .inp) and
`ex13_unified_model_io.py::run_build()` (builds the identical model as
Python objects, no .inp). Nothing enforced the two copies staying equal
-- exactly the kind of thing that already broke `build`/`read` parity
once this session (an `element_type` dict desync after a layer-count
change). Both now read from one `FoldModelConfig` instance instead of
carrying their own copy of the numbers.

What `DEFAULT_CONFIG` covers, and how it maps to the model
------------------------------------------------------------
- `geometry`: display half-length, hinge pivot/gap position, the layer
  stack (material family + thickness + element-row count per physical
  layer, repeated `n_layer_pairs` times), plate thickness/mesh density.
  Plate x-range is *derived* (`hinge_half_gap` .. `display_half_length`,
  mirrored) rather than a separate field -- it cannot disagree with the
  hinge/tip geometry because there is nowhere for a second, conflicting
  number to live.
- `grading`: element *size* (dx) in each x-zone of `_graded_display_x()`
  (tip cluster / hinge-edge cluster / free hinge span / under-plate
  body), not point counts -- segment point counts are derived from
  `length/dx` so geometry changes don't silently change element size
  in some zone the way hard-coded integer counts would.
- `materials`: PET (elastic-plastic), PSA (Arruda-Boyce + Prony + WLF
  viscoelastic), STEEL (rigid-plate placeholder, E is overridden near-
  zero at solve time regardless of what's set here -- see
  `run_folding_from_result`).
- `drive`: fold angle target and the adaptive-dt controller's time
  bounds.
- `solver`: Newton tolerances/iteration budget and which element
  formulation each material family uses -- none of this is expressible
  in Abaqus `.inp` text (no keyword carries Newton `tol`/`ul_mode`/...),
  so it only ever comes from this config, for every mode.

What editing this config does NOT do
-------------------------------------
For `read`/`roundtrip` modes, the `.inp` file (or its in-memory text)
is what actually gets parsed -- `gen_ex12_inp.generate(config)` must be
re-run (or `roundtrip` mode re-invoked, which calls it internally) for a
`geometry`/`materials`/`drive`-affecting config change to take effect.
Editing `DEFAULT_CONFIG` and re-running `ex13_unified_model_io.py
--mode read` on a *stale* `.inp` file changes nothing. `build` mode has
no intermediate file, so a config change takes effect on the very next
run. `solver` tuning fields affect all three modes immediately (they
never touch `.inp` text in the first place).

See `examples/README_folding_model.md` for the task-oriented "I want to
change X" guide built on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from dispsolver.material.type_tags import J2_PLASTIC, ARRUDA_BOYCE_VISCO, NEOHOOKEAN


@dataclass
class MaterialDef:
    """One entry in the materials registry (`MaterialsConfig.definitions`).

    `id` is a small stable display-order integer used only by the
    startup review print (`dispsolver/postprocess/model_review.py`) --
    it is NOT the solver pid. `name` is the literal *MATERIAL name
    written to .inp text and the exact string `LayerSpec.material_name`
    must reference. `type` is one of the canonical tags in
    `dispsolver/material/type_tags.py`, used by `material_factory.py` to
    build either a live object or .inp text, and later to pick which
    properties to show in the review print.
    """
    id: int
    name: str
    type: str
    params: Dict


@dataclass
class LayerSpec:
    """One physical layer within a repeating unit of the display stack.

    `material_name` must be an exact key into
    `MaterialsConfig.definitions` -- it is the literal *MATERIAL name
    that multiple physical layers share (e.g. every PET layer uses
    material_name="PET"), not a per-layer-unique generated name. Each
    physical layer still gets its own pid, but that now comes purely
    from *SOLID SECTION/ELSET structure (one section per physical
    layer) -- see `dispsolver/io/model_builder.py::_build_sections` --
    independent of whether its material name repeats across layers.
    """
    material_name: str
    thickness_mm: float
    n_rows: int


@dataclass
class GeometryConfig:
    display_half_length: float = 40.0
    # Plate inner edge / free-hinge-span boundary, |x|. The plate spans
    # from here out to display_half_length; the hinge pivot sits inside
    # this gap (hinge_pivot_x < hinge_half_gap) -- see AGENTS.md 1.0 for
    # why the pivot is set inside the plate's own footprint.
    hinge_half_gap: float = 10.0
    hinge_pivot_x: float = 3.0

    # One repeating unit of the layer stack + how many times to repeat
    # it. Total physical layers = n_layer_pairs * len(layer_pattern);
    # total thickness = n_layer_pairs * sum(l.thickness_mm for l in layer_pattern).
    # PSA first, then PET -- layer 1 = PSA, layer 2 = PET, layer 3 = PSA, ...
    layer_pattern: List[LayerSpec] = field(default_factory=lambda: [
        LayerSpec("PSA", 0.03, 1),
        LayerSpec("PET", 0.05, 3),
    ])
    n_layer_pairs: int = 7

    plate_thickness: float = 0.5
    plate_mesh_nx: int = 60
    plate_mesh_ny: int = 2


@dataclass
class MeshGradingConfig:
    """Element size (dx, mm) per x-zone of the graded display mesh.

    Zone layout (mirrored about x=0), outermost to innermost:
        tip cluster            [display_half_length - tip_cluster_width, display_half_length]
        plate body (coarse)    [hinge_half_gap + hinge_edge_cluster_width, display_half_length - tip_cluster_width]
        hinge-edge cluster     [hinge_half_gap - hinge_edge_cluster_width, hinge_half_gap + hinge_edge_cluster_width]  (roughly)
        free hinge span        [-hinge_span_half_width, hinge_span_half_width]
    `hinge_span_half_width` must be < `hinge_half_gap` (the moderately-
    fine free-hinge-span zone sits strictly inside the hinge-edge
    clusters, per the original tip-inversion fix, AGENTS.md 4.12).
    """
    tip_cluster_width: float = 2.0
    tip_dx: float = 0.125
    hinge_edge_cluster_width: float = 2.0
    hinge_edge_dx: float = 0.125
    hinge_span_half_width: float = 8.0
    hinge_span_dx: float = 0.25
    plate_body_dx: float = 0.5


@dataclass
class MaterialsConfig:
    """Registry of distinct materials, keyed by name -- the same string
    used as `*MATERIAL, NAME=...` in .inp text and as
    `LayerSpec.material_name`. A name appears here exactly once no
    matter how many physical layers/sections reference it: dedup is
    structural now, not a naming convention. `gen_ex12_inp.py` emits
    exactly one `*MATERIAL` block per entry; `dispsolver/io/
    model_builder.py`'s `_build_sections`/`_build_materials` assign one
    pid per `*SOLID SECTION` but share the built material object/params
    across every pid whose section names the same material.
    """
    definitions: Dict[str, MaterialDef] = field(default_factory=lambda: {
        "PET": MaterialDef(
            id=1, name="PET", type=J2_PLASTIC,
            params={"E": 4000.0, "nu": 0.3, "sigma_y0": 80.0, "H": 400.0},
        ),
        # Arruda-Boyce base (mu/lambda_m/K) + Prony (single term) + WLF.
        # lambda_m is an assumed locking-stretch shape parameter, not
        # measured -- see AGENTS.md 1.4. mu/K derived from a target small-
        # strain modulus E=0.05 MPa (nu=0.49, near-incompressible) via the
        # standard isotropic relations mu=E/(2*(1+nu)), K=E/(3*(1-2*nu)) --
        # same derivation/nu as the previous E=0.5 MPa values, just 10x
        # softer (both mu and K are linear in E for fixed nu).
        "PSA": MaterialDef(
            id=2, name="PSA", type=ARRUDA_BOYCE_VISCO,
            params={
                "mu": 0.016779, "lambda_m": 3.0, "K": 0.83333,
                "prony_g": [0.20], "prony_tau": [3.33],
                "wlf_T_ref": 25.0, "wlf_C1": 17.0, "wlf_C2": 51.6,
            },
        ),
        "STEEL": MaterialDef(
            id=3, name="STEEL", type=NEOHOOKEAN,
            params={"E": 20000.0, "nu": 0.3},
        ),
    })


@dataclass
class DriveConfig:
    theta_max_deg: float = 90.0
    t_total: float = 1.0
    dt_init: float = 0.005
    dt_max: float = 0.1
    dt_min: float = 1e-5


@dataclass
class SolverTuningConfig:
    tol: float = 1e-3
    rtol: float = 1e-4
    atol: float = 1e-6
    max_iter: int = 25
    ul_mode: bool = True
    alpha: float = -0.15
    theta_penalty_k: float = 1e8
    target_iters: int = 5
    # Element formulation per material name -- keyed the same way as
    # LayerSpec.material_name. Anything not PET/PSA (i.e. the rigid
    # plate's STEEL) is left to DynamicSolver's "Q4" default.
    pet_element_type: str = "Q4_COROTATIONAL"
    psa_element_type: str = "Q4_VISCO_SIMO"


@dataclass
class FoldModelConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    grading: MeshGradingConfig = field(default_factory=MeshGradingConfig)
    materials: MaterialsConfig = field(default_factory=MaterialsConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    solver: SolverTuningConfig = field(default_factory=SolverTuningConfig)


DEFAULT_CONFIG = FoldModelConfig()
