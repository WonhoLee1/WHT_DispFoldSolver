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

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import argparse

from dispsolver.material.type_tags import J2_PLASTIC, ARRUDA_BOYCE_VISCO, NEOHOOKEAN
from dispsolver.solver.abaqus_config import (
    AbaqusSolverConfig,
    AbaqusStepConfig,
    AbaqusTimeIncrementationControls,
    AbaqusFieldControls,
    AbaqusDynamicControls,
    AbaqusSectionControls,
    AbaqusStabilizeControls,
)


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
    # this gap (hinge_pivot_x < hinge_half_gap) so that at 90-degree fold,
    # the plates stand parallel ("11자") with neck gap = 2 * hinge_pivot_x
    # to form the teardrop (물방울) loop.
    hinge_half_gap: float = 7.5
    hinge_pivot_x: float = 0.8
    # Where the plate<->display surface tie actually starts, |x|. None (default)
    # = tie starts right at hinge_half_gap (the original, full-length tie
    # covering the whole plate-body span). Set higher (e.g. 12.0) to leave a
    # buffer strip [hinge_half_gap, tie_attach_start) UNTIED/free, so the
    # free-hinge span's compatibility requirement at the boundary is relaxed
    # rather than an abrupt "rigid the instant x >= hinge_half_gap" transition.
    tie_attach_start: Optional[float] = None
    # Pivot height. 0.0 = on the tie/attachment plane (plate top surface,
    # display bottom surface -- both sit at y=0, verified against the
    # generated .inp, see dev_log). Raising this lifts the rotation
    # center into the display stack (e.g. 0.25 = mid-thickness/neutral
    # axis of the 0.5mm stack) -- changes neck gap to
    # 2*hinge_pivot_x + 2*hinge_pivot_y (was 2*hinge_pivot_x at y=0).
    hinge_pivot_y: float = 0.0

    # One repeating unit of the layer stack + how many times to repeat
    # it. Total physical layers = n_layer_pairs * len(layer_pattern);
    # total thickness = n_layer_pairs * sum(l.thickness_mm for l in layer_pattern).
    # PSA first, then PET -- layer 1 = PSA, layer 2 = PET, layer 3 = PSA, ...
    layer_pattern: List[LayerSpec] = field(default_factory=lambda: [
        LayerSpec("PSA", 0.03, 1),
        LayerSpec("PET", 0.06, 3),
    ])
    n_layer_pairs: int = 7
    # Optional explicit per-layer list. If set, overrides layer_pattern * n_layer_pairs.
    # Default: 6 PSA/PET pairs + 1 PSA + 1 GLASS (top layer = 60um Glass 70 GPa)
    custom_layers: Optional[List[LayerSpec]] = field(default_factory=lambda: [
        LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.06, 3),
        LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.06, 3),
        LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.06, 3),
        LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.06, 3),
        LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.06, 3),
        LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.06, 3),
        LayerSpec("PSA", 0.03, 1), LayerSpec("GLASS", 0.06, 3),
    ])

    # Cutouts ("tape" voids): physical layer index (1-based, matching the
    # DISP_LAYER01..NN elsets and the Qt viewer's Part/Layer list) -> list
    # of (x_start, x_end) ABSOLUTE-x intervals where that layer has NO
    # elements. Layers absent from this dict are fully filled.
    layer_void_regions: Dict[int, List[Tuple[float, float]]] = field(
        default_factory=lambda: {1: [(-7.5, 0.0), (0.0, 7.5)]}
    )

    plate_thickness: float = 0.5
    plate_mesh_nx: int = 60
    plate_mesh_ny: int = 4


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
    tip_dx: float = 0.25
    hinge_edge_cluster_width: float = 1.0
    hinge_edge_dx: float = 0.25
    hinge_span_half_width: float = 5.0
    hinge_span_dx: float = 0.25
    plate_body_dx: float = 0.50
    uniform: bool = False

    def __post_init__(self):
        # Enforce graded-mesh invariant from display_builder: span must lie inside hinge cluster
        # hinge_lo = hinge_half_gap - hinge_edge_cluster_width (validated in display_builder)
        # This catches config drift before mesh generation.
        if self.hinge_span_half_width < -1e-12 or self.tip_cluster_width < -1e-12:
            raise ValueError("MeshGradingConfig: widths must be non-negative")
        if self.plate_body_dx <= 0 or self.tip_dx <= 0 or self.hinge_edge_dx <= 0 or self.hinge_span_dx <= 0:
            raise ValueError("MeshGradingConfig: dx values must be positive")


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
        "PSA": MaterialDef(
            id=2, name="PSA", type=ARRUDA_BOYCE_VISCO,
            params={
                # Arruda-Boyce 8-chain: the "mu" parameter is NOT the initial
                # shear modulus. For this code's series (_AB_C in
                # q4_visco_simo_fs_jax) G0 = mu * 1.074550 at lambda_m = 3.0,
                # so the parameter must be de-scaled to hit the target modulus:
                #   target E = 0.05 MPa, nu = 0.49
                #   -> G0 = E/(2(1+nu)) = 0.016779 MPa,  K = E/(3(1-2nu)) = 0.83333 MPa
                #   -> mu = G0 / 1.074550 = 0.015614 MPa
                # Using 0.016779 directly (as before) made the layer 7.5% stiff
                # (E_eff = 0.0537 MPa).
                "mu": 0.015614, "lambda_m": 3.0, "K": 0.83333,
                "prony_g": [0.12, 0.08], "prony_tau": [0.7, 7.0],
                "wlf_T_ref": 25.0, "wlf_C1": 17.0, "wlf_C2": 51.6,
            },
        ),
        "STEEL": MaterialDef(
            id=3, name="STEEL", type=NEOHOOKEAN,
            params={"E": 20000.0, "nu": 0.3},
        ),
        "GLASS": MaterialDef(
            id=4, name="GLASS", type=J2_PLASTIC,
            params={"E": 70000.0, "nu": 0.22, "sigma_y0": 1e9, "H": 0.0},
        ),
    })


@dataclass
class DriveConfig:
    theta_max_deg: float = 90.0
    t_total: float = 1.0
    dt_init: float = 0.0025
    dt_max: float = 0.01
    dt_min: float = 1e-5


@dataclass
class SolverTuningConfig:
    tol: float = 1e-2
    rtol: float = 1e-4
    atol: float = 1e-6
    max_iter: int = 50
    max_cutbacks: int = 20
    # NLGEOM -- the Abaqus-standard geometric-nonlinearity switch, and the
    # one you should set. Turning it on does NOT mean "every element runs
    # Updated Lagrangian": each element applies the large-deformation
    # treatment its own formulation requires (declared in
    # dynamic.py::_ELEMENT_LARGE_DEF and resolved per pid by
    # DynamicSolver._use_ul_for). The SRI family, for instance, stays Total
    # Lagrangian even with NLGEOM on because its shear sampling is written
    # in Cartesian components and is not frame-invariant -- forcing UL on it
    # silently corrupts the stiffness (dev_log/
    # plan_abaqus_element_consolidation_20260908.md sections 8-9).
    # Call `solver.element_large_deformation_report()` to see what each pid
    # actually ends up running.
    nlgeom: bool = True
    # DEPRECATED aliases, kept so existing configs/examples keep working.
    # They feed the same switch; prefer `nlgeom`.
    ul_large_rotation_mode: bool = True
    ul_mode: bool = True
    integration_mode: str = "moderate-4"
    alpha: float = -0.22
    theta_penalty_k: float = 1e8
    target_iters: int = 6
    # Element formulation per material name -- keyed the same way as
    # LayerSpec.material_name. Anything not PET/PSA/GLASS (i.e. the rigid
    # plate's STEEL) is left to DynamicSolver's "Q4" default.
    # PET/GLASS on Abaqus CPE4R (J2), PSA on CPE4H -- user decision 2026-10-10.
    # Was Q4_COROTATIONAL_SRI (PET/GLASS) / CPE4I (PSA); SRI's bending stiffness
    # swings 1.2x -> 20-316x with element orientation (AGENTS.md 4.1 C6 table).
    pet_element_type: str = "CPE4R"
    # *SECTION CONTROLS, HOURGLASS= for the CPE4R pids. ENHANCED is set here on
    # purpose: the Abaqus/Standard default for an elastic-plastic material is
    # STIFFNESS (DynamicSolver uses that when this is None), whose 0.005G hourglass
    # stiffness makes one-row-per-layer bending mesh dependent. ENHANCED reproduces
    # CPE4I's elastic bending stiffness exactly (element/cpe4r_jax.py).
    hourglass: Optional[str] = "ENHANCED"
    hourglass_scale: float = 1.0   # Abaqus data-line ss; STIFFNESS only
    # How the plate<->display surface tie is enforced.
    #   "penalty"   : SurfaceTieConstraint spring (gap = F/k_tie; the finite
    #                 stiffness cannot hold once the laminate bending reaction
    #                 grows -- measured gap 0 -> 34mm by 68 deg fold).
    #   "kinematic" : master-slave (MPC) elimination -- the slave node follows
    #                 N1*u_m1 + N2*u_m2 exactly. This is what Abaqus *TIE does
    #                 by default; the plate is rigid and driven by an exact
    #                 RBE2 condensation, so the tied display DOFs are a closed-
    #                 form function of theta and are prescribed directly. Zero
    #                 gap by construction, no penalty parameter.
    # Measured 2026-10-09 (dev_log/ex12_fold_wall_root_cause_20261009.md): on the deck with the
    # untied, contact-free strip every kinematic variant died at 1-4 deg because the display fell
    # through the plate there; with the tie over the whole plate and residual-checked convergence,
    # kinematic + CPE4I reaches 49.5 deg/side (penalty 39.0 deg before the convergence fix).
    tie_method: str = "kinematic"
    # Frictionless contact of the display bottom against each rigid plate top
    # (constraint/rigid_segment_contact2d.py). Needed wherever the display rests on
    # a plate without being tied -- the [hinge_half_gap, tie_attach_start) strip --
    # otherwise the display passes through the plate (measured 2026-10-09).
    plate_contact: bool = False
    plate_contact_k: float = 1.0e4   # penalty stiffness [N/mm per node, unit thickness]
    # open gap [mm] within which the contact TANGENT (not the force) still carries the penalty;
    # see RotatingRigidSegmentContact2D.stiffness_gap_tol. 0 = plain active-set tangent.
    plate_contact_gap_tol: float = 0.0
    # > 0: C1 softened pressure-overclosure over this clearance [mm] (contact pressure rises from 0
    # at g0 to k*g0/2 at touch, then linear); avoids open/closed chattering against the compliant display
    plate_contact_softening: float = 0.0
    # > 0: round the plate's inner (hinge-side) edge with this radius [mm] for contact, as Abaqus'
    # FILLET RADIUS on an analytical rigid surface; a sharp edge chatters contact on/off
    plate_contact_fillet: float = 0.0
    # Abaqus 2026 contact CONVERGENCE CHECK: STRICT (default, unchanged global test) | MODERATE |
    # RELAXED | CONCEPT DESIGN, applied to the DOFs the plate contact acts on
    # (DynamicSolver.set_contact_convergence_check).
    contact_convergence_check: str = "STRICT"
    psa_element_type: str = "CPE4H"   # see pet_element_type; CPE4H locks in bending at one row, see the node-id-overflow assert in gen_ex12_inp.py
    glass_element_type: str = "CPE4R"
    # PARDISO linear solver options:
    # "auto" (default: detects SPD vs Indefinite vs Nonsymmetric based on constraints),
    # "spd" (mtype=2, Cholesky LL^T, fastest, requires strictly SPD and upper triangle),
    # "indefinite" (mtype=-2, LDL^T, for saddle-point/Lagrange multipliers or softening),
    # "nonsymmetric" (mtype=11, LU, general fallback)
    pardiso_mtype: str = "auto"
    pardiso_phase_reuse: bool = True  # Reuse Phase 11 (symbolic analysis) across Newton iters
    max_displacement_corr: Optional[float] = None  # None = no artificial clipping during large-rotation folding
    # Abaqus-grade Distortion Control (*SECTION CONTROLS, DISTORTION CONTROL=YES/NO)
    distortion_control: bool = True  # Enable distortion control for 14-layer viscoelastic PSA
    distortion_j_crit: float = 0.20  # Critical volume ratio threshold

    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Generate kwargs dictionary for DynamicSolver.__init__."""
        # `nlgeom` is authoritative; the two `ul_*` fields are deprecated
        # aliases retained for backwards compatibility. Any of the three
        # being on turns geometric nonlinearity on -- the per-element
        # decision of what that MEANS (TL vs UL) is made in
        # DynamicSolver._use_ul_for, not here.
        use_nlgeom = self.nlgeom or self.ul_large_rotation_mode or self.ul_mode
        return {
            "mode": self.integration_mode,
            "tol": self.tol,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_iter": self.max_iter,
            "max_cutbacks": self.max_cutbacks,
            "nlgeom": use_nlgeom,
            "alpha": self.alpha,
            "pardiso_mtype": self.pardiso_mtype,
            "pardiso_phase_reuse": self.pardiso_phase_reuse,
            "max_displacement_corr": self.max_displacement_corr,
            "distortion_control": self.distortion_control,
            "distortion_j_crit": self.distortion_j_crit,
            "hourglass": self.hourglass,
            "hourglass_scale": self.hourglass_scale,
        }

    def to_abaqus_config(self) -> AbaqusSolverConfig:
        """Convert to AbaqusSolverConfig representation."""
        return AbaqusSolverConfig(
            time_incrementation=AbaqusTimeIncrementationControls(
                I_0=self.max_iter,
                I_C=self.max_cutbacks,
            ),
            field_controls=AbaqusFieldControls(
                R_n=self.tol,
                rtol=self.rtol,
                atol=self.atol,
                max_disp_corr=self.max_displacement_corr,
            ),
            dynamic_controls=AbaqusDynamicControls(
                integration_mode=self.integration_mode,
                alpha=self.alpha,
            ),
            section_controls=AbaqusSectionControls(
                distortion_control=self.distortion_control,
                distortion_j_crit=self.distortion_j_crit,
            ),
            pardiso_mtype=self.pardiso_mtype,
            pardiso_phase_reuse=self.pardiso_phase_reuse,
            pet_element_type=self.pet_element_type,
            psa_element_type=self.psa_element_type,
            glass_element_type=self.glass_element_type,
            theta_penalty_k=self.theta_penalty_k,
            ul_mode=self.ul_mode,
        )


@dataclass
class FoldModelConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    grading: MeshGradingConfig = field(default_factory=MeshGradingConfig)
    materials: MaterialsConfig = field(default_factory=MaterialsConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    solver: SolverTuningConfig = field(default_factory=SolverTuningConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert FoldModelConfig to a Python dict."""
        d = asdict(self)
        # Convert tuple void regions to plain lists for PyYAML compatibility
        if "geometry" in d and "layer_void_regions" in d["geometry"] and isinstance(d["geometry"]["layer_void_regions"], dict):
            d["geometry"]["layer_void_regions"] = {
                k: [list(v_item) for v_item in v_list]
                for k, v_list in d["geometry"]["layer_void_regions"].items()
            }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FoldModelConfig:
        """Construct FoldModelConfig from a Python dictionary."""
        d = dict(d or {})

        # Geometry
        geo_dict = d.get("geometry", {})
        if "layer_pattern" in geo_dict:
            geo_dict["layer_pattern"] = [
                LayerSpec(**l) if isinstance(l, dict) else l for l in geo_dict["layer_pattern"]
            ]
        if "custom_layers" in geo_dict and geo_dict["custom_layers"] is not None:
            geo_dict["custom_layers"] = [
                LayerSpec(**l) if isinstance(l, dict) else l for l in geo_dict["custom_layers"]
            ]
        if "layer_void_regions" in geo_dict and isinstance(geo_dict["layer_void_regions"], dict):
            # Parse void region keys back to int
            geo_dict["layer_void_regions"] = {
                int(k): [tuple(v_item) for v_item in v_list]
                for k, v_list in geo_dict["layer_void_regions"].items()
            }
        geo = GeometryConfig(**geo_dict) if isinstance(geo_dict, dict) else GeometryConfig()

        # Grading
        grading_dict = d.get("grading", {})
        grading = MeshGradingConfig(**grading_dict) if isinstance(grading_dict, dict) else MeshGradingConfig()

        # Materials
        mat_dict = d.get("materials", {})
        if "definitions" in mat_dict and isinstance(mat_dict["definitions"], dict):
            mat_defs = {}
            for name, m_data in mat_dict["definitions"].items():
                if isinstance(m_data, dict):
                    mat_defs[name] = MaterialDef(**m_data)
                else:
                    mat_defs[name] = m_data
            mat_dict["definitions"] = mat_defs
        materials = MaterialsConfig(**mat_dict) if isinstance(mat_dict, dict) else MaterialsConfig()

        # Drive
        drive_dict = d.get("drive", {})
        drive = DriveConfig(**drive_dict) if isinstance(drive_dict, dict) else DriveConfig()

        # Solver
        solver_dict = d.get("solver", {})
        solver = SolverTuningConfig(**solver_dict) if isinstance(solver_dict, dict) else SolverTuningConfig()

        return cls(geometry=geo, grading=grading, materials=materials, drive=drive, solver=solver)

    def to_json(self, path_or_file: Optional[Union[str, Path]] = None, indent: int = 2) -> Optional[str]:
        """Export config to JSON string or save to file path."""
        data_str = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path_or_file is not None:
            p = Path(path_or_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(data_str)
            return None
        return data_str

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> FoldModelConfig:
        """Import FoldModelConfig from JSON file path or JSON string."""
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(path_or_str))
        return cls.from_dict(data)

    def to_yaml(self, path_or_file: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Export config to YAML string or save to file path (falls back to JSON if PyYAML absent)."""
        try:
            import yaml
            data_str = yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        except ImportError:
            data_str = self.to_json(indent=2)

        if path_or_file is not None:
            p = Path(path_or_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(data_str)
            return None
        return data_str

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> FoldModelConfig:
        """Import FoldModelConfig from YAML file path or string."""
        try:
            import yaml
            p = Path(path_or_str)
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                data = yaml.safe_load(str(path_or_str))
            return cls.from_dict(data)
        except ImportError:
            return cls.from_json(path_or_str)

    def save(self, path: Union[str, Path]) -> None:
        """Save config to file, detecting format (.json, .yaml, .yml) from extension."""
        p = Path(path)
        if p.suffix.lower() in (".yaml", ".yml"):
            self.to_yaml(p)
        else:
            self.to_json(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> FoldModelConfig:
        """Load config from file, detecting format (.json, .yaml, .yml) from extension."""
        p = Path(path)
        if p.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(p)
        return cls.from_json(p)

    def set_path(self, key_path: str, value: Any) -> None:
        """Set nested attribute using dot-notation (e.g. 'solver.integration_mode', 'solver.max_cutbacks')."""
        parts = key_path.split(".")
        target = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise AttributeError(f"Invalid key path: '{key_path}' (missing '{part}')")
            target = getattr(target, part)
        attr_name = parts[-1]
        if not hasattr(target, attr_name):
            raise AttributeError(f"Invalid attribute: '{attr_name}' in key path '{key_path}'")

        curr_val = getattr(target, attr_name)
        if curr_val is not None:
            if isinstance(curr_val, bool):
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(value)
            elif isinstance(curr_val, int):
                value = int(value)
            elif isinstance(curr_val, float):
                value = float(value)
        elif isinstance(value, str):
            # Optional[...] field currently None: the declared type is gone, so parse the text
            # ("none"/"null" -> None, else int, float or bool when it reads as one)
            text = value.strip()
            if text.lower() in ("none", "null"):
                value = None
            elif text.lower() in ("true", "false"):
                value = text.lower() == "true"
            else:
                for conv in (int, float):
                    try:
                        value = conv(text)
                        break
                    except ValueError:
                        pass
        setattr(target, attr_name, value)

    def get_path(self, key_path: str) -> Any:
        """Get nested attribute value using dot-notation (e.g. 'solver.integration_mode')."""
        parts = key_path.split(".")
        target = self
        for part in parts:
            if not hasattr(target, part):
                raise AttributeError(f"Invalid key path: '{key_path}' (missing '{part}')")
            target = getattr(target, part)
        return target

    @classmethod
    def get_preset(cls, name: str, **overrides) -> FoldModelConfig:
        """Load a standard predefined configuration preset.

        Presets
        -------
        - "default": Default 14-layer PET/PSA display fold configuration.
        - "teardrop": Compact PET/PSA teardrop fold configuration.
        - "glass_fold": Thin glass substrate folding configuration.
        - "optistruct_galpha": OptiStruct GALPHA=0.8 integration mode configuration.
        - "moderate-4": Heavy damping moderate-4 integration mode configuration.
        - "quasistatic": Pure quasi-static analysis configuration (no inertia).
        """
        name_clean = name.lower().replace("_", "-")
        if name_clean in ("default", "standard"):
            cfg = cls()
        elif name_clean == "teardrop":
            cfg = make_teardrop_config()
        elif name_clean in ("glass", "glass-fold"):
            cfg = make_glass_config()
        elif name_clean == "optistruct-galpha":
            cfg = cls()
            cfg.solver.integration_mode = "generalized-0.8"
        elif name_clean == "moderate-4":
            cfg = cls()
            cfg.solver.integration_mode = "moderate-4"
            cfg.solver.alpha = -0.22
        elif name_clean == "quasistatic":
            cfg = cls()
            cfg.solver.integration_mode = "quasistatic"
        else:
            raise ValueError(f"Unknown preset '{name}'. Options: 'default', 'teardrop', 'glass_fold', 'optistruct_galpha', 'moderate-4', 'quasistatic'.")

        for k, v in overrides.items():
            cfg.set_path(k, v)
        return cfg

    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Generate kwargs dictionary for DynamicSolver.__init__."""
        return self.solver.to_solver_kwargs()

    def to_abaqus_config(self) -> AbaqusSolverConfig:
        """Convert solver tuning parameters to AbaqusSolverConfig representation."""
        abq = self.solver.to_abaqus_config()
        abq.step.t_total = self.drive.t_total
        abq.step.dt_init = self.drive.dt_init
        abq.step.dt_min = self.drive.dt_min
        abq.step.dt_max = self.drive.dt_max
        return abq


DEFAULT_CONFIG = FoldModelConfig()


def add_config_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add standard configuration command-line arguments (--config, --preset, --set) to an ArgumentParser."""
    group = parser.add_argument_group("Solver & Model Configuration Options")
    group.add_argument("--config", type=str, default=None, help="Path to external config file (.json or .yaml)")
    group.add_argument("--preset", type=str, default=None, help="Predefined config preset (e.g. 'teardrop', 'glass_fold', 'optistruct_galpha', 'moderate-4', 'quasistatic')")
    group.add_argument("--set", action="append", default=[], help="Override config parameter via key path (e.g. --set solver.integration_mode=generalized-0.8 --set solver.max_cutbacks=20)")
    return parser


def apply_cli_overrides(config: FoldModelConfig, args: argparse.Namespace) -> FoldModelConfig:
    """Apply CLI arguments (--config, --preset, --set) to a FoldModelConfig instance."""
    if getattr(args, "config", None):
        config = FoldModelConfig.load(args.config)
    elif getattr(args, "preset", None):
        config = FoldModelConfig.get_preset(args.preset)

    for item in getattr(args, "set", []):
        if "=" in item:
            key_path, val_str = item.split("=", 1)
            config.set_path(key_path.strip(), val_str.strip())

    return config


def make_glass_config(thickness_mm: float = 0.1, n_rows: int = 2) -> FoldModelConfig:
    """Convenience helper to create a monolithic thin glass substrate folding configuration."""
    cfg = FoldModelConfig()
    cfg.geometry.layer_pattern = [LayerSpec("GLASS", thickness_mm, n_rows)]
    cfg.geometry.n_layer_pairs = 1
    # Monolithic glass has no adhesive tape cutout in bottom layer
    cfg.geometry.layer_void_regions = {}
    return cfg


def make_teardrop_config(n_layer_pairs: int = 3, dt_max: float = 0.01) -> FoldModelConfig:
    """Convenience helper to create a PET-PSA multilayer teardrop folding configuration.

    - Tight plate gap: hinge_pivot_x = 0.8 mm -> closed plate gap = 1.6 mm.
    - Free hinge span: hinge_half_gap = 7.5 mm -> free span = 15.0 mm.
    - Yields natural minimum-energy teardrop (water-drop) loop at 90-degree folding.
    """
    cfg = FoldModelConfig()
    cfg.geometry.display_half_length = 40.0
    cfg.geometry.hinge_half_gap = 7.5
    cfg.geometry.hinge_pivot_x = 1.4
    cfg.geometry.hinge_pivot_y = 0.0
    # Tie over the whole plate. A [hinge_half_gap, 12.0) untied strip (96f8adb) had no contact,
    # so the display passed through the rigid plate there (1.35 mm deep at 41 deg, 2026-10-09);
    # bring the strip back only together with solver.plate_contact once that is validated.
    cfg.geometry.tie_attach_start = None
    # Display <-> plate contact on (user requirement 2026-10-10): softened onset over 1 um of
    # overclosure and a 0.25 mm fillet on the plate's hinge-side edge (Abaqus FILLET RADIUS),
    # which avoids open/closed chattering against the compliant display. With the tie over the
    # whole plate the contact guards the display bottom outside the tie (free span / loop).
    cfg.solver.plate_contact = True
    cfg.solver.plate_contact_softening = 0.001
    cfg.solver.plate_contact_fillet = 0.25
    cfg.geometry.n_layer_pairs = n_layer_pairs
    cfg.geometry.layer_pattern = [LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.05, 3)]
    cfg.geometry.layer_void_regions = {1: [(-7.5, 0.0), (0.0, 7.5)]}
    cfg.grading.hinge_span_half_width = 6.0
    cfg.grading.hinge_span_dx = 0.25
    cfg.drive.dt_init = 0.005
    cfg.drive.dt_max = dt_max
    cfg.drive.dt_min = 1e-5
    cfg.solver.integration_mode = "moderate-4"
    cfg.solver.ul_mode = True
    cfg.solver.max_displacement_corr = None
    cfg.solver.distortion_control = True
    cfg.solver.distortion_j_crit = 0.20
    cfg.solver.pet_element_type = "CPE4R"
    cfg.solver.glass_element_type = "CPE4R"
    cfg.solver.psa_element_type = "CPE4H"
    cfg.solver.hourglass = "ENHANCED"
    return cfg


def smoothstep_amp(t: float, t_total: float = 1.0) -> float:
    """C²-smoothstep amplitude ramp (10s^3 - 15s^4 + 6s^5). Zero velocity and acceleration at t=0 and t=t_total."""
    import numpy as np
    s = float(np.clip(t / t_total, 0.0, 1.0))
    return float(s**3 * (10.0 + s * (-15.0 + 6.0 * s)))



