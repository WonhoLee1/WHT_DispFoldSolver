"""
model_spec.py
=============
Single Source of Truth (SSOT) data structures and specifications for 2D/3D multi-layer
foldable display teardrop folding simulations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


# Canonical, verified element formulation cases (SSOT, standard Abaqus element types)
# Solvers automatically promote these to Co-Rotational kernels under nlgeom=True via ElementCatalog
ELEM_CASES_2D: Dict[str, Dict[str, str]] = {
    "2D-Opt1":   {"solid": "CPE4I", "psa": "CPE4H", "plate": "CPE4"},
    "2D-Opt2":   {"solid": "CPE4R", "psa": "CPE4H", "plate": "CPE4"},
    "2D-Base":   {"solid": "CPE4",  "psa": "CPE4H", "plate": "CPE4"},
}

ELEM_CASES_3D: Dict[str, Dict[str, str]] = {
    "3D-Opt1":   {"solid": "C3D8I", "psa": "C3D8H", "plate": "C3D8"},
    "3D-Opt2-H": {"solid": "C3D8R", "psa": "C3D8H", "plate": "C3D8"},
    "3D-Base":   {"solid": "C3D8",  "psa": "C3D8H", "plate": "C3D8"},
}


@dataclass
class LayerSpec:
    """Individual laminated display layer specification."""
    name: str                     # e.g. 'PSA_1', 'PET_1', 'UTG_GLASS'
    thickness: float              # Layer thickness in mm (e.g. 0.030, 0.060)
    material_name: str            # Referenced material key ('PSA', 'PET', 'GLASS')
    is_void_in_hinge: bool = False # Whether voided in free hinge span (e.g. layer 0 bottom PSA)
    elem_type_2d: str = "CPE4I"   # 2D solid formulation (standard Abaqus name)
    elem_type_3d: str = "C3D8I"   # 3D solid formulation (standard Abaqus name)
    ny_elements: int = 1          # Number of elements through layer thickness
    color: str = "#3498db"        # Hex color code for visualization


@dataclass
class MaterialSpec:
    """Rigorous continuum constitutive properties (Elastic, Plastic, Hyperelastic)."""
    name: str
    mat_type: str                 # 'ELASTIC', 'ELASTIC_PLASTIC', 'HYPERELASTIC_VISCO'
    E: float                      # Young's modulus (MPa)
    nu: float                     # Poisson's ratio
    density: float = 1.2e-9       # Density (tonne / mm^3)
    yield_stress: float = 0.0     # Initial yield stress (MPa)
    hardening_modulus: float = 0.0 # Hardening modulus (MPa)
    arruda_boyce_mu: float = 0.0  # Arruda-Boyce ground shear modulus mu_0 (MPa)
    arruda_boyce_lm: float = 7.0  # Arruda-Boyce locking stretch lambda_m
    prony_g: List[float] = field(default_factory=list)   # Viscoelastic shear ratios
    prony_tau: List[float] = field(default_factory=list) # Relaxation times (s)

    @property
    def plane_strain_E(self) -> float:
        """Effective plane strain Young's modulus E' = E / (1 - nu^2)."""
        return self.E / max(1e-9, 1.0 - self.nu ** 2)


@dataclass
class HingeGeometrySpec:
    """Kinematic dimensions and boundaries of foldable display hinge."""
    display_half_length: float = 40.0 # Half length of display sheet (mm)
    hinge_half_gap: float = 7.5       # Free span half gap (mm, total span = 15.0 mm)
    hinge_pivot_x: float = 1.4        # Rigid plate rotation pivot X (mm, neck gap = 2.8 mm)
    hinge_pivot_y: float = 0.0        # Rigid plate rotation pivot Y (mm)
    plate_thickness: float = 0.5      # Support plate thickness (mm)
    width_z: float = 1.0              # Display extrusion width in Z (mm)


@dataclass
class MeshSpec:
    """Discretization mesh resolution and progressive grading."""
    use_gmsh: bool = True             # True to use Gmsh C-API; False for fallback
    nx_hinge: int = 30                # Elements in hinge span [-7.5, 7.5]
    nx_plate: int = 20                # Elements along each rigid plate body
    nz_elements: int = 2              # Elements along Z-width (3D)
    bias_progression: float = 1.15    # Biased grading factor (fine in hinge, coarse at ends)


@dataclass
class SimulationCaseSpec:
    """Nonlinear solver runtime configuration and parameters."""
    dim: str = "3D"                   # "2D" or "3D"
    preset_name: str = "teardrop"     # Preset name identifier
    elem_case: Optional[str] = None   # "3D-Opt1", "3D-Opt2-H", "3D-Base", "2D-Opt1", "2D-Opt2", "2D-Base"
    theta_max_deg: float = 90.0       # Folding angle per plate in degrees (total 180°)
    total_time: float = 1.0           # Pseudo-time duration (s)
    dt_init: float = 0.005            # Initial time increment (s)
    dt_min: float = 1.0e-5            # Minimum time increment before abort (s)
    dt_max: float = 0.02              # Maximum allowed time increment (s)
    solution_preset: str = "ADAPTIVE" # "ADAPTIVE", "STANDARD", "RELAXED"
    enable_predictor: bool = True     # Kinematic rotation predictor On/Off switch
    tie_method: str = "MPC"           # "MPC" (algebraic elimination) or "PENALTY"
    contact_mode: str = "AUTO"        # "AUTO", "OFF", "DUAL_MORTAR"
    max_steps: Optional[int] = None   # Optional step limit for smoke tests


class UnifiedFoldModelConfig:
    """Single Source of Truth container uniting geometry, layers, materials, and case."""

    def __init__(self, preset: str = "teardrop", dim: str = "3D", elem_case: Optional[str] = None) -> None:
        self.geometry = HingeGeometrySpec()
        self.mesh = MeshSpec()
        self.materials: Dict[str, MaterialSpec] = {}
        self.layers: List[LayerSpec] = []
        self.simulation_case = SimulationCaseSpec(dim=dim, preset_name=preset, elem_case=elem_case)

        self._setup_standard_materials()
        self.load_preset(preset, elem_case=elem_case)

    def _setup_standard_materials(self) -> None:
        """Initializes canonical, literature-verified material definitions."""
        # 1. PET Substrate (J2 Elastoplastic)
        self.materials["PET"] = MaterialSpec(
            name="PET",
            mat_type="ELASTIC_PLASTIC",
            E=4000.0,
            nu=0.30,
            density=1.4e-9,
            yield_stress=80.0,
            hardening_modulus=400.0,
        )

        # 2. PSA Optically Clear Adhesive (Arruda-Boyce Hyperelastic + Viscoelastic)
        self.materials["PSA"] = MaterialSpec(
            name="PSA",
            mat_type="HYPERELASTIC_VISCO",
            E=0.05,
            nu=0.499,
            density=1.0e-9,
            arruda_boyce_mu=0.016779, # G_0 = 16.78 kPa
            arruda_boyce_lm=7.0,
            prony_g=[0.60, 0.25],
            prony_tau=[0.05, 0.50],
        )

        # 3. Ultra-Thin Glass / Cover Glass (Linear Elastic)
        self.materials["GLASS"] = MaterialSpec(
            name="GLASS",
            mat_type="ELASTIC",
            E=70000.0,
            nu=0.22,
            density=2.5e-9,
        )

        # 4. Support Steel Rigid Plate (High Stiffness)
        self.materials["PLATE"] = MaterialSpec(
            name="PLATE",
            mat_type="ELASTIC",
            E=200000.0,
            nu=0.30,
            density=7.85e-9,
        )

    def load_preset(self, preset: str, elem_case: Optional[str] = None) -> None:
        """Loads a predefined kinematic and stackup preset."""
        preset = preset.lower()
        self.simulation_case.preset_name = preset
        if elem_case is not None:
            self.simulation_case.elem_case = elem_case

        # Resolve element formulation case (Abaqus standard names promoted via ElementCatalog)
        case_name = self.simulation_case.elem_case
        if not case_name:
            case_name = "3D-Opt1" if self.simulation_case.dim == "3D" else "2D-Opt1"
        self.simulation_case.elem_case = case_name

        case_3d_key = case_name if case_name in ELEM_CASES_3D else "3D-Opt1"
        case_2d_key = case_name if case_name in ELEM_CASES_2D else "2D-Opt1"

        spec_3d = ELEM_CASES_3D[case_3d_key]
        spec_2d = ELEM_CASES_2D[case_2d_key]

        solid_2d = spec_2d["solid"]
        psa_2d = spec_2d["psa"]
        solid_3d = spec_3d["solid"]
        psa_3d = spec_3d["psa"]

        self.layers.clear()

        if preset == "teardrop":
            # Canonical 14-layer stackup (Total thickness 0.630 mm, 2.8 mm neck gap)
            self.geometry.hinge_half_gap = 7.5
            self.geometry.hinge_pivot_x = 1.4
            self.geometry.hinge_pivot_y = 0.0

            # 7 PSA (30 um) + 6 PET (60 um) + 1 GLASS (60 um)
            # Layer 0 bottom PSA has tape void in hinge [-7.5, 7.5]
            self.layers.append(LayerSpec("PSA_0_Voided", 0.030, "PSA", is_void_in_hinge=True,
                                         elem_type_2d=psa_2d, elem_type_3d=psa_3d, color="#f39c12"))
            for i in range(1, 6):
                self.layers.append(LayerSpec(f"PET_{i}", 0.060, "PET", elem_type_2d=solid_2d, elem_type_3d=solid_3d, color="#3498db"))
                self.layers.append(LayerSpec(f"PSA_{i}", 0.030, "PSA", elem_type_2d=psa_2d, elem_type_3d=psa_3d, color="#e67e22"))
            self.layers.append(LayerSpec("PET_6", 0.060, "PET", elem_type_2d=solid_2d, elem_type_3d=solid_3d, color="#2980b9"))
            self.layers.append(LayerSpec("PSA_6", 0.030, "PSA", elem_type_2d=psa_2d, elem_type_3d=psa_3d, color="#e67e22"))
            self.layers.append(LayerSpec("GLASS_Top", 0.060, "GLASS", elem_type_2d=solid_2d, elem_type_3d=solid_3d, color="#2ecc71"))

        elif preset == "compact_4":
            # 4-layer fast verification stackup: PET - PSA - PET - GLASS
            self.geometry.hinge_half_gap = 7.5
            self.geometry.hinge_pivot_x = 1.4
            self.geometry.hinge_pivot_y = 0.0

            self.layers.append(LayerSpec("PET_Bottom", 0.060, "PET", elem_type_2d=solid_2d, elem_type_3d=solid_3d, color="#3498db"))
            self.layers.append(LayerSpec("PSA_Mid", 0.030, "PSA", elem_type_2d=psa_2d, elem_type_3d=psa_3d, color="#e67e22"))
            self.layers.append(LayerSpec("PET_Mid", 0.060, "PET", elem_type_2d=solid_2d, elem_type_3d=solid_3d, color="#2980b9"))
            self.layers.append(LayerSpec("GLASS_Top", 0.060, "GLASS", elem_type_2d=solid_2d, elem_type_3d=solid_3d, color="#2ecc71"))

        elif preset == "u_shape":
            # Semicircular pure U-fold (Fixed pivot at origin)
            self.geometry.hinge_half_gap = 4.712  # pi * R / 2 for R=1.5
            self.geometry.hinge_pivot_x = 0.0
            self.geometry.hinge_pivot_y = 0.0
            self.load_preset("teardrop")
            self.geometry.hinge_pivot_x = 0.0

        elif preset == "compact_waterdrop":
            # Slim teardrop (1.5 mm neck gap)
            self.geometry.hinge_half_gap = 5.0
            self.geometry.hinge_pivot_x = 0.75
            self.geometry.hinge_pivot_y = 0.0
            self.load_preset("teardrop")
            self.geometry.hinge_pivot_x = 0.75
            self.geometry.hinge_half_gap = 5.0

        elif preset == "crease_zero_gap":
            # Tight crease fold with full face-to-face contact (4-layer compact stack)
            self.load_preset("compact_4")
            self.geometry.hinge_half_gap = 3.0
            self.geometry.hinge_pivot_x = 0.0
            self.geometry.hinge_pivot_y = 0.0
            self.simulation_case.contact_mode = "DUAL_MORTAR"

        else:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: teardrop, compact_4, u_shape, compact_waterdrop, crease_zero_gap.")

        # Intelligent Contact Guard evaluation
        self.resolve_contact_guard()

    def resolve_contact_guard(self) -> str:
        """Intelligently determines whether contact engine must be active."""
        if self.simulation_case.contact_mode in ("OFF", "DUAL_MORTAR"):
            return self.simulation_case.contact_mode

        # AUTO mode: If pivot is tighter than 0.8 mm, top surfaces collide -> activate DUAL_MORTAR
        if abs(self.geometry.hinge_pivot_x) < 0.8:
            resolved = "DUAL_MORTAR"
        else:
            resolved = "OFF"
        return resolved

    def requires_self_contact(self) -> bool:
        """Returns True if self-contact (Dual Mortar) is required."""
        return self.resolve_contact_guard() == "DUAL_MORTAR"

    def set_dimension(self, dim: str) -> None:
        """Switches model dimension ('2D' or '3D') seamlessly."""
        if dim.upper() not in ("2D", "3D"):
            raise ValueError("dim must be '2D' or '3D'")
        self.simulation_case.dim = dim.upper()

    def modify_layer(self, layer_idx: int, **kwargs) -> None:
        """Modifies properties of a specific layer for parameter studies."""
        if not (0 <= layer_idx < len(self.layers)):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(self.layers)-1}]")
        layer = self.layers[layer_idx]
        for key, val in kwargs.items():
            if hasattr(layer, key):
                setattr(layer, key, val)
            else:
                raise AttributeError(f"LayerSpec has no attribute '{key}'")

    @property
    def total_thickness(self) -> float:
        """Total composite stackup thickness in mm."""
        return sum(layer.thickness for layer in self.layers)

    def compute_composite_bending_stiffness(self, width: float = 1.0) -> Tuple[float, float, float]:
        """Computes analytical lower/upper bound composite bending stiffnesses.

        Returns
        -------
        D_slip : float
            Lower bound free-slip bending stiffness (N*mm^2).
        D_mono : float
            Upper bound monolithic fully-bonded bending stiffness (N*mm^2).
        y_NA : float
            Neutral axis elevation from laminate bottom (mm).
        """
        b = width
        # 1. Compute layer centers and neutral axis
        y_bottom = 0.0
        sum_EA = 0.0
        sum_EA_y = 0.0

        layer_data = []
        for l in self.layers:
            mat = self.materials[l.material_name]
            E_eff = mat.plane_strain_E
            t = l.thickness
            y_c = y_bottom + 0.5 * t
            A = b * t
            EA = E_eff * A
            sum_EA += EA
            sum_EA_y += EA * y_c
            layer_data.append((E_eff, t, y_c, A))
            y_bottom += t

        y_NA = sum_EA_y / max(1e-9, sum_EA)

        # 2. D_slip: Sum of individual bending rigidities about each layer's own centroid
        D_slip = sum(E_eff * b * (t ** 3) / 12.0 for E_eff, t, y_c, A in layer_data)

        # 3. D_mono: Parallel axis theorem about global composite neutral axis
        D_mono = sum(E_eff * (b * (t ** 3) / 12.0 + A * ((y_c - y_NA) ** 2)) for E_eff, t, y_c, A in layer_data)

        return D_slip, D_mono, y_NA


# Aliases for specification consistency with PRD and tests
DisplayStackConfig = UnifiedFoldModelConfig
FoldingCaseSpec = SimulationCaseSpec

