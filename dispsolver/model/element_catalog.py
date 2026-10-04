"""
element_catalog.py
==================
Canonical single source of truth for element types, kinematics, formulations,
and solver kernel dispatches across 2D and 3D solvers.

Resolves Architectural Issue P4 (dev_log/plan_inp_framework_modernization_20261004.md):
Unifies element catalog across assembly.py, dynamic2d.py, dynamic3d.py, and test contracts.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set, Any


@dataclass(frozen=True)
class ElementDescriptor:
    """Abaqus element specification and dispatch metadata."""
    type_name: str
    dimension: int                          # 2 or 3
    family: str                             # "CONTINUUM", "SHELL", "TRUSS", "SOLID_SHELL", "RIGID", "COUPLING"
    nodes: int                              # 2, 3, 4, 6, 8, 10, 20
    integration: str                        # "FULL", "REDUCED", "INCOMPATIBLE", "MODIFIED"
    is_hybrid: bool = False
    is_axisymmetric: bool = False
    kernel_nlgeom_yes: Optional[int] = None # Kernel ID in solver when NLGEOM=True (None if unsupported)
    kernel_nlgeom_no: Optional[int] = None  # Kernel ID in solver when NLGEOM=False
    cr_wrapped_yes: bool = False            # Wrapped by generic Co-Rotational outer hull in NLGEOM=True
    nlgeom_yes_supported: bool = True       # Whether NLGEOM=True is mathematically supported
    nlgeom_no_supported: bool = True        # Whether NLGEOM=False is mathematically supported
    substitution_note: Optional[str] = None # Warning/substitution explanation if applicable
    supported_materials: Optional[Tuple[str, ...]] = None # None means all standard materials


_ELEMENT_DESCRIPTORS: Tuple[ElementDescriptor, ...] = (
    # =========================================================================
    # 3D Continuum Hexahedral Elements
    # =========================================================================
    ElementDescriptor(
        type_name="C3D8",
        dimension=3,
        family="CONTINUUM",
        nodes=8,
        integration="FULL",
        kernel_nlgeom_yes=1,    # C3D8_CR: Hughes B-bar + Co-Rotational
        kernel_nlgeom_no=16,   # C3D8 B-bar core with R = I (any material)
    ),
    ElementDescriptor(
        type_name="C3D8R",
        dimension=3,
        family="CONTINUUM",
        nodes=8,
        integration="REDUCED",
        kernel_nlgeom_yes=13,   # C3D8R_CR: Reduced Integration + Co-Rotational
        kernel_nlgeom_no=8,    # C3D8_REDUCED: Small-strain reduced integration
    ),
    ElementDescriptor(
        type_name="C3D8RH",
        dimension=3,
        family="CONTINUUM",
        nodes=8,
        integration="REDUCED",
        is_hybrid=True,
        kernel_nlgeom_yes=13,
        kernel_nlgeom_no=8,
        substitution_note="runs as C3D8R (reduced integration); the hybrid pressure field is not included",
    ),
    ElementDescriptor(
        type_name="C3D8I",
        dimension=3,
        family="CONTINUUM",
        nodes=8,
        integration="INCOMPATIBLE",
        kernel_nlgeom_yes=11,   # C3D8I_CR: EAS-9 + Co-Rotational (or TL kernel 0 for inelastic)
        kernel_nlgeom_no=18,   # C3D8I small strain EAS (or TL kernel 0 for inelastic)
    ),
    ElementDescriptor(
        type_name="C3D8IH",
        dimension=3,
        family="CONTINUUM",
        nodes=8,
        integration="INCOMPATIBLE",
        is_hybrid=True,
        kernel_nlgeom_yes=11,
        kernel_nlgeom_no=18,
        substitution_note="runs as C3D8I (incompatible modes); the hybrid pressure field is not included",
    ),
    ElementDescriptor(
        type_name="C3D8H",
        dimension=3,
        family="CONTINUUM",
        nodes=8,
        integration="FULL",
        is_hybrid=True,
        kernel_nlgeom_yes=2,    # C3D8H: Herrmann u-P Mixed Hybrid + Co-Rotational
        kernel_nlgeom_no=17,   # C3D8H core with R = I
    ),
    # =========================================================================
    # 3D Quadratic Hexahedral Elements
    # =========================================================================
    ElementDescriptor(
        type_name="C3D20",
        dimension=3,
        family="CONTINUUM",
        nodes=20,
        integration="FULL",
        kernel_nlgeom_yes=15,   # C3D20 wrapped in generic CR
        kernel_nlgeom_no=15,    # C3D20 linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="C3D20R",
        dimension=3,
        family="CONTINUUM",
        nodes=20,
        integration="REDUCED",
        kernel_nlgeom_yes=15,
        kernel_nlgeom_no=15,
        cr_wrapped_yes=True,
        substitution_note="runs as full-integration C3D20; reduced integration is not implemented",
    ),
    ElementDescriptor(
        type_name="C3D20H",
        dimension=3,
        family="CONTINUUM",
        nodes=20,
        integration="FULL",
        is_hybrid=True,
        kernel_nlgeom_yes=15,
        kernel_nlgeom_no=15,
        cr_wrapped_yes=True,
        substitution_note="runs as C3D20; the hybrid pressure field is not included",
    ),
    ElementDescriptor(
        type_name="C3D20RH",
        dimension=3,
        family="CONTINUUM",
        nodes=20,
        integration="REDUCED",
        is_hybrid=True,
        kernel_nlgeom_yes=15,
        kernel_nlgeom_no=15,
        cr_wrapped_yes=True,
        substitution_note="runs as full-integration C3D20; neither reduced integration nor the hybrid field is included",
    ),
    # =========================================================================
    # 3D Tetrahedral & Wedge Elements
    # =========================================================================
    ElementDescriptor(
        type_name="C3D10",
        dimension=3,
        family="CONTINUUM",
        nodes=10,
        integration="FULL",
        kernel_nlgeom_yes=5,    # C3D10 wrapped in generic CR
        kernel_nlgeom_no=5,     # C3D10 linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="C3D10M",
        dimension=3,
        family="CONTINUUM",
        nodes=10,
        integration="MODIFIED",
        kernel_nlgeom_yes=4,    # C3D10M modified tet
        kernel_nlgeom_no=4,
        cr_wrapped_yes=False,
    ),
    ElementDescriptor(
        type_name="C3D10H",
        dimension=3,
        family="CONTINUUM",
        nodes=10,
        integration="FULL",
        is_hybrid=True,
        kernel_nlgeom_yes=5,
        kernel_nlgeom_no=5,
        cr_wrapped_yes=True,
        substitution_note="runs as C3D10M (nu > 0.48) or C3D10; the hybrid pressure field is not included",
    ),
    ElementDescriptor(
        type_name="C3D10MH",
        dimension=3,
        family="CONTINUUM",
        nodes=10,
        integration="MODIFIED",
        is_hybrid=True,
        kernel_nlgeom_yes=4,
        kernel_nlgeom_no=4,
        cr_wrapped_yes=False,
        substitution_note="runs as C3D10M; the hybrid pressure field is not included",
    ),
    ElementDescriptor(
        type_name="C3D4",
        dimension=3,
        family="CONTINUUM",
        nodes=4,
        integration="FULL",
        kernel_nlgeom_yes=6,    # C3D4 wrapped in generic CR
        kernel_nlgeom_no=6,     # C3D4 linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="C3D4H",
        dimension=3,
        family="CONTINUUM",
        nodes=4,
        integration="FULL",
        is_hybrid=True,
        kernel_nlgeom_yes=None, # ANP cannot be CR-wrapped; raises NotImplementedError
        kernel_nlgeom_no=9,     # C3D4_ANP
        nlgeom_yes_supported=False,
        substitution_note="runs as C3D4 with average nodal pressure (ANP), not the Abaqus hybrid formulation",
    ),
    ElementDescriptor(
        type_name="C3D6",
        dimension=3,
        family="CONTINUUM",
        nodes=6,
        integration="FULL",
        kernel_nlgeom_yes=7,    # C3D6 wrapped in generic CR
        kernel_nlgeom_no=7,     # C3D6 linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="SOLID_SHELL",
        dimension=3,
        family="SOLID_SHELL",
        nodes=8,
        integration="FULL",
        kernel_nlgeom_yes=14,   # SOLID_SHELL_CR
        kernel_nlgeom_no=12,    # SOLID_SHELL linear
    ),
    # =========================================================================
    # 2D Continuum Plane Strain / Stress Elements
    # =========================================================================
    ElementDescriptor(
        type_name="CPE4",
        dimension=2,
        family="CONTINUUM",
        nodes=4,
        integration="FULL",
        kernel_nlgeom_yes=5,    # CPE4_CR
        kernel_nlgeom_no=0,     # CPE4 linear small strain
    ),
    ElementDescriptor(
        type_name="CPE4I",
        dimension=2,
        family="CONTINUUM",
        nodes=4,
        integration="INCOMPATIBLE",
        kernel_nlgeom_yes=11,   # CPE4I_CR
        kernel_nlgeom_no=1,     # CPE4I linear small strain
    ),
    ElementDescriptor(
        type_name="CPE4H",
        dimension=2,
        family="CONTINUUM",
        nodes=4,
        integration="FULL",
        is_hybrid=True,
        kernel_nlgeom_yes=12,   # CPE4H_CR
        kernel_nlgeom_no=3,     # CPE4H linear small strain
    ),
    ElementDescriptor(
        type_name="CPE4R",
        dimension=2,
        family="CONTINUUM",
        nodes=4,
        integration="REDUCED",
        kernel_nlgeom_yes=13,   # CPE4R_CR
        kernel_nlgeom_no=2,     # CPE4R linear small strain
    ),
    ElementDescriptor(
        type_name="CPE6M",
        dimension=2,
        family="CONTINUUM",
        nodes=6,
        integration="MODIFIED",
        kernel_nlgeom_yes=14,   # CPE6M_CR
        kernel_nlgeom_no=8,     # CPE6M linear small strain
    ),
    ElementDescriptor(
        type_name="CPE3",
        dimension=2,
        family="CONTINUUM",
        nodes=3,
        integration="FULL",
        kernel_nlgeom_yes=6,    # CPE3 wrapped in 2D CR
        kernel_nlgeom_no=6,     # CPE3 linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="CPE6",
        dimension=2,
        family="CONTINUUM",
        nodes=6,
        integration="FULL",
        kernel_nlgeom_yes=7,    # CPE6 wrapped in 2D CR
        kernel_nlgeom_no=7,     # CPE6 linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="CPE8",
        dimension=2,
        family="CONTINUUM",
        nodes=8,
        integration="FULL",
        kernel_nlgeom_yes=9,    # CPE8 wrapped in 2D CR
        kernel_nlgeom_no=9,     # CPE8 linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="CPE8R",
        dimension=2,
        family="CONTINUUM",
        nodes=8,
        integration="REDUCED",
        kernel_nlgeom_yes=10,   # CPE8R wrapped in 2D CR
        kernel_nlgeom_no=10,    # CPE8R linear small strain
        cr_wrapped_yes=True,
    ),
    ElementDescriptor(
        type_name="T2D2",
        dimension=2,
        family="TRUSS",
        nodes=2,
        integration="FULL",
        kernel_nlgeom_yes=18,   # T2D2 wrapped in 2D CR
        kernel_nlgeom_no=18,    # T2D2 small strain / axial
        cr_wrapped_yes=True,
    ),
    # =========================================================================
    # 2D Axisymmetric Elements
    # =========================================================================
    ElementDescriptor(
        type_name="CAX4",
        dimension=2,
        family="CONTINUUM",
        nodes=4,
        integration="FULL",
        is_axisymmetric=True,
        kernel_nlgeom_yes=None, # Axisymmetric elements do not support NLGEOM=YES
        kernel_nlgeom_no=17,    # CAX4 small strain
        nlgeom_yes_supported=False,
    ),
    ElementDescriptor(
        type_name="CAX4R",
        dimension=2,
        family="CONTINUUM",
        nodes=4,
        integration="REDUCED",
        is_axisymmetric=True,
        kernel_nlgeom_yes=None,
        kernel_nlgeom_no=17,    # CAX4R runs small strain
        nlgeom_yes_supported=False,
    ),
    ElementDescriptor(
        type_name="CAX8",
        dimension=2,
        family="CONTINUUM",
        nodes=8,
        integration="FULL",
        is_axisymmetric=True,
        kernel_nlgeom_yes=None,
        kernel_nlgeom_no=15,    # CAX8 small strain
        nlgeom_yes_supported=False,
    ),
    ElementDescriptor(
        type_name="CAX8R",
        dimension=2,
        family="CONTINUUM",
        nodes=8,
        integration="REDUCED",
        is_axisymmetric=True,
        kernel_nlgeom_yes=None,
        kernel_nlgeom_no=16,    # CAX8R small strain
        nlgeom_yes_supported=False,
    ),
)


class ElementCatalog:
    """Catalog manager providing single-source query and dispatch verification."""
    _REGISTRY: Dict[str, ElementDescriptor] = {
        desc.type_name: desc for desc in _ELEMENT_DESCRIPTORS
    }

    # Aliases
    _ALIASES: Dict[str, str] = {
        "C3D8_CR": "C3D8",
        "C3D8_COROTATIONAL": "C3D8",
        "C3D8_FBAR_CR": "C3D8",
        "C3D8_REDUCED": "C3D8R",
        "C3D8R_CR": "C3D8R",
        "C3D8_EAS": "C3D8I",
        "C3D8I_CR": "C3D8I",
        "C3D8_HYBRID": "C3D8H",
        "C3D8H_CR": "C3D8H",
        "C3D10_MODIFIED": "C3D10M",
        "C3D4_ANP": "C3D4H",
        "C3D6_WEDGE": "C3D6",
        "WEDGE6": "C3D6",
        "C3D8_SS": "SOLID_SHELL",
        "SOLID_SHELL_CR": "SOLID_SHELL",
        "CPE4_CR": "CPE4",
        "CPE4_COROTATIONAL": "CPE4",
        "CPE4I_CR": "CPE4I",
        "CPE4H_CR": "CPE4H",
        "CPE4R_CR": "CPE4R",
        "CPE6M_CR": "CPE6M",
        "TRUSS2D": "T2D2",
        "LINK": "T2D2",
    }

    @classmethod
    def lookup(cls, etype: str) -> Optional[ElementDescriptor]:
        """Lookup element descriptor by canonical name or alias."""
        clean = etype.strip().upper()
        if clean in cls._REGISTRY:
            return cls._REGISTRY[clean]
        canonical = cls._ALIASES.get(clean)
        if canonical and canonical in cls._REGISTRY:
            return cls._REGISTRY[canonical]
        return None

    @classmethod
    def detect_dim(cls, etype: str) -> Optional[int]:
        """Detect spatial dimension (2 or 3) from element type name."""
        desc = cls.lookup(etype)
        if desc is not None:
            return desc.dimension
        clean = etype.strip().upper()
        if clean.startswith(("CPE", "CPS", "CAX", "CGAX", "CGPE", "R2D", "RAX", "T2D")):
            return 2
        if clean.startswith(("C3D", "S4", "S3", "S8", "M3D", "STRI", "DCOUP3D")):
            return 3
        return None

    @classmethod
    def get_substitutions(cls) -> Dict[str, str]:
        """Return dict of canonical substitution notes."""
        return {
            desc.type_name: desc.substitution_note
            for desc in cls._REGISTRY.values()
            if desc.substitution_note is not None
        }

    @classmethod
    def all_descriptors(cls) -> Tuple[ElementDescriptor, ...]:
        """Return all element descriptors."""
        return tuple(cls._REGISTRY.values())
