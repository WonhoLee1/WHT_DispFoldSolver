"""
element_catalog.py
==================
The single source of truth for which solver kernel runs an element type.

DynamicSolver2D/3D._build_kernel_groups call ElementCatalog.resolve(); the
kernel table (KERNELS) says what each kernel is -- small strain or not,
whether it is co-rotated under NLGEOM, whether it takes only one linear
elastic matrix, how many nodes it consumes. Nothing else in the solvers
decides element dispatch any more.

Before 2026-10-05 the dispatch lived in two hand-written if-chains (one per
solver, NLGEOM YES and NO each), plus a substitution table, a co-rotation
set, a connectivity-trimming list and this catalog -- which no solver read.
The copies had drifted: the catalog said C3D10M was not co-rotated while
the solver co-rotated it; 3D shells (S4R, S8R, S3...) were assembled with
solid-shell, hexahedral or tetrahedral kernels, 2D "shells" with plane
kernels; CPE4IH ran hybrid-without-EAS under NLGEOM and EAS-without-hybrid
without it, with no note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------- kernels
@dataclass(frozen=True)
class KernelInfo:
    name: str
    nodes: int                      # nodes the kernel consumes (connectivity is trimmed to this)
    small_strain: bool              # formulation without large-rotation kinematics of its own
    cr_wrap: bool = False           # co-rotated (cr_generic_2d/3d) under NLGEOM
    elastic_only: bool = False      # assembles one constant linear elastic matrix per material
    nlgeom_unsupported: Optional[str] = None   # why it cannot run under NLGEOM (cannot be co-rotated)


KERNELS: Dict[int, Dict[int, KernelInfo]] = {
    3: {
        0: KernelInfo("C3D8I total-Lagrangian EAS", 8, small_strain=False),
        1: KernelInfo("C3D8 co-rotational B-bar", 8, small_strain=False),
        2: KernelInfo("C3D8H co-rotational u-p hybrid", 8, small_strain=False),
        3: KernelInfo("C3D8 F-bar total-Lagrangian", 8, small_strain=False),
        4: KernelInfo("C3D10M modified tetrahedron", 10, small_strain=True, cr_wrap=True),
        5: KernelInfo("C3D10 quadratic tetrahedron", 10, small_strain=True, cr_wrap=True, elastic_only=True),
        6: KernelInfo("C3D4 linear tetrahedron", 4, small_strain=True, cr_wrap=True, elastic_only=True),
        7: KernelInfo("C3D6 wedge", 6, small_strain=True, cr_wrap=True),
        8: KernelInfo("C3D8R reduced integration", 8, small_strain=True),
        9: KernelInfo("C3D4 average nodal pressure", 4, small_strain=True,
                      nlgeom_unsupported="the average-nodal-pressure kernel couples elements through shared "
                                         "nodes, so it cannot be co-rotated element by element; use C3D10M or hexahedra"),
        10: KernelInfo("C3D8 linear small strain", 8, small_strain=True, elastic_only=True),
        11: KernelInfo("C3D8I co-rotational EAS", 8, small_strain=False, elastic_only=True),
        12: KernelInfo("solid shell small strain", 8, small_strain=True),
        13: KernelInfo("C3D8R co-rotational", 8, small_strain=False),
        14: KernelInfo("solid shell co-rotational", 8, small_strain=False),
        15: KernelInfo("C3D20 quadratic hexahedron", 20, small_strain=True, cr_wrap=True),
        16: KernelInfo("C3D8 B-bar small strain", 8, small_strain=True),
        17: KernelInfo("C3D8H u-p hybrid small strain", 8, small_strain=True),
        18: KernelInfo("C3D8I EAS small strain", 8, small_strain=True, elastic_only=True),
    },
    2: {
        0: KernelInfo("CPE4", 4, small_strain=True),
        1: KernelInfo("CPE4I", 4, small_strain=True),
        2: KernelInfo("CPE4R", 4, small_strain=True),
        3: KernelInfo("CPE4H", 4, small_strain=True),
        4: KernelInfo("CPE4 F-bar", 4, small_strain=False),
        5: KernelInfo("CPE4 co-rotational", 4, small_strain=False),
        6: KernelInfo("CPE3", 3, small_strain=True, cr_wrap=True),
        7: KernelInfo("CPE6", 6, small_strain=True, cr_wrap=True),
        8: KernelInfo("CPE6M", 6, small_strain=True),
        9: KernelInfo("CPE8", 8, small_strain=True, cr_wrap=True),
        10: KernelInfo("CPE8R", 8, small_strain=True, cr_wrap=True),
        11: KernelInfo("CPE4I co-rotational", 4, small_strain=False),
        12: KernelInfo("CPE4H co-rotational", 4, small_strain=False),
        13: KernelInfo("CPE4R co-rotational", 4, small_strain=False),
        14: KernelInfo("CPE6M co-rotational", 6, small_strain=False),
        15: KernelInfo("CAX8", 8, small_strain=True,
                       nlgeom_unsupported="axisymmetric kernels are small strain and a rotation in the r-z plane "
                                          "is not a rigid motion of the ring, so they cannot be co-rotated"),
        16: KernelInfo("CAX8R", 8, small_strain=True,
                       nlgeom_unsupported="axisymmetric kernels are small strain and cannot be co-rotated"),
        17: KernelInfo("CAX4", 4, small_strain=True,
                       nlgeom_unsupported="axisymmetric kernels are small strain and cannot be co-rotated"),
        18: KernelInfo("T2D2 truss", 2, small_strain=True, cr_wrap=True),
    },
}


def cr_wrapped_kernels(dim: int) -> frozenset:
    return frozenset(k for k, info in KERNELS[dim].items() if info.cr_wrap)


def elastic_only_kernels(dim: int) -> Dict[int, str]:
    return {k: info.name for k, info in KERNELS[dim].items() if info.elastic_only}


# ----------------------------------------------------------------------------- elements
@dataclass(frozen=True)
class ElementDescriptor:
    """Abaqus element type (or an internal formulation name) and its dispatch."""
    type_name: str
    dimension: int                          # 2 or 3
    family: str                             # CONTINUUM, PLANE_STRESS, AXISYMMETRIC, TRUSS, SOLID_SHELL, SHELL, SKIP
    nodes: int
    integration: str                        # FULL, REDUCED, INCOMPATIBLE, MODIFIED
    is_hybrid: bool = False
    is_axisymmetric: bool = False
    kernel_nlgeom_yes: Optional[int] = None
    kernel_nlgeom_no: Optional[int] = None
    kernel_yes_incompressible: Optional[int] = None   # nu > 0.48
    kernel_no_incompressible: Optional[int] = None
    kernel_yes_inelastic: Optional[int] = None        # material other than linear elastic
    kernel_no_inelastic: Optional[int] = None
    substitution_note: Optional[str] = None           # formulation the type name promises but does not get
    note_yes: Optional[str] = None
    note_no: Optional[str] = None
    note_no_inelastic: Optional[str] = None
    unsupported: Optional[str] = None                 # not runnable at all in this solver
    unsupported_yes: Optional[str] = None
    unsupported_no: Optional[str] = None
    plane_stress: bool = False
    aliases: Tuple[str, ...] = field(default_factory=tuple)

    # compatibility with the earlier catalog API
    @property
    def cr_wrapped_yes(self) -> bool:
        k = self.kernel_nlgeom_yes
        return k is not None and KERNELS[self.dimension][k].cr_wrap

    @property
    def nlgeom_yes_supported(self) -> bool:
        return self.unsupported is None and self.unsupported_yes is None and self.kernel_nlgeom_yes is not None

    @property
    def nlgeom_no_supported(self) -> bool:
        return self.unsupported is None and self.unsupported_no is None and self.kernel_nlgeom_no is not None


_SKIP = ("DCOUP3D", "DCOUP2D", "MASS", "ROTARYI", "SPRING1", "SPRING2", "GAPUNI")
_NO_HYBRID = "the hybrid pressure field is not included"


def _hex(name, yes, no, integ="FULL", hybrid=False, note=None, aliases=(), **kw):
    return ElementDescriptor(name, 3, "CONTINUUM", kw.pop("nodes", 8), integ, is_hybrid=hybrid,
                             kernel_nlgeom_yes=yes, kernel_nlgeom_no=no, substitution_note=note,
                             aliases=tuple(aliases), **kw)


def _plane(name, yes, no, nodes, integ="FULL", hybrid=False, note=None, aliases=(), **kw):
    stress = name.startswith("CPS")
    return ElementDescriptor(name, 2, "PLANE_STRESS" if stress else "CONTINUUM", nodes, integ, is_hybrid=hybrid,
                             kernel_nlgeom_yes=yes, kernel_nlgeom_no=no, substitution_note=note,
                             plane_stress=stress, aliases=tuple(aliases), **kw)


def _cax(name, no, nodes, integ="FULL", hybrid=False, note=None):
    return ElementDescriptor(name, 2, "AXISYMMETRIC", nodes, integ, is_hybrid=hybrid, is_axisymmetric=True,
                             kernel_nlgeom_yes=None, kernel_nlgeom_no=no, substitution_note=note,
                             unsupported_yes=KERNELS[2][no].nlgeom_unsupported)


_C3D8I_NO_INEL = ("NLGEOM=NO with a non-elastic material runs the total-Lagrangian (finite-strain) EAS "
                  "kernel; no small-strain EAS kernel supports it")
_SHELL_3D = "shell elements are handled by the shell solver (dispsolver/solver_shell), not the 3D solid solver"

_ELEMENT_DESCRIPTORS: Tuple[ElementDescriptor, ...] = (
    # ------------------------------------------------------------------ 3D hexahedra
    _hex("C3D8", 1, 16),
    _hex("C3D8R", 13, 8, "REDUCED", aliases=("C3D8_REDUCED",)),
    _hex("C3D8RH", 13, 8, "REDUCED", True, "runs as C3D8R (reduced integration); " + _NO_HYBRID),
    _hex("C3D8I", 11, 18, "INCOMPATIBLE", kernel_yes_inelastic=0, kernel_no_inelastic=0,
         note_no_inelastic=_C3D8I_NO_INEL, aliases=("C3D8_EAS",)),
    _hex("C3D8IH", 11, 18, "INCOMPATIBLE", True, "runs as C3D8I (incompatible modes); " + _NO_HYBRID,
         kernel_yes_inelastic=0, kernel_no_inelastic=0, note_no_inelastic=_C3D8I_NO_INEL),
    _hex("C3D8H", 2, 17, hybrid=True, aliases=("C3D8_HYBRID", "HYBRID")),
    # internal formulation names: the named kernel whatever NLGEOM says
    _hex("C3D8_CR", 1, 1, aliases=("C3D8_COROTATIONAL", "C3D8_FBAR_CR")),
    _hex("C3D8I_CR", 11, 11, "INCOMPATIBLE"),
    _hex("C3D8H_CR", 2, 2, hybrid=True),
    _hex("C3D8R_CR", 13, 13, "REDUCED"),
    _hex("C3D8_FBAR", 3, 3),
    _hex("C3D20", 15, 15, nodes=20),
    _hex("C3D20R", 15, 15, "REDUCED", nodes=20, note="runs as full-integration C3D20; reduced integration is not implemented"),
    _hex("C3D20H", 15, 15, hybrid=True, nodes=20, note="runs as C3D20; " + _NO_HYBRID),
    _hex("C3D20RH", 15, 15, "REDUCED", True, nodes=20,
         note="runs as full-integration C3D20; neither reduced integration nor the hybrid field is included"),
    # ------------------------------------------------------------------ 3D tetrahedra, wedges
    _hex("C3D10", 5, 5, nodes=10, kernel_yes_incompressible=4, kernel_no_incompressible=4),
    _hex("C3D10H", 5, 5, hybrid=True, nodes=10, kernel_yes_incompressible=4, kernel_no_incompressible=4,
         note="runs as C3D10M (nu > 0.48) or C3D10; " + _NO_HYBRID),
    _hex("C3D10M", 4, 4, "MODIFIED", nodes=10, aliases=("C3D10_MODIFIED",)),
    _hex("C3D10MH", 4, 4, "MODIFIED", True, nodes=10, note="runs as C3D10M; " + _NO_HYBRID),
    _hex("C3D4", 6, 6, nodes=4, kernel_yes_incompressible=9, kernel_no_incompressible=9),
    _hex("C3D4H", 9, 9, hybrid=True, nodes=4, aliases=("C3D4_ANP", "ANP"),
         note="runs as C3D4 with average nodal pressure (ANP), not the Abaqus hybrid formulation"),
    _hex("C3D6", 7, 7, nodes=6, aliases=("C3D6_WEDGE", "WEDGE6")),
    # ------------------------------------------------------------------ 3D solid shells (internal names)
    ElementDescriptor("SOLID_SHELL", 3, "SOLID_SHELL", 8, "INCOMPATIBLE", kernel_nlgeom_yes=14, kernel_nlgeom_no=12,
                      aliases=("C3D8_SS",)),
    ElementDescriptor("SOLID_SHELL_CR", 3, "SOLID_SHELL", 8, "INCOMPATIBLE", kernel_nlgeom_yes=14,
                      unsupported_no="SOLID_SHELL_CR names the co-rotational kernel; use SOLID_SHELL without NLGEOM"),
    ElementDescriptor("SOLID_SHELL_LINEAR", 3, "SOLID_SHELL", 8, "INCOMPATIBLE", kernel_nlgeom_yes=12,
                      unsupported_no="SOLID_SHELL_LINEAR is only dispatched under NLGEOM; use SOLID_SHELL"),
    # ------------------------------------------------------------------ shells: not in the solid solvers
    *[ElementDescriptor(n, 3, "SHELL", nn, "FULL", unsupported=_SHELL_3D)
      for n, nn in (("S4", 4), ("S4R", 4), ("S3", 3), ("S3R", 3), ("STRI3", 3), ("S8", 8), ("S8R", 8),
                    ("S9R5", 8), ("STRI65", 6), ("M3D4", 4), ("M3D4R", 4))],
    # ------------------------------------------------------------------ 2D plane strain / stress
    *[_plane(n, 5, 0, 4) for n in ("CPE4", "CPS4")],
    *[_plane(n, 11, 1, 4, "INCOMPATIBLE") for n in ("CPE4I", "CPS4I")],
    _plane("Q4_EAS", 11, 1, 4, "INCOMPATIBLE", aliases=("INCOMPATIBLE",)),
    *[_plane(n, 12, 1, 4, "INCOMPATIBLE", True,
             note_yes="NLGEOM=YES runs the co-rotational hybrid kernel (no incompatible modes)",
             note_no="NLGEOM=NO runs the incompatible-mode kernel (" + _NO_HYBRID + ")") for n in ("CPE4IH", "CPS4IH")],
    *[_plane(n, 12, 3, 4, hybrid=True) for n in ("CPE4H", "CPS4H")],
    _plane("Q4_HYBRID", 12, 3, 4, hybrid=True),
    *[_plane(n, 13, 2, 4, "REDUCED") for n in ("CPE4R", "CPS4R")],
    _plane("Q4_REDUCED", 13, 2, 4, "REDUCED"),
    *[_plane(n, 13, 2, 4, "REDUCED", True, "runs as CPE4R; " + _NO_HYBRID) for n in ("CPE4RH", "CPS4RH")],
    *[_plane(n, 14, 8, 6, "MODIFIED") for n in ("CPE6M", "CPS6M")],
    *[_plane(n, 14, 8, 6, "MODIFIED", True, "runs as CPE6M; " + _NO_HYBRID) for n in ("CPE6MH", "CPS6MH")],
    _plane("CPE4_CR", 5, 5, 4, aliases=("Q4_COROTATIONAL", "CPE4_COROTATIONAL")),
    _plane("CPE4_FBAR", 4, 4, 4, aliases=("FBAR",)),
    _plane("CPE4I_CR", 11, 11, 4, "INCOMPATIBLE"),
    _plane("CPE4H_CR", 12, 12, 4, hybrid=True),
    _plane("CPE4R_CR", 13, 13, 4, "REDUCED"),
    _plane("CPE6M_CR", 14, 14, 6, "MODIFIED"),
    *[_plane(n, 6, 6, 3) for n in ("CPE3", "CPS3")],
    _plane("T3", 6, 6, 3, aliases=("CST",)),
    *[_plane(n, 6, 6, 3, hybrid=True, note="runs as CPE3; " + _NO_HYBRID) for n in ("CPE3H", "CPS3H")],
    *[_plane(n, 7, 7, 6) for n in ("CPE6", "CPS6")],
    _plane("T6", 7, 7, 6),
    *[_plane(n, 7, 7, 6, hybrid=True, note="runs as CPE6; " + _NO_HYBRID) for n in ("CPE6H", "CPS6H")],
    *[_plane(n, 9, 9, 8) for n in ("CPE8", "CPS8")],
    _plane("Q8", 9, 9, 8),
    *[_plane(n, 9, 9, 8, hybrid=True, note="runs as CPE8; " + _NO_HYBRID) for n in ("CPE8H", "CPS8H")],
    *[_plane(n, 10, 10, 8, "REDUCED") for n in ("CPE8R", "CPS8R")],
    _plane("Q8R", 10, 10, 8, "REDUCED"),
    *[_plane(n, 10, 10, 8, "REDUCED", True, "runs as CPE8R; " + _NO_HYBRID) for n in ("CPE8RH", "CPS8RH")],
    ElementDescriptor("T2D2", 2, "TRUSS", 2, "FULL", kernel_nlgeom_yes=18, kernel_nlgeom_no=18, aliases=("TRUSS2D", "LINK")),
    # ------------------------------------------------------------------ 2D axisymmetric (small strain only)
    _cax("CAX4", 17, 4),
    _cax("CAX4I", 17, 4, "INCOMPATIBLE", note="runs as CAX4; incompatible modes are not implemented"),
    _cax("CAX4H", 17, 4, hybrid=True, note="runs as CAX4; " + _NO_HYBRID),
    _cax("CAX4IH", 17, 4, "INCOMPATIBLE", True, "runs as CAX4; neither incompatible modes nor the hybrid field is included"),
    _cax("CAX4R", 17, 4, "REDUCED", note="runs as full-integration CAX4; reduced integration is not implemented"),
    _cax("CAX4RH", 17, 4, "REDUCED", True, "runs as full-integration CAX4; neither reduced integration nor the hybrid field is included"),
    _cax("CAX8", 15, 8),
    _cax("CAX8H", 15, 8, hybrid=True, note="runs as CAX8; " + _NO_HYBRID),
    _cax("CAX8R", 16, 8, "REDUCED"),
    _cax("CAX8RH", 16, 8, "REDUCED", True, "runs as CAX8R; " + _NO_HYBRID),
)


class ElementCatalog:
    """Element type -> kernel resolution for the 2D and 3D solvers."""

    _BY_DIM: Dict[int, Dict[str, ElementDescriptor]] = {2: {}, 3: {}}
    for _d in _ELEMENT_DESCRIPTORS:
        for _n in (_d.type_name,) + _d.aliases:
            _BY_DIM[_d.dimension][_n] = _d
    _REGISTRY: Dict[str, ElementDescriptor] = {}
    for _d in _ELEMENT_DESCRIPTORS:
        _REGISTRY.setdefault(_d.type_name, _d)
    del _d, _n

    @classmethod
    def lookup(cls, etype: str, dim: Optional[int] = None) -> Optional[ElementDescriptor]:
        clean = etype.strip().upper()
        dims = (dim,) if dim in (2, 3) else (3, 2)
        for d in dims:
            desc = cls._BY_DIM[d].get(clean)
            if desc is not None:
                return desc
        return None

    @classmethod
    def resolve(cls, etype: str, dim: int, nlgeom: bool, nu: float = 0.3,
                linear_elastic: bool = True) -> Tuple[Optional[int], List[str]]:
        """(kernel id, notes) for an element of type `etype` in the `dim`D
        solver. Kernel None = not a solved element (point mass, coupling...).
        Raises NotImplementedError when the combination cannot run."""
        et = etype.strip().upper()
        if et in _SKIP:
            return None, []
        desc = cls._BY_DIM[dim].get(et)
        if desc is None:
            raise NotImplementedError(f"element type {et} is not implemented in the {dim}D solver")
        if desc.unsupported:
            raise NotImplementedError(f"element type {et}: {desc.unsupported}")
        reason = desc.unsupported_yes if nlgeom else desc.unsupported_no
        if reason:
            raise NotImplementedError(f"element type {et}: NLGEOM={'YES' if nlgeom else 'NO'} is not implemented ({reason})")
        if nlgeom:
            k, k_inc, k_inel, note = desc.kernel_nlgeom_yes, desc.kernel_yes_incompressible, desc.kernel_yes_inelastic, desc.note_yes
        else:
            k, k_inc, k_inel, note = desc.kernel_nlgeom_no, desc.kernel_no_incompressible, desc.kernel_no_inelastic, desc.note_no
        notes = [n for n in (desc.substitution_note, note) if n]
        if k_inc is not None and nu > 0.48:
            k = k_inc
        if k_inel is not None and not linear_elastic:
            k = k_inel
            if not nlgeom and desc.note_no_inelastic:
                notes.append(desc.note_no_inelastic)
        if k is None:
            raise NotImplementedError(f"element type {et}: no kernel for NLGEOM={'YES' if nlgeom else 'NO'}")
        info = KERNELS[dim][k]
        if nlgeom and info.nlgeom_unsupported:
            raise NotImplementedError(f"element type {et}: NLGEOM=YES is not implemented ({info.nlgeom_unsupported})")
        return k, notes

    @classmethod
    def detect_dim(cls, etype: str) -> Optional[int]:
        """Spatial dimension (2 or 3) from an element type name."""
        clean = etype.strip().upper()
        if clean in cls._BY_DIM[3] and clean not in cls._BY_DIM[2]:
            return 3
        if clean in cls._BY_DIM[2] and clean not in cls._BY_DIM[3]:
            return 2
        if clean.startswith(("CPE", "CPS", "CAX", "CGAX", "CGPE", "R2D", "RAX", "T2D")):
            return 2
        if clean.startswith(("C3D", "S4", "S3", "S8", "M3D", "STRI", "DCOUP3D")):
            return 3
        return None

    @classmethod
    def get_substitutions(cls) -> Dict[str, str]:
        return {d.type_name: d.substitution_note for d in cls._REGISTRY.values() if d.substitution_note}

    @classmethod
    def all_descriptors(cls) -> Tuple[ElementDescriptor, ...]:
        return tuple(cls._REGISTRY.values())
