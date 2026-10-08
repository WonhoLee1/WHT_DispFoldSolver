"""ContactCatalog -- the single table that turns a contact definition into a
runtime constraint class.

Before this module, ContactPair.build_runtime_constraint and GeneralContact
chose the constraint class through nested ``if constraint_enforcement ...`` /
``if sliding ...`` branches, and several combinations were substituted
silently (a rigid master with NITSCHE/MORTAR enforcement became a plain
penalty constraint; 2D classes ignore SMALL sliding).  Now:

    resolve_contact(dim, enforcement, sliding, nlgeom, master=, scope=, explicit_pairs=)
        -> ContactSpec

* the class (dotted path) and the contact mode (contact_geometry.contact_mode)
  come only from here;
* an unsupported combination raises NotImplementedError naming it -- never a
  silent substitute;
* a *supported* combination whose class does not implement part of the
  request says so in ``honors_sliding`` / ``honors_enforcement`` / ``note``
  (documented limitation, behavior unchanged).

``mode`` is what contact_geometry.contact_mode(sliding, nlgeom) yields.  The
solver's NLGEOM is only known at solve time, so the constraint classes that
consume it (Deformable3D, Nitsche3D) re-evaluate it in ``initialize``; the
spec's mode is the same function evaluated at resolve time.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from dispsolver.constraint3d.contact_geometry import contact_mode, normalize_sliding

ENFORCEMENTS = ("PENALTY", "AUGMENTED_LAGRANGE", "DUAL_MORTAR", "STANDARD_MORTAR", "NITSCHE")
_MORTAR = ("DUAL_MORTAR", "STANDARD_MORTAR")

_P3 = "dispsolver.constraint3d."
_P2 = "dispsolver.constraint."
C_DEF3D = _P3 + "surface_contact3d_deformable.DeformableSurfaceContactConstraint3D"
C_NITSCHE3D = _P3 + "nitsche_contact3d.NitscheContactConstraint3D"
C_MORTAR3D_SMALL = _P3 + "dual_mortar_clipping3d.DualMortarClippedContactConstraint3D"
C_MORTAR3D_FINITE = _P3 + "finite_sliding_mortar3d.FiniteSlidingDualMortar3D"
C_RIGID3D = _P3 + "surface_contact3d.SurfaceContactConstraint3D"
C_DEF2D = _P2 + "surface_contact2d_deformable.DeformableSurfaceContactConstraint2D"
C_MORTAR2D = _P2 + "dual_mortar_contact.DualMortarContactConstraint2D"
C_RIGID2D = _P2 + "surface_contact2d.SurfaceContactConstraint2D"
C_GENERAL3D = "dispsolver.contact.general_contact3d.GeneralContact3D"


@dataclass(frozen=True)
class ContactSpec:
    dim: int
    scope: str                 # "pair" (ContactPair) | "general" (GeneralContact)
    master: str                # "deformable" | "rigid"
    enforcement: str
    sliding: str               # normalized: SMALL | FINITE
    nlgeom: bool
    class_path: str            # dotted path of the concrete constraint class
    family: str                # builder argument family (see ContactPair.build_runtime_constraint)
    mode: str                  # contact_geometry.contact_mode(sliding, nlgeom)
    supported: bool = True
    reason: str = ""
    honors_sliding: bool = True
    honors_enforcement: bool = True
    note: str = ""

    def load(self) -> type:
        mod, _, name = self.class_path.rpartition(".")
        return getattr(importlib.import_module(mod), name)


def _key(dim, scope, master, enforcement, sliding, explicit_pairs):
    return (dim, scope, master, enforcement, sliding, explicit_pairs)


def _table() -> Dict[Tuple, Tuple[str, str, bool, bool, str]]:
    """key -> (class_path, family, honors_sliding, honors_enforcement, note)."""
    t: Dict[Tuple, Tuple[str, str, bool, bool, str]] = {}
    for sl in ("SMALL", "FINITE"):
        # ---- 3D deformable pair
        for enf in ("PENALTY", "AUGMENTED_LAGRANGE"):
            t[_key(3, "pair", "deformable", enf, sl, None)] = (C_DEF3D, "node_to_surface", True, True, "")
        t[_key(3, "pair", "deformable", "NITSCHE", sl, None)] = (C_NITSCHE3D, "nitsche3d", True, True, "")
        for enf in _MORTAR:
            if sl == "FINITE":
                t[_key(3, "pair", "deformable", enf, sl, None)] = (C_MORTAR3D_FINITE, "mortar3d_finite", True, True, "")
            else:
                t[_key(3, "pair", "deformable", enf, sl, None)] = (
                    C_MORTAR3D_SMALL, "mortar3d_small", True, True,
                    "reference-configuration clipped mortar; ignores the solver's NLGEOM contact mode")
        # ---- 2D deformable pair: the 2D classes always re-track in the current configuration
        for enf in ("PENALTY", "AUGMENTED_LAGRANGE"):
            t[_key(2, "pair", "deformable", enf, sl, None)] = (
                C_DEF2D, "node_to_segment2d", sl == "FINITE", True,
                "2D classes always track the contact point in the current configuration (finite sliding)")
        for enf in ("NITSCHE",) + _MORTAR:
            t[_key(2, "pair", "deformable", enf, sl, None)] = (
                C_MORTAR2D, "mortar2d", sl == "FINITE", True,
                "2D classes always track the contact point in the current configuration (finite sliding)")
        # ---- analytical rigid master (penalty / augmented Lagrange only)
        for enf in ("PENALTY", "AUGMENTED_LAGRANGE"):
            t[_key(3, "pair", "rigid", enf, sl, None)] = (
                C_RIGID3D, "rigid3d", False, True, "analytical rigid plane: sliding mode does not apply")
            t[_key(2, "pair", "rigid", enf, sl, None)] = (
                C_RIGID2D, "rigid2d", False, True, "analytical rigid line: sliding mode does not apply")
    # ---- GeneralContact (3D only; sliding is not a GeneralContact setting -> SMALL,
    # so there is no sliding request to honor).  GeneralContact3D stores
    # contact_enforcement / augmented_lagrange and never reads them: every
    # enforcement runs the same penalty formulation, so only PENALTY is honored.
    _NOT_HONORED = ("GeneralContact3D implements penalty only (contact_enforcement / augmented_lagrange "
                    "are stored, never read); the {enf} request cannot be applied")
    for ep in (True, False):
        t[_key(3, "general", "deformable", "PENALTY", "SMALL", ep)] = (C_GENERAL3D, "general3d", True, True,
                                                                     "GeneralContact has no sliding setting")
        for enf in ("AUGMENTED_LAGRANGE", "NITSCHE"):
            t[_key(3, "general", "deformable", enf, "SMALL", ep)] = (
                C_GENERAL3D, "general3d", True, False, _NOT_HONORED.format(enf=enf))
    for enf in _MORTAR:
        t[_key(3, "general", "deformable", enf, "SMALL", True)] = (
            C_MORTAR3D_SMALL, "mortar3d_general", True, True, "explicit included_pairs -> clipped mortar (one pair)")
        t[_key(3, "general", "deformable", enf, "SMALL", False)] = (
            C_GENERAL3D, "general3d", True, False,
            "mortar enforcement needs explicit included_pairs; it is NOT applied by GeneralContact3D")
    return t


_TABLE = _table()


def catalog_class_paths() -> frozenset:
    return frozenset(v[0] for v in _TABLE.values())


def catalog_keys():
    return tuple(_TABLE.keys())


def resolve_contact(dim: int, enforcement: str, sliding: str, nlgeom: bool = True, *,
                    master: str = "deformable", scope: str = "pair",
                    explicit_pairs: bool = False) -> ContactSpec:
    """Resolve (dim, enforcement, sliding, NLGEOM) -> ContactSpec, or raise NotImplementedError."""
    desc = (f"dim={dim}, scope={scope}, master={master}, enforcement={enforcement!r}, "
            f"sliding={sliding!r}, nlgeom={bool(nlgeom)}")
    enf = str(enforcement).upper()
    try:
        sl = normalize_sliding(sliding)
    except ValueError:
        raise NotImplementedError(f"contact combination not implemented ({desc}): "
                                  "sliding must be SMALL or FINITE") from None
    if enf not in ENFORCEMENTS:
        raise NotImplementedError(f"contact combination not implemented ({desc}): "
                                  f"enforcement must be one of {ENFORCEMENTS}")
    key = _key(int(dim), scope, master, enf, sl, bool(explicit_pairs) if scope == "general" else None)
    row = _TABLE.get(key)
    if row is None:
        why = ""
        if master == "rigid":
            why = (f"an analytical rigid master supports PENALTY / AUGMENTED_LAGRANGE only "
                   f"(it used to be substituted by penalty silently)")
        elif scope == "general" and int(dim) != 3:
            why = "GeneralContact is 3D only"
        raise NotImplementedError(f"contact combination not implemented ({desc})" + (f": {why}" if why else ""))
    cls_path, family, hs, he, note = row
    return ContactSpec(dim=int(dim), scope=scope, master=master, enforcement=enf, sliding=sl,
                       nlgeom=bool(nlgeom), class_path=cls_path, family=family,
                       mode=contact_mode(sl, bool(nlgeom)), honors_sliding=hs,
                       honors_enforcement=he, note=note)


def check_spec_honored(spec: ContactSpec, what: str) -> None:
    """Builders call this right after resolve_contact: a class that does not
    implement the requested enforcement raises; one that ignores the requested
    sliding mode warns (finite tracking is a superset of small sliding)."""
    if not spec.honors_enforcement:
        raise NotImplementedError(
            f"{what}: contact combination not implemented (dim={spec.dim}, scope={spec.scope}, "
            f"enforcement={spec.enforcement!r}): {spec.note}")
    if not spec.honors_sliding:
        import warnings
        warnings.warn(f"{what}: {spec.note} (requested sliding={spec.sliding})", stacklevel=3)
