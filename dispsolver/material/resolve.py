"""
dispsolver/material/resolve.py
==============================
The one place the 2D and 3D solvers turn a model material into (kernel id,
props vector).  It only *calls* the catalog (`material_id`, `pack_props`); it
never maps a material type to an id itself.

Contract (A5, audit O7):
- for a Material OBJECT the catalog is the only authority: id =
  material_id(mat), props = pack_props(mat).  `Material.to_numba_props()` (a
  second, independent type->id mapping in model/material.py) never decides the
  id or the props the kernel gets;
- for a property DICT the id is material_id(dict); an explicit integer
  `mat_type` that disagrees with it raises (no silent override either way);
- an unsupported material raises NotImplementedError naming the material, its
  type, the dimension and its deck location -- it never becomes linear elastic.

Where pack_props(Material) cannot read what the Material object carries
(TEMPORARY, the id is still the catalog's; props only, fed from the object's own
data via to_numba_props(), used purely as a flattener):
- ids 0, 2, 7, 12, 14 (linear elastic, J2, J2 Voce, tabular J2, Drucker-Prager):
  pack_props(Material) misses the linear hardening modulus taken from the
  *PLASTIC table, the nested `drucker_prager` dict, and the E default of a
  material without *ELASTIC; pack_props(flat dict) reads them.  The flat dict
  must resolve to the SAME catalog id, else NotImplementedError.
- id 13 (ductile damage): flat damage keys are ignored by pack_props
  (catalog.py:392-420) -> `_pack_ductile_damage`.
- id 6 (hyper-viscoelastic): `_pack_hyper_viscoelastic`.
Remove each once pack_props reads the Material for that id (agent B).

A pid with NO material at all is handled by the solvers (P2 raises for deck
models, 3D keeps its constructor C_mat default).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from dispsolver.material.catalog import MATERIALS, material_id, moduli_factor, pack_props, prony_check
from dispsolver.material3d.numba_materials import (
    MAT_LINEAR_ELASTIC,
    MAT_J2_PLASTICITY,
    MAT_J2_PLASTICITY_NONLINEAR,
    MAT_J2_PLASTICITY_TABULAR,
    MAT_J2_DUCTILE_DAMAGE,
    MAT_DRUCKER_PRAGER,
    MAT_VISCOELASTIC_PRONY,
    MAT_CONCRETE_DP,
    MAT_COHESIVE,
    MAT_HYPER_VISCOELASTIC,
    MAT_CUSTOM_ELASTIC,
    MAT_HYPERELASTIC_NEOHOOKEAN,
    MAT_HYPERELASTIC_ARRUDA_BOYCE,
    MAT_HYPERELASTIC_YEOH,
    MAT_HYPERELASTIC_MOONEY_RIVLIN,
    MAT_HYPERELASTIC_OGDEN,
    MAT_HYPERELASTIC_POLYNOMIAL,
    MAT_HYPERELASTIC_MARLOW,
)


class MaterialProps(dict):
    """A to_numba_props() dict that remembers the Material it came from, so the
    solvers (which receive these dicts from the deck path) still resolve the id
    and props from the Material object, not from the dict's own mat_type."""

    def __init__(self, props: dict, material: Any):
        super().__init__(props)
        self.material = material


def material_nu(mat_obj: Any, default: float = 0.3) -> float:
    """Poisson ratio of a Material object or property dict, for the solvers'
    element-kernel choice (near-incompressibility)."""
    if isinstance(mat_obj, dict):
        return float(mat_obj.get("nu", default))
    if hasattr(mat_obj, "to_numba_props"):
        return float(mat_obj.to_numba_props().get("nu", default))
    props = getattr(mat_obj, "props", None)
    return float(props[1]) if props is not None and len(props) > 1 else default


def _pack_hyper_viscoelastic(md: dict, where: str) -> np.ndarray:
    """Props of MAT_HYPER_VISCOELASTIC in the layout its kernel reads
    (numba_materials.py, `mat_type == MAT_HYPER_VISCOELASTIC`):
    [base_model, p0, p1, p2, n_terms, g1, tau1, ..., g6, tau6] with
    NH: p0=mu p1=K; AB: p0=mu p1=lambda_m p2=K.
    Any other base model raises: the Yeoh layout of that kernel puts d1 in the
    slot of the first Prony term, and Mooney-Rivlin constants are not carried
    by the material."""
    prony = md.get("prony")
    if not prony:
        raise NotImplementedError(f"{where}: hyper-viscoelastic material without Prony terms")
    base = int(md.get("base_model", MAT_HYPERELASTIC_NEOHOOKEAN))
    props = np.zeros(64, dtype=np.float64)
    props[0] = float(base)
    prony_check(prony, capacity=6, kernel="hyper-viscoelastic (id 6)")
    # LONG TERM constants (Keywords Guide default MODULI) -> the kernel's instantaneous shear constants;
    # the volumetric constant K is not relaxed by the kernel and is not scaled
    f_mod = moduli_factor(md, prony)
    if base == MAT_HYPERELASTIC_NEOHOOKEAN:
        props[1] = f_mod * float(md.get("mu", md.get("c10", 0.2)))
        props[2] = float(md.get("K", 10.0))
    elif base == MAT_HYPERELASTIC_ARRUDA_BOYCE:
        props[1] = f_mod * float(md.get("mu", 0.16785))
        props[2] = float(md.get("lambda_m", 3.0))
        if props[2] <= 0.0:
            raise NotImplementedError(f"{where}: Arruda-Boyce lambda_m = {props[2]} <= 0 (the kernel would use 3.0)")
        props[3] = float(md.get("K", 8.3333))
    else:
        raise NotImplementedError(
            f"{where}: hyper-viscoelastic with base model id {base} is not implemented "
            f"(only Neo-Hookean and Arruda-Boyce bases)")
    n_terms = min(len(prony), 6)
    props[4] = float(n_terms)
    for m_idx in range(n_terms):
        item = prony[m_idx]
        props[5 + 2 * m_idx] = float(item[0])
        props[6 + 2 * m_idx] = float(item[2] if len(item) > 2 else item[1])
    return props


def _pack_ductile_damage(md: dict) -> np.ndarray:
    """Props of MAT_J2_DUCTILE_DAMAGE from the flat to_numba_props keys, in the
    layout the kernel reads (unchanged from the pre-A5 chain)."""
    dp = list(md.get("damage_init_p", [0.05, 0, 0, 0, 0]))
    while len(dp) < 5:
        dp.append(0.0)
    props = np.zeros(64, dtype=np.float64)
    props[:17] = [
        float(md.get("E", 200000.0)), float(md.get("nu", 0.3)),
        float(md.get("yield_stress", 250.0)), float(md.get("hardening_modulus", 0.0)),
        float(md.get("damage_init_type", 0)),
        float(dp[0]), float(dp[1]), float(dp[2]), float(dp[3]), float(dp[4]),
        float(md.get("damage_Gf", 1.0)), float(md.get("damage_L", 1.0)), float(md.get("damage_evol", 0)),
        float(md.get("damage_T", 20.0)), float(md.get("damage_Troom", 20.0)),
        float(md.get("damage_Tmelt", 1000.0)), float(md.get("damage_eps0", 1.0)),
    ]
    return props


# ids whose packing reads E / nu (their kernels need the elastic constants)
_NEEDS_ELASTIC = {MAT_LINEAR_ELASTIC, MAT_VISCOELASTIC_PRONY, MAT_J2_PLASTICITY, MAT_J2_PLASTICITY_NONLINEAR,
                  MAT_J2_PLASTICITY_TABULAR, MAT_J2_DUCTILE_DAMAGE, MAT_DRUCKER_PRAGER, MAT_CONCRETE_DP}

_FLAT_PACKED = {MAT_LINEAR_ELASTIC, MAT_J2_PLASTICITY, MAT_J2_PLASTICITY_NONLINEAR,
                MAT_J2_PLASTICITY_TABULAR, MAT_DRUCKER_PRAGER}


def _resolve(mat_obj: Any, dim: int, label: str = "material") -> Tuple[int, np.ndarray]:
    """Return (mat_id, props[64]) for a Material object or property dict.

    Raises NotImplementedError (never falls back to linear elastic) when the
    material has no catalog entry for `dim`, or no stiffness definition, or a
    dict's explicit integer mat_type disagrees with the catalog.
    """
    from dispsolver.model.regions import describe

    if isinstance(mat_obj, MaterialProps):
        mat_obj = mat_obj.material
    where = describe(mat_obj, label)
    name = getattr(mat_obj, "name", None)
    if name:
        where = f"{where} (material {name!r})"

    is_dict = isinstance(mat_obj, dict)
    extra = mat_obj if is_dict else (getattr(mat_obj, "extra_params", None) or {})
    shown_type = mat_obj.get("mat_type", mat_obj.get("type")) if is_dict else getattr(mat_obj, "mat_type", None)

    try:
        mid = material_id(mat_obj)
    except ValueError as exc:
        raise NotImplementedError(f"{where}: {shown_type!r} is not implemented in {dim}D ({exc})") from exc

    spec = MATERIALS[mid]
    if dim not in spec.dims:
        raise NotImplementedError(f"{where}: {spec.name} ({shown_type!r}) is not implemented in {dim}D")
    if dim == 2 and "engineering_constants" in extra:
        raise NotImplementedError(
            f"{where}: orthotropic / lamina / anisotropic elasticity (ENGINEERING CONSTANTS) "
            f"is not implemented in 2D")

    if is_dict:
        explicit = mat_obj.get("mat_type")
        if isinstance(explicit, (int, np.integer)) and not isinstance(explicit, bool) and int(explicit) != mid:
            raise NotImplementedError(
                f"{where}: the property dict carries mat_type {int(explicit)} but the catalog "
                f"resolves its content to id {mid} ({spec.name}); pass the Material object, or a "
                f"dict whose mat_type agrees with its content")

    if mid == MAT_CUSTOM_ELASTIC and getattr(mat_obj, "C_mat", extra.get("C_mat")) is None \
            and "engineering_constants" not in extra:
        raise NotImplementedError(
            f"{where}: {shown_type!r} has no solver material model "
            f"(no C_mat / engineering constants to build a stiffness from)")

    if mid in _NEEDS_ELASTIC:
        if is_dict:
            missing = [k for k in ("E", "nu") if k not in mat_obj]
        else:
            missing = [] if (getattr(mat_obj, "elastic", None) is not None
                             or getattr(mat_obj, "E", None) is not None) else ["E", "nu"]
        refused = (getattr(mat_obj, "extra_params", None) or extra or {}).get("refused_definitions") \
            if missing else None
        if missing and refused:
            raise NotImplementedError(
                f"{where}: its {'; '.join(refused)} definition was refused by the reader; "
                f"nothing to compute it with (material has no *ELASTIC ({'/'.join(missing)} missing): "
                f"{spec.name} needs E, nu -- no default is assumed)")
        if missing:
            raise NotImplementedError(
                f"{where}: material has no *ELASTIC ({'/'.join(missing)} missing): "
                f"{spec.name} needs E, nu -- no default is assumed")

    if mid == MAT_HYPER_VISCOELASTIC:
        if is_dict:
            md = mat_obj
            if isinstance(md.get("hyperelastic"), dict):
                # _pack_hyper_viscoelastic reads top-level keys only: a nested
                # hyperelastic block (model + constants) would be ignored and the
                # Neo-Hookean base with default mu, K packed instead
                raise NotImplementedError(
                    f"{where}: a hyper-viscoelastic dict with a nested 'hyperelastic' block is not "
                    f"packed (its model and constants would be ignored); pass the Material object")
        else:
            md = mat_obj.to_numba_props()
            if md.get("mat_type") != MAT_HYPER_VISCOELASTIC:
                raise NotImplementedError(
                    f"{where}: the catalog resolves this material to {spec.name} but the material's "
                    f"own data cannot be packed as one (its type is {shown_type!r})")
        return mid, _pack_hyper_viscoelastic(md, where)

    if is_dict:
        if mid == MAT_J2_DUCTILE_DAMAGE and not isinstance(mat_obj.get("damage_init"), dict):
            # the flat damage_* layout (the documented DynamicSolver3D(materials={pid: dict})
            # form): pack_props reads only a nested damage_init dict and would silently
            # turn Johnson-Cook initiation into the constant-strain criterion
            return mid, _pack_ductile_damage(mat_obj)
        return mid, pack_props(mat_obj, dim)

    if mid == MAT_J2_DUCTILE_DAMAGE or mid in _FLAT_PACKED:
        flat = mat_obj.to_numba_props()
        if mid == MAT_J2_PLASTICITY_NONLINEAR and not ("R_inf" in flat and "b_exp" in flat):
            raise NotImplementedError(
                f"{where}: Voce hardening needs both R_inf and b_exp (flattened data has "
                f"{sorted(k for k in ('R_inf', 'b_exp') if k in flat)}); it is not packed with an invented default")
        flat_id = material_id(flat)
        if flat_id != mid:
            raise NotImplementedError(
                f"{where}: the catalog resolves this material to {spec.name} but its flattened "
                f"data resolves to {MATERIALS[flat_id].name}; refusing to pack it as either")
        return mid, (_pack_ductile_damage(flat) if mid == MAT_J2_DUCTILE_DAMAGE else pack_props(flat, dim))

    return mid, pack_props(mat_obj, dim)


# Constitutive definitions a Material can store, and which ids consume them. A stored definition the
# chosen id does not consume would be silently dropped (the id is picked from the content, so the result
# must not depend on keyword order): refuse it instead.


_HYPER_IDS = {MAT_HYPERELASTIC_NEOHOOKEAN, MAT_HYPERELASTIC_ARRUDA_BOYCE, MAT_HYPERELASTIC_YEOH,
              MAT_HYPERELASTIC_MOONEY_RIVLIN, MAT_HYPERELASTIC_OGDEN, MAT_HYPERELASTIC_POLYNOMIAL,
              MAT_HYPERELASTIC_MARLOW}


def _stored_definitions(m) -> dict:
    ex = m.extra_params
    out = {}
    if ex.get("creep"):
        out["*CREEP"] = None
    if m.plastic or ex.get("tabular_plastic"):
        out["*PLASTIC"] = None
    if m.viscoelastic_prony:
        out["*VISCOELASTIC"] = None
    if m.hyperelastic:
        out["*HYPERELASTIC"] = None
    if ex.get("mullins"):
        out["*MULLINS EFFECT"] = None
    if "damage_init" in ex:
        out["*DAMAGE INITIATION"] = None
    if "damage_evol" in ex:
        out["*DAMAGE EVOLUTION"] = None
    if ex.get("johnson_cook"):
        out["*PLASTIC, HARDENING=JOHNSON COOK"] = None
    if "drucker_prager" in ex:
        out["*DRUCKER PRAGER"] = None
    if "concrete_dp" in ex:
        out["*CONCRETE DAMAGED PLASTICITY"] = None
    if ex.get("rate_dependent"):
        out["*RATE DEPENDENT"] = None
    if ex.get("trs"):
        out["*TRS"] = None
    return out


def _consumed_by(mid: int) -> set:
    if mid in _HYPER_IDS:
        return {"*HYPERELASTIC"}
    return {
        MAT_HYPER_VISCOELASTIC: {"*HYPERELASTIC", "*VISCOELASTIC"},
        MAT_VISCOELASTIC_PRONY: {"*VISCOELASTIC"},
        MAT_J2_PLASTICITY: {"*PLASTIC"},
        MAT_J2_PLASTICITY_NONLINEAR: {"*PLASTIC"},
        MAT_J2_PLASTICITY_TABULAR: {"*PLASTIC"},
        MAT_J2_DUCTILE_DAMAGE: {"*PLASTIC", "*DAMAGE INITIATION", "*DAMAGE EVOLUTION"},
        MAT_DRUCKER_PRAGER: {"*DRUCKER PRAGER"},
        MAT_CONCRETE_DP: {"*CONCRETE DAMAGED PLASTICITY"},
        MAT_COHESIVE: {"*DAMAGE INITIATION", "*DAMAGE EVOLUTION"},
    }.get(mid, set())


def _refuse_unconsumed_definitions(m, mid: int, label: str) -> None:
    from dispsolver.model.regions import describe

    stored = _stored_definitions(m)
    dropped = sorted(set(stored) - _consumed_by(mid))
    where = None
    if dropped:
        where = describe(m, label) + (f" (material {m.name!r})" if getattr(m, "name", None) else "")
        raise NotImplementedError(
            f"{where}: it defines {', '.join(dropped)}, which {MATERIALS[mid].name} does not consume; "
            f"running it would silently drop the definition (no combined constitutive model exists for it)")
    if mid == MAT_HYPER_VISCOELASTIC:
        base = (m.hyperelastic or {}).get("model", "").upper()
        if not (base.startswith("NEO") or base.startswith("ARRUDA")):
            where = describe(m, label) + (f" (material {m.name!r})" if getattr(m, "name", None) else "")
            raise NotImplementedError(
                f"{where}: *VISCOELASTIC on a {base or 'missing'} hyperelastic base is not implemented (the "
                f"hyper-viscoelastic kernel carries Neo-Hooke and Arruda-Boyce bases; Mooney-Rivlin, Yeoh, "
                f"Ogden, polynomial constants would be replaced by Neo-Hooke defaults)")
    if mid == MAT_J2_DUCTILE_DAMAGE and m.plastic and len({float(r[0]) for r in m.plastic}) >= 2:
        # the damage kernel hardens linearly WITHOUT bound from the first two points
        # (numba_materials.py ~1270); Abaqus holds the last yield stress after the last point.
        # A flat table (one yield stress throughout) gives H=0 there: perfect plasticity, accepted.
        where = describe(m, label) + (f" (material {m.name!r})" if getattr(m, "name", None) else "")
        raise NotImplementedError(
            f"{where}: ductile damage with a tabular *PLASTIC ({len(m.plastic)} points) is not implemented: "
            f"the damage kernel extrapolates linearly without bound where Abaqus holds the last yield stress "
            f"(a single yield stress, perfect plasticity, is accepted)")
    if (mid in _HYPER_IDS or mid == MAT_HYPER_VISCOELASTIC) and getattr(m, "elastic", None) is not None:
        # Abaqus rejects *ELASTIC together with *HYPERELASTIC; the hyperelastic kernel would ignore it
        where = describe(m, label) + (f" (material {m.name!r})" if getattr(m, "name", None) else "")
        raise NotImplementedError(
            f"{where}: *ELASTIC together with *HYPERELASTIC is not a valid material (the *ELASTIC would be "
            f"ignored)")


def resolve_material(mat_obj: Any, dim: int, label: str = "material") -> Tuple[int, np.ndarray]:
    """Return (mat_id, props[64]) for a Material object or property dict (see `_resolve`).

    Two refusals apply to every material whatever its id:
      * a definition the deck gave and the reader refused (extra_params['refused_definitions'])
        -- running what is left would be a different model than the deck;
      * a non-finite packed value (e.g. the NaN a multilinear Drucker-Prager hardening leaves
        in d0) -- it must never reach a kernel.
    """
    from dispsolver.model.regions import describe

    obj = mat_obj.material if isinstance(mat_obj, MaterialProps) else mat_obj
    extra = obj if isinstance(obj, dict) else (getattr(obj, "extra_params", None) or {})
    refused = extra.get("refused_definitions")
    if refused:
        where = describe(obj, label)
        name = getattr(obj, "name", None)
        if name:
            where = f"{where} (material {name!r})"
        raise NotImplementedError(
            f"{where}: its {'; '.join(refused)} definition was refused by the reader; "
            f"nothing to compute it with -- running the rest would be a different model than the deck")
    mid, props = _resolve(mat_obj, dim, label)
    if not isinstance(obj, dict):
        _refuse_unconsumed_definitions(obj, mid, label)
    bad = np.flatnonzero(~np.isfinite(np.asarray(props, dtype=float)))
    if bad.size:
        where = describe(obj, label)
        name = getattr(obj, "name", None)
        if name:
            where = f"{where} (material {name!r})"
        raise NotImplementedError(
            f"{where}: packed {MATERIALS[mid].name} props are not finite at slot(s) "
            f"{', '.join(str(int(i)) for i in bad[:6])} (an undefined input value, e.g. a refused or "
            f"multilinear hardening table); refusing to hand them to a kernel")
    return mid, props
