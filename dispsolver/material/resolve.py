"""
dispsolver/material/resolve.py
==============================
The one place the 2D and 3D solvers turn a model material into (kernel id,
props vector).  It only *calls* the catalog (`material_id`, `pack_props`); it
never maps a material type to an id itself.

Contract (A5, audit O7):
- an unmapped material raises NotImplementedError naming the material and its
  deck location (regions.describe) -- it never silently becomes linear elastic;
- 2D and 3D go through the same function, so the same material gets the same id.

A pid with NO material at all is a different case and is handled by the
solvers (P2 raises for deck models, 3D keeps its constructor C_mat default).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from dispsolver.material.catalog import MATERIALS, material_id, pack_props
from dispsolver.material3d.numba_materials import MAT_HYPER_VISCOELASTIC, MAT_CUSTOM_ELASTIC


def _pack_hyper_viscoelastic(md: dict) -> np.ndarray:
    """Props layout of MAT_HYPER_VISCOELASTIC, exactly as the solvers packed it
    before A5.  TEMPORARY: the catalog resolves a to_numba_props dict that
    carries `prony` (and no `hyperelastic` key) to MAT_VISCOELASTIC_PRONY, so
    `pack_props` would write the wrong layout under id 6.  Remove once the
    catalog honours an explicit integer mat_type (reported to agent B)."""
    props = np.zeros(64, dtype=np.float64)
    prony = md["prony"]
    props[0] = float(md.get("base_model", 1))
    props[1] = float(md.get("mu", md.get("c10", 0.2)))
    props[2] = float(md.get("K", md.get("lambda_m", 10.0)))
    props[3] = float(md.get("p2", 0.0))
    n_terms = min(len(prony), 6)
    props[4] = float(n_terms)
    for m_idx in range(n_terms):
        item = prony[m_idx]
        props[5 + 2 * m_idx] = float(item[0])
        props[6 + 2 * m_idx] = float(item[2] if len(item) > 2 else item[1])
    return props


def resolve_material(mat_obj: Any, dim: int, label: str = "material") -> Tuple[int, np.ndarray]:
    """Return (mat_id, props[64]) for a Material object or property dict.

    Raises NotImplementedError (never falls back to linear elastic) when the
    material has no catalog entry for `dim`, or no stiffness definition.
    """
    from dispsolver.model.regions import describe

    where = describe(mat_obj, label)
    name = getattr(mat_obj, "name", None)
    if name:
        where = f"{where} (material {name!r})"

    if not isinstance(mat_obj, dict) and hasattr(mat_obj, "to_numba_props"):
        # The Material itself must be resolvable: its own mat_type is what the
        # deck said, and to_numba_props turns an unknown one into id 99.
        try:
            material_id(mat_obj)
        except ValueError as exc:
            raise NotImplementedError(f"{where}: {exc}") from exc
        md = mat_obj.to_numba_props()
    else:
        md = mat_obj
    try:
        mid = material_id(md)
    except ValueError as exc:
        raise NotImplementedError(f"{where}: {exc}") from exc

    spec = MATERIALS[mid]
    if dim not in spec.dims:
        raise NotImplementedError(f"{where}: material {spec.name} is not implemented for {dim}D")

    explicit = md.get("mat_type") if isinstance(md, dict) else None
    if isinstance(explicit, (int, np.integer)) and not isinstance(explicit, bool) \
            and int(explicit) in MATERIALS and int(explicit) != mid:
        if int(explicit) == MAT_HYPER_VISCOELASTIC and md.get("prony"):
            return MAT_HYPER_VISCOELASTIC, _pack_hyper_viscoelastic(md)
        raise NotImplementedError(
            f"{where}: the material carries id {int(explicit)} but the catalog resolves it to "
            f"{mid} ({spec.name})")

    extra = md if isinstance(md, dict) else (getattr(md, "extra_params", None) or {})
    if mid == MAT_CUSTOM_ELASTIC and getattr(md, "C_mat", extra.get("C_mat")) is None             and "engineering_constants" not in extra:
        raise NotImplementedError(
            f"{where}: mat_type {getattr(md, 'mat_type', extra.get('mat_type'))!r} has no solver material model "
            f"(no C_mat / engineering constants to build a stiffness from)")

    return mid, pack_props(md, dim)
