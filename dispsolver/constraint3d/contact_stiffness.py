"""
contact_stiffness.py
======================
Auto-derived representative element stiffness for contact (design doc
sec A.1/B.7.1, dev_log/contact_abaqus_grade_design_20260915.md). Replaces
the hardcoded `1.0e8` fallback ContactPair.build_runtime_constraint used
when no penalty_stiffness was given, and is the single shared basis both
hard and soft contact's auto-scaled parameters draw from (`k_ref`).

First-cut estimator, not a literal Abaqus-internal formula (Abaqus does
not publish its own): per slave node, walk its parent element(s), take
`E_eff * L_elem` (L_elem = cbrt(bounding-box volume), a cheap proxy for
element size -- no proper min-edge-length computation for this first
cut), then the MINIMUM over all slave-node parent elements (conservative:
a stiffness default should not be dominated by one unusually stiff
element in a mixed-material contact zone).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np


def _material_effective_E(mat_obj: Any, default_E: float = 200000.0) -> float:
    """Best-effort scalar effective Young's modulus from whatever shape a
    material object/dict happens to be (mirrors the permissiveness of
    DynamicSolver3D._setup_numba_topology's own material-shape handling,
    but only needs a single scalar here, not a full constitutive dispatch).
    """
    if mat_obj is None:
        return default_E
    E = getattr(mat_obj, "E", None)
    if E is not None:
        return float(E)
    K = getattr(mat_obj, "K", None)
    mu = getattr(mat_obj, "mu", None)
    if K is not None and mu is not None:
        K, mu = float(K), float(mu)
        denom = 3.0 * K + mu
        if denom > 1e-30:
            return 9.0 * K * mu / denom
    if isinstance(mat_obj, dict):
        if "E" in mat_obj:
            return float(mat_obj["E"])
        if "K" in mat_obj and "mu" in mat_obj:
            K, mu = float(mat_obj["K"]), float(mat_obj["mu"])
            denom = 3.0 * K + mu
            if denom > 1e-30:
                return 9.0 * K * mu / denom
    return default_E


def _element_characteristic_length(mesh: Any, elem: Any) -> float:
    coords = np.array(
        [[mesh.nodes[nid].x, mesh.nodes[nid].y, mesh.nodes[nid].z] for nid in elem.node_ids],
        dtype=np.float64,
    )
    extents = coords.max(axis=0) - coords.min(axis=0)
    volume = float(np.prod(np.maximum(extents, 1e-30)))
    return volume ** (1.0 / 3.0)


def representative_element_stiffness(
    mesh: Any,
    materials: Optional[Dict[int, Any]],
    slave_node_ids,
) -> Optional[float]:
    """k_ref = min over the slave nodes' parent elements of (E_eff *
    L_elem). Returns None if `mesh` has no elements touching any slave
    node (e.g. a constraint built with a stub/empty mesh in a unit test)
    -- callers should fall back to a documented last-resort constant in
    that case, not silently produce 0 or NaN.
    """
    slave_set = set(slave_node_ids)
    k_values = []
    for elem in mesh.elements.values():
        if slave_set.isdisjoint(elem.node_ids):
            continue
        pid = getattr(elem, "pid", 0)
        mat_obj = (materials or {}).get(pid)
        E_eff = _material_effective_E(mat_obj)
        L_elem = _element_characteristic_length(mesh, elem)
        k_values.append(E_eff * L_elem)
    if not k_values:
        return None
    return min(k_values)


def representative_element_length(mesh: Any, slave_node_ids) -> Optional[float]:
    """L_char = min over the slave nodes' parent elements of L_elem
    (same characteristic-length proxy as representative_element_stiffness,
    without the material scaling) -- the shared length basis for
    nonlinear-penalty's `e`/`d` breakpoints (design doc sec A.2) and the
    default `allowable_penetration` soft contact falls back to (sec B.7.5)
    when the caller leaves it at None. Returns None under the same
    condition representative_element_stiffness does (no element found
    touching any slave node)."""
    slave_set = set(slave_node_ids)
    lengths = []
    for elem in mesh.elements.values():
        if slave_set.isdisjoint(elem.node_ids):
            continue
        lengths.append(_element_characteristic_length(mesh, elem))
    if not lengths:
        return None
    return min(lengths)
