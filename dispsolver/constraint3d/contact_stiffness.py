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
    if hasattr(mat_obj, "hyperelastic") and mat_obj.hyperelastic:
        h = mat_obj.hyperelastic
        c10 = float(h.get("c10", 0.0))
        c01 = float(h.get("c01", 0.0))
        mu_val = 2.0 * (c10 + c01) if (c10 + c01) > 0 else float(h.get("mu", 1.0))
        d1_val = float(h.get("d1", h.get("d", 0.01)))
        K_val = 2.0 / d1_val if d1_val > 1e-12 else 1e6
        return float((9.0 * K_val * mu_val) / (3.0 * K_val + mu_val))

    if isinstance(mat_obj, dict):
        if "E" in mat_obj:
            return float(mat_obj["E"])
        if "K" in mat_obj and "mu" in mat_obj:
            K, mu = float(mat_obj["K"]), float(mat_obj["mu"])
            denom = 3.0 * K + mu
            if denom > 1e-30:
                return 9.0 * K * mu / denom
        if "c10" in mat_obj:
            c10 = float(mat_obj.get("c10", 0.0))
            c01 = float(mat_obj.get("c01", 0.0))
            mu_val = 2.0 * (c10 + c01) if (c10 + c01) > 0 else 1.0
            d1_val = float(mat_obj.get("d1", 0.01))
            K_val = 2.0 / d1_val if d1_val > 1e-12 else 1e6
            return float((9.0 * K_val * mu_val) / (3.0 * K_val + mu_val))
    return default_E


def _get_elem_nodes(elem: Any) -> list:
    if hasattr(elem, "node_ids"):
        return list(elem.node_ids)
    if isinstance(elem, (list, tuple)) and len(elem) >= 2:
        if isinstance(elem[1], (list, tuple)):
            return list(elem[1])
        return list(elem)
    return []


def _get_node_coords(mesh: Any, nid: int) -> np.ndarray:
    if hasattr(mesh, "nodes"):
        node = mesh.nodes[nid]
        if hasattr(node, "x"):
            z = getattr(node, "z", 0.0)
            return np.array([node.x, node.y, z], dtype=np.float64)
        arr = np.asarray(node, dtype=np.float64)
        if len(arr) == 2:
            return np.array([arr[0], arr[1], 0.0], dtype=np.float64)
        return arr[:3]
    return np.zeros(3, dtype=np.float64)


def _element_characteristic_length(mesh: Any, elem: Any) -> float:
    e_nids = _get_elem_nodes(elem)
    if not e_nids:
        return 1.0
    pts = np.array([_get_node_coords(mesh, nid) for nid in e_nids], dtype=np.float64)
    is_2d = np.all(np.abs(pts[:, 2]) < 1e-12)
    if is_2d:
        extents = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        area = float(np.prod(np.maximum(extents, 1e-30)))
        return max(area, 1e-30) ** 0.5
    else:
        extents = pts.max(axis=0) - pts.min(axis=0)
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
    if not hasattr(mesh, "elements") or not mesh.elements:
        return None
    slave_set = set(slave_node_ids)
    k_values = []
    for elem in mesh.elements.values():
        e_nids = _get_elem_nodes(elem)
        if slave_set.isdisjoint(e_nids):
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
    if not hasattr(mesh, "elements") or not mesh.elements:
        return None
    slave_set = set(slave_node_ids)
    lengths = []
    for elem in mesh.elements.values():
        e_nids = _get_elem_nodes(elem)
        if slave_set.isdisjoint(e_nids):
            continue
        lengths.append(_element_characteristic_length(mesh, elem))
    if not lengths:
        return None
    return min(lengths)
