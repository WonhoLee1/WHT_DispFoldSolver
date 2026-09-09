"""
model_review.py
================
Console-only startup review of a model's material/part structure --
printed once at solver startup (examples/ex12_abaqus_inp_plate_fold.py)
and once when a saved result is loaded into the Qt viewer
(dispsolver/postprocess/viewer.py). Generic over plain pid-keyed dicts
so the exact same formatter works from a live ModelBuilderResult/
SimpleNamespace and from a loaded Result (see the two adapter functions
below format_model_review()).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..material.type_tags import J2_PLASTIC, ARRUDA_BOYCE_VISCO, NEOHOOKEAN

_SEP = "=" * 100


def _material_repr(params: dict, type_tag: Optional[str]) -> str:
    """One representative-properties line for a material, keyed off its
    canonical type tag. Falls back to just listing param-dict key names
    for an unknown/missing tag (e.g. a saved result from before
    `pid_types` existed in Meta) rather than guessing from key presence.
    """
    if type_tag == J2_PLASTIC:
        return (f"J2 plasticity: E={params.get('E')}, nu={params.get('nu')}, "
                f"sigma_y0={params.get('sigma_y0')}, H={params.get('H')}")
    if type_tag == ARRUDA_BOYCE_VISCO:
        return (f"Arruda-Boyce + Prony viscoelastic: mu={params.get('mu')}, "
                f"lambda_m={params.get('lambda_m')}, K={params.get('K')}")
    if type_tag == NEOHOOKEAN:
        return f"NeoHookean: E={params.get('E')}, nu={params.get('nu')}"
    keys = [k for k in params if k not in ("base", "prony", "wlf")]
    return f"(unknown type; params: {', '.join(sorted(keys))})"


def compute_pid_thickness_from_mesh(mesh) -> Dict[int, float]:
    """Compute physical y-thickness (max_y - min_y) for each element property ID (pid) in a Mesh.

    Parameters
    ----------
    mesh : Mesh
        A mesh object containing `.elements` (dict of Element) and `.nodes` (dict of Node).

    Returns
    -------
    dict[int, float]
        Mapping from pid to thickness in millimeters (Delta y = max(y) - min(y)).
    """
    if not hasattr(mesh, "elements") or not hasattr(mesh, "nodes"):
        return {}
    pid_ys: Dict[int, List[float]] = {}
    for elem in mesh.elements.values():
        pid = getattr(elem, "pid", 0)
        ys = []
        for nid in getattr(elem, "node_ids", []):
            if nid in mesh.nodes:
                node = mesh.nodes[nid]
                y_val = getattr(node, "y", None)
                if y_val is None and hasattr(node, "coords"):
                    y_val = float(node.coords[1])
                if y_val is not None:
                    ys.append(float(y_val))
        if not ys:
            continue
        min_y, max_y = min(ys), max(ys)
        if pid not in pid_ys:
            pid_ys[pid] = [min_y, max_y]
        else:
            if min_y < pid_ys[pid][0]:
                pid_ys[pid][0] = min_y
            if max_y > pid_ys[pid][1]:
                pid_ys[pid][1] = max_y
    return {pid: ymax - ymin for pid, (ymin, ymax) in pid_ys.items()}


def compute_pid_thickness_from_result(result) -> Dict[int, float]:
    """Compute physical y-thickness (max_y - min_y) for each element property ID (pid) in a Result.

    Parameters
    ----------
    result : Result
        A loaded `Result` instance, optionally containing `meta["pid_thickness"]` or
        raw topology/points arrays.

    Returns
    -------
    dict[int, float]
        Mapping from pid to thickness in millimeters (Delta y = max(y) - min(y)).
    """
    import numpy as np

    if hasattr(result, "meta") and isinstance(result.meta, dict) and "pid_thickness" in result.meta:
        return {int(k): float(v) for k, v in result.meta["pid_thickness"].items()}

    if (
        not hasattr(result, "element_pid")
        or not hasattr(result, "points")
        or not hasattr(result, "connectivity")
        or not hasattr(result, "offsets")
    ):
        return {}

    pid_thickness: Dict[int, float] = {}
    pids = np.unique(result.element_pid)
    for pid in pids:
        e_idx = np.where(result.element_pid == pid)[0]
        if len(e_idx) > 0:
            node_idx = np.concatenate(
                [result.connectivity[result.offsets[e]:result.offsets[e + 1]] for e in e_idx]
            )
            ys = result.points[node_idx, 1]
            pid_thickness[int(pid)] = float(ys.max() - ys.min())
    return pid_thickness


def format_model_review(
    pid_element_count: Dict[int, int],
    pid_names: Dict[int, str],
    pid_params: Dict[int, dict],
    pid_types: Optional[Dict[int, str]] = None,
    pid_part_names: Optional[Dict[int, str]] = None,
    pid_element_type: Optional[Dict[int, str]] = None,
    pid_thickness: Optional[Dict[int, float]] = None,
) -> str:
    """Build the console review text. Dict keys may be int or str
    (Result.meta round-trips pid keys as strings through JSON) --
    normalized here so callers don't have to.

    Parameters
    ----------
    pid_element_count : dict[int, int]
        Number of elements per pid.
    pid_names : dict[int, str]
        Material name per pid.
    pid_params : dict[int, dict]
        Constitutive parameters per pid.
    pid_types : dict[int, str], optional
        Canonical type tags per pid.
    pid_part_names : dict[int, str], optional
        Human-readable part name overrides per pid (e.g. 'Plate Left').
    pid_element_type : dict[int, str], optional
        Element formulation name per pid (e.g. 'Q4_UP', 'Q4').
    pid_thickness : dict[int, float], optional
        Physical thickness (Delta y in mm) per pid.

    Returns
    -------
    str
        Formatted console review banner.
    """
    counts = {int(k): v for k, v in pid_element_count.items()}
    names = {int(k): v for k, v in pid_names.items()}
    params = {int(k): v for k, v in pid_params.items()}
    types = {int(k): v for k, v in (pid_types or {}).items()}
    part_names = {int(k): v for k, v in (pid_part_names or {}).items()}
    elem_types = {int(k): v for k, v in (pid_element_type or {}).items()}
    thicknesses = {int(k): float(v) for k, v in (pid_thickness or {}).items()}

    lines = [_SEP, " MODEL REVIEW", _SEP]

    # -- Materials: dedup by name --
    by_name: Dict[str, int] = {}  # name -> a representative pid
    for pid in sorted(names):
        by_name.setdefault(names[pid], pid)
    lines.append(f"Materials ({len(by_name)} distinct):")
    for name, pid in by_name.items():
        lines.append(f"  - {name}: {_material_repr(params.get(pid, {}), types.get(pid))}")

    # -- Parts/Layers: one line per pid -- non-layer parts (e.g. rigid
    # plates) show their part_names override ("Plate Left") instead of
    # a bare pid number.
    lines.append(f"Parts / Layers ({len(counts)} total):")
    for pid in sorted(counts):
        n_elem = counts[pid]
        name = names.get(pid, f"pid{pid}")
        label = part_names.get(pid, f"pid {pid}")
        thick = thicknesses.get(pid)
        if thick is not None:
            thick_str = f", thickness={thick:.4f} mm ({thick * 1e3:.1f} um)"
        else:
            thick_str = ""
        etype = elem_types.get(pid, "")
        etype_str = f", element={etype}" if etype else ""
        lines.append(f"  - {label}: {n_elem} elements, material={name}{thick_str}{etype_str}")

    lines.append(f"Total elements: {sum(counts.values())}, total layers/parts: {len(counts)}")
    lines.append(_SEP)
    return "\n".join(lines)


def print_model_review_from_builder_result(
    result,
    pid_element_type: Optional[Dict[int, str]] = None,
    pid_thickness: Optional[Dict[int, float]] = None,
) -> None:
    """Adapter for a live ModelBuilderResult / SimpleNamespace (has
    .mesh, .material_names, .material_params, and optionally
    .material_types).

    Parameters
    ----------
    result : ModelBuilderResult or SimpleNamespace
        Model builder result containing .mesh, .material_names, etc.
    pid_element_type : dict[int, str], optional
        Element formulation name per pid.
    pid_thickness : dict[int, float], optional
        Explicit thickness mapping. If omitted, computed from result.mesh.
    """
    counts: Dict[int, int] = {}
    for elem in result.mesh.elements.values():
        pid = getattr(elem, "pid", 0)
        counts[pid] = counts.get(pid, 0) + 1
    if pid_thickness is None and hasattr(result, "mesh"):
        pid_thickness = compute_pid_thickness_from_mesh(result.mesh)
    print(format_model_review(
        counts, result.material_names, result.material_params,
        getattr(result, "material_types", None),
        getattr(result, "part_names", None),
        pid_element_type=pid_element_type,
        pid_thickness=pid_thickness,
    ))


def print_model_review_from_result(result) -> None:
    """Adapter for a loaded dispsolver.postprocess.result_io.Result (has
    .element_pid array, .meta['pid_names']/['pid_types'], .materials).

    Parameters
    ----------
    result : Result
        Loaded solve result object.
    """
    import numpy as np
    pids, cnts = np.unique(result.element_pid, return_counts=True)
    counts = {int(p): int(c) for p, c in zip(pids, cnts)}
    elem_types = result.meta.get("pid_element_types", {}) if hasattr(result, "meta") else {}
    pid_thickness = compute_pid_thickness_from_result(result)
    print(format_model_review(
        counts, result.meta.get("pid_names", {}), result.materials,
        result.meta.get("pid_types", {}),
        result.meta.get("pid_part_names", {}),
        pid_element_type=elem_types,
        pid_thickness=pid_thickness,
    ))
