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

from typing import Dict, Optional

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


def format_model_review(
    pid_element_count: Dict[int, int],
    pid_names: Dict[int, str],
    pid_params: Dict[int, dict],
    pid_types: Optional[Dict[int, str]] = None,
    pid_part_names: Optional[Dict[int, str]] = None,
) -> str:
    """Build the console review text. Dict keys may be int or str
    (Result.meta round-trips pid keys as strings through JSON) --
    normalized here so callers don't have to."""
    counts = {int(k): v for k, v in pid_element_count.items()}
    names = {int(k): v for k, v in pid_names.items()}
    params = {int(k): v for k, v in pid_params.items()}
    types = {int(k): v for k, v in (pid_types or {}).items()}
    part_names = {int(k): v for k, v in (pid_part_names or {}).items()}

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
        lines.append(f"  - {label}: {n_elem} elements, material={name}")

    lines.append(f"Total elements: {sum(counts.values())}, total layers/parts: {len(counts)}")
    lines.append(_SEP)
    return "\n".join(lines)


def print_model_review_from_builder_result(result) -> None:
    """Adapter for a live ModelBuilderResult / SimpleNamespace (has
    .mesh, .material_names, .material_params, and optionally
    .material_types)."""
    counts: Dict[int, int] = {}
    for elem in result.mesh.elements.values():
        pid = getattr(elem, "pid", 0)
        counts[pid] = counts.get(pid, 0) + 1
    print(format_model_review(
        counts, result.material_names, result.material_params,
        getattr(result, "material_types", None),
        getattr(result, "part_names", None),
    ))


def print_model_review_from_result(result) -> None:
    """Adapter for a loaded dispsolver.postprocess.result_io.Result (has
    .element_pid array, .meta['pid_names']/['pid_types'], .materials)."""
    import numpy as np
    pids, cnts = np.unique(result.element_pid, return_counts=True)
    counts = {int(p): int(c) for p, c in zip(pids, cnts)}
    print(format_model_review(
        counts, result.meta.get("pid_names", {}), result.materials,
        result.meta.get("pid_types", {}),
        result.meta.get("pid_part_names", {}),
    ))
