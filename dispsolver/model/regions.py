"""Resolve a BC / load / field ``region`` string to solver indices.

One resolver for every consumer (2D and 3D solvers, Model.create_solver*),
with Abaqus semantics. It replaces five private copies that disagreed with
each other and, worse, returned "nothing" for a name they could not match,
which every caller then treated as "skip this BC/load" -- the run went on
without it (keyword audit 2026-10-04).

Accepted forms, in order:
  * a node/element set name -- assembly-level or instance-qualified
    ``INST.SET`` -- matched case-insensitively (Abaqus labels are);
  * a bare part-level set name, only when it is unambiguous (the flattened
    system registers ``SET`` as an alias of ``INST.SET`` only if a single
    instance defines it; otherwise it is listed in ``sys.ambiguous_names``);
  * a node/element number: ``INST.<n>``, or bare ``<n>`` in a model with a
    single instance. The number is the deck's (part-local) label, never the
    solver's internal global id -- the old code tried the internal id first,
    which picks a different node whenever the deck numbering is not 1..N;
  * a step-level overlay node set (``*NSET`` written inside a ``*STEP``).

Anything else raises ``RegionError``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def describe(obj: Any, label: str) -> str:
    """`label` plus the deck location the object came from, when known
    (AbaqusInputReader stamps `source_location` = 'file:line')."""
    loc = getattr(obj, "source_location", None)
    return f"{label} [{loc}]" if loc else label


class RegionError(ValueError):
    """A region name in the model does not resolve to any node/element."""


def _strip_assembly(sys: Any, key: str) -> str:
    """ASSEMBLY.INSTANCE.SET -> INSTANCE.SET (Abaqus writes the assembly name in
    front of instance-qualified names in history data)."""
    asm = getattr(sys, "assembly_name", None)
    if asm and key.upper().startswith(str(asm).upper() + "."):
        return key[len(asm) + 1:]
    return key


def _instance_names(sys: Any) -> set:
    return {k[0] for k in getattr(sys, "node_local_to_global", {}).keys()}


def _ambiguous(sys: Any, region: str, kind: str, what: str) -> None:
    amb = getattr(sys, "ambiguous_names", {}).get(kind, {})
    for bare, quals in amb.items():
        if bare.upper() == region.upper():
            raise RegionError(
                f"{what}: region {region!r} is defined in several instances ({sorted(quals)}); "
                f"qualify it with the instance name, e.g. {sorted(quals)[0]!r}")


def resolve_node_region(sys: Any, region: Any, step: Any = None, what: str = "region") -> np.ndarray:
    """0-based node indices of `region`. Raises RegionError if unresolvable."""
    if region is None or (isinstance(region, str) and not region.strip()):
        raise RegionError(f"{what}: empty node region")
    gn = getattr(sys, "global_nsets", {})
    key = _strip_assembly(sys, str(region).strip())
    if key in gn:
        return np.asarray(gn[key], dtype=np.int64)
    _ambiguous(sys, key, "nset", what)
    if key.upper() == "ALL":
        return np.arange(len(sys.nid_to_idx), dtype=np.int64)
    l2g = getattr(sys, "node_local_to_global", {})
    if key.isdigit():
        insts = _instance_names(sys)
        if len(insts) != 1:
            raise RegionError(f"{what}: bare node number {key} in a model with instances {sorted(insts)}; "
                              f"write it as INSTANCE.{key}")
        g = l2g.get((next(iter(insts)), int(key)))
        if g is None:
            raise RegionError(f"{what}: node {key} does not exist")
        return np.array([sys.nid_to_idx[g]], dtype=np.int64)
    if "." in key:
        inst, tail = key.rsplit(".", 1)
        if tail.isdigit():
            for (iname, nid), g in l2g.items():
                if nid == int(tail) and iname.upper() == inst.upper():
                    return np.array([sys.nid_to_idx[g]], dtype=np.int64)
            raise RegionError(f"{what}: node {key!r} does not exist")
    if step is not None:
        overlay = getattr(step, "step_nsets", {})
        for name, ids in overlay.items():
            if name.upper() == key.upper():
                insts = _instance_names(sys)
                if len(insts) != 1:
                    raise RegionError(f"{what}: step-level *NSET {key!r} in a multi-instance model")
                inst = next(iter(insts))
                out = []
                for nid in ids:
                    g = l2g.get((inst, int(nid)))
                    if g is None:
                        raise RegionError(f"{what}: step-level *NSET {key!r} lists missing node {nid}")
                    out.append(sys.nid_to_idx[g])
                return np.array(sorted(out), dtype=np.int64)
    raise RegionError(f"{what}: node region {key!r} is not a node set, surface or node number of this model")


def resolve_elem_region(sys: Any, region: Any, what: str = "region", allow_blank_all: bool = False) -> np.ndarray:
    """0-based element indices of `region`. A blank region means the whole
    model only where Abaqus allows it (allow_blank_all, e.g. *DLOAD GRAV)."""
    n_elem = len(getattr(sys, "elem_conn_0based", []))
    if region is None or (isinstance(region, str) and not region.strip()):
        if allow_blank_all:
            return np.arange(n_elem, dtype=np.int64)
        raise RegionError(f"{what}: empty element region")
    ge = getattr(sys, "global_elsets", {})
    key = _strip_assembly(sys, str(region).strip())
    if key in ge:
        return np.asarray(ge[key], dtype=np.int64)
    _ambiguous(sys, key, "elset", what)
    if key.upper() == "ALL":
        return np.arange(n_elem, dtype=np.int64)
    l2g = getattr(sys, "elem_local_to_global", {})
    if key.isdigit():
        insts = {k[0] for k in l2g}
        if len(insts) != 1:
            raise RegionError(f"{what}: bare element number {key} in a model with instances {sorted(insts)}")
        g = l2g.get((next(iter(insts)), int(key)))
        if g is None:
            raise RegionError(f"{what}: element {key} does not exist")
        return np.array([g - 1], dtype=np.int64)
    if "." in key:
        inst, tail = key.rsplit(".", 1)
        if tail.isdigit():
            for (iname, eid), g in l2g.items():
                if eid == int(tail) and iname.upper() == inst.upper():
                    return np.array([g - 1], dtype=np.int64)
            raise RegionError(f"{what}: element {key!r} does not exist")
    raise RegionError(f"{what}: element region {key!r} is not an element set or element number of this model")


def node_ids_of(sys: Any, idx: np.ndarray) -> list:
    """Solver global node ids for 0-based node indices."""
    inv = getattr(sys, "_idx_to_nid", None)
    if inv is None or len(inv) != len(sys.nid_to_idx):
        inv = {i: g for g, i in sys.nid_to_idx.items()}
        try:
            sys._idx_to_nid = inv
        except Exception:
            pass
    return [inv[int(i)] for i in idx]
