from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple
import numpy as np

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_base import _parse_data_rows
from dispsolver.io.schema import (InputError, Keyword, Note, Param, Record, RowSchema, Schema, Unsupported,
                                  applies, keyword)
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    ParamSpec,
    RowSpec,
)

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── MESH DEFINITION (NODES, ELEMENTS, SETS, SURFACES) ──────────────────────

@GLOBAL_KEYWORD_REGISTRY.register(
    "NODE",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE},
    support="FULL",
    params={
        "NSET": ParamSpec("NSET"),
        "SYSTEM": ParamSpec("SYSTEM", default="RC", choices=["RC", "R", "C", "S"]),
        "INPUT": ParamSpec("INPUT", supported=False),
    },
    rows=RowSpec(column_types=[int, float, float, float], min_columns=2, max_columns=4),
    description="Define node coordinates and assign to node sets",
)
def _handle_node(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    import numpy as np
    part = reader.get_active_part()
    nset_name = block.params.get("nset", None)
    coord_system = block.params.get("system", "RC").upper()
    added_nids = []

    for line in block.data_lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("**"):
            continue
        parts = [p.strip() for p in line_str.split(",")]
        if not parts or not parts[0]:
            continue
        try:
            nid = int(parts[0])
        except ValueError:
            continue
        coords = []
        for x in parts[1:]:
            if not x:
                coords.append(0.0)
            else:
                try:
                    coords.append(float(x))
                except ValueError:
                    coords.append(0.0)

        if coord_system == "C":
            # Cylindrical: r, theta (in degrees), [z]
            r_val = coords[0] if len(coords) > 0 else 0.0
            theta_rad = np.radians(coords[1]) if len(coords) > 1 else 0.0
            x_val = r_val * np.cos(theta_rad)
            y_val = r_val * np.sin(theta_rad)
            z_val = coords[2] if len(coords) > 2 else 0.0
            coords = [x_val, y_val, z_val] if part.dim == 3 else [x_val, y_val]
        elif coord_system == "S":
            # Spherical: r, theta (polar deg), phi (azimuth deg)
            r_val = coords[0] if len(coords) > 0 else 0.0
            th_rad = np.radians(coords[1]) if len(coords) > 1 else 0.0
            ph_rad = np.radians(coords[2]) if len(coords) > 2 else 0.0
            x_val = r_val * np.sin(th_rad) * np.cos(ph_rad)
            y_val = r_val * np.sin(th_rad) * np.sin(ph_rad)
            z_val = r_val * np.cos(th_rad)
            coords = [x_val, y_val, z_val] if part.dim == 3 else [x_val, y_val]

        if len(coords) < part.dim:
            coords.extend([0.0] * (part.dim - len(coords)))
        part.Node(nid, *coords[:part.dim])
        added_nids.append(nid)

    if nset_name and added_nids:
        part.create_set(nset_name, nodes=added_nids)


@GLOBAL_KEYWORD_REGISTRY.register(
    "ELEMENT",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE},
    support="FULL",
    params={
        "TYPE": ParamSpec("TYPE", required=True),
        "ELSET": ParamSpec("ELSET"),
    },
    rows=RowSpec(column_types=[int], min_columns=2),
    description="Define elements by connectivity and assign to element sets",
)
def _handle_element(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    part = reader.get_active_part()
    etype = block.params.get("type", "C3D8").upper()
    elset_name = block.params.get("elset", None)
    rows = _parse_data_rows(block.data_lines, dtype=int)
    added_eids = []

    for r in rows:
        if not r:
            continue
        eid = int(r[0])
        conn = [int(n) for n in r[1:]]
        part.Element(eid, etype, conn)
        added_eids.append(eid)

    if elset_name and added_eids:
        part.create_set(elset_name, elements=added_eids)


# ── *NSET / *ELSET (schema; member resolution stays here until the BIND pass) ──

@dataclass(frozen=True)
class SetRecord(Record):
    kind: str                                   # node / element
    name: str
    instance: Optional[str]                     # INSTANCE= (assembly level only)
    tokens: Tuple[str, ...] = ()                # labels and set names, as written
    ranges: Tuple[Tuple[int, int, int], ...] = ()   # GENERATE: (first, last, increment)
    of_elset: Optional[str] = None              # *NSET, ELSET=: the nodes of these elements


class _Members(RowSchema):
    items: Tuple[str, ...] = ()


def _tokens(row) -> list:
    """Members of one data line; blank-separated members are tolerated as
    the legacy reader did (Abaqus itself requires commas)."""
    return [t for item in row.items if item for t in item.split()]


def _ranges(kw: str, name: str, rows) -> Tuple[Tuple[int, int, int], ...]:
    """GENERATE data lines 'first, last[, increment]'; (last - first) / increment
    must be a whole number (Abaqus Keywords Guide, *NSET / *ELSET)."""
    out = []
    for r in rows:
        vals = _tokens(r)
        try:
            nums = [int(v) for v in vals]
        except ValueError:
            raise InputError(f"*{kw} {name} at {r.loc}: GENERATE line must hold integers, got {vals}") from None
        if not 2 <= len(nums) <= 3:
            raise InputError(f"*{kw} {name} at {r.loc}: GENERATE line is 'first, last[, increment]', got {vals}")
        a, b, inc = nums[0], nums[1], (nums[2] if len(nums) == 3 else 1)
        if inc == 0 or (b - a) % inc:
            raise InputError(f"*{kw} {name} at {r.loc}: ({b} - {a}) / {inc} is not a whole number")
        out.append((a, b, inc))
    return tuple(out)


def _set_records(kw: str, kind: str, name: str, p, rows, ctx):
    if p.unsorted:
        # the model stores sets sorted; an order-dependent consumer (*EQUATION
        # node-set terms) then pairs the nodes in ascending order
        yield Note("warning", "set-order-sorted",
                         f"*{kw} {name} at {ctx.loc}: UNSORTED order is not kept (sets are stored sorted)",
                         loc=ctx.loc)
    if p.generate:
        yield SetRecord(kind, name, p.instance, ranges=_ranges(kw, name, rows), loc=ctx.loc)
    else:
        yield SetRecord(kind, name, p.instance, tokens=tuple(t for r in rows for t in _tokens(r)), loc=ctx.loc)


@keyword("NSET", support="FULL")
class NSet(Keyword):
    """Node set: node numbers / node sets, GENERATE ranges, or the nodes of an element set."""

    class Params(Schema):
        nset: str
        elset: Optional[str] = None
        generate: bool = False
        instance: Optional[str] = None
        internal: bool = False                  # Abaqus/CAE bookkeeping; no effect on the model
        unsorted: bool = False

    Row = _Members

    def to_records(self, p, rows, ctx):
        if p.elset is not None:
            # legacy: ELSET was declared supported and ignored -> an empty set
            if p.generate:
                raise InputError(f"*NSET {p.nset} at {ctx.loc}: ELSET and GENERATE are mutually exclusive")
            if rows:
                raise InputError(f"*NSET {p.nset} at {ctx.loc}: no data lines with ELSET")
            yield SetRecord("node", p.nset, p.instance, of_elset=p.elset, loc=ctx.loc)
            return
        yield from _set_records("NSET", "node", p.nset, p, rows, ctx)


@keyword("ELSET", support="FULL")
class ElSet(Keyword):
    """Element set: element numbers / element sets, or GENERATE ranges."""

    class Params(Schema):
        elset: str
        generate: bool = False
        instance: Optional[str] = None
        internal: bool = False
        unsorted: bool = False

    Row = _Members

    def to_records(self, p, rows, ctx):
        yield from _set_records("ELSET", "element", p.elset, p, rows, ctx)


@applies(SetRecord)
def _apply_set(reader: AbaqusInputReader, rec: SetRecord):
    _store_set(reader, rec.kind, rec.name, _set_members(reader, rec))


def _assembly_scope(reader: AbaqusInputReader) -> bool:
    """Sets are assembly sets inside *ASSEMBLY, and also at model level (after
    *END ASSEMBLY) in a deck that has an assembly: Abaqus decks refer to
    instance sets there as INSTANCE.SET / ASSEMBLY.INSTANCE.SET."""
    return (reader.context == KeywordContext.ASSEMBLY
            or (reader.context == KeywordContext.GLOBAL and getattr(reader, "has_assembly", False)))


def _set_members(reader: AbaqusInputReader, rec: SetRecord):
    """Members of a *NSET/*ELSET as (instance name or None, label) pairs.

    Labels are instance-local; the instance is None outside the assembly.
    In the assembly a member may be a label (of INSTANCE=), an assembly set,
    a set of the INSTANCE= instance, or an instance-qualified set INST.SET
    (also written ASSEMBLY.INST.SET). Qualified names used to be rejected
    (and before that silently dropped)."""
    kind, name = rec.kind, rec.name
    ids_attr = "node_ids" if kind == "node" else "element_ids"
    part_sets = "node_sets" if kind == "node" else "element_sets"
    kw = "*NSET" if kind == "node" else "*ELSET"
    asm = reader.model.root_assembly
    in_asm = _assembly_scope(reader)
    inst_key = None
    if in_asm and rec.instance:
        if rec.instance not in asm.instances:
            raise KeyError(f"{kw} {name}: instance {rec.instance!r} is not defined")
        inst_key = asm.instances[rec.instance].name

    def part_set(part, tok, attr=ids_attr, tables=part_sets):
        for table in (getattr(part, tables), part.sets):
            if tok in table:
                return list(getattr(table[tok], attr) or [])
        return None

    if rec.ranges:
        return [(inst_key, i) for a, b, inc in rec.ranges for i in range(a, b + (1 if inc > 0 else -1), inc)]

    if rec.of_elset is not None:
        # the nodes of the elements of a previously defined element set
        if in_asm:
            raise NotImplementedError(f"*NSET {name}, ELSET={rec.of_elset} at assembly level is not implemented")
        part = reader.get_active_part()
        eids = part_set(part, rec.of_elset, "element_ids", "element_sets")
        if eids is None:
            raise KeyError(f"*NSET {name}: element set {rec.of_elset!r} is not defined")
        return [(None, n) for n in sorted({n for e in eids for n in part.elements[int(e)][1]})]

    part_ctx = None if in_asm else reader.get_active_part()
    out = []
    for tok in rec.tokens:
        if tok.lstrip("-").isdigit():
            out.append((inst_key, int(tok)))
            continue
        if not in_asm:
            ids = part_set(part_ctx, tok) if part_ctx is not None else None
            if ids is None:
                raise KeyError(f"{kw} {name}: member {tok!r} is neither a {kind} number nor a defined {kind} set")
            out += [(None, i) for i in ids]
            continue
        # assembly level
        aset = getattr(asm, part_sets).get(tok)
        if aset is not None:
            mem = getattr(aset, "members", None)
            out += list(mem) if mem is not None else [(None, int(i)) for i in getattr(aset, ids_attr)]
            continue
        if inst_key is not None:
            ids = part_set(asm.instances[inst_key].part, tok)
            if ids is not None:
                out += [(inst_key, i) for i in ids]
                continue
        pieces = tok.split(".")
        if len(pieces) >= 2:
            if len(pieces) == 3 and pieces[0].upper() == str(asm.name).upper():
                pieces = pieces[1:]
            iname, sname = ".".join(pieces[:-1]), pieces[-1]
            if iname in asm.instances:
                inst = asm.instances[iname]
                ids = part_set(inst.part, sname)
                if ids is not None:
                    out += [(inst.name, i) for i in ids]
                    continue
        raise KeyError(f"{kw} {name}: member {tok!r} is neither a {kind} number, an assembly {kind} set "
                       f"nor an instance-qualified set (INSTANCE.SET)")
    return out


def _store_set(reader: AbaqusInputReader, kind: str, name: str, members):
    """Abaqus adds to a set that already exists ("assigned to the set, or added
    to the set if it already exists"); a part-level set used to be replaced."""
    asm = reader.model.root_assembly
    labels = [i for _, i in members]
    if _assembly_scope(reader):
        # An assembly set lists INSTANCE-local labels. It is stored on the
        # assembly with its instance and mapped per instance at flattening.
        table = asm.node_sets if kind == "node" else asm.element_sets
        prev = table.get(name)
        allm = (list(getattr(prev, "members", [])) if prev is not None else []) + list(members)
        aset = asm.Set(name, nodes=[i for _, i in allm]) if kind == "node" else asm.Set(name, elements=[i for _, i in allm])
        aset.members = allm
    elif reader.context == KeywordContext.STEP and reader.current_step is not None:
        cur = (reader.current_step.step_nsets if kind == "node" else reader.current_step.step_elsets).setdefault(name, [])
        for i in labels:
            if i not in cur:
                cur.append(i)
    else:
        # PART, GLOBAL and INSTANCE (get_active_part returns the instance's
        # own part there; the set keeps its deck name so sections and
        # INST.SET references find it -- it used to be renamed INST__SET).
        part = reader.get_active_part()
        legacy = part.node_sets if kind == "node" else part.element_sets
        if name in legacy:
            old = list(getattr(legacy[name], "node_ids" if kind == "node" else "element_ids") or [])
            seen = set(old)
            labels = old + [i for i in labels if i not in seen]
        if kind == "node":
            part.create_set(name, nodes=labels)
        else:
            part.create_set(name, elements=labels)


# ── *SURFACE (schema) ───────────────────────────────────────────────────────
#
# Legacy defects closed here: a missing face label became S1 (Abaqus uses all
# free faces of the elements), unresolved element / node / surface names were
# skipped without a word, a SEGMENTS surface kept only its first START and
# last LINE (CIRCL ignored), COMBINE=INTERSECTION / DIFFERENCE fell through to
# the element path, and at assembly level an element set without a face label
# became "all nodes of those elements".
#
# What a data-line name means (a surface to merge, an element set to take
# faces of, a node set) is only known against the model, so to_records keeps
# the lines as written and the applier resolves each name.

_FACE_RE = re.compile(r"^S([1-6])$")


@dataclass(frozen=True)
class SurfaceRecord(Record):
    name: str
    type: str                                   # ELEMENT / NODE
    items: Tuple[Tuple[str, Optional[int]], ...]   # (name or label, face 1..6 / None)


@dataclass(frozen=True)
class SegmentSurfaceRecord(Record):
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]


@keyword("SURFACE", support="PARTIAL")
class SurfaceKw(Keyword):
    """Element-face, node, single-line SEGMENTS and UNION surfaces."""

    class Params(Schema):
        name: str
        type: Literal["ELEMENT", "NODE", "SEGMENTS", "CYLINDER", "REVOLUTION", "CUTTING SURFACE",
                      "EULERIAN MATERIAL", "USER", "XFEM"] = Param(
            "ELEMENT", supported=("ELEMENT", "NODE", "SEGMENTS"))
        combine: Optional[Literal["UNION", "INTERSECTION", "DIFFERENCE"]] = Param(None, supported=("UNION",))
        internal: bool = False                  # Abaqus/CAE bookkeeping; no effect on the model

    Row = _Members

    def to_records(self, p, rows, ctx):
        if p.type == "SEGMENTS":
            yield _segments(p.name, rows, ctx)
            return
        lines = [(r, _tokens(r)) for r in rows]
        lines = [(r, t) for r, t in lines if t]
        if not lines:
            raise InputError(f"*SURFACE {p.name} at {ctx.loc}: no data lines")
        items = []
        for r, t in lines:
            if p.type == "NODE":
                # "node or node set[, area]": the area weights node-based contact
                if len(t) > 1 and float(t[1]) != 1.0:
                    raise Unsupported(f"*SURFACE {p.name} at {r.loc}: node area {t[1]} is not implemented "
                                      f"(1.0 only)")
                items.append((t[0], None))
                continue
            if len(t) == 1:
                items.append((t[0], None))      # a surface to merge, or an element set's free faces
                continue
            m = _FACE_RE.match(t[1].upper())
            if m is None:
                raise Unsupported(f"*SURFACE {p.name} at {r.loc}: face label {t[1]!r} is not implemented "
                                  f"(solid faces S1-S6 only)")
            items.append((t[0], int(m.group(1))))
        yield SurfaceRecord(p.name, p.type, tuple(items), loc=ctx.loc)


def _segments(name: str, rows, ctx) -> SegmentSurfaceRecord:
    """A 2D analytical rigid surface of ONE straight segment: START, x, y then
    LINE, x, y. More segments or a CIRCL is a different surface -> gap."""
    cmds = [_tokens(r) for r in rows]
    cmds = [[c[0].upper()] + c[1:] for c in cmds if c]
    if [c[0] for c in cmds] != ["START", "LINE"]:
        raise Unsupported(f"*SURFACE {name}, TYPE=SEGMENTS at {ctx.loc}: only a single START + LINE segment "
                          f"is implemented, got {[c[0] for c in cmds]}")
    try:
        pts = [(float(c[1]), float(c[2])) for c in cmds]
    except (IndexError, ValueError):
        raise InputError(f"*SURFACE {name}, TYPE=SEGMENTS at {ctx.loc}: START/LINE need x, y") from None
    return SegmentSurfaceRecord(name, pts[0], pts[1], loc=ctx.loc)


@applies(SegmentSurfaceRecord)
def _apply_segment_surface(reader: AbaqusInputReader, rec: SegmentSurfaceRecord):
    from dispsolver.model.interaction import AnalyticalRigidSurface
    start_pt = np.array(rec.start, dtype=np.float64)
    tangent = np.array(rec.end, dtype=np.float64) - start_pt
    norm_t = np.linalg.norm(tangent)
    if norm_t <= 1e-12:
        raise ValueError(f"*SURFACE {rec.name}: START and LINE points coincide")
    t_hat = tangent / norm_t
    normal = np.array([-t_hat[1], t_hat[0]], dtype=np.float64)
    ref_nid = getattr(reader, "pending_analytical_surfaces", {}).get(rec.name)
    rigid_surf = AnalyticalRigidSurface(name=rec.name, point=start_pt, normal=normal, reference_node=ref_nid)
    if hasattr(reader.model, "analytical_rigid_surfaces"):
        reader.model.analytical_rigid_surfaces[rec.name] = rigid_surf
    part = reader.get_active_part()
    if part is not None:
        part.surfaces[rec.name] = rigid_surf
    if hasattr(reader.model.root_assembly, "surfaces"):
        reader.model.root_assembly.surfaces[rec.name] = rigid_surf


def _face_pairs(surf):
    out = []
    for ef in getattr(surf, "faces", None) or []:
        out.append((ef.element_id, ef.face_id) if hasattr(ef, "element_id") else (ef[0], ef[1]))
    return out


def _free_faces_gap(rec, tok):
    return NotImplementedError(f"*SURFACE {rec.name}: {tok!r} without a face label (all free faces of the "
                               f"elements) is not implemented")


def _part_element_ids(part, tok):
    for table in (part.element_sets, part.sets):
        if tok in table and getattr(table[tok], "element_ids", None):
            return list(table[tok].element_ids)
    return None


def _part_node_ids(part, tok):
    for table in (part.node_sets, part.sets):
        if tok in table and getattr(table[tok], "node_ids", None):
            return list(table[tok].node_ids)
    return None


@applies(SurfaceRecord)
def _apply_surface(reader: AbaqusInputReader, rec: SurfaceRecord):
    if _assembly_scope(reader):
        _apply_assembly_surface(reader, rec)
        return
    part = reader.get_active_part()   # the instance's own part inside *INSTANCE
    if rec.type == "NODE":
        nodes = []
        for tok, _ in rec.items:
            ids = _part_node_ids(part, tok)
            if ids is not None:
                nodes += ids
            elif tok.isdigit():
                nodes.append(int(tok))
            else:
                raise KeyError(f"*SURFACE {rec.name}: {tok!r} is neither a node number nor a node set")
        part.create_set(rec.name, nodes=nodes)
        return
    faces = []
    for tok, fid in rec.items:
        if fid is None:
            if tok in getattr(part, "surfaces", {}):
                pairs = _face_pairs(part.surfaces[tok])           # a surface of surfaces
                if not pairs:
                    raise NotImplementedError(f"*SURFACE {rec.name}: member surface {tok!r} has no element "
                                              f"faces (analytical or node surface); merging it is not implemented")
                faces += pairs
                continue
            if _part_element_ids(part, tok) is not None or tok.isdigit():
                raise _free_faces_gap(rec, tok)
            raise KeyError(f"*SURFACE {rec.name}: {tok!r} is neither a surface nor an element set")
        eids = _part_element_ids(part, tok)
        if eids is not None:
            faces += [(e, fid) for e in eids]
        elif tok.isdigit():
            faces.append((int(tok), fid))
        else:
            raise KeyError(f"*SURFACE {rec.name}: {tok!r} is neither an element number nor an element set")
    part.create_surface(rec.name, faces=faces)


def _apply_assembly_surface(reader: AbaqusInputReader, rec: SurfaceRecord):
    """An assembly surface lists instance-local faces / nodes: (instance,
    element, face) in inst_faces and (instance, node) in inst_nodes."""
    from dispsolver.model.set import SetScope, Surface
    asm = reader.model.root_assembly
    inst_faces, inst_nodes = [], []

    def qualified(tok):
        """(instance, local name) of INST.NAME / ASSEMBLY.INST.NAME, or None."""
        pieces = tok.split(".")
        if len(pieces) == 3 and pieces[0].upper() == str(asm.name).upper():
            pieces = pieces[1:]
        if len(pieces) >= 2 and ".".join(pieces[:-1]) in asm.instances:
            return asm.instances[".".join(pieces[:-1])], pieces[-1]
        return None

    def _by_instance(rec, tok, aset):
        """{instance: [labels]} of an assembly set. Every member must belong to
        a defined instance and the set must not be empty -- dropping members
        here built an empty surface without a word (a *DSLOAD on it then
        loaded nothing)."""
        by_inst = {}
        for inst_name, label in getattr(aset, "members", None) or []:
            if inst_name is None:
                raise NotImplementedError(f"*SURFACE {rec.name}: set {tok!r} lists assembly-level labels "
                                          f"(no INSTANCE=); assembly-level nodes/elements are not implemented")
            if inst_name not in asm.instances:
                raise KeyError(f"*SURFACE {rec.name}: set {tok!r} refers to undefined instance {inst_name!r}")
            by_inst.setdefault(inst_name, []).append(label)
        if not by_inst:
            raise KeyError(f"*SURFACE {rec.name}: set {tok!r} is empty")
        return by_inst

    def add_faces(inst_name, part, pairs):
        inst_faces.extend((inst_name, e, f) for e, f in pairs)
        tmp = Surface(name=rec.name, faces=[])
        for e, f in pairs:
            tmp.add_face(e, f)
        inst_nodes.extend((inst_name, n) for n in tmp.get_nodes(part))

    def elements_of(tok):
        """[(instance name, part, [element ids])] of an assembly or qualified element set."""
        aset = asm.element_sets.get(tok)
        if aset is not None:
            return [(n, asm.instances[n].part, ids) for n, ids in _by_instance(rec, tok, aset).items()]
        q = qualified(tok)
        if q is not None:
            ids = _part_element_ids(q[0].part, q[1])
            if ids is not None:
                return [(q[0].name, q[0].part, ids)]
        return None

    def nodes_of(tok):
        aset = asm.node_sets.get(tok)
        if aset is not None:
            return [(n, i) for n, ids in _by_instance(rec, tok, aset).items() for i in ids]
        q = qualified(tok)
        if q is not None:
            ids = _part_node_ids(q[0].part, q[1])
            if ids is not None:
                return [(q[0].name, i) for i in ids]
        return None

    for tok, fid in rec.items:
        if rec.type == "NODE":
            got = nodes_of(tok)
            if got is None:
                raise KeyError(f"*SURFACE {rec.name}: {tok!r} is not an assembly or instance node set")
            inst_nodes += got
            continue
        if fid is None:
            other = asm.surfaces.get(tok)
            if other is not None and (getattr(other, "inst_faces", None) or getattr(other, "inst_nodes", None)):
                inst_faces += list(getattr(other, "inst_faces", []) or [])
                inst_nodes += list(getattr(other, "inst_nodes", []) or [])
                continue
            q = qualified(tok)
            if q is not None and tok.split(".")[-1] in getattr(q[0].part, "surfaces", {}):
                surf = q[0].part.surfaces[tok.split(".")[-1]]
                if _face_pairs(surf):
                    add_faces(q[0].name, q[0].part, _face_pairs(surf))
                else:
                    inst_nodes += [(q[0].name, n) for n in getattr(surf, "node_ids", None) or []]
                continue
            if elements_of(tok) is not None:
                raise _free_faces_gap(rec, tok)
            raise KeyError(f"*SURFACE {rec.name}: {tok!r} is not a defined surface")
        groups = elements_of(tok)
        if groups is None:
            raise KeyError(f"*SURFACE {rec.name}: {tok!r} is not an assembly or instance element set")
        for inst_name, part, ids in groups:
            add_faces(inst_name, part, [(e, fid) for e in ids])
    surf_obj = Surface(name=rec.name, node_ids=set(), scope=SetScope.ASSEMBLY)
    surf_obj.inst_nodes = inst_nodes
    surf_obj.inst_faces = inst_faces
    asm.surfaces[rec.name] = surf_obj


@GLOBAL_KEYWORD_REGISTRY.register(
    "NGEN",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE},
    support="FULL",
    params={
        "NSET": ParamSpec("NSET"),
        "LINE": ParamSpec("LINE", default="C", choices=["C", "P"], aliases=["L"]),
        "SYSTEM": ParamSpec("SYSTEM", default="RC", choices=["RC", "C", "S"]),
    },
    rows=RowSpec(min_columns=2, max_columns=10),
    description="Generate nodes between two nodes along a line or circular arc",
)
def _handle_ngen(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    import numpy as np
    part = reader.get_active_part()
    nset_name = block.params.get("nset", None)
    line_param = block.params.get("line", block.params.get("l", "")).upper()
    is_circle = line_param.startswith("C")
    added_nids = []

    for line in block.data_lines:
        parts_comma = [p.strip() for p in line.split(",")]
        parts_space = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if len(parts_space) >= 2:
            n_start = int(parts_space[0])
            n_end = int(parts_space[1])
            n_inc = int(parts_space[2]) if len(parts_space) >= 3 else 1

            missing = [n for n in (n_start, n_end) if n not in part.nodes]
            if missing:
                raise KeyError(f"*NGEN: end node(s) {missing} are not defined in part {part.name!r}")
            if True:
                coords_start = np.array(part.nodes[n_start], dtype=float)
                coords_end = np.array(part.nodes[n_end], dtype=float)

                n_steps = abs(n_end - n_start) // n_inc
                if n_steps > 0:
                    line_nids = [n_start]
                    if is_circle:
                        center = None
                        if len(parts_comma) >= 4 and parts_comma[3]:
                            try:
                                c_nid = int(parts_comma[3])
                            except ValueError:
                                c_nid = None
                            if c_nid:                      # 0 / blank: centre given by coordinates instead
                                if c_nid not in part.nodes:
                                    raise KeyError(f"*NGEN: circle-centre node {c_nid} is not defined in part {part.name!r}")
                                center = np.array(part.nodes[c_nid], dtype=float)
                        if center is None and len(parts_comma) >= 6 and parts_comma[4] and parts_comma[5]:
                            cx = float(parts_comma[4])
                            cy = float(parts_comma[5])
                            cz = float(parts_comma[6]) if len(parts_comma) >= 7 and parts_comma[6] else (coords_start[2] if len(coords_start) > 2 else 0.0)
                            center = np.array([cx, cy, cz]) if part.dim == 3 else np.array([cx, cy])
                        elif center is None and len(parts_space) >= 5:
                            try:
                                cx = float(parts_space[3])
                                cy = float(parts_space[4])
                                cz = float(parts_space[5]) if len(parts_space) >= 6 else (coords_start[2] if len(coords_start) > 2 else 0.0)
                                center = np.array([cx, cy, cz]) if part.dim == 3 else np.array([cx, cy])
                            except ValueError:
                                pass

                        if center is None:
                            center = np.zeros(part.dim, dtype=float)

                        normal_z = 1.0
                        if len(parts_comma) >= 10 and parts_comma[9]:
                            try:
                                normal_z = float(parts_comma[9])
                            except ValueError:
                                normal_z = 1.0

                        cx = center[0]
                        cy = center[1]
                        r1 = np.linalg.norm(coords_start[:2] - np.array([cx, cy]))
                        a1 = np.arctan2(coords_start[1] - cy, coords_start[0] - cx)
                        a2 = np.arctan2(coords_end[1] - cy, coords_end[0] - cx)

                        if normal_z < 0:
                            if a2 >= a1:
                                a2 -= 2.0 * np.pi
                        else:
                            if a2 <= a1:
                                a2 += 2.0 * np.pi

                        for i in range(1, n_steps):
                            nid = n_start + i * n_inc
                            frac = i / n_steps
                            ang = a1 + frac * (a2 - a1)
                            new_coords = coords_start.copy()
                            new_coords[0] = cx + r1 * np.cos(ang)
                            new_coords[1] = cy + r1 * np.sin(ang)
                            if len(coords_start) > 2:
                                new_coords[2] = coords_start[2] + frac * (coords_end[2] - coords_start[2])
                            part.Node(nid, *new_coords[:part.dim])
                            line_nids.append(nid)
                    else:
                        for i in range(1, n_steps):
                            nid = n_start + i * n_inc
                            frac = i / n_steps
                            coords = coords_start + frac * (coords_end - coords_start)
                            part.Node(nid, *coords[:part.dim])
                            line_nids.append(nid)
                    line_nids.append(n_end)
                    added_nids.extend(line_nids)

    if nset_name and added_nids:
        part.create_set(nset_name, nodes=sorted(list(set(added_nids))))


@GLOBAL_KEYWORD_REGISTRY.register(
    "NFILL",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE},
    support="FULL",
    params={
        "NSET": ParamSpec("NSET"),
        "BIAS": ParamSpec("BIAS", type=float, default=1.0),
        "SINGULAR": ParamSpec("SINGULAR", type=int, default=0),
        "TWO STEP": ParamSpec("TWO STEP", aliases=["TWO_STEP"], supported=False),
    },
    rows=RowSpec(min_columns=3, max_columns=4),
    description="Generate nodes between two node sets by interpolation",
)
def _handle_nfill(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    import numpy as np
    part = reader.get_active_part()
    nset_name = block.params.get("nset", None)
    bias_val = float(block.params.get("bias", 1.0))
    added_nids = []
    bounding_nids = []

    for line in block.data_lines:
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if len(parts) >= 3:
            s1_name = parts[0]
            s2_name = parts[1]
            n_intervals = int(parts[2])
            n_inc = int(parts[3]) if len(parts) >= 4 else None

            for _sn in (s1_name, s2_name):
                if _sn not in part.node_sets and _sn not in part.sets:
                    raise KeyError(f"*NFILL: node set {_sn!r} is not defined in part {part.name!r}")
            s1_nids = sorted(list(part.node_sets[s1_name].node_ids if s1_name in part.node_sets else part.sets[s1_name].node_ids))
            s2_nids = sorted(list(part.node_sets[s2_name].node_ids if s2_name in part.node_sets else part.sets[s2_name].node_ids))
            bounding_nids.extend(s1_nids)
            bounding_nids.extend(s2_nids)

            if s1_nids and s2_nids and len(s1_nids) == len(s2_nids) and n_intervals > 0:
                for nid1, nid2 in zip(s1_nids, s2_nids):
                    cur_inc = n_inc if n_inc is not None else (nid2 - nid1) // n_intervals
                    _miss = [n for n in (nid1, nid2) if n not in part.nodes]
                    if _miss:
                        raise KeyError(f"*NFILL: node(s) {_miss} of sets {s1_name!r}/{s2_name!r} are not defined in part {part.name!r}")
                    if True:
                        c1 = np.array(part.nodes[nid1], dtype=float)
                        c2 = np.array(part.nodes[nid2], dtype=float)
                        for i in range(1, n_intervals):
                            new_nid = nid1 + i * cur_inc
                            if bias_val != 1.0 and abs(bias_val - 1.0) > 1e-6:
                                frac = (1.0 - bias_val**i) / (1.0 - bias_val**n_intervals)
                            else:
                                frac = i / float(n_intervals)
                            new_c = c1 + frac * (c2 - c1)
                            part.Node(new_nid, *new_c[:part.dim])
                            added_nids.append(new_nid)

    if nset_name:
        all_set_nids = sorted(list(set(bounding_nids + added_nids)))
        if all_set_nids:
            part.create_set(nset_name, nodes=all_set_nids)


@GLOBAL_KEYWORD_REGISTRY.register(
    "NCOPY",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE},
    support="FULL",
    params={
        "OLD SET": ParamSpec("OLD SET", required=True, aliases=["OLD_SET"]),
        "CHANGE NUMBER": ParamSpec("CHANGE NUMBER", type=int, required=True, aliases=["CHANGE_NUMBER"]),
        "SHIFT": ParamSpec("SHIFT", default="NO"),
        "NEW SET": ParamSpec("NEW SET", aliases=["NEW_SET"]),
        "MULTIPLE": ParamSpec("MULTIPLE", type=int, supported=False),
        "POLE": ParamSpec("POLE", supported=False),
    },
    rows=RowSpec(column_types=[float], min_columns=0, max_columns=7),
    description="Copy a node set with an optional translation and coordinate shift",
)
def _handle_ncopy(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    import numpy as np
    part = reader.get_active_part()

    old_set = block.params.get("old set", block.params.get("old_set", block.params.get("oldset", "")))
    new_set = block.params.get("new set", block.params.get("new_set", block.params.get("newset", None)))
    change_num = int(block.params.get("change number", block.params.get("change_number", block.params.get("changenumber", 0))))

    if old_set in part.node_sets:
        old_nids = part.node_sets[old_set].node_ids
    elif old_set in part.sets:
        old_nids = part.sets[old_set].node_ids
    else:
        raise KeyError(f"*NCOPY: OLD SET {old_set!r} is not defined in part {part.name!r}")

    if not old_nids:
        return

    shift = np.zeros(part.dim, dtype=float)
    if block.data_lines:
        vals = []
        for token in block.data_lines[0].replace(",", " ").split():
            try:
                vals.append(float(token))
            except ValueError:
                pass
        if vals:
            shift[:min(len(vals), part.dim)] = vals[:part.dim]

    new_nids = []
    _miss = [n for n in old_nids if n not in part.nodes]
    if _miss:
        raise KeyError(f"*NCOPY: node(s) {sorted(_miss)} of set {old_set!r} are not defined in part {part.name!r}")
    for nid in old_nids:
        if True:
            coords = np.array(part.nodes[nid], dtype=float) + shift
            new_nid = nid + change_num
            part.Node(new_nid, *coords[:part.dim])
            new_nids.append(new_nid)

    if new_set and new_nids:
        part.create_set(new_set, nodes=new_nids)


@GLOBAL_KEYWORD_REGISTRY.register(
    "ELGEN",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE},
    support="FULL",
    params={
        "ELSET": ParamSpec("ELSET"),
    },
    rows=RowSpec(column_types=[int], min_columns=2, max_columns=7),
    description="Generate elements by replicating an existing element pattern",
)
def _handle_elgen(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    part = reader.get_active_part()
    elset_name = block.params.get("elset", None)
    added_eids = []

    for line in block.data_lines:
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if len(parts) >= 2:
            master_eid = int(parts[0])
            n_col = int(parts[1])
            node_inc_col = int(parts[2]) if len(parts) >= 3 else 1
            elem_inc_col = int(parts[3]) if len(parts) >= 4 else 1

            n_row = int(parts[4]) if len(parts) >= 5 else 1
            node_inc_row = int(parts[5]) if len(parts) >= 6 else 0
            elem_inc_row = int(parts[6]) if len(parts) >= 7 else node_inc_row

            if master_eid not in part.elements:
                raise KeyError(f"*ELGEN: master element {master_eid} is not defined in part {part.name!r}")
            if True:
                etype, master_conn = part.elements[master_eid]
                for r in range(n_row):
                    for c in range(n_col):
                        if r == 0 and c == 0:
                            added_eids.append(master_eid)
                            continue
                        new_eid = master_eid + c * elem_inc_col + r * elem_inc_row
                        new_conn = [nid + c * node_inc_col + r * node_inc_row for nid in master_conn]
                        part.Element(new_eid, etype, new_conn)
                        added_eids.append(new_eid)

    if elset_name and added_eids:
        part.create_set(elset_name, elements=added_eids)




