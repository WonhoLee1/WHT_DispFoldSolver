from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_base import _parse_data_rows
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


@GLOBAL_KEYWORD_REGISTRY.register(
    "NSET",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE, KeywordContext.STEP},
    support="FULL",
    params={
        "NSET": ParamSpec("NSET", required=True),
        "GENERATE": ParamSpec("GENERATE", type=bool, default=False, aliases=["GEN"]),
        "INSTANCE": ParamSpec("INSTANCE"),
        "ELSET": ParamSpec("ELSET"),
        "INTERNAL": ParamSpec("INTERNAL", type=bool, default=False),
        "UNSORTED": ParamSpec("UNSORTED", type=bool, default=False),
    },
    rows=RowSpec(column_types=[int], min_columns=1),
    description="Define a set of nodes",
)
def _handle_nset(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("nset", "")
    if not name:
        return
    _store_set(reader, "node", name, _set_members(reader, block, "node", name))


def _assembly_scope(reader: AbaqusInputReader) -> bool:
    """Sets are assembly sets inside *ASSEMBLY, and also at model level (after
    *END ASSEMBLY) in a deck that has an assembly: Abaqus decks refer to
    instance sets there as INSTANCE.SET / ASSEMBLY.INSTANCE.SET."""
    return (reader.context == KeywordContext.ASSEMBLY
            or (reader.context == KeywordContext.GLOBAL and getattr(reader, "has_assembly", False)))


def _set_members(reader: AbaqusInputReader, block: AbaqusKeywordBlock, kind: str, name: str):
    """Members of a *NSET/*ELSET as (instance name or None, label) pairs.

    Labels are instance-local; the instance is None outside the assembly.
    In the assembly a member may be a label (of INSTANCE=), an assembly set,
    a set of the INSTANCE= instance, or an instance-qualified set INST.SET
    (also written ASSEMBLY.INST.SET). Qualified names used to be rejected
    (and before that silently dropped)."""
    ids_attr = "node_ids" if kind == "node" else "element_ids"
    part_sets = "node_sets" if kind == "node" else "element_sets"
    kw = "*NSET" if kind == "node" else "*ELSET"
    asm = reader.model.root_assembly
    in_asm = _assembly_scope(reader)
    inst_param = block.params.get("instance", None)
    inst_key = None
    if in_asm and inst_param:
        if inst_param not in asm.instances:
            raise KeyError(f"{kw} {name}: instance {inst_param!r} is not defined")
        inst_key = asm.instances[inst_param].name

    def part_set(part, tok):
        for table in (getattr(part, part_sets), part.sets):
            if tok in table:
                return list(getattr(table[tok], ids_attr))
        return None

    generate = ("generate" in block.params or "gen" in block.params)
    out = []
    if generate:
        for line in block.data_lines:
            vals = [int(v) for v in line.replace(",", " ").split() if v.strip()]
            if len(vals) >= 2:
                step = vals[2] if len(vals) >= 3 else 1
                out += [(inst_key, i) for i in range(vals[0], vals[1] + 1, step)]
        return out

    part_ctx = None if in_asm else reader.get_active_part()
    for line in block.data_lines:
        for tok in (t.strip() for t in line.replace(",", " ").split() if t.strip()):
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
        if kind == "node":
            part.create_set(name, nodes=labels)
        else:
            part.create_set(name, elements=labels)


@GLOBAL_KEYWORD_REGISTRY.register(
    "ELSET",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE, KeywordContext.STEP},
    support="FULL",
    params={
        "ELSET": ParamSpec("ELSET", required=True),
        "GENERATE": ParamSpec("GENERATE", type=bool, default=False, aliases=["GEN"]),
        "INSTANCE": ParamSpec("INSTANCE"),
        "INTERNAL": ParamSpec("INTERNAL", type=bool, default=False),
    },
    rows=RowSpec(column_types=[int], min_columns=1),
    description="Define a set of elements",
)
def _handle_elset(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("elset", "")
    if not name:
        return
    _store_set(reader, "element", name, _set_members(reader, block, "element", name))


@GLOBAL_KEYWORD_REGISTRY.register(
    "SURFACE",
    valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE},
    support="FULL",
    params={
        "NAME": ParamSpec("NAME", required=True),
        "TYPE": ParamSpec("TYPE", default="ELEMENT", choices=["ELEMENT", "NODE", "SEGMENTS", "CYLINDER"]),
        "COMBINE": ParamSpec("COMBINE"),
        "TRIM": ParamSpec("TRIM", supported=False),
        "REGION TYPE": ParamSpec("REGION TYPE", aliases=["REGION_TYPE"], supported=False),
    },
    rows=RowSpec(column_types=[str, str], min_columns=1, max_columns=2),
    description="Define a geometry or element surface",
)
def _handle_surface(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", "")
    stype = block.params.get("type", "ELEMENT").upper()
    if stype == "SEGMENTS":
        # Parse 2D analytical rigid line segments (START, x, y / LINE, x, y)
        start_pt = None
        end_pt = None
        for line in block.data_lines:
            tokens = [t.strip().upper() for t in line.split(",") if t.strip()]
            if not tokens:
                continue
            cmd = tokens[0]
            if cmd == "START" and len(tokens) >= 3:
                start_pt = np.array([float(tokens[1]), float(tokens[2])], dtype=np.float64)
            elif cmd == "LINE" and len(tokens) >= 3:
                end_pt = np.array([float(tokens[1]), float(tokens[2])], dtype=np.float64)

        if start_pt is not None and end_pt is not None:
            tangent = end_pt - start_pt
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-12:
                t_hat = tangent / norm_t
                normal = np.array([-t_hat[1], t_hat[0]], dtype=np.float64)
            else:
                normal = np.array([0.0, 1.0], dtype=np.float64)

            from dispsolver.model.interaction import AnalyticalRigidSurface
            ref_nid = None
            if hasattr(reader, "pending_analytical_surfaces") and name in reader.pending_analytical_surfaces:
                ref_nid = reader.pending_analytical_surfaces[name]
            rigid_surf = AnalyticalRigidSurface(name=name, point=start_pt, normal=normal, reference_node=ref_nid)
            if hasattr(reader.model, "analytical_rigid_surfaces"):
                reader.model.analytical_rigid_surfaces[name] = rigid_surf
            part = reader.get_active_part()
            if part is not None:
                part.surfaces[name] = rigid_surf
            if hasattr(reader.model, "root_assembly") and hasattr(reader.model.root_assembly, "surfaces"):
                reader.model.root_assembly.surfaces[name] = rigid_surf
        return

    if reader.context == KeywordContext.ASSEMBLY or block.params.get("combine", "").upper() == "UNION":
        assembly = reader.model.root_assembly
        from dispsolver.model.set import Surface, SetScope, NodeSet
        collected_nids = set()
        collected_inst_nids = []
        collected_faces = []
        for line in block.data_lines:
            tokens = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
            for tok in tokens:
                if "." in tok:
                    inst_name, sub_name = tok.split(".", 1)
                    if inst_name in assembly.instances:
                        inst = assembly.instances[inst_name]
                        p = inst.part
                        sub_nids = []
                        if hasattr(p, "surfaces") and sub_name in p.surfaces:
                            surf = p.surfaces[sub_name]
                            sub_nids = surf.get_nodes(p) if hasattr(surf, "get_nodes") else list(getattr(surf, "node_ids", []))
                            if hasattr(surf, "faces"):
                                for ef in surf.faces:
                                    eid = ef.element_id if hasattr(ef, "element_id") else ef[0]
                                    fid = ef.face_id if hasattr(ef, "face_id") else ef[1]
                                    collected_faces.append((inst_name, eid, fid))
                        elif hasattr(p, "sets") and sub_name in p.sets:
                            sub_nids = p.sets[sub_name].get_nodes(include_elements=True)
                        elif hasattr(p, "node_sets") and sub_name in p.node_sets:
                            sub_nids = p.node_sets[sub_name].node_ids
                        for nid in sub_nids:
                            collected_inst_nids.append((inst_name, nid))
                else:
                    for p in reader.model.parts.values():
                        sub_nids = []
                        if hasattr(p, "surfaces") and tok in p.surfaces:
                            surf = p.surfaces[tok]
                            sub_nids = surf.get_nodes(p) if hasattr(surf, "get_nodes") else list(getattr(surf, "node_ids", []))
                            if hasattr(surf, "faces"):
                                for ef in surf.faces:
                                    eid = ef.element_id if hasattr(ef, "element_id") else ef[0]
                                    fid = ef.face_id if hasattr(ef, "face_id") else ef[1]
                                    collected_faces.append((None, eid, fid))
                        elif hasattr(p, "sets") and tok in p.sets:
                            sub_nids = p.sets[tok].get_nodes(include_elements=True)
                        elif hasattr(p, "node_sets") and tok in p.node_sets:
                            sub_nids = p.node_sets[tok].node_ids
                        for nid in sub_nids:
                            collected_nids.add(nid)
        surf_obj = Surface(name=name, node_ids=collected_nids, scope=SetScope.ASSEMBLY)
        surf_obj.inst_nodes = collected_inst_nids
        surf_obj.inst_faces = collected_faces
        assembly.surfaces[name] = surf_obj
        return

    part = reader.get_active_part()   # the instance's own part inside *INSTANCE

    faces = []
    nodes = []

    for line in block.data_lines:
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if not parts:
            continue
        if stype == "NODE":
            # Node set or node ID
            target = parts[0]
            if target in part.sets:
                nodes.extend(part.sets[target].node_ids)
            elif target in part.node_sets:
                nodes.extend(part.node_sets[target].node_ids)
            elif target.isdigit():
                nodes.append(int(target))
        else:
            # Element face definition: ELSET_NAME, S1..S6 or EID, S1..S6
            target = parts[0]
            face_id_str = parts[1] if len(parts) >= 2 else "S1"
            # S1 -> face index 1
            fid = int(face_id_str.upper().replace("S", "")) if face_id_str.upper().startswith("S") else 1
            if target in part.element_sets:
                for eid in part.element_sets[target].element_ids:
                    faces.append((eid, fid))
            elif target in part.sets:
                for eid in part.sets[target].element_ids:
                    faces.append((eid, fid))
            elif target.isdigit():
                faces.append((int(target), fid))

    if stype == "NODE":
        part.create_set(name, nodes=nodes)
    else:
        part.create_surface(name, faces=faces)


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

            if n_start in part.nodes and n_end in part.nodes:
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
                                if c_nid in part.nodes:
                                    center = np.array(part.nodes[c_nid], dtype=float)
                            except ValueError:
                                pass
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

            s1_nids = sorted(list(part.node_sets[s1_name].node_ids if s1_name in part.node_sets else (part.sets[s1_name].node_ids if s1_name in part.sets else [])))
            s2_nids = sorted(list(part.node_sets[s2_name].node_ids if s2_name in part.node_sets else (part.sets[s2_name].node_ids if s2_name in part.sets else [])))
            bounding_nids.extend(s1_nids)
            bounding_nids.extend(s2_nids)

            if s1_nids and s2_nids and len(s1_nids) == len(s2_nids) and n_intervals > 0:
                for nid1, nid2 in zip(s1_nids, s2_nids):
                    cur_inc = n_inc if n_inc is not None else (nid2 - nid1) // n_intervals
                    if nid1 in part.nodes and nid2 in part.nodes:
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

    old_nids = []
    if old_set in part.node_sets:
        old_nids = part.node_sets[old_set].node_ids
    elif old_set in part.sets:
        old_nids = part.sets[old_set].node_ids

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
    for nid in old_nids:
        if nid in part.nodes:
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

            if master_eid in part.elements:
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




