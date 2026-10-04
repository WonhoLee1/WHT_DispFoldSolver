from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_base import _parse_data_rows
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── MESH DEFINITION (NODES, ELEMENTS, SETS, SURFACES) ──────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("NODE", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE})
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


@GLOBAL_KEYWORD_REGISTRY.register("ELEMENT", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE})
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


@GLOBAL_KEYWORD_REGISTRY.register("NSET", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE, KeywordContext.STEP})
def _handle_nset(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("nset", "")
    if not name:
        return

    generate = "generate" in block.params or "gen" in block.params or block.params.get("generate", "").lower() in ["yes", "true", "1"] or block.params.get("gen", "").lower() in ["yes", "true", "1"]
    nids = []

    if generate and block.data_lines:
        for line in block.data_lines:
            parts = [int(p) for p in line.replace(",", " ").split() if p.strip()]
            if len(parts) >= 2:
                start = parts[0]
                end = parts[1]
                step = parts[2] if len(parts) >= 3 else 1
                nids.extend(list(range(start, end + 1, step)))
    else:
        part_ctx = reader.get_active_part()
        for line in block.data_lines:
            tokens = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
            for tok in tokens:
                clean_tok = tok.replace("-", "")
                if clean_tok.isdigit():
                    nids.append(int(tok))
                else:
                    # Resolve referenced set name
                    resolved = False
                    if part_ctx is not None:
                        if tok in part_ctx.node_sets:
                            nids.extend(part_ctx.node_sets[tok].node_ids)
                            resolved = True
                        elif tok in part_ctx.sets:
                            nids.extend(part_ctx.sets[tok].node_ids)
                            resolved = True
                    if not resolved and hasattr(reader.model, "root_assembly"):
                        if tok in reader.model.root_assembly.node_sets:
                            nids.extend(reader.model.root_assembly.node_sets[tok].node_ids)
                            resolved = True
                        elif tok in reader.model.root_assembly.sets:
                            nids.extend(reader.model.root_assembly.sets[tok].node_ids)
                            resolved = True

    inst_name = block.params.get("instance", None)
    if reader.context in [KeywordContext.ASSEMBLY]:
        if inst_name:
            if inst_name in reader.model.root_assembly.instances:
                inst = reader.model.root_assembly.instances[inst_name]
                inst.part.create_set(name, nodes=nids)
            reader.model.root_assembly.Set(f"{inst_name}.{name}", nodes=nids)
        else:
            reader.model.root_assembly.Set(name, nodes=nids)
    elif reader.context == KeywordContext.STEP and reader.current_step is not None:
        cur = reader.current_step.step_nsets.setdefault(name, [])
        for nid in nids:
            if nid not in cur:
                cur.append(nid)
    elif reader.context == KeywordContext.INSTANCE and reader.current_instance is not None:
        inst = reader.current_instance
        inst.part.create_set(f"{inst.name}__{name}", nodes=nids)
    else:
        part = reader.get_active_part()
        part.create_set(name, nodes=nids)


@GLOBAL_KEYWORD_REGISTRY.register("ELSET", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE, KeywordContext.STEP})
def _handle_elset(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("elset", "")
    if not name:
        return

    generate = "generate" in block.params or "gen" in block.params or block.params.get("generate", "").lower() in ["yes", "true", "1"] or block.params.get("gen", "").lower() in ["yes", "true", "1"]
    eids = []

    if generate and block.data_lines:
        for line in block.data_lines:
            parts = [int(p) for p in line.replace(",", " ").split() if p.strip()]
            if len(parts) >= 2:
                start = parts[0]
                end = parts[1]
                step = parts[2] if len(parts) >= 3 else 1
                eids.extend(list(range(start, end + 1, step)))
    else:
        part_ctx = reader.get_active_part()
        for line in block.data_lines:
            tokens = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
            for tok in tokens:
                clean_tok = tok.replace("-", "")
                if clean_tok.isdigit():
                    eids.append(int(tok))
                else:
                    # Resolve referenced element set name
                    resolved = False
                    if part_ctx is not None:
                        if tok in part_ctx.element_sets:
                            eids.extend(part_ctx.element_sets[tok].element_ids)
                            resolved = True
                        elif tok in part_ctx.sets:
                            eids.extend(part_ctx.sets[tok].element_ids)
                            resolved = True
                    if not resolved and hasattr(reader.model, "root_assembly"):
                        if tok in reader.model.root_assembly.element_sets:
                            eids.extend(reader.model.root_assembly.element_sets[tok].element_ids)
                            resolved = True
                        elif tok in reader.model.root_assembly.sets:
                            eids.extend(reader.model.root_assembly.sets[tok].element_ids)
                            resolved = True

    inst_name = block.params.get("instance", None)
    if reader.context in [KeywordContext.ASSEMBLY]:
        if inst_name:
            if inst_name in reader.model.root_assembly.instances:
                inst = reader.model.root_assembly.instances[inst_name]
                inst.part.create_set(name, elements=eids)
            reader.model.root_assembly.Set(f"{inst_name}.{name}", elements=eids)
        else:
            reader.model.root_assembly.Set(name, elements=eids)
    elif reader.context == KeywordContext.STEP and reader.current_step is not None:
        cur = reader.current_step.step_elsets.setdefault(name, [])
        for eid in eids:
            if eid not in cur:
                cur.append(eid)
    elif reader.context == KeywordContext.INSTANCE and reader.current_instance is not None:
        inst = reader.current_instance
        inst.part.create_set(f"{inst.name}__{name}", elements=eids)
    else:
        part = reader.get_active_part()
        part.create_set(name, elements=eids)


@GLOBAL_KEYWORD_REGISTRY.register("SURFACE", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE})
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

    if reader.context == KeywordContext.INSTANCE and reader.current_instance is not None:
        inst = reader.current_instance
        name = f"{inst.name}__{name}"
        part = inst.part
    else:
        part = reader.get_active_part()

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


@GLOBAL_KEYWORD_REGISTRY.register("NGEN", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE})
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


@GLOBAL_KEYWORD_REGISTRY.register("NFILL", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE})
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


@GLOBAL_KEYWORD_REGISTRY.register("NCOPY", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE})
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


@GLOBAL_KEYWORD_REGISTRY.register("ELGEN", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE})
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




