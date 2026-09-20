from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_base import _parse_data_rows
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── MESH DEFINITION (NODES, ELEMENTS, SETS, SURFACES) ──────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("NODE", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE})
def _handle_node(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    part = reader.get_active_part()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    nset_name = block.params.get("nset", None)
    added_nids = []

    for r in rows:
        if not r:
            continue
        nid = int(r[0])
        coords = [float(x) for x in r[1:]]
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

    generate = "generate" in block.params or block.params.get("generate", "").lower() in ["yes", "true", "1"]
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
        for line in block.data_lines:
            parts = [int(p.replace(",", "")) for p in line.replace(",", " ").split() if p.strip().replace("-", "").isdigit()]
            nids.extend(parts)

    if reader.context in [KeywordContext.ASSEMBLY]:
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

    generate = "generate" in block.params or block.params.get("generate", "").lower() in ["yes", "true", "1"]
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
        for line in block.data_lines:
            parts = [int(p.replace(",", "")) for p in line.replace(",", " ").split() if p.strip().replace("-", "").isdigit()]
            eids.extend(parts)

    if reader.context in [KeywordContext.ASSEMBLY]:
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



