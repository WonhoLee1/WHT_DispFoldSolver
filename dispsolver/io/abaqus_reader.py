"""
abaqus_reader.py
================
Robust, production-grade Abaqus .inp reader and executor engine.
Builds dispsolver.model.Model objects directly from parsed keyword blocks,
supporting both Part/Assembly hierarchies and Non-Assembly (flat) decks.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import numpy as np

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock, tokenize, tokenize_string
from dispsolver.io.keyword_registry import (
    KeywordContext,
    KeywordSpec,
    KeywordRegistry,
    GLOBAL_KEYWORD_REGISTRY,
)
from dispsolver.model.model import Model
from dispsolver.model.part import Part
from dispsolver.model.instance import Instance
from dispsolver.model.material import Material
from dispsolver.model.section import SolidSection, ShellSection
from dispsolver.model.step import Step, PredefinedField
from dispsolver.model.set import GeneralSet, SetScope


class _AbortParse(Exception):
    """Hard parse abort for keywords with no internal landing.

    Unlike ordinary handler errors (warn + continue), this is re-raised
    by parse_blocks: continuing past e.g. *INCLUDE would silently solve
    a different model than the deck describes.
    """


def _parse_data_rows(data_lines: List[str], dtype=float) -> List[List[Any]]:
    """Parse comma/whitespace-separated data lines into a table of values."""
    rows = []
    for line in data_lines:
        line_str = line.strip()
        if not line_str:
            continue
        parts = [p.strip() for p in line_str.split(",") if p.strip()]
        if not parts:
            # Fallback to whitespace split if no commas present
            parts = [p.strip() for p in line_str.split() if p.strip()]
        if not parts:
            continue
        row = []
        for p in parts:
            try:
                row.append(dtype(p))
            except ValueError:
                row.append(p)  # string
        rows.append(row)
    return rows


class AbaqusInputReader:
    """Production reader converting Abaqus keyword blocks into a complete dispsolver.model.Model."""

    def __init__(self, dim: int = 3, model_name: str = "AbaqusModel"):
        self.dim = int(dim)
        self.model = Model(name=model_name, dim=self.dim)
        self.registry = GLOBAL_KEYWORD_REGISTRY

        # Parsing states
        self.context: KeywordContext = KeywordContext.GLOBAL
        self.current_part: Optional[Part] = None
        self.current_instance: Optional[Instance] = None
        self.current_step: Optional[Step] = None
        self.current_material: Optional[Material] = None
        self.current_interaction: Optional[str] = None
        self.current_coupling: Optional[Dict[str, Any]] = None

        # Flat / Non-assembly tracking
        self.has_assembly: bool = False
        self._default_part: Optional[Part] = None
        self._default_instance: Optional[Instance] = None

        # Statistics & warnings
        self.unsupported_keywords: List[str] = []
        self.warnings: List[str] = []

    def get_active_part(self) -> Part:
        """Return currently active part (or create/return default part for non-assembly models)."""
        if self.current_part is not None:
            return self.current_part
        if self._default_part is None:
            self._default_part = self.model.Part("PART_DEFAULT", dim=self.dim)
        return self._default_part

    def ensure_default_instance(self):
        """Ensure default instance exists if model has flat/non-assembly elements."""
        if not self.has_assembly and self._default_instance is None:
            part = self.get_active_part()
            self._default_instance = self.model.root_assembly.Instance("INST_DEFAULT", part=part)

    def parse_blocks(self, blocks: List[AbaqusKeywordBlock]) -> Model:
        """Parse all keyword blocks into the internal Model."""
        for block in blocks:
            kw = block.keyword
            spec = self.registry.get_handler(kw)
            if spec is not None:
                try:
                    spec.handler(self, block)
                except _AbortParse:
                    raise
                except Exception as e:
                    msg = f"Error processing keyword *{kw} at line {block.line_number}: {e}"
                    warnings.warn(msg)
                    self.warnings.append(msg)
            else:
                self.unsupported_keywords.append(kw)

        # If non-assembly deck was loaded, bind the default instance
        if not self.has_assembly and self._default_part is not None:
            self.ensure_default_instance()

        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD HANDLERS IMPLEMENTATION (Standard Abaqus Keywords)
# ─────────────────────────────────────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("HEADING", valid_contexts={KeywordContext.GLOBAL})
def _handle_heading(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if block.data_lines:
        reader.model.name = block.data_lines[0].strip()


# ─── PART & ASSEMBLY SCOPE ──────────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("PART", valid_contexts={KeywordContext.GLOBAL})
def _handle_part(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    part_name = block.params.get("name", f"PART-{len(reader.model.parts) + 1}")
    reader.current_part = reader.model.Part(part_name, dim=reader.dim)
    reader.context = KeywordContext.PART


@GLOBAL_KEYWORD_REGISTRY.register("END PART", valid_contexts={KeywordContext.PART}, aliases=["ENDPART"])
def _handle_end_part(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.current_part = None
    reader.context = KeywordContext.GLOBAL


@GLOBAL_KEYWORD_REGISTRY.register("ASSEMBLY", valid_contexts={KeywordContext.GLOBAL})
def _handle_assembly(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.has_assembly = True
    reader.context = KeywordContext.ASSEMBLY
    if "name" in block.params:
        reader.model.root_assembly.name = block.params["name"]


@GLOBAL_KEYWORD_REGISTRY.register("END ASSEMBLY", valid_contexts={KeywordContext.ASSEMBLY}, aliases=["ENDASSEMBLY"])
def _handle_end_assembly(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.context = KeywordContext.GLOBAL


@GLOBAL_KEYWORD_REGISTRY.register("INSTANCE", valid_contexts={KeywordContext.ASSEMBLY})
def _handle_instance(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    inst_name = block.params.get("name", "")
    part_name = block.params.get("part", "")
    if not part_name and inst_name in reader.model.parts:
        part_name = inst_name

    if part_name not in reader.model.parts:
        # Auto-create empty part if not found
        part = reader.model.Part(part_name, dim=reader.dim)
    else:
        part = reader.model.parts[part_name]

    inst = reader.model.root_assembly.Instance(inst_name, part=part)
    reader.current_instance = inst
    reader.context = KeywordContext.INSTANCE

    # Check for translation / rotation data lines
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if len(rows) >= 1:
        # First line: translation vector [dx, dy, dz]
        trans = rows[0]
        if len(trans) >= 2:
            t_vec = [float(trans[0]), float(trans[1]), float(trans[2]) if len(trans) >= 3 else 0.0]
            inst.translate(t_vec)
    if len(rows) >= 2:
        # Second line: rotation axis [x_a, y_a, z_a, x_b, y_b, z_b, angle_deg]
        rot_row = rows[1]
        if len(rot_row) >= 7:
            p_a = np.array([float(rot_row[0]), float(rot_row[1]), float(rot_row[2])])
            p_b = np.array([float(rot_row[3]), float(rot_row[4]), float(rot_row[5])])
            axis = p_b - p_a
            angle_deg = float(rot_row[6])
            inst.rotate(axis=axis, angle_deg=angle_deg, center=p_a)


@GLOBAL_KEYWORD_REGISTRY.register("END INSTANCE", valid_contexts={KeywordContext.INSTANCE}, aliases=["ENDINSTANCE"])
def _handle_end_instance(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.current_instance = None
    reader.context = KeywordContext.ASSEMBLY


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


@GLOBAL_KEYWORD_REGISTRY.register("NSET", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE})
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
    else:
        part = reader.get_active_part()
        part.create_set(name, nodes=nids)


@GLOBAL_KEYWORD_REGISTRY.register("ELSET", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE})
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
    else:
        part = reader.get_active_part()
        part.create_set(name, elements=eids)


@GLOBAL_KEYWORD_REGISTRY.register("SURFACE", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.INSTANCE})
def _handle_surface(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", "")
    stype = block.params.get("type", "ELEMENT").upper()
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


# ─── MATERIAL & SECTION DEFINITIONS ─────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("MATERIAL", valid_contexts={KeywordContext.GLOBAL})
def _handle_material(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"MAT-{len(reader.model.materials) + 1}")
    mat = reader.model.Material(name)
    reader.current_material = mat
    reader.context = KeywordContext.MATERIAL


@GLOBAL_KEYWORD_REGISTRY.register("ELASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_elastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        mat = reader.model.Material("DEFAULT_ELASTIC")
        reader.current_material = mat

    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        # Table of (E, nu[, temp, ...])
        table = [tuple(r) for r in rows]
        mat.Elastic(table=tuple(table))


@GLOBAL_KEYWORD_REGISTRY.register("EXPANSION", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_expansion(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    exp_type = block.params.get("type", "ISO").upper()
    zero_val = float(block.params.get("zero", 0.0))
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        table = [tuple(r) for r in rows]
        mat.Expansion(type=exp_type, table=tuple(table), zero=zero_val)


@GLOBAL_KEYWORD_REGISTRY.register("PLASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_plastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    hardening = block.params.get("hardening", "ISOTROPIC").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        table = [tuple(r) for r in rows]
        mat.Plastic(table=tuple(table), hardening=hardening)


@GLOBAL_KEYWORD_REGISTRY.register("CREEP", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_creep(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    law = block.params.get("law", "STRAIN").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        table = [tuple(r) for r in rows]
        mat.Creep(law=law, table=tuple(table))


@GLOBAL_KEYWORD_REGISTRY.register("HYPERELASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_hyperelastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    # Model type can be bare keyword or TYPE=...
    model_type = "NEO HOOKE"
    for known in ["ARRUDA-BOYCE", "ARRUDA BOYCE", "YEOH", "MOONEY-RIVLIN", "OGDEN", "POLYNOMIAL", "NEO HOOKE"]:
        if known.lower() in block.params or known.lower().replace("-", " ") in block.params:
            model_type = known.upper()
            break
    if "type" in block.params:
        model_type = block.params["type"].upper()

    rows = _parse_data_rows(block.data_lines, dtype=float)
    table = [tuple(r) for r in rows] if rows else ()
    mat.Hyperelastic(type=model_type, table=table)


@GLOBAL_KEYWORD_REGISTRY.register("VISCOELASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_viscoelastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    time_domain = block.params.get("time", "PRONY").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    table = [tuple(r) for r in rows] if rows else ()
    mat.Viscoelastic(time=time_domain, table=table)


@GLOBAL_KEYWORD_REGISTRY.register("SOLID SECTION", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE}, aliases=["SOLIDSECTION"])
def _handle_solid_section(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    elset = block.params.get("elset", "ALL")
    mat_name = block.params.get("material", "DEFAULT")
    sec_name = f"SEC_SOLID_{elset}"

    thickness = 1.0
    if block.data_lines:
        rows = _parse_data_rows(block.data_lines, dtype=float)
        if rows and len(rows[0]) >= 1:
            thickness = float(rows[0][0])

    sec = reader.model.HomogeneousSolidSection(name=sec_name, material=mat_name, thickness=thickness)
    
    # Locate part containing this elset
    part = reader.get_active_part()
    if elset not in part.element_sets and elset not in part.sets:
        # Search all parts in model
        for p in reader.model.parts.values():
            if elset in p.element_sets or elset in p.sets:
                part = p
                break
    part.SectionAssignment(region=elset, sectionName=sec_name)


@GLOBAL_KEYWORD_REGISTRY.register("SHELL SECTION", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE}, aliases=["SHELLSECTION"])
def _handle_shell_section(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    elset = block.params.get("elset", "ALL")
    mat_name = block.params.get("material", "DEFAULT")
    sec_name = f"SEC_SHELL_{elset}"

    thickness = 1.0
    if block.data_lines:
        rows = _parse_data_rows(block.data_lines, dtype=float)
        if rows and len(rows[0]) >= 1:
            thickness = float(rows[0][0])

    sec = reader.model.ShellSection(name=sec_name, material=mat_name, thickness=thickness)
    
    # Locate part containing this elset
    part = reader.get_active_part()
    if elset not in part.element_sets and elset not in part.sets:
        for p in reader.model.parts.values():
            if elset in p.element_sets or elset in p.sets:
                part = p
                break
    part.SectionAssignment(region=elset, sectionName=sec_name)


# ─── CONSTRAINTS & INTERACTIONS ──────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("TIE", valid_contexts={KeywordContext.ASSEMBLY, KeywordContext.GLOBAL})
def _handle_tie(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"TIE-{len(reader.model.root_assembly.ties) + 1}")
    pos_tol = float(block.params.get("position tolerance", block.params.get("position_tolerance", 0.0)))
    adjust = block.params.get("adjust", "yes").lower() == "yes"

    master = None
    slave = None
    for line in block.data_lines:
        tokens = [t.strip() for t in line.split(",") if t.strip()]
        if len(tokens) >= 2:
            slave = tokens[0]
            master = tokens[1]
            break

    if master and slave:
        reader.model.Tie(name=name, master=master, slave=slave, positionTolerance=pos_tol, adjust=adjust)


@GLOBAL_KEYWORD_REGISTRY.register("RIGID BODY", valid_contexts={KeywordContext.PART, KeywordContext.ASSEMBLY, KeywordContext.GLOBAL}, aliases=["RIGIDBODY"])
def _handle_rigid_body(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"RB_{len(reader.model.constraints) + 1}")
    rp = block.params.get("ref node", block.params.get("ref_node", None))
    elset = block.params.get("elset", None)
    tie_nset = block.params.get("tie nset", block.params.get("tie_nset", None))
    pin_nset = block.params.get("pin nset", block.params.get("pin_nset", None))

    reader.model.RigidBody(name=name, refPoint=rp, elset=elset, tieNset=tie_nset, pinNset=pin_nset)


@GLOBAL_KEYWORD_REGISTRY.register("SURFACE INTERACTION", valid_contexts={KeywordContext.GLOBAL}, aliases=["SURFACEINTERACTION"])
def _handle_surface_interaction(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"INTERACTION-{len(reader.model.contact_properties) + 1}")
    reader.model.ContactProperty(name=name)


@GLOBAL_KEYWORD_REGISTRY.register("CONTACT PAIR", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY}, aliases=["CONTACTPAIR"])
def _handle_contact_pair(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    interaction = block.params.get("interaction", "DEFAULT")
    if block.data_lines:
        for line in block.data_lines:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                slave, master = parts[0], parts[1]
                name = f"CP_{slave}_{master}"
                reader.model.ContactPair(name=name, main=master, secondary=slave, interactionProperty=interaction)


# ─── STEPS, BCS & LOADS ──────────────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("STEP", valid_contexts={KeywordContext.GLOBAL})
def _handle_step(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    step_name = block.params.get("name", f"Step-{len(reader.model.steps)}")
    nlgeom = block.params.get("nlgeom", "yes").lower() == "yes"
    time_period = float(block.params.get("time period", block.params.get("time_period", 1.0)))

    # Previous step
    prev_step = reader.current_step.name if reader.current_step is not None else None
    reader.current_step = reader.model.Step(
        name=step_name,
        previous=prev_step,
        procedure="STATIC",
        time_period=time_period,
        nlgeom=nlgeom
    )


@GLOBAL_KEYWORD_REGISTRY.register("STATIC", valid_contexts={KeywordContext.STEP})
def _handle_static(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is not None:
        reader.current_step.procedure = "STATIC"
        if block.data_lines:
            rows = _parse_data_rows(block.data_lines, dtype=float)
            if rows and len(rows[0]) >= 1:
                r = rows[0]
                reader.current_step.dt_init = float(r[0])
                if len(r) >= 2:
                    reader.current_step.time_period = float(r[1])
                if len(r) >= 3:
                    reader.current_step.dt_min = float(r[2])
                if len(r) >= 4:
                    reader.current_step.dt_max = float(r[3])


@GLOBAL_KEYWORD_REGISTRY.register("DYNAMIC", valid_contexts={KeywordContext.STEP})
def _handle_dynamic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is not None:
        reader.current_step.procedure = "DYNAMIC"
        if block.data_lines:
            rows = _parse_data_rows(block.data_lines, dtype=float)
            if rows and len(rows[0]) >= 1:
                r = rows[0]
                reader.current_step.dt_init = float(r[0])
                if len(r) >= 2:
                    reader.current_step.time_period = float(r[1])


@GLOBAL_KEYWORD_REGISTRY.register("END STEP", valid_contexts={KeywordContext.STEP}, aliases=["ENDSTEP"])
def _handle_end_step(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    # Step remains active until next STEP or EOF
    pass


@GLOBAL_KEYWORD_REGISTRY.register("BOUNDARY", valid_contexts={KeywordContext.STEP, KeywordContext.GLOBAL})
def _handle_boundary(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    rows = _parse_data_rows(block.data_lines, dtype=str)
    target_step = reader.current_step if reader.current_step is not None else reader.model.initial_step

    for r in rows:
        if len(r) < 2:
            continue
        region = str(r[0])
        first_dof = int(r[1])
        last_dof = int(r[2]) if len(r) >= 3 and r[2] != "" else first_dof
        val = float(r[3]) if len(r) >= 4 else 0.0

        kwargs = {}
        for dof in range(first_dof, last_dof + 1):
            if dof == 1:
                kwargs["u1"] = val
            elif dof == 2:
                kwargs["u2"] = val
            elif dof == 3:
                kwargs["u3"] = val
            elif dof == 4:
                kwargs["ur1"] = val
            elif dof == 5:
                kwargs["ur2"] = val
            elif dof == 6:
                kwargs["ur3"] = val

        bc_name = f"BC_{region}_DOF{first_dof}_{last_dof}"
        reader.model.DisplacementBC(name=bc_name, createStepName=target_step.name, region=region, **kwargs)


@GLOBAL_KEYWORD_REGISTRY.register("CLOAD", valid_contexts={KeywordContext.STEP})
def _handle_cload(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is None:
        return
    rows = _parse_data_rows(block.data_lines, dtype=str)
    for r in rows:
        if len(r) >= 3:
            region = str(r[0])
            dof = int(r[1])
            mag = float(r[2])
            load_name = f"CLOAD_{region}_DOF{dof}"
            cf_kwargs = {}
            if dof == 1:
                cf_kwargs["cf1"] = mag
            elif dof == 2:
                cf_kwargs["cf2"] = mag
            elif dof == 3:
                cf_kwargs["cf3"] = mag
            reader.model.ConcentratedForce(name=load_name, createStepName=reader.current_step.name, region=region, **cf_kwargs)


@GLOBAL_KEYWORD_REGISTRY.register("TEMPERATURE", valid_contexts={KeywordContext.STEP, KeywordContext.GLOBAL})
def _handle_temperature(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    rows = _parse_data_rows(block.data_lines, dtype=str)
    target_step = reader.current_step if reader.current_step is not None else reader.model.initial_step
    for r in rows:
        if len(r) >= 2:
            region = str(r[0])
            temp_val = float(r[1])
            pf = PredefinedField(name=f"TEMP_{region}", field_type="TEMPERATURE", region=region, values=temp_val)
            target_step.predefined_fields.append(pf)


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def read_inp(source: Union[str, os.PathLike], dim: int = 3, model_name: str = "AbaqusModel") -> Model:
    """Read an Abaqus .inp file or raw string and return a populated Model instance.
    
    Parameters
    ----------
    source : str or PathLike
        Path to .inp file on disk, or raw INP text string containing '*HEADING' or '*NODE'.
    dim : int
        Spatial dimensionality (2 or 3).
    model_name : str
        Name for the constructed Model.
        
    Returns
    -------
    Model
        Populated Model container ready for solver creation and inspection.
    """
    source_str = str(source)
    if os.path.exists(source_str):
        blocks = tokenize(source_str)
    else:
        blocks = tokenize_string(source_str)

    reader = AbaqusInputReader(dim=dim, model_name=model_name)
    model = reader.parse_blocks(blocks)
    return model


def execute_inp(
    source: Union[str, os.PathLike],
    dim: int = 3,
    step_name: Optional[str] = None,
    dt: Optional[float] = None,
    verbose: bool = True
) -> Any:
    """Read an Abaqus .inp file, compile to solver system, and execute analysis steps.
    
    Parameters
    ----------
    source : str or PathLike
        Path to .inp file or raw string.
    dim : int
        Spatial dimensionality (2 or 3).
    step_name : Optional[str]
        Specific step to run (defaults to first non-initial step).
    dt : Optional[float]
        Time increment override.
    verbose : bool
        If True, prints execution progress.
        
    Returns
    -------
    dict
        Execution summary dictionary with solver state, displacements, and iteration count.
    """
    model = read_inp(source, dim=dim)
    
    # Determine step to run
    if step_name is None:
        steps_list = [s for s in model.steps.keys() if s != "Initial"]
        target_step = steps_list[0] if steps_list else None
    else:
        target_step = step_name

    if dim == 3:
        solver, sys = model.create_solver(step_name=target_step)
    else:
        solver, sys = model.create_solver2d(step_name=target_step)

    step_obj = model.steps.get(target_step) if target_step else None
    step_dt = dt if dt is not None else (step_obj.dt_init if step_obj else 1.0)

    if verbose:
        print(f"Executing Abaqus Model '{model.name}' [{dim}D] on Step '{target_step}' (dt={step_dt})...")

    conv, iters = solver.solve_step(dt=step_dt)

    if verbose:
        print(f"Execution completed: Converged={conv}, Newton Iterations={iters}")

    return {
        "model": model,
        "solver": solver,
        "system": sys,
        "converged": conv,
        "iterations": iters,
        "u": solver.u.copy() if hasattr(solver, "u") else None,
    }
