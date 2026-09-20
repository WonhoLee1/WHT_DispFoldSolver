from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext
from dispsolver.io.keyword_base import _parse_data_rows

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

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



