from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    ParamSpec,
    RowSpec,
)

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader
from dispsolver.io.keyword_base import _parse_data_rows

# KEYWORD HANDLERS IMPLEMENTATION (Standard Abaqus Keywords)
# ─────────────────────────────────────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register(
    "HEADING",
    valid_contexts={KeywordContext.GLOBAL},
    support="IGNORED",
    params={},
    rows=str,
    description="Model title or comment heading",
)
def _handle_heading(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if block.data_lines:
        reader.model.name = block.data_lines[0].strip()


# ─── PART & ASSEMBLY SCOPE ──────────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register(
    "PART",
    valid_contexts={KeywordContext.GLOBAL},
    support="FULL",
    params={"NAME": ParamSpec("NAME", default="PART-1")},
    description="Begin a part definition",
)
def _handle_part(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    part_name = block.params.get("name", f"PART-{len(reader.model.parts) + 1}")
    reader.current_part = reader.model.Part(part_name, dim=reader.dim)
    reader.context = KeywordContext.PART


@GLOBAL_KEYWORD_REGISTRY.register(
    "END PART",
    valid_contexts={KeywordContext.PART},
    aliases=["ENDPART"],
    support="FULL",
    params={},
    description="End part definition",
)
def _handle_end_part(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.current_part = None
    reader.context = KeywordContext.GLOBAL


@GLOBAL_KEYWORD_REGISTRY.register(
    "ASSEMBLY",
    valid_contexts={KeywordContext.GLOBAL},
    support="FULL",
    params={"NAME": ParamSpec("NAME", default="ASSEMBLY")},
    description="Begin the root assembly definition",
)
def _handle_assembly(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.has_assembly = True
    reader.context = KeywordContext.ASSEMBLY
    if "name" in block.params:
        reader.model.root_assembly.name = block.params["name"]


@GLOBAL_KEYWORD_REGISTRY.register(
    "END ASSEMBLY",
    valid_contexts={KeywordContext.ASSEMBLY},
    aliases=["ENDASSEMBLY"],
    support="FULL",
    params={},
    description="End assembly definition",
)
def _handle_end_assembly(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.context = KeywordContext.GLOBAL


@GLOBAL_KEYWORD_REGISTRY.register(
    "INSTANCE",
    valid_contexts={KeywordContext.ASSEMBLY},
    support="FULL",
    params={
        "NAME": ParamSpec("NAME", required=True),
        "PART": ParamSpec("PART"),
    },
    rows=RowSpec(column_types=[float], min_columns=2, max_columns=7),
    description="Create an instance of a part within the assembly",
)
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
            if float(np.linalg.norm(axis)) < 1e-12:
                msg = f"*INSTANCE {inst_name}: degenerate rotation axis - rotation ignored"
                warnings.warn(msg)
                reader.warnings.append(msg)
            else:
                if reader.dim == 2 and (abs(float(axis[0])) > 1e-12 or abs(float(axis[1])) > 1e-12):
                    msg = (f"*INSTANCE {inst_name}: in-plane rotation axis in 2D - "
                           f"in-plane rotation applied about the axis point")
                    warnings.warn(msg)
                    reader.warnings.append(msg)
                inst.rotate(axis=axis, angle_deg=angle_deg, center=p_a)


@GLOBAL_KEYWORD_REGISTRY.register(
    "END INSTANCE",
    valid_contexts={KeywordContext.INSTANCE},
    aliases=["ENDINSTANCE"],
    support="FULL",
    params={},
    description="End instance definition",
)
def _handle_end_instance(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.current_instance = None
    reader.context = KeywordContext.ASSEMBLY




