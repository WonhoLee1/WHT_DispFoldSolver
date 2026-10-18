"""Spring and dashpot element property keywords.

Implements Abaqus keywords:
- *SPRING: Spring stiffness definition for SPRING1/SPRING2/SPRINGA element sets
- *DASHPOT: Dashpot damping coefficient for DASHPOT1/DASHPOT2 element sets
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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


@GLOBAL_KEYWORD_REGISTRY.register(
    "SPRING",
    valid_contexts={KeywordContext.PART, KeywordContext.ASSEMBLY, KeywordContext.GLOBAL},
    support="PARTIAL",
    params={
        "ELSET": ParamSpec("ELSET", required=True),
        "NONLINEAR": ParamSpec("NONLINEAR", type=bool, default=False),
        "ORIENTATION": ParamSpec("ORIENTATION", default=None, supported=False),
        "DEPENDENCIES": ParamSpec("DEPENDENCIES", type=int, default=0, supported=False),
        "RTOL": ParamSpec("RTOL", type=float, default=None, supported=False),
        "COMPLEX STIFFNESS": ParamSpec(
            "COMPLEX STIFFNESS",
            type=bool,
            default=False,
            supported=False,
            aliases=["COMPLEX_STIFFNESS"],
        ),
    },
    rows=RowSpec(min_columns=0, max_columns=2),
    description="Define spring stiffness behavior for spring elements",
)
def _handle_spring(reader: AbaqusInputReader, block: AbaqusKeywordBlock) -> None:
    elset = block.params.get("elset")
    if not elset:
        reader.report("error", "missing-elset", "*SPRING: missing required ELSET parameter", block)
        return

    nonlinear = bool(block.params.get("nonlinear", False))
    rows = _parse_data_rows(block.data_lines, dtype=float, default=None)

    dof = None
    stiffness = 0.0
    curve = []

    if rows:
        # First row may specify the degree of freedom
        first_row = rows[0]
        start_idx = 0
        if len(first_row) == 1 and first_row[0] is not None:
            # Single value on line 1 is typically the DOF (e.g., 1, 2, 3)
            # or stiffness if DOF omitted for SPRINGA
            val = first_row[0]
            if float(val).is_integer() and 1 <= int(val) <= 6 and len(rows) > 1:
                dof = int(val)
                start_idx = 1
            else:
                stiffness = float(val)
                start_idx = 1
        elif len(first_row) >= 2:
            if first_row[0] is not None and float(first_row[0]).is_integer() and 1 <= int(first_row[0]) <= 6:
                dof = int(first_row[0])
                stiffness = float(first_row[1]) if first_row[1] is not None else 0.0
                start_idx = 1
            else:
                stiffness = float(first_row[0]) if first_row[0] is not None else 0.0

        if not nonlinear and start_idx < len(rows):
            # Second line provides stiffness if line 1 was DOF
            stiff_row = rows[start_idx]
            if len(stiff_row) > 0 and stiff_row[0] is not None:
                stiffness = float(stiff_row[0])
        elif nonlinear:
            for r in rows[start_idx:]:
                if len(r) >= 2 and r[0] is not None and r[1] is not None:
                    curve.append((float(r[0]), float(r[1])))

    spring_entry = {
        "elset": elset,
        "dof": dof,
        "stiffness": stiffness,
        "nonlinear": nonlinear,
        "curve": curve if nonlinear else None,
    }

    # Store in active part if inside *PART, and in model
    part = reader.get_active_part()
    if part is not None:
        if not hasattr(part, "springs"):
            part.springs = {}
        part.springs[elset] = spring_entry

    if not hasattr(reader.model, "springs"):
        reader.model.springs = {}
    reader.model.springs[elset] = spring_entry


@GLOBAL_KEYWORD_REGISTRY.register(
    "DASHPOT",
    valid_contexts={KeywordContext.PART, KeywordContext.ASSEMBLY, KeywordContext.GLOBAL},
    support="PARTIAL",
    params={
        "ELSET": ParamSpec("ELSET", required=True),
        "NONLINEAR": ParamSpec("NONLINEAR", type=bool, default=False),
        "ORIENTATION": ParamSpec("ORIENTATION", default=None, supported=False),
        "DEPENDENCIES": ParamSpec("DEPENDENCIES", type=int, default=0, supported=False),
        "RTOL": ParamSpec("RTOL", type=float, default=None, supported=False),
    },
    rows=RowSpec(min_columns=0, max_columns=2),
    description="Define dashpot behavior for dashpot elements",
)
def _handle_dashpot(reader: AbaqusInputReader, block: AbaqusKeywordBlock) -> None:
    elset = block.params.get("elset")
    if not elset:
        reader.report("error", "missing-elset", "*DASHPOT: missing required ELSET parameter", block)
        return

    nonlinear = bool(block.params.get("nonlinear", False))
    rows = _parse_data_rows(block.data_lines, dtype=float, default=None)

    dof = None
    coefficient = 0.0

    if rows:
        first_row = rows[0]
        start_idx = 0
        if len(first_row) == 1 and first_row[0] is not None:
            val = first_row[0]
            if float(val).is_integer() and 1 <= int(val) <= 6 and len(rows) > 1:
                dof = int(val)
                start_idx = 1
            else:
                coefficient = float(val)
                start_idx = 1
        elif len(first_row) >= 2:
            if first_row[0] is not None and float(first_row[0]).is_integer() and 1 <= int(first_row[0]) <= 6:
                dof = int(first_row[0])
                coefficient = float(first_row[1]) if first_row[1] is not None else 0.0
                start_idx = 1
            else:
                coefficient = float(first_row[0]) if first_row[0] is not None else 0.0

        if not nonlinear and start_idx < len(rows):
            c_row = rows[start_idx]
            if len(c_row) > 0 and c_row[0] is not None:
                coefficient = float(c_row[0])

    dashpot_entry = {
        "elset": elset,
        "dof": dof,
        "coefficient": coefficient,
        "nonlinear": nonlinear,
    }

    part = reader.get_active_part()
    if part is not None:
        if not hasattr(part, "dashpots"):
            part.dashpots = {}
        part.dashpots[elset] = dashpot_entry

    if not hasattr(reader.model, "dashpots"):
        reader.model.dashpots = {}
    reader.model.dashpots[elset] = dashpot_entry
