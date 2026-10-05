from __future__ import annotations

import warnings
from dataclasses import dataclass
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
from dispsolver.io.schema import InputError, Keyword, Note, Record, RowSchema, Schema, applies, keyword

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


# ─── PART & ASSEMBLY SCOPE (schema) ─────────────────────────────────────────
#
# Scope keywords are applied even when a parameter is a gap or an error
# (`refuse_on_gap = False`): refusing *PART or *INSTANCE pushes every following
# keyword out of its context, and the real problem drowns in bad-context errors.


@dataclass(frozen=True)
class ScopeRecord(Record):
    action: str                           # PART END_PART ASSEMBLY END_ASSEMBLY END_INSTANCE
    name: Optional[str] = None


@keyword("PART", support="FULL")
class PartKw(Keyword):
    """Begin a part definition."""

    refuse_on_gap = False

    class Params(Schema):
        name: Optional[str] = None

    def to_records(self, p, rows, ctx):
        if p.name is None:
            yield Note("error", "missing-param", f"*PART at {ctx.loc}: missing required parameter 'NAME'",
                       loc=ctx.loc)
        yield ScopeRecord("PART", p.name, loc=ctx.loc)


@keyword("END PART", support="FULL", contexts={KeywordContext.PART})
class EndPart(Keyword):
    """End of a part definition."""

    refuse_on_gap = False

    def to_records(self, p, rows, ctx):
        yield ScopeRecord("END_PART", loc=ctx.loc)


@keyword("ASSEMBLY", support="FULL")
class AssemblyKw(Keyword):
    """Begin the root assembly definition."""

    refuse_on_gap = False

    class Params(Schema):
        name: Optional[str] = None

    def to_records(self, p, rows, ctx):
        if p.name is None:
            yield Note("error", "missing-param", f"*ASSEMBLY at {ctx.loc}: missing required parameter 'NAME'",
                       loc=ctx.loc)
        yield ScopeRecord("ASSEMBLY", p.name, loc=ctx.loc)


@keyword("END ASSEMBLY", support="FULL", contexts={KeywordContext.ASSEMBLY})
class EndAssembly(Keyword):
    """End of the assembly definition."""

    refuse_on_gap = False

    def to_records(self, p, rows, ctx):
        yield ScopeRecord("END_ASSEMBLY", loc=ctx.loc)


@keyword("END INSTANCE", support="FULL", contexts={KeywordContext.INSTANCE})
class EndInstance(Keyword):
    """End of an instance definition."""

    refuse_on_gap = False

    def to_records(self, p, rows, ctx):
        yield ScopeRecord("END_INSTANCE", loc=ctx.loc)


@applies(ScopeRecord)
def _apply_scope(reader: AbaqusInputReader, rec: ScopeRecord):
    if rec.action == "PART":
        reader.current_part = reader.model.Part(rec.name or f"PART-{len(reader.model.parts) + 1}", dim=reader.dim)
        reader.context = KeywordContext.PART
    elif rec.action == "END_PART":
        reader.current_part = None
        reader.context = KeywordContext.GLOBAL
    elif rec.action == "ASSEMBLY":
        reader.has_assembly = True
        reader.context = KeywordContext.ASSEMBLY
        if rec.name is not None:
            reader.model.root_assembly.name = rec.name
    elif rec.action == "END_ASSEMBLY":
        reader.context = KeywordContext.GLOBAL
    elif rec.action == "END_INSTANCE":
        reader.current_instance = None
        reader.context = KeywordContext.ASSEMBLY
    else:
        raise RuntimeError(rec.action)


@dataclass(frozen=True)
class InstanceRecord(Record):
    name: str
    part: str
    translation: Optional[Tuple[float, float, float]]
    rotation: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]]   # (a, b, angle deg)


@keyword("INSTANCE", support="FULL")
class InstanceKw(Keyword):
    """Part instance with an optional translation line and rotation line."""

    refuse_on_gap = False

    class Params(Schema):
        name: str
        part: Optional[str] = None

    class Row(RowSchema):
        v: Tuple[float, ...] = ()

    def to_records(self, p, rows, ctx):
        if len(rows) > 2:
            raise InputError(f"*INSTANCE {p.name} at {ctx.loc}: at most a translation and a rotation line, "
                             f"got {len(rows)} data lines")
        trans = rot = None
        if rows:
            t = [x or 0.0 for x in rows[0].v]
            if not 2 <= len(t) <= 3:
                raise InputError(f"*INSTANCE {p.name} at {rows[0].loc}: translation needs 2 or 3 values")
            trans = (t[0], t[1], t[2] if len(t) == 3 else 0.0)
        if len(rows) == 2:
            r = [x or 0.0 for x in rows[1].v]
            if len(r) != 7:
                # used to be ignored without a word
                raise InputError(f"*INSTANCE {p.name} at {rows[1].loc}: rotation needs 7 values "
                                 f"(axis point a, axis point b, angle), got {len(r)}")
            rot = ((r[0], r[1], r[2]), (r[3], r[4], r[5]), r[6])
        yield InstanceRecord(p.name, p.part or p.name, trans, rot, loc=ctx.loc)


@applies(InstanceRecord)
def _apply_instance(reader: AbaqusInputReader, rec: InstanceRecord):
    if rec.part in reader.model.parts:
        part = reader.model.parts[rec.part]
    else:
        # An instance may carry its own mesh (instance-level *NODE / *ELEMENT);
        # a PART= nobody defines is otherwise a typo, so say so.
        part = reader.model.Part(rec.part, dim=reader.dim)
        reader.diagnostics.add("warning", "undefined-part",
                               f"*INSTANCE {rec.name}: PART={rec.part} is not defined; an empty part was "
                               f"created (expected only for an instance that defines its own mesh)", rec.loc)
    inst = reader.model.root_assembly.Instance(rec.name, part=part)
    reader.current_instance = inst
    reader.context = KeywordContext.INSTANCE
    if rec.translation is not None:
        inst.translate(list(rec.translation))
    if rec.rotation is not None:
        p_a, p_b = np.array(rec.rotation[0]), np.array(rec.rotation[1])
        axis = p_b - p_a
        if float(np.linalg.norm(axis)) < 1e-12:
            reader.diagnostics.add("error", "bad-instance-rotation",
                                   f"*INSTANCE {rec.name}: degenerate rotation axis - rotation ignored", rec.loc)
            return
        if reader.dim == 2 and (abs(float(axis[0])) > 1e-12 or abs(float(axis[1])) > 1e-12):
            reader.diagnostics.add("info", "instance-2d-rotation",
                                   f"*INSTANCE {rec.name}: in-plane rotation axis in 2D - in-plane rotation "
                                   f"applied about the axis point", rec.loc)
        inst.rotate(axis=axis, angle_deg=rec.rotation[2], center=p_a)
