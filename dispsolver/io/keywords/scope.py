from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
)

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader
from dispsolver.io.schema import Keyword, Note, Record, RowSchema, Schema, applies, keyword

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
    closes_with = "END PART"

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
    closes_with = "END ASSEMBLY"

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
        inst = reader.current_instance
        if inst is not None and getattr(inst, "_auto_part", False) and not inst.part.elements:
            # PART= named nothing and the instance brought no mesh of its own
            reader.diagnostics.add("error", "undefined-part",
                                   f"*INSTANCE {inst.name}: PART={inst.part.name} is not defined and the "
                                   f"instance defines no elements", rec.loc)
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
    """Part instance with an optional translation line and rotation line.
    A bad transform line is reported and the transform dropped; the instance
    itself is kept so the scope stays intact."""

    refuse_on_gap = False
    closes_with = "END INSTANCE"

    class Params(Schema):
        name: str
        part: Optional[str] = None

    class Row(RowSchema):
        v: Tuple[float, ...] = ()

    def to_records(self, p, rows, ctx):
        def bad(loc, msg):
            return Note("error", "bad-instance-transform", f"*INSTANCE {p.name} at {loc}: {msg}", loc=loc)

        if p.part is None:
            yield Note("error", "missing-param",
                       f"*INSTANCE {p.name} at {ctx.loc}: missing required parameter 'PART' "
                       f"(the part named {p.name!r} is used)", loc=ctx.loc)
        trans = rot = None
        if len(rows) > 2:
            yield bad(ctx.loc, f"at most a translation and a rotation line, got {len(rows)} data lines; "
                               f"transform ignored")
        else:
            if rows:
                t = [x or 0.0 for x in rows[0].v]
                if 2 <= len(t) <= 3:
                    trans = (t[0], t[1], t[2] if len(t) == 3 else 0.0)
                else:
                    yield bad(rows[0].loc, "translation needs 2 or 3 values; translation ignored")
            if len(rows) == 2:
                r = [x or 0.0 for x in rows[1].v]
                if len(r) == 7:
                    rot = ((r[0], r[1], r[2]), (r[3], r[4], r[5]), r[6])
                else:
                    # used to be ignored without a word
                    yield bad(rows[1].loc, f"rotation needs 7 values (axis point a, axis point b, angle), "
                                           f"got {len(r)}; rotation ignored")
        yield InstanceRecord(p.name, p.part or p.name, trans, rot, loc=ctx.loc)


@applies(InstanceRecord)
def _apply_instance(reader: AbaqusInputReader, rec: InstanceRecord):
    # Labels are case-insensitive and blanks on a keyword line are ignored, so
    # PART=PART -1 names part PART-1 (Abaqus verification decks write it so;
    # the legacy lookup created an empty part "PART -1" and lost the mesh).
    key = rec.part.replace(" ", "").upper()
    hits = [n for n in reader.model.parts if n.replace(" ", "").upper() == key]
    name = rec.part if rec.part in reader.model.parts else (hits[0] if len(hits) == 1 else None)
    auto = name is None
    # An instance may carry its own mesh (instance-level *NODE / *ELEMENT), so
    # an undefined PART= is checked at *END INSTANCE, not here.
    part = reader.model.Part(rec.part, dim=reader.dim) if auto else reader.model.parts[name]
    inst = reader.model.root_assembly.Instance(rec.name, part=part)
    inst._auto_part = auto
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
