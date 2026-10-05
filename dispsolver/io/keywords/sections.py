"""Section keywords on the schema framework: *SOLID SECTION, *SHELL SECTION.

Moved out of keywords/material.py (the material keywords there are agent B's
B4 migration). A section is applied even when one of its parameters is a gap
(`refuse_on_gap = False`): refusing it leaves its elements without a material,
and every later error would be about that instead of the real cause. The gap
is still reported and still blocks a strict run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from dispsolver.io.keyword_registry import KeywordContext
from dispsolver.io.schema import (InputError, Keyword, Record, RowSchema, Schema, Unsupported, applies,
                                  keyword)

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader


def _section_part(reader, elset: str, kind: str):
    """Part a *SOLID/*SHELL SECTION applies to. Inside a *PART: that part,
    which must define the ELSET. Outside any part (written after *END PART):
    the one part that defines the ELSET; several candidates are ambiguous.
    The old fallback took the FIRST part with that ELSET name."""
    if reader.context in (KeywordContext.PART, KeywordContext.INSTANCE) or elset.upper() == "ALL":
        part = reader.get_active_part()
        if elset.upper() != "ALL" and elset not in part.element_sets and elset not in part.sets:
            raise KeyError(f"*{kind} SECTION: ELSET {elset!r} is not defined in part {part.name!r}")
        return part
    owners = [p for p in reader.model.parts.values() if elset in p.element_sets or elset in p.sets]
    if len(owners) == 1:
        return owners[0]
    if not owners:
        raise KeyError(f"*{kind} SECTION: ELSET {elset!r} is not defined in any part")
    raise KeyError(f"*{kind} SECTION: ELSET {elset!r} is defined in parts {[p.name for p in owners]}; "
                   f"put the section inside the *PART it belongs to")


class _Cells(RowSchema):
    items: Tuple[str, ...] = ()


def _cells(rows):
    return [[c.strip() if c is not None else "" for c in r.items] for r in rows]


@dataclass(frozen=True)
class SectionRecord(Record):
    kind: str                             # SOLID / SHELL
    elset: str
    material: str
    orientation: Optional[str]
    thickness: float


@keyword("SOLID SECTION", support="FULL")
class SolidSection(Keyword):
    """Homogeneous solid section; a composite solid whose layers share one
    material and orientation (exactly the homogeneous section)."""

    refuse_on_gap = False

    class Params(Schema):
        elset: str
        material: Optional[str] = None
        orientation: Optional[str] = None
        composite: bool = False

    Row = _Cells

    def to_records(self, p, rows, ctx):
        cells = _cells(rows)
        if not p.composite:
            if p.material is None:
                raise InputError(f"*SOLID SECTION, ELSET={p.elset} at {ctx.loc}: missing required parameter 'MATERIAL'")
            if len(cells) > 1:
                raise InputError(f"*SOLID SECTION, ELSET={p.elset} at {ctx.loc}: one data line expected")
            thickness = 1.0
            if cells and cells[0] and cells[0][0]:
                try:
                    thickness = float(cells[0][0])   # plane stress / strain thickness
                except ValueError:
                    raise InputError(f"*SOLID SECTION, ELSET={p.elset} at {ctx.loc}: thickness "
                                     f"{cells[0][0]!r} is not a number (distributions are not supported)") from None
            yield SectionRecord("SOLID", p.elset, p.material, p.orientation, thickness, loc=ctx.loc)
            return
        # composite: per layer "thickness, integration points, material[, orientation[, ply]]"
        layers = []
        for r, c in zip(rows, cells):
            if len(c) < 3 or not c[2]:
                raise InputError(f"*SOLID SECTION, COMPOSITE at {r.loc}: a layer needs thickness, "
                                 f"integration points and material")
            layers.append((c[2], (c[3] if len(c) > 3 and c[3] else None) or p.orientation))
        if not layers:
            raise InputError(f"*SOLID SECTION, COMPOSITE, ELSET={p.elset} at {ctx.loc}: no layers")
        if len(set(layers)) > 1:
            # the legacy handler kept the first layer and dropped the rest
            raise Unsupported(f"*SOLID SECTION, COMPOSITE, ELSET={p.elset}: {len(set(layers))} different "
                              f"layers (material, orientation) are not implemented")
        mat, ori = layers[0]
        # one material and orientation through the thickness is that homogeneous
        # solid; layer thicknesses are relative for solid elements
        yield SectionRecord("SOLID", p.elset, mat, ori, 1.0, loc=ctx.loc)


@keyword("SHELL SECTION", support="PARTIAL")
class ShellSection(Keyword):
    """Homogeneous shell section (thickness). Shell elements themselves are
    refused by the element catalog; the section is kept for completeness."""

    refuse_on_gap = False

    class Params(Schema):
        elset: str
        material: Optional[str] = None
        composite: bool = False

    Row = _Cells

    def to_records(self, p, rows, ctx):
        if p.composite:
            raise Unsupported(f"*SHELL SECTION, COMPOSITE, ELSET={p.elset} is not implemented")
        if p.material is None:
            raise InputError(f"*SHELL SECTION, ELSET={p.elset} at {ctx.loc}: missing required parameter 'MATERIAL'")
        cells = _cells(rows)
        thickness = 1.0
        if cells and cells[0] and cells[0][0]:
            try:
                thickness = float(cells[0][0])
            except ValueError:
                raise InputError(f"*SHELL SECTION, ELSET={p.elset} at {ctx.loc}: thickness {cells[0][0]!r} "
                                 f"is not a number (distributions are not supported)") from None
        yield SectionRecord("SHELL", p.elset, p.material, None, thickness, loc=ctx.loc)


@applies(SectionRecord)
def _apply_section(reader: AbaqusInputReader, rec: SectionRecord):
    # Section names are per part: Abaqus CAE writes the same ELSET name
    # (e.g. _PickedSet2) in every part, and a model-wide "SEC_SOLID_<elset>"
    # made the last part's section overwrite every earlier part's material.
    part = _section_part(reader, rec.elset, rec.kind)
    sec_name = f"SEC_{rec.kind}_{part.name}_{rec.elset}"
    if rec.kind == "SOLID":
        reader.model.HomogeneousSolidSection(name=sec_name, material=rec.material,
                                             thickness=rec.thickness, orientation=rec.orientation)
    else:
        reader.model.ShellSection(name=sec_name, material=rec.material, thickness=rec.thickness)
    part.SectionAssignment(region=rec.elset, sectionName=sec_name)
