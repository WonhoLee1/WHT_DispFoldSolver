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
from dispsolver.io.schema import (InputError, Keyword, Param, Record, RowSchema, Schema, Unsupported, applies,
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
    controls: Optional[str] = None


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
        controls: Optional[str] = None

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
            yield SectionRecord("SOLID", p.elset, p.material, p.orientation, thickness, controls=p.controls, loc=ctx.loc)
            return
        # composite: per layer "thickness, integration points, material[, orientation[, ply]]"
        layers = []
        for r, c in zip(rows, cells):
            if len(c) < 3 or not c[2]:
                raise InputError(f"*SOLID SECTION, COMPOSITE at {r.loc}: a layer needs thickness, "
                                 f"integration points and material")
            ori = c[3] if len(c) > 3 and c[3] else None
            if ori is not None:
                try:
                    float(ori)
                except ValueError:
                    pass
                else:
                    # an angle relative to the section orientation, not a name
                    raise Unsupported(f"*SOLID SECTION, COMPOSITE at {r.loc}: layer orientation angle {ori} "
                                      f"is not implemented")
            layers.append((c[2], ori or p.orientation))
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
                                             thickness=rec.thickness, orientation=rec.orientation,
                                             controls=rec.controls)
    elif rec.kind == "COHESIVE":
        reader.model.CohesiveSection(name=sec_name, material=rec.material,
                                     out_of_plane_thickness=rec.thickness,
                                     controls=rec.controls)
    else:
        reader.model.ShellSection(name=sec_name, material=rec.material, thickness=rec.thickness)
    part.SectionAssignment(region=rec.elset, sectionName=sec_name)


@keyword("COHESIVE SECTION", support="FULL")
class CohesiveSectionKeyword(Keyword):
    """Cohesive section for interface elements (*COHESIVE SECTION in Abaqus)."""

    refuse_on_gap = False

    class Params(Schema):
        elset: str
        material: str
        response: str = "TRACTION SEPARATION"
        controls: Optional[str] = None

    Row = _Cells

    def to_records(self, p, rows, ctx):
        cells = _cells(rows)
        thick = 1.0
        if cells and cells[0] and cells[0][0]:
            try:
                thick = float(cells[0][0])
            except ValueError:
                pass
        yield SectionRecord("COHESIVE", p.elset, p.material, None, thick, controls=p.controls, loc=ctx.loc)



@dataclass(frozen=True)
class SectionControlsRecord(Record):
    name: str
    distortion_control: bool = True
    length_ratio: float = 0.1
    hourglass_control: str = "ENHANCED"
    viscous_damping: float = 0.0


@keyword("SECTION CONTROLS", support="FULL")
class SectionControlsKeyword(Keyword):
    """Controls for continuum solid elements (*SECTION CONTROLS in Abaqus)."""

    refuse_on_gap = False

    class Params(Schema):
        name: str
        distortion_control: Optional[str] = None
        length_ratio: Optional[float] = None
        hourglass: Optional[str] = None
        viscous_damping: float = Param(0.0, undocumented="in-house parameter for viscous damping")

    Row = _Cells

    def to_records(self, p, rows, ctx):
        dist_on = True
        if p.distortion_control is not None:
            dist_str = str(p.distortion_control).strip().upper()
            dist_on = dist_str in ("YES", "Y", "TRUE", "ON", "1")
        lr = float(p.length_ratio) if p.length_ratio is not None else 0.1
        hg = str(p.hourglass).strip().upper() if p.hourglass is not None else "ENHANCED"
        yield SectionControlsRecord(
            name=p.name,
            distortion_control=dist_on,
            length_ratio=lr,
            hourglass_control=hg,
            viscous_damping=float(p.viscous_damping),
            loc=ctx.loc,
        )


@applies(SectionControlsRecord)
def _apply_section_controls(reader: AbaqusInputReader, rec: SectionControlsRecord):
    reader.model.SectionControls(
        name=rec.name,
        distortion_control=rec.distortion_control,
        length_ratio=rec.length_ratio,
        hourglass_control=rec.hourglass_control,
        viscous_damping=rec.viscous_damping,
    )

