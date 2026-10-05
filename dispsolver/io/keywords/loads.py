"""Load keywords on the schema framework: *CLOAD, *DLOAD, *DSLOAD.

OP=NEW removes the loads of the SAME kind only (Abaqus Keywords Guide: "all
concentrated loads" for *CLOAD, "all distributed loads" for *DLOAD, "all
distributed surface loads" for *DSLOAD). The legacy handlers deactivated every
load in the step, so a *DSLOAD, OP=NEW also removed the step's *CLOADs.
Load names carry their kind as prefix (CLOAD_, DLOAD_, DSLOAD_), which is
what the reset matches on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from dispsolver.io.schema import (InputError, Keyword, Record, RowSchema, Schema, Unsupported,
                                  applies, keyword)

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader


@dataclass(frozen=True)
class LoadResetRecord(Record):
    step: str
    kind: str                             # CLOAD / DLOAD / DSLOAD


@applies(LoadResetRecord)
def _apply_load_reset(reader: AbaqusInputReader, rec: LoadResetRecord):
    step = reader.model.steps[rec.step]
    for name in list(step.loads.keys()):
        if name.startswith(rec.kind + "_"):
            step.deactivate_load(name)


def _overlay_note(reader: AbaqusInputReader, step_name: str, load_name: str, kw: str, loc) -> None:
    step = reader.model.steps[step_name]
    entry = step.loads.get(load_name)
    if entry is not None and entry.entity.region in step.step_nsets:
        reader.diagnostics.add("info", "step-overlay-region",
                               f"*{kw} {load_name}: step-overlay region; solver load path "
                               f"resolves global sets only (P2)", loc)


# ── *CLOAD ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConcentratedLoadRecord(Record):
    step: str
    region: str                           # node number or node set
    dof: int
    magnitude: float
    amplitude: Optional[str]
    op: str


@keyword("CLOAD", support="FULL")
class CLoad(Keyword):
    """Concentrated nodal forces."""

    class Params(Schema):
        op: Literal["MOD", "NEW"] = "MOD"
        amplitude: Optional[str] = None

    class Row(RowSchema):
        region: str
        dof: int
        magnitude: float

    def to_records(self, p, rows, ctx):
        if p.op == "NEW":
            yield LoadResetRecord(ctx.step, "CLOAD", loc=ctx.loc)
        for r in rows:
            if r.dof not in (1, 2, 3):
                # moments used to be accepted and then dropped
                raise Unsupported(f"*CLOAD {r.region}: degree of freedom {r.dof} is not implemented "
                                  f"(continuum nodes carry dofs 1-3)")
            yield ConcentratedLoadRecord(ctx.step, r.region, r.dof, r.magnitude, p.amplitude, p.op, loc=r.loc)


@applies(ConcentratedLoadRecord)
def _apply_cload(reader: AbaqusInputReader, rec: ConcentratedLoadRecord):
    name = f"CLOAD_{rec.region}_DOF{rec.dof}"
    reader.model.ConcentratedForce(name=name, createStepName=rec.step, region=rec.region,
                                   amplitude=rec.amplitude, **{f"cf{rec.dof}": rec.magnitude})
    reader.model.steps[rec.step].loads[name].modified_params["op"] = rec.op
    _overlay_note(reader, rec.step, name, "CLOAD", rec.loc)


# ── *DLOAD ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GravityRecord(Record):
    step: str
    region: str                           # "" = whole model
    g: Tuple[float, float, float]         # magnitude * direction
    amplitude: Optional[str]


@dataclass(frozen=True)
class BodyForceRecord(Record):
    step: str
    region: str
    label: str                            # BX / BY / BZ
    b: Tuple[float, float, float]
    amplitude: Optional[str]


@dataclass(frozen=True)
class PressureRecord(Record):
    step: str
    kw: str                               # DLOAD / DSLOAD (name prefix)
    region: str                           # element set (face given) or surface
    face: Optional[str]                   # P1..P6 for an element-face pressure, P for an edge/surface
    label: str
    magnitude: float
    amplitude: Optional[str]


@keyword("DLOAD", support="PARTIAL")
class DLoad(Keyword):
    """Distributed loads on elements: GRAV, BX/BY/BZ, P / P1..P6."""

    class Params(Schema):
        op: Literal["MOD", "NEW"] = "MOD"
        amplitude: Optional[str] = None

    class Row(RowSchema):
        region: Optional[str]             # blank: the whole model (GRAV)
        load_type: str
        magnitude: Optional[float] = None  # absent for *NU (user subroutine) types
        comps: Tuple[float, ...] = ()

    def to_records(self, p, rows, ctx):
        if p.op == "NEW":
            yield LoadResetRecord(ctx.step, "DLOAD", loc=ctx.loc)
        for r in rows:
            region, lt, mag = r.region or "", r.load_type.upper(), r.magnitude
            if lt != "GRAV" and lt not in ("BX", "BY", "BZ") and not re.fullmatch(r"P[1-6]?", lt):
                raise Unsupported(f"*DLOAD load type {lt} on {region!r} is not implemented")
            if mag is None:
                raise InputError(f"*DLOAD {lt} on {region!r} at {r.loc}: magnitude missing")
            if lt == "GRAV":
                if len(r.comps) < 3:
                    raise InputError(f"*DLOAD GRAV on {region!r} at {r.loc}: direction vector missing")
                d = [c or 0.0 for c in r.comps[:3]]
                yield GravityRecord(ctx.step, region, (mag * d[0], mag * d[1], mag * d[2]), p.amplitude, loc=r.loc)
            elif lt in ("BX", "BY", "BZ"):
                b = {"BX": (mag, 0.0, 0.0), "BY": (0.0, mag, 0.0), "BZ": (0.0, 0.0, mag)}[lt]
                yield BodyForceRecord(ctx.step, region, lt, b, p.amplitude, loc=r.loc)
            else:
                yield PressureRecord(ctx.step, "DLOAD", region, lt, lt, mag, p.amplitude, loc=r.loc)


@applies(GravityRecord)
def _apply_gravity(reader: AbaqusInputReader, rec: GravityRecord):
    name = f"DLOAD_GRAV_{rec.region}"
    reader.model.Gravity(name=name, createStepName=rec.step, region=rec.region,
                         comp1=rec.g[0], comp2=rec.g[1], comp3=rec.g[2], amplitude=rec.amplitude)
    _overlay_note(reader, rec.step, name, "DLOAD", rec.loc)


@applies(BodyForceRecord)
def _apply_body_force(reader: AbaqusInputReader, rec: BodyForceRecord):
    name = f"DLOAD_{rec.region}_{rec.label}"
    reader.model.BodyForce(name=name, createStepName=rec.step, region=rec.region,
                           b1=rec.b[0], b2=rec.b[1], b3=rec.b[2], amplitude=rec.amplitude)
    _overlay_note(reader, rec.step, name, "DLOAD", rec.loc)


@applies(PressureRecord)
def _apply_pressure(reader: AbaqusInputReader, rec: PressureRecord):
    name = f"{rec.kw}_{rec.region}_{rec.label}"
    kwargs = {"face": rec.face} if rec.face is not None else {}
    reader.model.Pressure(name=name, createStepName=rec.step, region=rec.region,
                          magnitude=rec.magnitude, amplitude=rec.amplitude, **kwargs)
    _overlay_note(reader, rec.step, name, rec.kw, rec.loc)


# ── *DSLOAD ─────────────────────────────────────────────────────────────────

@keyword("DSLOAD", support="PARTIAL")
class DSLoad(Keyword):
    """Distributed surface loads: uniform pressure P.

    The legacy handler also landed HP and PN as a uniform pressure and
    skipped every other type without a word; HP (hydrostatic) needs its
    zero-pressure height and gradient, so it was a different load."""

    class Params(Schema):
        op: Literal["MOD", "NEW"] = "MOD"
        amplitude: Optional[str] = None

    class Row(RowSchema):
        surface: str
        load_type: str
        magnitude: Optional[float] = None  # absent for PNU (user subroutine)
        comps: Tuple[float, ...] = ()

    def to_records(self, p, rows, ctx):
        if p.op == "NEW":
            yield LoadResetRecord(ctx.step, "DSLOAD", loc=ctx.loc)
        for r in rows:
            lt = r.load_type.upper()
            if lt != "P":
                raise Unsupported(f"*DSLOAD load type {lt} on {r.surface!r} is not implemented (P only)")
            if r.magnitude is None:
                raise InputError(f"*DSLOAD P on {r.surface!r} at {r.loc}: magnitude missing")
            if any(c is not None for c in r.comps):
                raise InputError(f"*DSLOAD P on {r.surface!r} at {r.loc}: a uniform pressure takes one value")
            yield PressureRecord(ctx.step, "DSLOAD", r.surface, None, "P", r.magnitude, p.amplitude, loc=r.loc)
