"""Material keywords on the schema framework (B4 migration).

Every material keyword is a Keyword class with typed Params and Row schemas,
a pure to_records, and an applier registered per record type. Material options
map to numerical IDs and runtime props vectors strictly through
dispsolver.material.catalog.material_id and pack_props.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from dispsolver.io.diagnostics import Loc
from dispsolver.io.keyword_registry import KeywordContext
from dispsolver.io.schema import (
    InputError,
    Keyword,
    KeywordCtx,
    Note,
    Param,
    Record,
    RowSchema,
    Schema,
    Unsupported,
    applies,
    keyword,
)

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader


class FloatRow(RowSchema):
    """Positional data line accepting a variable number of float values (or blanks)."""
    items: Tuple[float, ...] = ()


def _cells(row, n: int) -> list:
    """The first n data items of a line by position (None for a blank or missing one)."""
    v = list(row.items)[:n]
    return v + [None] * (n - len(v))


# ─── 1. *MATERIAL ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaterialRecord(Record):
    name: str


@keyword("MATERIAL", support="FULL", contexts={KeywordContext.GLOBAL})
class Material(Keyword):
    """Defines a material by name and opens the MATERIAL context scope."""

    class Params(Schema):
        name: str
        rtol: Optional[float] = Param(None, control=True)
        srate_factor: Optional[float] = Param(None, control=True)
        srate_time: Optional[float] = Param(None, control=True)
        strain_rate_interpolation: Optional[str] = Param(None, control=True)

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        yield MaterialRecord(name=p.name, loc=ctx.loc)


@applies(MaterialRecord)
def _apply_material(reader: AbaqusInputReader, rec: MaterialRecord) -> None:
    mat = reader.model.Material(rec.name)
    reader.current_material = mat
    reader.context = KeywordContext.MATERIAL


def _tabulate(kw: str, rows, ncols: int, dependencies: int, names: Tuple[str, ...], curve: bool = False):
    """Data lines of `ncols` items [+ temperature + field variables] -> (kind, table).

    Keywords Guide layout: items 1..ncols, then temperature, then the field
    variables. kind is 'const' (one line), 'temp' (lines with temperature) or
    'field' (DEPENDENCIES=1; rows are (items..., FV1)). Blank cells keep their
    position. Several lines without temperature, several field variables, and
    temperature together with a field variable are not represented.
    """
    if not rows:
        raise InputError(f"*{kw}: no data line")
    if dependencies > 1:
        raise Unsupported(f"*{kw}: DEPENDENCIES={dependencies} (several field variables) is not represented")
    width = ncols + 1 + dependencies
    grid = [_cells(r, width) for r in rows]
    for g in grid:
        for i in range(ncols):
            if g[i] is None:
                raise InputError(f"*{kw}: data item {i + 1} ({names[i]}) is missing; the Keywords Guide gives no default")
    temps = [g[ncols] for g in grid]
    if dependencies == 1:
        if any(g[ncols + 1] is None for g in grid):
            raise InputError(f"*{kw}: DEPENDENCIES=1 needs the field variable (data item {ncols + 2}) on every line")
        if any(t is not None and t != temps[0] for t in temps):
            raise Unsupported(f"*{kw}: joint temperature and field-variable dependence is not represented")
        return "field", tuple(tuple(float(v) for v in g[:ncols]) + (float(g[ncols + 1]),) for g in grid)
    if all(t is None for t in temps):
        if len(grid) > 1 and not curve:
            raise InputError(f"*{kw}: several data lines without temperature")
        return "const", tuple(tuple(float(v) for v in g[:ncols]) for g in grid)
    if any(t is None for t in temps):
        raise InputError(f"*{kw}: temperature is missing on some data lines")
    return "temp", tuple(tuple(float(v) for v in g[:ncols + 1]) for g in grid)


# ─── 2. *ELASTIC ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ElasticRecord(Record):
    type: str
    kind: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("ELASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Elastic(Keyword):
    """Linear elasticity: isotropic (E, nu[, T]) and engineering constants."""

    class Params(Schema):
        type: Literal[
            "ISOTROPIC", "ENGINEERING CONSTANTS", "ORTHOTROPIC", "ANISOTROPIC",
            "TRACTION", "BILAMINA", "LAMINA", "SHORT FIBER", "COUPLED TRACTION",
            "TRANSVERSELY ISOTROPIC", "SHEAR"
        ] = Param("ISOTROPIC", supported=("ISOTROPIC", "ENGINEERING CONSTANTS"))
        moduli: Literal["LONG TERM", "INSTANTANEOUS"] = Param("LONG TERM", unsupported="MODULI not supported")
        dependencies: int = 0
        cohesive_offset: Optional[str] = Param(None, unsupported="COHESIVE OFFSET not supported")
        compression_factor: Optional[float] = Param(None, unsupported="COMPRESSION FACTOR not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if p.type == "ENGINEERING CONSTANTS":
            # line 1: E1 E2 E3 nu12 nu13 nu23 G12 G13; line 2: G23, temperature, fields
            if p.dependencies:
                raise Unsupported("*ELASTIC, TYPE=ENGINEERING CONSTANTS: DEPENDENCIES not represented")
            if len(rows) != 2:
                raise InputError("*ELASTIC, TYPE=ENGINEERING CONSTANTS needs two data lines (a single temperature set)")
            a, b = _cells(rows[0], 8), _cells(rows[1], 2)
            vals = a + [b[0]]
            if any(v is None for v in vals):
                raise InputError("*ELASTIC, TYPE=ENGINEERING CONSTANTS: E1 E2 E3 nu12 nu13 nu23 G12 G13 G23 are all required")
            if b[1] is not None:
                raise Unsupported("*ELASTIC, TYPE=ENGINEERING CONSTANTS: temperature dependence not represented")
            yield ElasticRecord(type=p.type, kind="const", table=(tuple(float(v) for v in vals),), loc=ctx.loc)
            return
        kind, table = _tabulate("ELASTIC", rows, 2, p.dependencies, ("Young's modulus E", "Poisson's ratio nu"))
        yield ElasticRecord(type=p.type, kind=kind, table=table, loc=ctx.loc)


@applies(ElasticRecord)
def _apply_elastic(reader: AbaqusInputReader, rec: ElasticRecord) -> None:
    mat = reader.current_material
    if mat is None:
        mat = reader.model.Material("DEFAULT_ELASTIC")
        reader.current_material = mat
    if rec.type == "ENGINEERING CONSTANTS":
        mat.Elastic(table=(), type=rec.type, flat_tokens=list(rec.table[0]))
    elif rec.kind == "field":
        mat.Elastic(field_table=rec.table, type=rec.type)
    else:
        mat.Elastic(table=rec.table, type=rec.type)


# ─── 3. *EXPANSION ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExpansionRecord(Record):
    type: str
    zero: float
    table: Tuple[Tuple[float, ...], ...]


@keyword("EXPANSION", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Expansion(Keyword):
    """Isotropic thermal expansion coefficient alpha (and temperature)."""

    class Params(Schema):
        # the model carries one scalar alpha
        type: Literal["ISO", "ORTHO", "ANISO", "SHORT FIBER", "TRANSVERSELY ISOTROPIC"] = Param("ISO", supported=("ISO",))
        zero: float = 0.0
        dependencies: int = Param(0, unsupported="EXPANSION DEPENDENCIES not supported")
        user: bool = Param(False, unsupported="EXPANSION USER not supported")
        pore_fluid: bool = Param(False, unsupported="EXPANSION PORE FLUID not supported")
        field: Optional[str] = Param(None, unsupported="EXPANSION FIELD not supported")
        liquid: bool = Param(False, unsupported="EXPANSION LIQUID not supported")
        properties: Optional[int] = Param(None, unsupported="EXPANSION PROPERTIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        _, table = _tabulate("EXPANSION", rows, 1, 0, ("alpha",))
        yield ExpansionRecord(type=p.type, zero=p.zero, table=table, loc=ctx.loc)


@applies(ExpansionRecord)
def _apply_expansion(reader: AbaqusInputReader, rec: ExpansionRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.Expansion(type=rec.type, table=rec.table, zero=rec.zero)


# ─── 4. *PLASTIC ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlasticRecord(Record):
    hardening: str
    kind: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("PLASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Plastic(Keyword):
    """Metal plasticity: isotropic hardening table (yield stress, plastic strain) or Johnson-Cook."""

    class Params(Schema):
        # kinematic hardening is not represented (the model hardens isotropically)
        hardening: Literal["ISOTROPIC", "KINEMATIC", "MULTILINEAR KINEMATIC", "COMBINED", "JOHNSON COOK", "USER"] = Param(
            "ISOTROPIC", supported=("ISOTROPIC", "JOHNSON COOK"))
        rate: Optional[str] = Param(None, unsupported="PLASTIC RATE not supported")
        dependencies: int = 0
        data_type: Optional[str] = Param(None, unsupported="PLASTIC DATA TYPE not supported")
        extrapolation: Optional[str] = Param(None, control=True)
        number_backstresses: Optional[int] = Param(None, unsupported="PLASTIC NUMBER BACKSTRESSES not supported")
        properties: Optional[int] = Param(None, unsupported="PLASTIC PROPERTIES not supported")
        scalestress: Optional[float] = Param(None, unsupported="PLASTIC SCALESTRESS not supported")
        static_recovery: Optional[str] = Param(None, unsupported="PLASTIC STATIC RECOVERY not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        hard = p.hardening.replace(" ", "_")
        if hard == "JOHNSON_COOK":
            # one line: A B n m theta_melt theta_transition (m blank or 0: no temperature dependence)
            if len(rows) != 1:
                raise InputError("*PLASTIC, HARDENING=JOHNSON COOK takes exactly one data line")
            v = _cells(rows[0], 6)
            for i, nm in enumerate(("A", "B", "n")):
                if v[i] is None:
                    raise InputError(f"*PLASTIC, HARDENING=JOHNSON COOK: {nm} (data item {i + 1}) is required")
            m = 0.0 if v[3] is None else float(v[3])
            if m != 0.0 and (v[4] is None or v[5] is None):
                raise InputError("*PLASTIC, HARDENING=JOHNSON COOK: melting and transition temperature are required when m is nonzero")
            yield Note("gap", "rate-effect-dropped",
                       "*PLASTIC HARDENING=JOHNSON_COOK: quasi-static isothermal landing (C=0, dispatcher carries no rate)",
                       loc=ctx.loc)
            yield PlasticRecord(hardening=hard, kind="const",
                                table=((float(v[0]), float(v[1]), float(v[2]), m, v[4], v[5]),), loc=ctx.loc)
            return
        kind, table = _tabulate("PLASTIC", rows, 2, p.dependencies, ("yield stress", "plastic strain"), curve=True)
        yield PlasticRecord(hardening=hard, kind=kind, table=table, loc=ctx.loc)


@applies(PlasticRecord)
def _apply_plastic(reader: AbaqusInputReader, rec: PlasticRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    if rec.hardening == "JOHNSON_COOK":
        A, B, n, m, t_melt, t_trans = rec.table[0]
        kw = {}
        if t_melt is not None and t_trans is not None:
            kw = {"T_melt": float(t_melt), "T_room": float(t_trans)}
        mat.JohnsonCook(A=A, B=B, n=n, m=m, **kw)
    elif rec.kind == "field":
        mat.Plastic(table=rec.table, hardening=rec.hardening, field_table=rec.table)
    else:
        mat.Plastic(table=rec.table, hardening=rec.hardening)


# ─── 5. *CREEP ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CreepRecord(Record):
    law: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("CREEP", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Creep(Keyword):
    """Creep material behavior (the kernel is the time-hardening power law A q^n t^m, sinh law, Anand)."""

    class Params(Schema):
        law: Literal["STRAIN", "TIME", "HYPERB", "ANAND", "DARVEAUX", "DOUBLE POWER", "POWER LAW",
                     "TIME POWER LAW", "USER"] = Param("STRAIN", supported=("STRAIN", "TIME", "HYPERB", "ANAND"))
        dependencies: int = Param(0, unsupported="CREEP DEPENDENCIES not supported")
        time: Optional[Literal["TOTAL", "CREEP"]] = Param("TOTAL", supported=("TOTAL",))

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError("*CREEP: no data line")
        law = p.law
        if law in ("STRAIN", "TIME"):
            if len(rows) > 1:
                raise Unsupported("*CREEP: several data lines (temperature dependence) are not represented")
            v = _cells(rows[0], 4)
            for i, nm in enumerate(("A", "n", "m")):
                if v[i] is None:
                    raise InputError(f"*CREEP LAW={law}: {nm} (data item {i + 1}) is required; the Keywords Guide gives no default")
            if v[3] is not None:
                raise Unsupported("*CREEP: temperature-dependent constants are not represented")
            if law == "STRAIN" and v[2] != 0.0:
                raise Unsupported("*CREEP LAW=STRAIN with m != 0 (strain hardening) is not represented: "
                                  "the kernel is the time-hardening form")
            table = ((float(v[0]), float(v[1]), float(v[2])),)
        elif law == "HYPERB":
            if len(rows) > 1:
                raise Unsupported("*CREEP: several data lines (field dependence) are not represented")
            v = _cells(rows[0], 5)
            for i, nm in enumerate(("A", "B", "n")):
                if v[i] is None:
                    raise InputError(f"*CREEP LAW=HYPERB: {nm} (data item {i + 1}) is required")
            if v[3] not in (None, 0.0) or v[4] not in (None, 0.0):
                raise Unsupported("*CREEP LAW=HYPERB: temperature dependence (delta H, R) is not represented")
            table = ((float(v[0]), float(v[1]), float(v[2])),)
        else:  # ANAND: S1 Q/R A xi m A0 shat n / a S2 S3 A1 A2 A3 A4
            if len(rows) != 2:
                raise InputError("*CREEP LAW=ANAND needs two data lines")
            a, b = _cells(rows[0], 8), _cells(rows[1], 8)
            if any(x is None for x in a) or b[0] is None:
                raise InputError("*CREEP LAW=ANAND: S1 Q/R A xi m A0 shat n / a are required")
            if any(x not in (None, 0.0) for x in b[1:]):
                raise Unsupported("*CREEP LAW=ANAND: S2, S3, A1..A4 are not represented")
            table = (tuple(float(x) for x in a + [b[0]]),)
        yield CreepRecord(law=law, table=table, loc=ctx.loc)


@applies(CreepRecord)
def _apply_creep(reader: AbaqusInputReader, rec: CreepRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.Creep(law=rec.law, table=rec.table)


# ─── 6. *HYPERELASTIC ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HyperelasticRecord(Record):
    model: str
    n: int
    table: Tuple[Tuple[float, ...], ...]
    coef: Tuple[Tuple[str, float], ...] = ()


def _coef_line(kw: str, rows, k_req: int, k_all: int, what: str) -> list:
    """One hyperelastic data line of k_all coefficients, the first k_req required.

    The trailing D_i may be omitted: the Keywords Guide treats the compressibility as
    given only "by entering nonzero values for D_i" (omitted == 0, incompressible).
    Temperature (the item after the coefficients) and several lines are not represented.
    """
    if len(rows) != 1:
        raise Unsupported(f"*{kw}: several data lines (temperature sets) are not represented")
    cells = _cells(rows[0], k_all + 1)
    if any(c is None for c in cells[:k_req]):
        raise InputError(f"*{kw}: {what}: the first {k_req} values are required; the Keywords Guide gives no default")
    if cells[k_all] is not None or len(rows[0].items) > k_all + 1:
        raise Unsupported(f"*{kw}: temperature-dependent constants are not represented")
    return [0.0 if c is None else float(c) for c in cells[:k_all]]


@keyword("HYPERELASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Hyperelastic(Keyword):
    """Hyperelastic strain-energy potentials. POLYNOMIAL (N=1) is the Abaqus default form."""

    class Params(Schema):
        # TYPE belongs to USER hyperelasticity only (Keywords Guide); the model is chosen by the flags
        type: Optional[str] = Param(None, unsupported="HYPERELASTIC TYPE applies only to USER hyperelasticity")
        n: int = 1
        moduli: Literal["INSTANTANEOUS", "LONG TERM"] = Param("LONG TERM", unsupported="HYPERELASTIC MODULI not supported")
        properties: Optional[int] = Param(None, unsupported="HYPERELASTIC PROPERTIES not supported")
        test_data_input: bool = Param(False, unsupported="HYPERELASTIC TEST DATA INPUT not supported")
        arruda_boyce: bool = False
        yeoh: bool = False
        mooney_rivlin: bool = False
        ogden: bool = False
        polynomial: bool = False
        neo_hooke: bool = False
        marlow: bool = Param(False, unsupported="MARLOW needs *UNIAXIAL TEST DATA, which is not read")
        reduced_polynomial: bool = False
        van_der_waals: bool = Param(False, unsupported="VAN DER WAALS not supported")
        valanis_landel: bool = Param(False, unsupported="VALANIS LANDEL not supported")
        hencky: bool = Param(False, unsupported="HYPERELASTIC HENCKY not supported")
        user: bool = Param(False, unsupported="HYPERELASTIC USER not supported")
        formulation: Optional[str] = Param(None, control=True, undocumented="Abaqus verification decks use FORMULATION")
        poisson: Optional[float] = Param(None, unsupported="HYPERELASTIC POISSON not supported")
        beta: Optional[float] = Param(None, unsupported="HYPERELASTIC BETA not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        flags = [nm for nm in ("arruda_boyce", "yeoh", "mooney_rivlin", "ogden", "polynomial",
                               "neo_hooke", "reduced_polynomial") if getattr(p, nm)]
        if len(flags) > 1:
            raise InputError(f"*HYPERELASTIC: parameters {', '.join(f.upper() for f in flags)} are mutually exclusive")
        form = flags[0] if flags else "polynomial"       # Keywords Guide: POLYNOMIAL is the default
        n = p.n
        if form in ("arruda_boyce", "yeoh", "mooney_rivlin", "neo_hooke") and "n" in p.given:
            raise InputError(f"*HYPERELASTIC: N can be used only with OGDEN, POLYNOMIAL and REDUCED POLYNOMIAL")
        if not rows:
            raise InputError("*HYPERELASTIC: no data line")
        kw = "HYPERELASTIC"
        coef: dict = {}
        table: tuple
        if form == "neo_hooke" or (form == "reduced_polynomial" and n == 1):
            v = _coef_line(kw, rows, 1, 2, "C10, D1")
            model, table = "NEO_HOOKEAN", (tuple(v),)
        elif form == "mooney_rivlin" or (form == "polynomial" and n == 1):
            v = _coef_line(kw, rows, 2, 3, "C10, C01, D1")
            model, table = "MOONEY_RIVLIN", (tuple(v),)
            coef = dict(zip(("c10", "c01", "d1"), v))
        elif form == "polynomial" and n == 2:
            v = _coef_line(kw, rows, 5, 7, "C10 C01 C20 C11 C02 D1 D2")
            model, table = "POLYNOMIAL", (tuple(v),)
            coef = dict(zip(("c10", "c01", "c20", "c11", "c02", "d1", "d2"), v))
        elif form == "polynomial":
            raise Unsupported(f"*HYPERELASTIC, POLYNOMIAL, N={n}: only N=1 and N=2 are represented")
        elif form == "yeoh" or (form == "reduced_polynomial" and n == 3):
            v = _coef_line(kw, rows, 3, 6, "C10 C20 C30 D1 D2 D3")
            model, table = "YEOH", (tuple(v),)
            coef = dict(zip(("c10", "c20", "c30", "d1", "d2", "d3"), v))
        elif form == "reduced_polynomial" and n == 2:
            v = _coef_line(kw, rows, 2, 4, "C10 C20 D1 D2")
            model, table = "YEOH", (tuple(v),)          # reduced polynomial N=2 == Yeoh with C30 = 0
            coef = {"c10": v[0], "c20": v[1], "c30": 0.0, "d1": v[2], "d2": v[3], "d3": 0.0}
        elif form == "reduced_polynomial":
            raise Unsupported(f"*HYPERELASTIC, REDUCED POLYNOMIAL, N={n}: only N=1..3 are represented")
        elif form == "arruda_boyce":
            v = _coef_line(kw, rows, 2, 3, "mu, lambda_m, D")
            model, table = "ARRUDA_BOYCE", (tuple(v),)
        else:  # ogden
            if not 1 <= n <= 6:
                raise InputError(f"*HYPERELASTIC, OGDEN, N={n}: N must be 1..6")
            vals = [float(x) for r in rows for x in r.items if x is not None]
            if len(vals) < 2 * n:
                raise InputError(f"*HYPERELASTIC, OGDEN, N={n}: mu_i, alpha_i (i=1..{n}) are required")
            if len(vals) > 3 * n:
                raise Unsupported("*HYPERELASTIC, OGDEN: temperature-dependent constants are not represented")
            vals = vals + [0.0] * (3 * n - len(vals))          # omitted D_i == 0
            mus, als, ds = vals[0:2 * n:2], vals[1:2 * n:2], vals[2 * n:]
            model, table = "OGDEN", tuple((mus[i], als[i], ds[i]) for i in range(n))
        yield HyperelasticRecord(model=model, n=n, table=table, coef=tuple(coef.items()), loc=ctx.loc)


@applies(HyperelasticRecord)
def _apply_hyperelastic(reader: AbaqusInputReader, rec: HyperelasticRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.Hyperelastic(model=rec.model, n=rec.n, table=rec.table, **dict(rec.coef))


# ─── 7. *VISCOELASTIC ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ViscoelasticRecord(Record):
    time: str
    type: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("VISCOELASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Viscoelastic(Keyword):
    """Time-domain Prony-series viscoelasticity (g_i, k_i, tau_i per term)."""

    class Params(Schema):
        # one of TIME / FREQUENCY / NONLINEAR is required (Keywords Guide); no default
        time: Optional[Literal["CREEP TEST DATA", "FREQUENCY DATA", "PRONY", "RELAXATION TEST DATA"]] = Param(
            None, supported=("PRONY",))
        frequency: Optional[Literal["CREEP TEST DATA", "RELAXATION TEST DATA", "FORMULA", "PRONY", "TABULAR"]] = Param(
            None, unsupported="VISCOELASTIC FREQUENCY not supported")
        type: Literal["ISOTROPIC", "TRANSVERSELY ISOTROPIC", "ORTHOTROPIC", "TRACTION"] = Param(
            "ISOTROPIC", supported=("ISOTROPIC",))
        dependencies: int = Param(0, unsupported="VISCOELASTIC DEPENDENCIES not supported")
        errtol: Optional[float] = Param(None, control=True)
        nmax: Optional[int] = Param(None, control=True)
        law: Optional[str] = Param(None, unsupported="VISCOELASTIC LAW not supported")
        preload: Optional[str] = Param(None, unsupported="VISCOELASTIC PRELOAD not supported")
        properties: Optional[int] = Param(None, unsupported="VISCOELASTIC PROPERTIES not supported")
        sratio: Optional[float] = Param(None, unsupported="VISCOELASTIC SRATIO not supported")
        nonlinear: bool = Param(False, unsupported="VISCOELASTIC NONLINEAR not supported")
        networkid: Optional[int] = Param(None, unsupported="VISCOELASTIC NETWORKID not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if p.time is None:
            raise InputError("*VISCOELASTIC: one of TIME, FREQUENCY or NONLINEAR is required")
        if not rows:
            raise InputError("*VISCOELASTIC: no Prony data line")
        table = []
        for r in rows:
            v = _cells(r, 3)
            if any(x is None for x in v) or len(r.items) > 3:
                raise InputError("*VISCOELASTIC: each line needs g_i, k_i and tau_i (the Keywords Guide gives no default)")
            table.append(tuple(float(x) for x in v))
        yield ViscoelasticRecord(time=p.time, type=p.type, table=tuple(table), loc=ctx.loc)


@applies(ViscoelasticRecord)
def _apply_viscoelastic(reader: AbaqusInputReader, rec: ViscoelasticRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.Viscoelastic(time=rec.time, table=rec.table)


# ─── 8. *RATE DEPENDENT ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RateDependentRecord(Record):
    type: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("RATE DEPENDENT", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class RateDependent(Keyword):
    """Strain-rate dependence: stored on the material, not yet consumed by any kernel."""

    class Params(Schema):
        type: Literal["POWER LAW", "CHABOCHE", "JOHNSON COOK", "YIELD RATIO"] = Param(
            "POWER LAW", supported=("POWER LAW", "JOHNSON COOK", "YIELD RATIO"))
        dependencies: int = Param(0, unsupported="RATE DEPENDENT DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError("*RATE DEPENDENT: no data line")
        rd_type = p.type.replace(" ", "_")
        names = {"POWER_LAW": ("D", "n"), "JOHNSON_COOK": ("C", "reference strain rate"),
                 "YIELD_RATIO": ("yield stress ratio", "plastic strain rate")}[rd_type]
        if rd_type == "JOHNSON_COOK" and len(rows) != 1:
            raise InputError("*RATE DEPENDENT, TYPE=JOHNSON COOK takes one data line")
        if rd_type == "POWER_LAW" and len(rows) != 1:
            raise Unsupported("*RATE DEPENDENT, TYPE=POWER LAW: several data lines (temperature dependence) not represented")
        table = []
        for r in rows:
            v = _cells(r, 3)
            if v[0] is None or v[1] is None:
                raise InputError(f"*RATE DEPENDENT: {names[0]} and {names[1]} are required; the Keywords Guide gives no default")
            if v[2] is not None:
                raise Unsupported("*RATE DEPENDENT: temperature-dependent data is not represented")
            table.append((float(v[0]), float(v[1])))
        yield Note("gap", "rate-effect-dropped",
                   "*RATE DEPENDENT is stored on the material but no kernel consumes it: the solve is rate independent",
                   loc=ctx.loc)
        yield RateDependentRecord(type=rd_type, table=tuple(table), loc=ctx.loc)


@applies(RateDependentRecord)
def _apply_rate_dependent(reader: AbaqusInputReader, rec: RateDependentRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.RateDependent(type=rec.type, table=rec.table)


# ─── 9. *DAMAGE INITIATION ───────────────────────────────────────────────────

@dataclass(frozen=True)
class DamageInitiationRecord(Record):
    criterion: str
    table: Tuple[Tuple[float, ...], ...]
    extra: Tuple[Tuple[str, float], ...] = ()


@keyword("DAMAGE INITIATION", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DamageInitiation(Keyword):
    """Damage initiation: ductile (eD [, triaxiality]) and Johnson-Cook (d1..d5, theta_melt, theta_transition, eps0)."""

    class Params(Schema):
        # CRITERION is required (Keywords Guide); there is no default
        criterion: Literal[
            "DUCTILE", "HC", "JOHNSON COOK", "SHEAR", "FLD", "FLSD", "MK", "MSFLD", "HASHIN", "LARC",
            "PLY FABRIC", "HSNFIBER", "HSNMATRIX", "TSINVMATRIX", "MAX STRAIN", "MAX STRESS", "TSAIWU",
            "TSAIWUE", "HYSTERESIS ENERGY", "MAXE", "MAXPE", "MAXPS", "MAXS", "MAXU", "QUADE", "QUADS",
            "QUADU", "USER"
        ] = Param(supported=("DUCTILE", "JOHNSON COOK"))
        definition: Optional[str] = Param(None, unsupported="DAMAGE INITIATION DEFINITION not supported")
        dependencies: int = Param(0, unsupported="DAMAGE INITIATION DEPENDENCIES not supported")
        frequency: Optional[int] = Param(None, control=True)
        position: Optional[str] = Param(None, control=True)
        properties: Optional[int] = Param(None, unsupported="DAMAGE INITIATION PROPERTIES not supported")
        tolerance: Optional[float] = Param(None, control=True)

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError("*DAMAGE INITIATION: no data line")
        if p.criterion == "JOHNSON COOK":
            if len(rows) != 1:
                raise InputError("*DAMAGE INITIATION, CRITERION=JOHNSON COOK takes one data line")
            v = _cells(rows[0], 8)
            d = v[:5]
            if any(x is None for x in d):
                raise InputError("*DAMAGE INITIATION, CRITERION=JOHNSON COOK: d1..d5 are all required; "
                                 "the Keywords Guide gives no default")
            extra = {}
            if d[4] != 0.0:          # temperature term in use
                if v[5] is None or v[6] is None:
                    raise InputError("*DAMAGE INITIATION, CRITERION=JOHNSON COOK: theta_melt and theta_transition "
                                     "are required when d5 is nonzero")
                extra.update(Tmelt=float(v[5]), Troom=float(v[6]))
            if d[3] != 0.0:          # rate term in use
                if v[7] is None:
                    raise InputError("*DAMAGE INITIATION, CRITERION=JOHNSON COOK: the reference strain rate is "
                                     "required when d4 is nonzero")
                extra["eps0"] = float(v[7])
            yield DamageInitiationRecord(criterion="JOHNSON_COOK", table=(tuple(float(x) for x in d),),
                                         extra=tuple(extra.items()), loc=ctx.loc)
            return
        # DUCTILE: (equivalent plastic strain eD, triaxiality, strain rate, temperature)
        pts = []
        for r in rows:
            v = _cells(r, 4)
            if v[0] is None:
                raise InputError("*DAMAGE INITIATION, CRITERION=DUCTILE: the equivalent plastic strain at "
                                 "initiation (data item 1) is required")
            if v[2] not in (None, 0.0):
                raise Unsupported("*DAMAGE INITIATION, CRITERION=DUCTILE: strain-rate dependence is not represented")
            if v[3] is not None:
                raise Unsupported("*DAMAGE INITIATION, CRITERION=DUCTILE: temperature dependence is not represented")
            pts.append((v[0], v[1]))
        if len(pts) == 1:
            table = ((float(pts[0][0]),),)                       # constant eD
        elif len(pts) == 2:
            if any(t is None for _, t in pts) or pts[0][1] == pts[1][1]:
                raise InputError("*DAMAGE INITIATION, CRITERION=DUCTILE: two points need distinct triaxialities")
            pts.sort(key=lambda x: x[1])
            table = tuple((float(t), float(e)) for e, t in pts)  # Material convention: (triaxiality, eD)
        else:
            raise Unsupported("*DAMAGE INITIATION, CRITERION=DUCTILE: more than two triaxiality points are not represented")
        yield DamageInitiationRecord(criterion="DUCTILE", table=table, loc=ctx.loc)


@applies(DamageInitiationRecord)
def _apply_damage_initiation(reader: AbaqusInputReader, rec: DamageInitiationRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.DamageInitiation(type=rec.criterion, table=rec.table, **dict(rec.extra))


# ─── 10. *DAMAGE EVOLUTION ───────────────────────────────────────────────────

@dataclass(frozen=True)
class DamageEvolutionRecord(Record):
    type: str
    Gf: float


@keyword("DAMAGE EVOLUTION", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DamageEvolution(Keyword):
    """Damage evolution by fracture energy (TYPE=ENERGY, linear or exponential softening)."""

    class Params(Schema):
        # TYPE is required (Keywords Guide); SOFTENING defaults to LINEAR
        type: Literal["DISCONTINUOUS", "DISPLACEMENT", "ENERGY", "HYSTERESIS ENERGY"] = Param(supported=("ENERGY",))
        softening: Literal["LINEAR", "EXPONENTIAL", "TABULAR"] = Param("LINEAR", supported=("LINEAR", "EXPONENTIAL"))
        degradation: Literal["MAXIMUM", "MULTIPLICATIVE"] = Param("MAXIMUM", unsupported="DAMAGE EVOLUTION DEGRADATION not supported")
        dependencies: int = Param(0, unsupported="DAMAGE EVOLUTION DEPENDENCIES not supported")
        mixed_mode_behavior: Optional[str] = Param(None, unsupported="MIXED MODE BEHAVIOR not supported")
        mode_mix_ratio: Optional[str] = Param(None, unsupported="MODE MIX RATIO not supported")
        power: Optional[float] = Param(None, unsupported="DAMAGE EVOLUTION POWER not supported")
        rate_dependent: bool = Param(False, unsupported="DAMAGE EVOLUTION RATE DEPENDENT not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError("*DAMAGE EVOLUTION: no data line (the fracture energy is required)")
        if len(rows) > 1:
            raise Unsupported("*DAMAGE EVOLUTION: several data lines (temperature dependence) are not represented")
        v = _cells(rows[0], 2)
        if v[0] is None or v[0] <= 0.0:
            raise InputError("*DAMAGE EVOLUTION: a positive fracture energy (data item 1) is required; "
                             "the Keywords Guide gives no default")
        if v[1] is not None:
            raise Unsupported("*DAMAGE EVOLUTION: temperature-dependent fracture energy is not represented")
        yield DamageEvolutionRecord(type=f"ENERGY_{p.softening}", Gf=float(v[0]), loc=ctx.loc)


@applies(DamageEvolutionRecord)
def _apply_damage_evolution(reader: AbaqusInputReader, rec: DamageEvolutionRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.DamageEvolution(type=rec.type, Gf=rec.Gf)


# ─── 11. *DRUCKER PRAGER ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DruckerPragerRecord(Record):
    beta: float
    psi: float


@keyword("DRUCKER PRAGER", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DruckerPrager(Keyword):
    """Extended Drucker-Prager plasticity model (linear criterion).

    Keywords Guide data line (SHEAR CRITERION=LINEAR): 1 beta [deg], 2 K (blank
    or 0.0 -> 1.0), 3 psi [deg], 4 temperature, 5.. field variables.
    """

    class Params(Schema):
        shear_criterion: Literal["LINEAR", "HYPERBOLIC", "EXPONENT FORM"] = Param("LINEAR", supported=("LINEAR",))
        eccentricity: Optional[float] = Param(None, unsupported="DRUCKER PRAGER ECCENTRICITY not supported")
        test_data: bool = Param(False, unsupported="DRUCKER PRAGER TEST DATA not supported")
        dependencies: int = Param(0, unsupported="DRUCKER PRAGER DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError("*DRUCKER PRAGER: no data line (friction angle and dilation angle are required)")
        if len(rows) > 1:
            raise Unsupported("*DRUCKER PRAGER: several data lines (temperature dependence) are not represented")
        v = _cells(rows[0], 4)
        if v[0] is None:
            raise InputError("*DRUCKER PRAGER: friction angle beta (data item 1) is required; the Keywords Guide gives no default")
        if v[2] is None:
            raise InputError("*DRUCKER PRAGER: dilation angle psi (data item 3) is required; the Keywords Guide gives no default")
        k = 1.0 if v[1] is None or v[1] == 0.0 else v[1]
        if k != 1.0:
            raise Unsupported(f"*DRUCKER PRAGER: K={k} (flow-stress ratio) is not represented; only K=1.0")
        if v[3] is not None:
            raise Unsupported("*DRUCKER PRAGER: temperature-dependent data is not represented")
        yield DruckerPragerRecord(beta=float(v[0]), psi=float(v[2]), loc=ctx.loc)


@applies(DruckerPragerRecord)
def _apply_drucker_prager(reader: AbaqusInputReader, rec: DruckerPragerRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        # cohesion d0 and hardening H come from *DRUCKER PRAGER HARDENING; the
        # Material default (d0=100) is not a deck value, so mark it unset (NaN)
        mat.DruckerPrager(beta=rec.beta, psi=rec.psi, d0=float("nan"), H=0.0)
        mat.extra_params["dp_hardening_missing"] = True


# ─── 12. *DRUCKER PRAGER HARDENING ───────────────────────────────────────────

@dataclass(frozen=True)
class DruckerPragerHardeningRecord(Record):
    type: str
    table: Tuple[Tuple[float, float], ...]


@keyword("DRUCKER PRAGER HARDENING", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DruckerPragerHardening(Keyword):
    """Hardening data (yield stress, plastic strain) for Drucker-Prager plasticity."""

    class Params(Schema):
        type: Literal["COMPRESSION", "TENSION", "SHEAR"] = "COMPRESSION"
        dependencies: int = Param(0, unsupported="DRUCKER PRAGER HARDENING DEPENDENCIES not supported")
        rate: Optional[str] = Param(None, unsupported="DRUCKER PRAGER HARDENING RATE not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError("*DRUCKER PRAGER HARDENING: no data line")
        table = []
        for r in rows:
            v = _cells(r, 3)
            if v[0] is None or v[1] is None:
                raise InputError("*DRUCKER PRAGER HARDENING: each line needs yield stress and plastic strain")
            if v[2] is not None:
                raise Unsupported("*DRUCKER PRAGER HARDENING: temperature-dependent data is not represented")
            table.append((float(v[0]), float(v[1])))
        if table[0][1] != 0.0:
            raise InputError("*DRUCKER PRAGER HARDENING: the first plastic strain must be zero (Keywords Guide)")
        yield DruckerPragerHardeningRecord(type=p.type, table=tuple(table), loc=ctx.loc)


@applies(DruckerPragerHardeningRecord)
def _apply_drucker_prager_hardening(reader: AbaqusInputReader, rec: DruckerPragerHardeningRecord) -> None:
    mat = reader.current_material
    if mat is None or "drucker_prager" not in mat.extra_params:
        reader.diagnostics.add("gap", "dp-hardening-without-drucker-prager",
                               "*DRUCKER PRAGER HARDENING: no *DRUCKER PRAGER landed in this material "
                               "(absent or refused above)", rec.loc)
        return
    dp = mat.extra_params["drucker_prager"]
    # linear model F = t - p tan(beta) - d: cohesion d from the test yield stress.
    # The DRUCKER PRAGER HARDENING page states only d = (sqrt3/2) tau (1+1/K) for
    # SHEAR; COMPRESSION / TENSION use the standard Abaqus Analysis Guide relations
    # (Extended Drucker-Prager models), not on the cached pages:
    #   COMPRESSION d = (1 - tan(beta)/3) sigma_c
    #   TENSION     d = (1/K + tan(beta)/3) sigma_t
    #   SHEAR       d = (sqrt(3)/2) (1 + 1/K) tau      (K = 1 here)
    tb = math.tan(math.radians(dp["beta"]))
    if rec.type == "COMPRESSION":
        f = 1.0 - tb / 3.0
    elif rec.type == "TENSION":
        f = 1.0 + tb / 3.0
    else:
        f = math.sqrt(3.0)
    d = [(f * s, e) for s, e in rec.table]
    if len(d) > 2:
        reader.diagnostics.add("gap", "dp-hardening-multilinear",
                               "*DRUCKER PRAGER HARDENING: more than two points; the model has linear hardening only",
                               rec.loc)
        mat.extra_params.pop("dp_hardening_missing", None)     # already reported
        return
    dp["d0"] = d[0][0]
    if len(d) == 1:
        dp["H"] = 0.0
    else:
        if d[1][1] <= d[0][1]:
            reader.diagnostics.add("error", "dp-hardening-strain",
                                   "*DRUCKER PRAGER HARDENING: plastic strain must increase", rec.loc)
            return
        dp["H"] = (d[1][0] - d[0][0]) / (d[1][1] - d[0][1])
    mat.extra_params.pop("dp_hardening_missing", None)


# ─── 13-17. CONCRETE DAMAGED PLASTICITY SUITE ────────────────────────────────
#
# The material's tables are (strain, value) pairs; the Keywords Guide gives
# (value, strain, rate, temperature), so every table is swapped on landing.
# An absent *CONCRETE TENSION/COMPRESSION DAMAGE means no damage (zero table).

_NO_DAMAGE = [(0.0, 0.0)]


def _try_finalize_cdp(reader: AbaqusInputReader, mat) -> None:
    ep = mat.extra_params
    if "cdp_params" not in ep or "cdp_compression" not in ep or "cdp_tension" not in ep:
        return
    p = ep["cdp_params"]
    try:
        mat.ConcreteDP(
            beta=p["beta"],
            psi=p["psi"],
            compression=ep["cdp_compression"],
            tension=ep["cdp_tension"],
            dt_table=ep.get("cdp_dt", _NO_DAMAGE),
            dc_table=ep.get("cdp_dc", _NO_DAMAGE),
        )
    except ValueError as e:
        reader.unsupported_keyword(f"CONCRETE_DAMAGED_PLASTICITY table rejected ({e})")


def _cdp_pairs(kw: str, rows, names: Tuple[str, str]) -> Tuple[Tuple[float, float], ...]:
    """Guide lines (value, strain, rate, temperature) -> ((strain, value), ...)."""
    if not rows:
        raise InputError(f"*{kw}: no data line")
    out = []
    for r in rows:
        v = _cells(r, 4)
        if v[0] is None or v[1] is None:
            raise InputError(f"*{kw}: each line needs {names[0]} and {names[1]}")
        if v[2] not in (None, 0.0):
            raise Unsupported(f"*{kw}: rate-dependent data is not represented")
        if v[3] is not None:
            raise Unsupported(f"*{kw}: temperature-dependent data is not represented")
        out.append((float(v[1]), float(v[0])))
    return tuple(out)


@dataclass(frozen=True)
class ConcreteDPRecord(Record):
    psi: float
    beta: float


@keyword("CONCRETE DAMAGED PLASTICITY", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteDamagedPlasticity(Keyword):
    """CDP flow potential / yield parameters: psi, eccentricity, fb0/fc0, Kc, viscosity."""

    class Params(Schema):
        dependencies: int = Param(0, unsupported="CDP DEPENDENCIES not supported")
        ref_length: Optional[float] = Param(None, control=True)

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError("*CONCRETE DAMAGED PLASTICITY: no data line (the dilation angle is required)")
        if len(rows) > 1:
            raise Unsupported("*CONCRETE DAMAGED PLASTICITY: several data lines (temperature dependence) not represented")
        v = _cells(rows[0], 6)
        if v[0] is None:
            raise InputError("*CONCRETE DAMAGED PLASTICITY: the dilation angle psi (data item 1) is required; "
                             "the Keywords Guide gives no default")
        # items 2-5: eccentricity, fb0/fc0, Kc, viscosity -- the material is a Drucker-Prager
        # backbone that does not use them; only the Guide defaults (blank, 0.0 or the default
        # value itself) can be accepted
        for i, nm, dflt in ((1, "eccentricity", 0.1), (2, "fb0/fc0", 1.16), (3, "Kc", 2.0 / 3.0), (4, "viscosity", 0.0)):
            if v[i] not in (None, 0.0) and abs(v[i] - dflt) > 1e-3:
                raise Unsupported(f"*CONCRETE DAMAGED PLASTICITY: {nm}={v[i]} is not represented "
                                  f"(Drucker-Prager backbone)")
        if v[5] is not None:
            raise Unsupported("*CONCRETE DAMAGED PLASTICITY: temperature-dependent data is not represented")
        yield Note("gap", "cdp-backbone-approximation",
                   "*CONCRETE DAMAGED PLASTICITY runs on a Drucker-Prager backbone (friction angle fixed at 20 deg, "
                   "no Lubliner/Kc yield surface, no cyclic stiffness recovery); the Abaqus defaults of eccentricity, "
                   "fb0/fc0, Kc and viscosity are not applied", loc=ctx.loc)
        yield ConcreteDPRecord(psi=float(v[0]), beta=20.0, loc=ctx.loc)


@applies(ConcreteDPRecord)
def _apply_cdp(reader: AbaqusInputReader, rec: ConcreteDPRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_params"] = {"psi": rec.psi, "beta": rec.beta}
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPCompressionRecord(Record):
    table: Tuple[Tuple[float, float], ...]


@keyword("CONCRETE COMPRESSION HARDENING", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteCompressionHardening(Keyword):
    """Compression hardening (yield stress, inelastic strain) for CDP."""

    class Params(Schema):
        dependencies: int = Param(0, unsupported="CDP COMPRESSION DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        yield CDPCompressionRecord(table=_cdp_pairs("CONCRETE COMPRESSION HARDENING", rows,
                                                    ("yield stress", "inelastic strain")), loc=ctx.loc)


@applies(CDPCompressionRecord)
def _apply_cdp_compression(reader: AbaqusInputReader, rec: CDPCompressionRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_compression"] = list(rec.table)
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPTensionRecord(Record):
    type: str
    table: Tuple[Tuple[float, float], ...]


@keyword("CONCRETE TENSION STIFFENING", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteTensionStiffening(Keyword):
    """Post-cracking (stress, cracking strain) for CDP."""

    class Params(Schema):
        type: Literal["STRAIN", "DISPLACEMENT", "GFI"] = Param("STRAIN", supported=("STRAIN",))
        dependencies: int = Param(0, unsupported="CDP TENSION DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        yield CDPTensionRecord(type=p.type, table=_cdp_pairs("CONCRETE TENSION STIFFENING", rows,
                                                             ("remaining stress", "cracking strain")), loc=ctx.loc)


@applies(CDPTensionRecord)
def _apply_cdp_tension(reader: AbaqusInputReader, rec: CDPTensionRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_tension"] = list(rec.table)
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPCompressionDamageRecord(Record):
    table: Tuple[Tuple[float, float], ...]


@keyword("CONCRETE COMPRESSION DAMAGE", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteCompressionDamage(Keyword):
    """Compression damage variable dc versus inelastic strain."""

    class Params(Schema):
        dependencies: int = Param(0, unsupported="CDP COMPRESSION DAMAGE DEPENDENCIES not supported")
        tension_recovery: Optional[float] = Param(None, unsupported="TENSION RECOVERY not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        yield CDPCompressionDamageRecord(table=_cdp_pairs("CONCRETE COMPRESSION DAMAGE", rows,
                                                          ("damage dc", "inelastic strain")), loc=ctx.loc)


@applies(CDPCompressionDamageRecord)
def _apply_cdp_dc(reader: AbaqusInputReader, rec: CDPCompressionDamageRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_dc"] = list(rec.table)
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPTensionDamageRecord(Record):
    type: str
    table: Tuple[Tuple[float, float], ...]


@keyword("CONCRETE TENSION DAMAGE", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteTensionDamage(Keyword):
    """Tension damage variable dt versus cracking strain."""

    class Params(Schema):
        type: Literal["STRAIN", "DISPLACEMENT"] = Param("STRAIN", supported=("STRAIN",))
        dependencies: int = Param(0, unsupported="CDP TENSION DAMAGE DEPENDENCIES not supported")
        compression_recovery: Optional[float] = Param(None, unsupported="COMPRESSION RECOVERY not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        yield CDPTensionDamageRecord(type=p.type, table=_cdp_pairs("CONCRETE TENSION DAMAGE", rows,
                                                                   ("damage dt", "cracking strain")), loc=ctx.loc)


@applies(CDPTensionDamageRecord)
def _apply_cdp_dt(reader: AbaqusInputReader, rec: CDPTensionDamageRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_dt"] = list(rec.table)
    _try_finalize_cdp(reader, mat)


# ─── 18. *MULLINS EFFECT ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MullinsRecord(Record):
    r: float
    m: float
    beta: float


@keyword("MULLINS EFFECT", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class MullinsEffect(Keyword):
    """Mullins effect constants r, m, beta (data line); stored, no kernel consumes them yet."""

    class Params(Schema):
        # R, M, BETA parameters are for TEST DATA INPUT only (Keywords Guide); the constants are data items
        r: Optional[float] = Param(None, unsupported="MULLINS EFFECT R applies only with TEST DATA INPUT")
        m: Optional[float] = Param(None, unsupported="MULLINS EFFECT M applies only with TEST DATA INPUT")
        beta: Optional[float] = Param(None, unsupported="MULLINS EFFECT BETA applies only with TEST DATA INPUT")
        dependencies: int = Param(0, unsupported="MULLINS EFFECT DEPENDENCIES not supported")
        properties: Optional[int] = Param(None, unsupported="MULLINS EFFECT PROPERTIES not supported")
        test_data_input: bool = Param(False, unsupported="MULLINS EFFECT TEST DATA INPUT not supported")
        user: bool = Param(False, unsupported="MULLINS EFFECT USER not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if len(rows) != 1:
            raise InputError("*MULLINS EFFECT takes one data line (r, m, beta)")
        v = _cells(rows[0], 4)
        if v[0] is None or v[1] is None:
            raise InputError("*MULLINS EFFECT: r and m (data items 1, 2) are required; the Keywords Guide gives no default")
        if v[3] is not None:
            raise Unsupported("*MULLINS EFFECT: temperature-dependent constants are not represented")
        # Guide: a blank beta is 0.0 in Abaqus/Standard (0.1 in Abaqus/Explicit); this is the implicit solver
        beta = 0.0 if v[2] is None else float(v[2])
        yield Note("gap", "mullins-not-connected",
                   "*MULLINS EFFECT: parsed and stored; no material kernel consumes it, so the solve has no Mullins softening",
                   loc=ctx.loc)
        yield MullinsRecord(r=float(v[0]), m=float(v[1]), beta=beta, loc=ctx.loc)


@applies(MullinsRecord)
def _apply_mullins(reader: AbaqusInputReader, rec: MullinsRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.MullinsEffect(table=((rec.r, rec.m, rec.beta),))


# ─── 19. *DENSITY ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DensityRecord(Record):
    density: float


@keyword("DENSITY", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Density(Keyword):
    """Material mass density."""

    class Params(Schema):
        dependencies: int = Param(0, unsupported="DENSITY DEPENDENCIES not supported")
        pore_fluid: bool = Param(False, unsupported="DENSITY PORE FLUID not supported")
        slurry: bool = Param(False, unsupported="DENSITY SLURRY not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError(f"*DENSITY at {ctx.loc}: missing density value")
        if len(rows) > 1:
            raise Unsupported("*DENSITY: several data lines (temperature dependence) are not represented")
        v = _cells(rows[0], 2)
        if v[0] is None:
            raise InputError(f"*DENSITY at {ctx.loc}: missing density value")
        if v[1] is not None:
            raise Unsupported("*DENSITY: temperature-dependent density is not represented")
        yield DensityRecord(density=float(v[0]), loc=ctx.loc)


@applies(DensityRecord)
def _apply_density(reader: AbaqusInputReader, rec: DensityRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.density = rec.density


# ─── 20. *TRS ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TRSRecord(Record):
    definition: str
    values: Tuple[float, float, float]


@keyword("TRS", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class TRS(Keyword):
    """Williams-Landel-Ferry shift (theta0, C1, C2); stored, no kernel consumes it yet."""

    class Params(Schema):
        definition: Literal["WLF", "ARRHENIUS", "USER", "TNM", "TABULAR"] = Param("WLF", supported=("WLF",))
        dependencies: int = Param(0, unsupported="TRS DEPENDENCIES not supported")
        properties: Optional[int] = Param(None, unsupported="TRS PROPERTIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if len(rows) != 1:
            raise InputError("*TRS, DEFINITION=WLF takes one data line (theta0, C1, C2)")
        v = _cells(rows[0], 3)
        if any(x is None for x in v) or len(rows[0].items) > 3:
            raise InputError("*TRS, DEFINITION=WLF: reference temperature, C1 and C2 are all required")
        yield Note("gap", "trs-not-connected",
                   "*TRS: parsed and stored; no material kernel consumes the shift function",
                   loc=ctx.loc)
        yield TRSRecord(definition=p.definition, values=tuple(float(x) for x in v), loc=ctx.loc)


@applies(TRSRecord)
def _apply_trs(reader: AbaqusInputReader, rec: TRSRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        t0, c1, c2 = rec.values
        mat.extra_params["trs"] = {"definition": rec.definition, "T_ref": t0, "C1": c1, "C2": c2}


# ─── 21. *ORIENTATION ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrientationRecord(Record):
    name: str
    system: str
    definition: str
    local_directions: int
    raw_rows: Tuple[Tuple[float, ...], ...]
    node_ids: Tuple[int, ...] = ()


@keyword("ORIENTATION", support="PARTIAL",
         contexts={KeywordContext.GLOBAL, KeywordContext.PART, KeywordContext.INSTANCE, KeywordContext.ASSEMBLY})
class Orientation(Keyword):
    """Local material orientation: points a, b (and origin c) by coordinates or node numbers."""

    class Params(Schema):
        name: str
        system: Literal["RECTANGULAR", "Z RECTANGULAR", "CYLINDRICAL", "SPHERICAL", "USER"] = Param(
            "RECTANGULAR", supported=("RECTANGULAR", "Z RECTANGULAR", "CYLINDRICAL", "SPHERICAL"))
        definition: Literal["COORDINATES", "NODES", "OFFSET TO NODES"] = "COORDINATES"
        local_directions: Optional[int] = None
        dispersion: Optional[str] = Param(None, unsupported="ORIENTATION DISPERSION not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        if not rows:
            raise InputError(f"*ORIENTATION {p.name}: no data line")
        raw_rows = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        first = _cells(rows[0], 9)
        n_pts = 6 if p.definition == "COORDINATES" else 2
        if any(x is None for x in first[:n_pts]):
            raise InputError(f"*ORIENTATION {p.name}: points a and b are required (first data line)")
        # second line: local direction and additional rotation alpha (defaults: direction 1, 0 degrees)
        if len(rows) > 1:
            second = _cells(rows[1], 2)
            if second[1] not in (None, 0.0):
                raise Unsupported(f"*ORIENTATION {p.name}: an additional rotation of {second[1]} deg is not represented")
        node_ids: tuple = ()
        if p.definition == "NODES":
            node_ids = tuple(int(v) for v in first[:3] if v is not None)
        yield OrientationRecord(
            name=p.name,
            system=p.system,
            definition=p.definition,
            local_directions=0 if p.local_directions is None else p.local_directions,
            raw_rows=raw_rows,
            node_ids=node_ids,
            loc=ctx.loc,
        )


@applies(OrientationRecord)
def _apply_orientation(reader: AbaqusInputReader, rec: OrientationRecord) -> None:
    frame = None
    if rec.system == "RECTANGULAR" and rec.definition == "COORDINATES" and rec.raw_rows and len(rec.raw_rows[0]) >= 6:
        row = rec.raw_rows[0]
        c = np.array(row[6:9], dtype=np.float64) if len(row) >= 9 else np.zeros(3)   # default origin: global origin
        a = np.array(row[:3], dtype=np.float64) - c
        b = np.array(row[3:6], dtype=np.float64) - c
        na = float(np.linalg.norm(a))
        if na > 1e-12:
            a = a / na
        nb = float(np.linalg.norm(b - a * float(np.dot(b, a))))
        if nb > 1e-12:
            b = (b - a * float(np.dot(b, a))) / nb
        cz = np.cross(a, b)
        frame = np.array([a, b, cz], dtype=np.float64)

    reader.model.Orientation(
        name=rec.name,
        system=rec.system,
        definition=rec.definition,
        frame=frame,
        raw_rows=rec.raw_rows,
        node_ids=rec.node_ids,
        local_directions=rec.local_directions,
    )
    if frame is None or rec.system != "RECTANGULAR" or rec.definition != "COORDINATES":
        msg = f"*ORIENTATION {rec.name}: parsed and stored; material-kernel consumption not connected"
        reader.diagnostics.add("gap", "orientation-not-connected", msg, rec.loc)
