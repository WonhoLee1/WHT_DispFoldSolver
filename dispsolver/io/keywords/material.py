"""Material keywords on the schema framework (B4 migration).

Every material keyword is a Keyword class with typed Params and Row schemas,
a pure to_records, and an applier registered per record type. Material options
map to numerical IDs and runtime props vectors strictly through
dispsolver.material.catalog.material_id and pack_props.
"""

from __future__ import annotations

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


# ─── 2. *ELASTIC ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ElasticRecord(Record):
    type: str
    dependencies: int
    table: Tuple[Tuple[float, ...], ...]


@keyword("ELASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Elastic(Keyword):
    """Linear elasticity: Young's modulus, Poisson's ratio, engineering constants, orthotropy."""

    class Params(Schema):
        type: Literal[
            "ISOTROPIC", "ENGINEERING CONSTANTS", "ORTHOTROPIC", "ANISOTROPIC",
            "TRACTION", "BILAMINA", "LAMINA", "SHORT FIBER", "COUPLED TRACTION",
            "TRANSVERSELY ISOTROPIC", "SHEAR"
        ] = Param("ISOTROPIC", supported=("ISOTROPIC", "ENGINEERING CONSTANTS", "ORTHOTROPIC", "ANISOTROPIC", "TRACTION", "LAMINA"))
        moduli: Literal["LONG TERM", "INSTANTANEOUS"] = Param("LONG TERM", unsupported="MODULI not supported")
        dependencies: int = 0
        cohesive_offset: Optional[str] = Param(None, unsupported="COHESIVE OFFSET not supported")
        compression_factor: Optional[float] = Param(None, unsupported="COMPRESSION FACTOR not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield ElasticRecord(type=p.type, dependencies=p.dependencies, table=table, loc=ctx.loc)


@applies(ElasticRecord)
def _apply_elastic(reader: AbaqusInputReader, rec: ElasticRecord) -> None:
    mat = reader.current_material
    if mat is None:
        mat = reader.model.Material("DEFAULT_ELASTIC")
        reader.current_material = mat
    flat_tokens = [v for r in rec.table for v in r]
    if rec.dependencies > 0:
        mat.Elastic(field_table=rec.table, type=rec.type, flat_tokens=flat_tokens)
    else:
        mat.Elastic(table=rec.table, type=rec.type, flat_tokens=flat_tokens)


# ─── 3. *EXPANSION ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExpansionRecord(Record):
    type: str
    zero: float
    table: Tuple[Tuple[float, ...], ...]


@keyword("EXPANSION", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Expansion(Keyword):
    """Thermal expansion coefficients."""

    class Params(Schema):
        type: Literal["ISO", "ORTHO", "ANISO"] = "ISO"
        zero: float = 0.0
        dependencies: int = Param(0, unsupported="EXPANSION DEPENDENCIES not supported")
        user: bool = Param(False, unsupported="EXPANSION USER not supported")
        pore_fluid: bool = Param(False, unsupported="EXPANSION PORE FLUID not supported")
        field: Optional[str] = Param(None, unsupported="EXPANSION FIELD not supported")
        liquid: bool = Param(False, unsupported="EXPANSION LIQUID not supported")
        properties: Optional[int] = Param(None, unsupported="EXPANSION PROPERTIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
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
    dependencies: int
    table: Tuple[Tuple[float, ...], ...]


@keyword("PLASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Plastic(Keyword):
    """Metal plasticity (yield stress, plastic strain, Johnson-Cook)."""

    class Params(Schema):
        hardening: Literal["ISOTROPIC", "KINEMATIC", "COMBINED", "JOHNSON COOK", "JOHNSON_COOK"] = "ISOTROPIC"
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
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        hard = p.hardening.replace(" ", "_")
        if hard == "JOHNSON_COOK":
            yield Note("gap", "rate-effect-dropped",
                       "*PLASTIC HARDENING=JOHNSON_COOK: quasi-static isothermal landing (C=0, dispatcher carries no rate)",
                       loc=ctx.loc)
        yield PlasticRecord(hardening=hard, dependencies=p.dependencies, table=table, loc=ctx.loc)


@applies(PlasticRecord)
def _apply_plastic(reader: AbaqusInputReader, rec: PlasticRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    if rec.hardening == "JOHNSON_COOK" and rec.table:
        r0 = rec.table[0]
        mat.JohnsonCook(
            A=float(r0[0]),
            B=float(r0[1]),
            n=float(r0[2]),
            m=float(r0[3]) if len(r0) > 3 else 0.0,
            T_melt=float(r0[4]) if len(r0) > 4 else 1793.15,
            T_room=float(r0[5]) if len(r0) > 5 else 293.15,
        )
    elif rec.dependencies > 0:
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
    """Creep material behavior."""

    class Params(Schema):
        law: Literal["STRAIN", "TIME", "HYPERBOLIC", "USER"] = "STRAIN"
        dependencies: int = Param(0, unsupported="CREEP DEPENDENCIES not supported")
        time: Optional[Literal["TOTAL", "CREEP"]] = Param("TOTAL", control=True)

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield CreepRecord(law=p.law, table=table, loc=ctx.loc)


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


@keyword("HYPERELASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Hyperelastic(Keyword):
    """Hyperelastic material models (Neo-Hookean, Mooney-Rivlin, Ogden, Yeoh, Arruda-Boyce, Marlow, etc.)."""

    class Params(Schema):
        type: Optional[Literal[
            "NEO HOOKE", "NEO_HOOKE", "NEO HOOKEAN", "NEO_HOOKEAN",
            "MOONEY RIVLIN", "MOONEY_RIVLIN", "POLYNOMIAL", "OGDEN", "YEOH",
            "ARRUDA BOYCE", "ARRUDA_BOYCE", "MARLOW",
            "REDUCED POLYNOMIAL", "REDUCED_POLYNOMIAL",
            "VAN DER WAALS", "VALANIS LANDEL"
        ]] = None
        n: int = 1
        moduli: Literal["INSTANTANEOUS", "LONG TERM"] = Param("INSTANTANEOUS", unsupported="HYPERELASTIC MODULI not supported")
        properties: Optional[int] = Param(None, unsupported="HYPERELASTIC PROPERTIES not supported")
        test_data_input: bool = Param(False, control=True)
        arruda_boyce: bool = False
        yeoh: bool = False
        mooney_rivlin: bool = False
        ogden: bool = False
        polynomial: bool = False
        neo_hooke: bool = False
        marlow: bool = False
        reduced_polynomial: bool = False
        van_der_waals: bool = Param(False, unsupported="VAN DER WAALS not supported")
        valanis_landel: bool = Param(False, unsupported="VALANIS LANDEL not supported")
        user: bool = Param(False, unsupported="HYPERELASTIC USER not supported")
        formulation: Optional[str] = Param(None, control=True, undocumented="Abaqus verification decks use FORMULATION")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        model = "NEO_HOOKEAN"
        if p.arruda_boyce:
            model = "ARRUDA_BOYCE"
        elif p.yeoh:
            model = "YEOH"
        elif p.mooney_rivlin:
            model = "MOONEY_RIVLIN"
        elif p.ogden:
            model = "OGDEN"
        elif p.polynomial:
            model = "POLYNOMIAL"
        elif p.neo_hooke:
            model = "NEO_HOOKEAN"
        elif p.marlow:
            model = "MARLOW"
        elif p.reduced_polynomial:
            model = "REDUCED_POLYNOMIAL"

        if p.type is not None:
            model = p.type.replace(" ", "_")
            if model == "NEO_HOOKE":
                model = "NEO_HOOKEAN"

        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield HyperelasticRecord(model=model, n=p.n, table=table, loc=ctx.loc)


@applies(HyperelasticRecord)
def _apply_hyperelastic(reader: AbaqusInputReader, rec: HyperelasticRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.Hyperelastic(model=rec.model, n=rec.n, table=rec.table)


# ─── 7. *VISCOELASTIC ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ViscoelasticRecord(Record):
    time: str
    type: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("VISCOELASTIC", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class Viscoelastic(Keyword):
    """Viscoelastic material behavior with Prony series."""

    class Params(Schema):
        time: Literal["PRONY", "FREQUENCY DATA", "TABULAR"] = "PRONY"
        frequency: Optional[Literal["CREEP TEST DATA", "RELAXATION TEST DATA", "FORMULA"]] = Param(None, unsupported="VISCOELASTIC FREQUENCY not supported")
        type: Literal["ISOTROPIC", "TRACTION"] = "ISOTROPIC"
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
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield ViscoelasticRecord(time=p.time, type=p.type, table=table, loc=ctx.loc)


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
    """Strain rate dependence for plastic materials."""

    class Params(Schema):
        type: Literal["POWER LAW", "POWER_LAW", "YIELD RATIO", "JOHNSON COOK"] = "POWER LAW"
        dependencies: int = Param(0, unsupported="RATE DEPENDENT DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        rd_type = p.type.replace(" ", "_")
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield RateDependentRecord(type=rd_type, table=table, loc=ctx.loc)


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


@keyword("DAMAGE INITIATION", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DamageInitiation(Keyword):
    """Damage initiation criteria (ductile, shear, Johnson-Cook)."""

    class Params(Schema):
        criterion: Literal[
            "DUCTILE", "JOHNSON COOK", "JOHNSON_COOK", "JC", "SHEAR", "MAXS", "MAXPE",
            "FRACTURE ENERGY", "HASHIN", "MK", "MSFLD", "FLD", "FLSD", "HOOSIEPOUR"
        ] = Param("DUCTILE", supported=("DUCTILE", "JOHNSON COOK", "JOHNSON_COOK", "JC"))
        definition: Optional[str] = Param(None, unsupported="DAMAGE INITIATION DEFINITION not supported")
        dependencies: int = Param(0, unsupported="DAMAGE INITIATION DEPENDENCIES not supported")
        frequency: Optional[int] = Param(None, control=True)
        position: Optional[str] = Param(None, control=True)
        properties: Optional[int] = Param(None, unsupported="DAMAGE INITIATION PROPERTIES not supported")
        tolerance: Optional[float] = Param(None, control=True)

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        crit = p.criterion.replace(" ", "_")
        if crit in ("JOHNSONCOOK", "JC"):
            crit = "JOHNSON_COOK"
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield DamageInitiationRecord(criterion=crit, table=table, loc=ctx.loc)


@applies(DamageInitiationRecord)
def _apply_damage_initiation(reader: AbaqusInputReader, rec: DamageInitiationRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.DamageInitiation(type=rec.criterion, table=rec.table)


# ─── 10. *DAMAGE EVOLUTION ───────────────────────────────────────────────────

@dataclass(frozen=True)
class DamageEvolutionRecord(Record):
    type: str
    Gf: float


@keyword("DAMAGE EVOLUTION", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DamageEvolution(Keyword):
    """Damage evolution after damage initiation."""

    class Params(Schema):
        type: Literal["ENERGY", "DISPLACEMENT"] = Param("ENERGY", supported=("ENERGY",))
        softening: Literal["LINEAR", "EXPONENTIAL", "TABULAR"] = Param("LINEAR", supported=("LINEAR", "EXPONENTIAL"))
        degradation: Literal["MAXIMUM", "MULTIPLICATIVE"] = Param("MAXIMUM", unsupported="DAMAGE EVOLUTION DEGRADATION not supported")
        dependencies: int = Param(0, unsupported="DAMAGE EVOLUTION DEPENDENCIES not supported")
        mixed_mode_behavior: Optional[str] = Param(None, unsupported="MIXED MODE BEHAVIOR not supported")
        mode_mix_ratio: Optional[str] = Param(None, unsupported="MODE MIX RATIO not supported")
        power: Optional[float] = Param(None, unsupported="DAMAGE EVOLUTION POWER not supported")
        rate_dependent: bool = Param(False, unsupported="DAMAGE EVOLUTION RATE DEPENDENT not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        gf = float(rows[0].items[0]) if rows and rows[0].items and rows[0].items[0] is not None else 1.0
        yield DamageEvolutionRecord(type=f"ENERGY_{p.softening}", Gf=gf, loc=ctx.loc)


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
    d0: float


@keyword("DRUCKER PRAGER", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DruckerPrager(Keyword):
    """Extended Drucker-Prager plasticity model."""

    class Params(Schema):
        shear_criterion: Optional[str] = Param(None, unsupported="DRUCKER PRAGER SHEAR CRITERION not supported")
        eccentricity: Optional[float] = Param(None, unsupported="DRUCKER PRAGER ECCENTRICITY not supported")
        test_data: bool = Param(False, unsupported="DRUCKER PRAGER TEST DATA not supported")
        dependencies: int = Param(0, unsupported="DRUCKER PRAGER DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        vals = rows[0].items if rows and rows[0].items else ()
        beta = float(vals[0]) if len(vals) > 0 and vals[0] is not None else 20.0
        psi = float(vals[1]) if len(vals) > 1 and vals[1] is not None else beta
        d0 = float(vals[2]) if len(vals) > 2 and vals[2] is not None else 100.0
        yield DruckerPragerRecord(beta=beta, psi=psi, d0=d0, loc=ctx.loc)


@applies(DruckerPragerRecord)
def _apply_drucker_prager(reader: AbaqusInputReader, rec: DruckerPragerRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.DruckerPrager(beta=rec.beta, psi=rec.psi, d0=rec.d0)


# ─── 12. *DRUCKER PRAGER HARDENING ───────────────────────────────────────────

@dataclass(frozen=True)
class DruckerPragerHardeningRecord(Record):
    type: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("DRUCKER PRAGER HARDENING", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class DruckerPragerHardening(Keyword):
    """Hardening data for Drucker-Prager plasticity."""

    class Params(Schema):
        type: Literal["COMPRESSION", "TENSION", "SHEAR"] = "COMPRESSION"
        dependencies: int = Param(0, unsupported="DRUCKER PRAGER HARDENING DEPENDENCIES not supported")
        rate: Optional[str] = Param(None, unsupported="DRUCKER PRAGER HARDENING RATE not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield DruckerPragerHardeningRecord(type=p.type, table=table, loc=ctx.loc)


@applies(DruckerPragerHardeningRecord)
def _apply_drucker_prager_hardening(reader: AbaqusInputReader, rec: DruckerPragerHardeningRecord) -> None:
    mat = reader.current_material
    if mat is None or "drucker_prager" not in mat.extra_params:
        return
    rows = rec.table
    if len(rows) >= 2 and len(rows[0]) >= 1 and len(rows[1]) >= 1:
        d0 = float(rows[0][0])
        e0 = float(rows[0][1]) if len(rows[0]) > 1 else 0.0
        d1 = float(rows[1][0])
        e1 = float(rows[1][1]) if len(rows[1]) > 1 else 0.0
        if e1 > e0:
            mat.extra_params["drucker_prager"]["H"] = (d1 - d0) / (e1 - e0)


# ─── 13-17. CONCRETE DAMAGED PLASTICITY SUITE ────────────────────────────────

def _try_finalize_cdp(reader: AbaqusInputReader, mat) -> None:
    ep = mat.extra_params
    if "concrete_dp" in ep or "cdp_params" not in ep:
        return
    if "cdp_compression" not in ep or "cdp_tension" not in ep:
        return
    if "cdp_dc" not in ep or "cdp_dt" not in ep:
        return
    p = ep["cdp_params"]
    try:
        mat.ConcreteDP(
            beta=p.get("beta", 20.0),
            psi=p.get("psi", 15.0),
            compression=ep["cdp_compression"],
            tension=ep["cdp_tension"],
            dt_table=ep["cdp_dt"],
            dc_table=ep["cdp_dc"],
        )
    except ValueError as e:
        reader.unsupported_keyword(f"CONCRETE_DAMAGED_PLASTICITY table rejected ({e})")


@dataclass(frozen=True)
class ConcreteDPRecord(Record):
    psi: float
    beta: float


@keyword("CONCRETE DAMAGED PLASTICITY", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteDamagedPlasticity(Keyword):
    """Concrete damaged plasticity yield and flow potential parameters."""

    class Params(Schema):
        dependencies: int = Param(0, unsupported="CDP DEPENDENCIES not supported")
        ref_length: Optional[float] = Param(None, control=True)

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        vals = rows[0].items if rows and rows[0].items else ()
        psi = float(vals[0]) if len(vals) > 0 and vals[0] is not None else 15.0
        yield ConcreteDPRecord(psi=psi, beta=20.0, loc=ctx.loc)


@applies(ConcreteDPRecord)
def _apply_cdp(reader: AbaqusInputReader, rec: ConcreteDPRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_params"] = {
        "psi": rec.psi,
        "beta": rec.beta,
    }
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPCompressionRecord(Record):
    table: Tuple[Tuple[float, ...], ...]


@keyword("CONCRETE COMPRESSION HARDENING", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteCompressionHardening(Keyword):
    """Compression hardening data for CDP."""

    class Params(Schema):
        dependencies: int = Param(0, unsupported="CDP COMPRESSION DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield CDPCompressionRecord(table=table, loc=ctx.loc)


@applies(CDPCompressionRecord)
def _apply_cdp_compression(reader: AbaqusInputReader, rec: CDPCompressionRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_compression"] = [tuple(v for v in r[:2] if v is not None) for r in rec.table]
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPTensionRecord(Record):
    type: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("CONCRETE TENSION STIFFENING", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteTensionStiffening(Keyword):
    """Tension stiffening data for CDP."""

    class Params(Schema):
        type: Literal["STRAIN", "DISPLACEMENT", "GFI"] = Param("STRAIN", supported=("STRAIN",))
        dependencies: int = Param(0, unsupported="CDP TENSION DEPENDENCIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield CDPTensionRecord(type=p.type, table=table, loc=ctx.loc)


@applies(CDPTensionRecord)
def _apply_cdp_tension(reader: AbaqusInputReader, rec: CDPTensionRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_tension"] = [tuple(v for v in r[:2] if v is not None) for r in rec.table]
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPCompressionDamageRecord(Record):
    table: Tuple[Tuple[float, ...], ...]


@keyword("CONCRETE COMPRESSION DAMAGE", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteCompressionDamage(Keyword):
    """Compression damage variable for CDP."""

    class Params(Schema):
        dependencies: int = Param(0, unsupported="CDP COMPRESSION DAMAGE DEPENDENCIES not supported")
        tension_recovery: Optional[float] = Param(None, unsupported="TENSION RECOVERY not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield CDPCompressionDamageRecord(table=table, loc=ctx.loc)


@applies(CDPCompressionDamageRecord)
def _apply_cdp_dc(reader: AbaqusInputReader, rec: CDPCompressionDamageRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_dc"] = [tuple(v for v in r[:2] if v is not None) for r in rec.table]
    _try_finalize_cdp(reader, mat)


@dataclass(frozen=True)
class CDPTensionDamageRecord(Record):
    type: str
    table: Tuple[Tuple[float, ...], ...]


@keyword("CONCRETE TENSION DAMAGE", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class ConcreteTensionDamage(Keyword):
    """Tension damage variable for CDP."""

    class Params(Schema):
        type: Literal["STRAIN", "DISPLACEMENT"] = Param("STRAIN", supported=("STRAIN",))
        dependencies: int = Param(0, unsupported="CDP TENSION DAMAGE DEPENDENCIES not supported")
        compression_recovery: Optional[float] = Param(None, unsupported="COMPRESSION RECOVERY not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield CDPTensionDamageRecord(type=p.type, table=table, loc=ctx.loc)


@applies(CDPTensionDamageRecord)
def _apply_cdp_dt(reader: AbaqusInputReader, rec: CDPTensionDamageRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    mat.extra_params["cdp_dt"] = [tuple(v for v in r[:2] if v is not None) for r in rec.table]
    _try_finalize_cdp(reader, mat)


# ─── 18. *MULLINS EFFECT ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MullinsRecord(Record):
    r: float
    m: float
    beta: float
    table: Tuple[Tuple[float, ...], ...]


@keyword("MULLINS EFFECT", support="FULL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class MullinsEffect(Keyword):
    """Mullins effect damage model for filled elastomers."""

    class Params(Schema):
        r: float = 2.0
        m: float = 0.5
        beta: float = 0.1
        dependencies: int = Param(0, unsupported="MULLINS EFFECT DEPENDENCIES not supported")
        properties: Optional[int] = Param(None, unsupported="MULLINS EFFECT PROPERTIES not supported")
        test_data_input: bool = Param(False, control=True)
        user: bool = Param(False, unsupported="MULLINS EFFECT USER not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        table = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        yield MullinsRecord(r=p.r, m=p.m, beta=p.beta, table=table, loc=ctx.loc)


@applies(MullinsRecord)
def _apply_mullins(reader: AbaqusInputReader, rec: MullinsRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.MullinsEffect(r=rec.r, m=rec.m, beta=rec.beta, table=rec.table if rec.table else None)


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
        if not rows or not rows[0].items or rows[0].items[0] is None:
            raise InputError(f"*DENSITY at {ctx.loc}: missing density value")
        yield DensityRecord(density=float(rows[0].items[0]), loc=ctx.loc)


@applies(DensityRecord)
def _apply_density(reader: AbaqusInputReader, rec: DensityRecord) -> None:
    mat = reader.current_material
    if mat is not None:
        mat.density = rec.density


# ─── 20. *TRS ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TRSRecord(Record):
    definition: str
    values: Tuple[float, ...]


@keyword("TRS", support="PARTIAL", contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
class TRS(Keyword):
    """Temperature-time shift for viscoelastic materials (WLF, Arrhenius)."""

    class Params(Schema):
        definition: Literal["WLF", "ARRHENIUS", "USER"] = "WLF"
        dependencies: int = Param(0, unsupported="TRS DEPENDENCIES not supported")
        properties: Optional[int] = Param(None, unsupported="TRS PROPERTIES not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        vals = tuple(float(v) for r in rows for v in r.items if v is not None)
        yield TRSRecord(definition=p.definition, values=vals, loc=ctx.loc)


@applies(TRSRecord)
def _apply_trs(reader: AbaqusInputReader, rec: TRSRecord) -> None:
    mat = reader.current_material
    if mat is None:
        return
    vals = rec.values
    if len(vals) >= 3:
        mat.extra_params["trs"] = {"definition": rec.definition, "T_ref": vals[0], "C1": vals[1], "C2": vals[2]}
    else:
        mat.extra_params["trs"] = {"definition": rec.definition, "values": vals}


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
    """Local material orientation coordinate system."""

    class Params(Schema):
        name: str
        system: Literal["RECTANGULAR", "CYLINDRICAL", "SPHERICAL", "ZCYLINDRICAL"] = "RECTANGULAR"
        definition: Literal["COORDINATES", "NODES", "OFFSET TO NODES"] = "COORDINATES"
        local_directions: int = 1
        dispersion: Optional[str] = Param(None, unsupported="ORIENTATION DISPERSION not supported")

    Row = FloatRow

    def to_records(self, p, rows, ctx) -> Iterable[Record]:
        raw_rows = tuple(tuple(v for v in r.items if v is not None) for r in rows)
        node_ids: tuple = ()
        if p.definition == "NODES":
            node_ids = tuple(int(v) for r in raw_rows for v in r)
        yield OrientationRecord(
            name=p.name,
            system=p.system,
            definition=p.definition,
            local_directions=p.local_directions,
            raw_rows=raw_rows,
            node_ids=node_ids,
            loc=ctx.loc,
        )


@applies(OrientationRecord)
def _apply_orientation(reader: AbaqusInputReader, rec: OrientationRecord) -> None:
    frame = None
    if rec.system == "RECTANGULAR" and rec.raw_rows and len(rec.raw_rows[0]) >= 6:
        a = np.array(rec.raw_rows[0][:3], dtype=np.float64)
        b = np.array(rec.raw_rows[0][3:6], dtype=np.float64)
        na = float(np.linalg.norm(a))
        if na > 1e-12:
            a = a / na
        nb = float(np.linalg.norm(b - a * float(np.dot(b, a))))
        if nb > 1e-12:
            b = (b - a * float(np.dot(b, a))) / nb
        c = np.cross(a, b)
        frame = np.array([a, b, c], dtype=np.float64)

    reader.model.Orientation(
        name=rec.name,
        system=rec.system,
        definition=rec.definition,
        frame=frame,
        raw_rows=rec.raw_rows,
        node_ids=rec.node_ids,
        local_directions=rec.local_directions,
    )
    if frame is None or rec.system != "RECTANGULAR" or rec.definition == "NODES":
        msg = f"*ORIENTATION {rec.name}: parsed and stored; material-kernel consumption not connected"
        reader.diagnostics.add("gap", "orientation-not-connected", msg, rec.loc)
