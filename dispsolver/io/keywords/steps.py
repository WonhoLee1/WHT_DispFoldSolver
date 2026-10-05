from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    ParamSpec,
    RowSpec,
)
from dispsolver.io.keyword_base import _parse_data_rows
from dispsolver.io.schema import InputError, Keyword, Lines, Param, Record, RowSchema, Schema, applies, keyword
from dispsolver.model.step import PredefinedField

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── STEPS, BCS & LOADS ──────────────────────────────────────────────────────

# ── *STEP / *STATIC / *DYNAMIC / *END STEP ─────────────────────────────────

@dataclass(frozen=True)
class StepRecord(Record):
    name: Optional[str]
    subheading: Optional[str]
    nlgeom: Optional[bool]                # None: inherit (Abaqus: once YES, stays YES)
    amplitude: Optional[str]              # requested default load variation (STEP / RAMP)


@keyword("STEP", support="FULL")
class StepKw(Keyword):
    """Begin an analysis step."""

    refuse_on_gap = False                 # see Keyword.refuse_on_gap
    Data = Lines()                        # optional subheading line

    class Params(Schema):
        name: Optional[str] = None
        nlgeom: Optional[bool] = None
        amplitude: Optional[Literal["STEP", "RAMP"]] = None
        # solution technique only: the increment cap, the unsymmetric matrix
        # switch, the initial-guess extrapolation and the SDI conversion
        inc: int = Param(100, control=True)
        unsymm: Optional[bool] = Param(None, control=True)
        extrapolation: Optional[Literal["LINEAR", "PARABOLIC", "VELOCITY PARABOLIC", "NO"]] = Param(
            None, control=True)
        convert_sdi: Optional[bool] = Param(None, control=True)

    def to_records(self, p, rows, ctx):
        yield StepRecord(p.name, rows[0] if rows else None, p.nlgeom, p.amplitude, loc=ctx.loc)


@applies(StepRecord)
def _apply_step(reader: AbaqusInputReader, rec: StepRecord):
    prev = reader.current_step
    nlgeom = rec.nlgeom if rec.nlgeom is not None else bool(getattr(prev, "nlgeom", False))
    reader.current_step = reader.model.Step(
        name=rec.name or f"Step-{len(reader.model.steps)}",
        previous=prev.name if prev is not None else None,
        procedure="STATIC",
        time_period=1.0,                  # set by the procedure's data line
        nlgeom=nlgeom)
    reader.current_step._amplitude_param = rec.amplitude
    reader.current_step.subheading = rec.subheading
    reader.context = KeywordContext.STEP
    reader.current_interaction = None


def _check_step_amplitude(reader: AbaqusInputReader, procedure: str, applied: str) -> None:
    """*STEP, AMPLITUDE= sets the default load/BC variation over the step.
    The solver applies RAMP in STATIC and STEP in DYNAMIC steps whatever the
    deck says, so a different request is recorded as a physics gap (it used to
    be accepted and ignored)."""
    want = str(getattr(reader.current_step, "_amplitude_param", "") or "").upper()
    if want and want != applied:
        reader.unsupported_param("STEP", f"AMPLITUDE={want} (with *{procedure})", "unsupported")


@dataclass(frozen=True)
class ProcedureRecord(Record):
    step: str
    procedure: str                        # STATIC / DYNAMIC
    dt_init: Optional[float]
    time_period: Optional[float]
    dt_min: Optional[float]
    dt_max: Optional[float]


class _IncrementRow(RowSchema):
    dt_init: Optional[float] = None
    time_period: Optional[float] = None
    dt_min: Optional[float] = None
    dt_max: Optional[float] = None


def _procedure(proc: str, rows, ctx):
    if len(rows) > 1:
        raise InputError(f"*{proc} at {ctx.loc}: one data line expected, got {len(rows)}")
    r = rows[0] if rows else _IncrementRow()
    yield ProcedureRecord(ctx.step, proc, r.dt_init, r.time_period, r.dt_min, r.dt_max, loc=ctx.loc)


@keyword("STATIC", support="FULL")
class Static(Keyword):
    """Static stress/displacement procedure."""

    class Params(Schema):
        direct: bool = Param(False, control=True)   # fixed increments: a solution technique

    Row = _IncrementRow

    def to_records(self, p, rows, ctx):
        return _procedure("STATIC", rows, ctx)


@keyword("DYNAMIC", support="FULL")
class Dynamic(Keyword):
    """Implicit dynamic procedure (Abaqus/Standard)."""

    class Params(Schema):
        direct: bool = Param(False, control=True)
        haftol: Optional[float] = Param(None, control=True)

    Row = _IncrementRow

    def to_records(self, p, rows, ctx):
        return _procedure("DYNAMIC", rows, ctx)


@applies(ProcedureRecord)
def _apply_procedure(reader: AbaqusInputReader, rec: ProcedureRecord):
    step = reader.model.steps[rec.step]
    step.procedure = rec.procedure
    _check_step_amplitude(reader, rec.procedure, "RAMP" if rec.procedure == "STATIC" else "STEP")
    for attr in ("dt_init", "time_period", "dt_min", "dt_max"):
        v = getattr(rec, attr)
        if v is not None:
            setattr(step, attr, float(v))


@dataclass(frozen=True)
class EndStepRecord(Record):
    pass


@keyword("END STEP", support="FULL", contexts={KeywordContext.STEP})
class EndStep(Keyword):
    """End of the step definition."""

    refuse_on_gap = False

    def to_records(self, p, rows, ctx):
        yield EndStepRecord(loc=ctx.loc)


@applies(EndStepRecord)
def _apply_end_step(reader: AbaqusInputReader, rec: EndStepRecord):
    reader.context = KeywordContext.GLOBAL


# ── *BOUNDARY: schema reference migration (see dispsolver/io/schema.py) ──

# Named boundary types -> constrained degrees of freedom (Abaqus Keywords
# Guide, *BOUNDARY). Only the translational dofs are listed: continuum nodes
# carry no rotation dof, and shell / beam elements are refused upstream. A
# rotational part of a symmetry type therefore has nothing to act on.
_NAMED_BC = {
    "ENCASTRE": (1, 2, 3, 4, 5, 6),
    "PINNED": (1, 2, 3),
    "XSYMM": (1,),
    "YSYMM": (2,),
    "ZSYMM": (3,),
    "XASYMM": (2, 3),
    "YASYMM": (1, 3),
    "ZASYMM": (1, 2),
}


@dataclass(frozen=True)
class BcResetRecord(Record):
    """OP=NEW: release every boundary condition active in the step first."""
    step: Optional[str]                   # None: model data (initial step)


@dataclass(frozen=True)
class BoundaryRecord(Record):
    step: Optional[str]
    region: str                           # node number or node set name (BIND resolves it)
    dofs: Tuple[int, ...]                 # 1-based
    value: float
    amplitude: Optional[str]
    op: str
    label: str                            # dof label used in the boundary condition name


@keyword("BOUNDARY", support="FULL")
class Boundary(Keyword):
    """Prescribed displacements on nodes or node sets."""

    class Params(Schema):
        op: Literal["MOD", "NEW"] = "MOD"
        amplitude: Optional[str] = None
        name: Optional[str] = None        # labels the condition; no effect on the model
        # VELOCITY / ACCELERATION used to be imposed as a displacement
        type: Literal["DISPLACEMENT", "VELOCITY", "ACCELERATION"] = Param(
            "DISPLACEMENT", supported=("DISPLACEMENT",))

    class Row(RowSchema):
        region: str
        first: str                        # first dof, or a named type (ENCASTRE, XSYMM, ...)
        last: Optional[int] = None
        value: Optional[float] = None

    def to_records(self, p, rows, ctx):
        if p.op == "NEW":
            yield BcResetRecord(ctx.step, loc=ctx.loc)
        for r in rows:
            named = r.first.upper()
            if named in _NAMED_BC:
                if r.last is not None or r.value is not None:
                    raise InputError(f"*BOUNDARY {r.region}, {named}: a named type takes no further values")
                dofs, value = _NAMED_BC[named], 0.0
                lo, hi = dofs[0], dofs[-1]
                label = f"DOF{lo}_{hi}" if dofs == tuple(range(lo, hi + 1)) else named
            else:
                try:
                    first = int(r.first)
                except ValueError:
                    raise InputError(f"*BOUNDARY {r.region}: {r.first!r} is neither a degree of freedom "
                                     f"nor a named type ({', '.join(_NAMED_BC)})") from None
                last = first if r.last is None else r.last
                if not 1 <= first <= last:
                    raise InputError(f"*BOUNDARY {r.region}: bad dof range {first}..{last}")
                dofs, value, label = tuple(range(first, last + 1)), (r.value or 0.0), f"DOF{first}_{last}"
            yield BoundaryRecord(ctx.step, r.region, dofs, value, p.amplitude, p.op, label, loc=r.loc)


def _bc_step(reader: AbaqusInputReader, step: Optional[str]):
    return reader.model.initial_step if step is None else reader.model.steps[step]


@applies(BcResetRecord)
def _apply_bc_reset(reader: AbaqusInputReader, rec: BcResetRecord):
    target = _bc_step(reader, rec.step)
    for bc_name in list(target.boundary_conditions.keys()):
        target.deactivate_bc(bc_name)


_DOF_KW = {1: "u1", 2: "u2", 3: "u3", 4: "ur1", 5: "ur2", 6: "ur3"}


@applies(BoundaryRecord)
def _apply_boundary(reader: AbaqusInputReader, rec: BoundaryRecord):
    target = _bc_step(reader, rec.step)
    kwargs = {_DOF_KW[d]: rec.value for d in rec.dofs if d in _DOF_KW}
    bc_name = f"BC_{rec.region}_{rec.label}"
    reader.model.DisplacementBC(name=bc_name, createStepName=target.name, region=rec.region,
                                amplitude=rec.amplitude, **kwargs)
    target.boundary_conditions[bc_name].modified_params["op"] = rec.op
    if any(d > 3 for d in rec.dofs):
        reader.diagnostics.add("info", "rotation-dofs-ignored",
                               f"*BOUNDARY {bc_name}: rotational DOFs stored but continuum nodes have no "
                               f"rotation target in the solver; translational part applied", rec.loc)


@GLOBAL_KEYWORD_REGISTRY.register(
    "TEMPERATURE",
    valid_contexts={KeywordContext.STEP, KeywordContext.GLOBAL},
    support="PARTIAL",
    params={
        "OP": ParamSpec("OP", default="MOD", choices=["MOD", "NEW"]),
        "AMPLITUDE": ParamSpec("AMPLITUDE", aliases=["AMP"]),
        "INPUT": ParamSpec("INPUT", supported=False),
        "SECTION SPECIFICATION": ParamSpec("SECTION SPECIFICATION", supported=False),
    },
    rows=RowSpec(column_types=[str, float], min_columns=2, max_columns=2),
    description="Prescribe nodal temperature field",
)
def _handle_temperature(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    rows = _parse_data_rows(block.data_lines, dtype=str)
    target_step = reader.current_step if reader.current_step is not None else reader.model.initial_step
    op = block.params.get("op", "MOD").upper()
    if op == "NEW":
        target_step.predefined_fields.clear()
    for r in rows:
        if len(r) >= 2:
            region = str(r[0])
            temp_val = float(r[1])
            pf = PredefinedField(name=f"TEMP_{region}", field_type="TEMPERATURE", region=region, values=temp_val)
            target_step.add_predefined_field(pf)


# ─── OUTPUT REQUESTS (FIELD & HISTORY) ───────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register(
    "OUTPUT",
    valid_contexts={KeywordContext.STEP},
    support="STUB",
    params={
        "FIELD": ParamSpec("FIELD", supported=True),
        "HISTORY": ParamSpec("HISTORY", supported=True, aliases=["HIST"]),
        "VARIABLE": ParamSpec("VARIABLE", aliases=["VAR"]),
        "FREQUENCY": ParamSpec("FREQUENCY", aliases=["FREQ", "FRE"]),
        "NUMBER INTERVAL": ParamSpec("NUMBER INTERVAL"),
        "TIME MARKS": ParamSpec("TIME MARKS", aliases=["TIME_MARKS", "TIMEMARKS"]),
        "TIME INTERVAL": ParamSpec("TIME INTERVAL", aliases=["TIME_INTERVAL", "TIME INT", "TIMEINT", "TIME INTERV"]),
        "TIME POINTS": ParamSpec("TIME POINTS", aliases=["TIME_POINTS"]),
        "FILTER": ParamSpec("FILTER", supported=False),
        "MODE LIST": ParamSpec("MODE LIST", aliases=["MODELIST"], supported=False),
        "OP": ParamSpec("OP", supported=False),
    },
    description="Output requests container (field or history)",
)
def _handle_output(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is None:
        return
    out_type = "FIELD"
    for k in block.params:
        if k.upper() == "HISTORY":
            out_type = "HISTORY"
            break
        elif k.upper() == "FIELD":
            out_type = "FIELD"
            break

    freq = block.params.get("frequency", None)
    frequency = int(freq) if freq is not None else 1
    num_intervals = block.params.get("number interval")
    num_intervals = int(num_intervals) if num_intervals is not None else None

    reader.current_step._current_output_type = out_type
    reader.current_step._current_output_freq = frequency
    reader.current_step._current_output_intervals = num_intervals


@GLOBAL_KEYWORD_REGISTRY.register(
    "NODE OUTPUT",
    valid_contexts={KeywordContext.STEP},
    aliases=["NODEOUTPUT"],
    support="STUB",
    params={
        "VARIABLE": ParamSpec("VARIABLE", aliases=["VAR"]),
        "NSET": ParamSpec("NSET"),
    },
    rows=RowSpec(column_types=[str]),
    description="Nodal output variable request",
)
def _handle_node_output(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is None:
        return
    out_type = getattr(reader.current_step, "_current_output_type", "FIELD")
    freq = getattr(reader.current_step, "_current_output_freq", 1)
    num_intervals = getattr(reader.current_step, "_current_output_intervals", None)

    nset = block.params.get("nset", None)
    variables = []
    for line in block.data_lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("**"):
            continue
        vars_in_line = [v.strip().upper() for v in line_clean.split(",") if v.strip()]
        variables.extend(vars_in_line)

    if not variables:
        variables = ["U", "RF"]

    if out_type == "FIELD":
        req_name = f"F-Output-Node-{reader.current_step.name}"
        if req_name in reader.current_step.field_output_requests:
            req = reader.current_step.field_output_requests[req_name]
            req.variables = list(dict.fromkeys(req.variables + variables))
        else:
            reader.current_step.FieldOutputRequest(
                name=req_name,
                frequency=freq,
                num_intervals=num_intervals,
                variables=variables,
                region=nset,
            )
    else:
        req_name = f"H-Output-Node-{reader.current_step.name}-{len(reader.current_step.history_output_requests)}"
        reader.current_step.HistoryOutputRequest(
            name=req_name,
            frequency=freq,
            variables=variables,
            region=nset,
        )


@GLOBAL_KEYWORD_REGISTRY.register(
    "ELEMENT OUTPUT",
    valid_contexts={KeywordContext.STEP},
    aliases=["ELEMENTOUTPUT"],
    support="STUB",
    params={
        "DIRECTIONS": ParamSpec("DIRECTIONS"),
        "POSITION": ParamSpec("POSITION", aliases=["POS"]),
        "VARIABLE": ParamSpec("VARIABLE", aliases=["VAR"]),
        "ELSET": ParamSpec("ELSET"),
        "REBAR": ParamSpec("REBAR", supported=False),
    },
    rows=RowSpec(column_types=[str]),
    description="Element output variable request",
)
def _handle_element_output(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is None:
        return
    out_type = getattr(reader.current_step, "_current_output_type", "FIELD")
    freq = getattr(reader.current_step, "_current_output_freq", 1)
    num_intervals = getattr(reader.current_step, "_current_output_intervals", None)

    elset = block.params.get("elset", None)
    variables = []
    for line in block.data_lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("**"):
            continue
        vars_in_line = [v.strip().upper() for v in line_clean.split(",") if v.strip()]
        variables.extend(vars_in_line)

    if not variables:
        variables = ["S", "E"]

    if out_type == "FIELD":
        req_name = f"F-Output-Elem-{reader.current_step.name}"
        if req_name in reader.current_step.field_output_requests:
            req = reader.current_step.field_output_requests[req_name]
            req.variables = list(dict.fromkeys(req.variables + variables))
        else:
            reader.current_step.FieldOutputRequest(
                name=req_name,
                frequency=freq,
                num_intervals=num_intervals,
                variables=variables,
                region=elset,
            )
    else:
        req_name = f"H-Output-Elem-{reader.current_step.name}-{len(reader.current_step.history_output_requests)}"
        reader.current_step.HistoryOutputRequest(
            name=req_name,
            frequency=freq,
            variables=variables,
            region=elset,
        )




# ─────────────────────────────────────────────────────────────────────────────

