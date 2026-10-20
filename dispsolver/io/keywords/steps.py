from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple

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
    closes_with = "END STEP"
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
    was_on = bool(getattr(prev, "nlgeom", False))
    nlgeom = rec.nlgeom if rec.nlgeom is not None else was_on
    if was_on and rec.nlgeom is False:
        # "Once the NLGEOM option has been switched on, it will be active during
        # all subsequent steps in the analysis." (Keywords Guide, *STEP)
        nlgeom = True
        reader.diagnostics.add("warning", "nlgeom-stays-on",
                               f"*STEP {rec.name or ''}: NLGEOM=NO after a step with NLGEOM=YES; "
                               f"Abaqus keeps geometric nonlinearity on", rec.loc)
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
    # *DYNAMIC only: the deck's integrator request, resolved against the
    # Keywords Guide defaults by _resolve_time_integration (None = omitted)
    application: Optional[str] = None
    time_integrator: Optional[str] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    initial_acceleration: bool = True
    # *STATIC only:
    stabilize: bool = False
    stabilize_factor: Optional[float] = None
    allsdvtol: Optional[float] = None


class _IncrementRow(RowSchema):
    dt_init: Optional[float] = None
    time_period: Optional[float] = None
    dt_min: Optional[float] = None
    dt_max: Optional[float] = None


def _procedure(proc: str, rows, ctx, p=None):
    if len(rows) > 1:
        raise InputError(f"*{proc} at {ctx.loc}: one data line expected, got {len(rows)}")
    r = rows[0] if rows else _IncrementRow()
    if proc == "DYNAMIC":
        # The Keywords Guide documents a default for neither the suggested
        # initial increment nor the time period of a transient dynamic step
        # (it does for the minimum and maximum increment), so an omitted one is
        # the deck's error, not a value to invent.
        for label, v in (("initial time increment", r.dt_init), ("time period", r.time_period)):
            if not v:
                raise InputError(f"*DYNAMIC at {ctx.loc}: {label} omitted; the Abaqus Keywords Guide "
                                 f"documents no default for it")
    if proc == "DYNAMIC" and p is not None:
        # Keywords Guide *DYNAMIC: "-1/2 <= alpha <= 0", "beta > 0", "gamma >= 1/2"
        if p.alpha is not None and not (-0.5 <= p.alpha <= 0.0):
            raise InputError(f"*DYNAMIC at {ctx.loc}: ALPHA={p.alpha} outside -1/2 <= alpha <= 0")
        if p.beta is not None and not p.beta > 0.0:
            raise InputError(f"*DYNAMIC at {ctx.loc}: BETA={p.beta} must be > 0")
        if p.gamma is not None and not p.gamma >= 0.5:
            raise InputError(f"*DYNAMIC at {ctx.loc}: GAMMA={p.gamma} must be >= 1/2")
        yield ProcedureRecord(ctx.step, proc, r.dt_init, r.time_period, r.dt_min, r.dt_max,
                              application=p.application, time_integrator=p.time_integrator,
                              alpha=p.alpha, beta=p.beta, gamma=p.gamma,
                              initial_acceleration=p.initial != "NO", loc=ctx.loc)
        return
    if proc == "STATIC" and p is not None:
        stab_enabled = False
        stab_factor = None
        allsdvtol = p.allsdtol if p.allsdtol is not None else p.allsdvtol
        if p.stabilize is not None:
            stab_enabled = True
            raw_s = str(p.stabilize).strip().upper()
            if raw_s in ("YES", "ON", "TRUE"):
                stab_factor = p.factor if p.factor is not None else 2e-4
            else:
                try:
                    stab_factor = float(raw_s)
                except ValueError:
                    raise InputError(f"*STATIC at {ctx.loc}: invalid value for STABILIZE: {p.stabilize}")
                if p.factor is not None:
                    stab_factor = float(p.factor)
        elif p.factor is not None:
            stab_enabled = True
            stab_factor = float(p.factor)

        yield ProcedureRecord(ctx.step, proc, r.dt_init, r.time_period, r.dt_min, r.dt_max,
                              stabilize=stab_enabled, stabilize_factor=stab_factor,
                              allsdvtol=allsdvtol, loc=ctx.loc)
        return
    yield ProcedureRecord(ctx.step, proc, r.dt_init, r.time_period, r.dt_min, r.dt_max, loc=ctx.loc)


@keyword("STATIC", support="FULL")
class Static(Keyword):
    """Static stress/displacement procedure."""

    class Params(Schema):
        direct: bool = Param(False, control=True)   # fixed increments: a solution technique
        stabilize: Optional[str] = None
        factor: Optional[float] = None
        allsdtol: Optional[float] = None
        allsdvtol: Optional[float] = Param(None, undocumented="common alias for ALLSDTOL")

    Row = _IncrementRow

    def to_records(self, p, rows, ctx):
        return _procedure("STATIC", rows, ctx, p)


@keyword("DYNAMIC", support="FULL")
class Dynamic(Keyword):
    """Implicit dynamic procedure (Abaqus/Standard)."""

    class Params(Schema):
        # The integrator (Analysis Guide "Implicit dynamic analysis", Table 1). Only the
        # HHT operators are implemented; the backward Euler operator of APPLICATION=
        # QUASI-STATIC / TIME INTEGRATOR=BWE is a gap (the block is refused).
        application: Optional[Literal["TRANSIENT FIDELITY", "MODERATE DISSIPATION", "QUASI-STATIC"]] = Param(
            None, supported=("TRANSIENT FIDELITY", "MODERATE DISSIPATION"))
        time_integrator: Optional[Literal["BWE", "HHT-TF", "HHT-MD"]] = Param(
            None, supported=("HHT-TF", "HHT-MD"))
        alpha: Optional[float] = None
        beta: Optional[float] = None
        gamma: Optional[float] = None
        initial: Optional[Literal["YES", "NO"]] = None     # YES (the default): computed at the step start
        # solution technique only: increment size control, half-increment residual
        # tolerance and the impact (velocity-jump) treatment of the increment
        direct: bool = Param(False, control=True)
        haftol: Optional[float] = Param(None, control=True)
        nohaf: bool = Param(False, control=True)
        halfinc_scale_factor: Optional[float] = Param(None, control=True)
        impact: Optional[Literal["AVERAGE TIME", "CURRENT TIME", "NO"]] = Param(None, control=True)
        incrementation: Optional[Literal["CONSERVATIVE", "AGGRESSIVE"]] = Param(None, control=True)

    Row = _IncrementRow

    def to_records(self, p, rows, ctx):
        return _procedure("DYNAMIC", rows, ctx, p)


def _has_contact(reader: AbaqusInputReader) -> bool:
    """Any contact interaction defined in the model or in a step read so far."""
    return bool(reader.model.interactions) or any(s.interactions for s in reader.model.steps.values())


def _dynamic_application(reader: AbaqusInputReader, step) -> str:
    """APPLICATION as written, else its Keywords Guide default: TRANSIENT FIDELITY "for
    problems without contact in the model", MODERATE DISSIPATION "for problems with
    contact in the model"."""
    req = getattr(step, "_dynamic_request", None) or {}
    if req.get("application"):
        return req["application"]
    return "MODERATE DISSIPATION" if _has_contact(reader) else "TRANSIENT FIDELITY"


def _dynamic_dt_max_default(reader: AbaqusInputReader, step, time_period: float) -> float:
    # *DYNAMIC, maximum increment: "If this value is zero, the default depends on the
    # APPLICATION setting. If APPLICATION = TRANSIENT FIDELITY, the maximum time increment
    # allowed is the time period of the step divided by 100. If APPLICATION = MODERATE
    # DISSIPATION, it is the time period of the step divided by 10."
    app = _dynamic_application(reader, step)
    return time_period / (10.0 if app == "MODERATE DISSIPATION" else 100.0)


# Analysis Guide "Implicit dynamic analysis", Table 1: the alpha of each HHT operator.
# beta and gamma follow from alpha (Keywords Guide *DYNAMIC: beta = (1-alpha)^2/4,
# gamma = 1/2 - alpha), which reproduces the table's 0.275625/0.55 and 0.5/0.91421.
_HHT_ALPHA = {"HHT TF": -0.05, "HHT MD": -0.41421}
_APPLICATION_INTEGRATOR = {"TRANSIENT FIDELITY": "HHT TF", "MODERATE DISSIPATION": "HHT MD"}


def _resolve_time_integration(reader: AbaqusInputReader, step) -> None:
    """The step's integrator: TIME INTEGRATOR as written, else the one APPLICATION
    selects; ALPHA as written, else the operator's; BETA / GAMMA as written, else
    computed from alpha. Stored as step.time_integration, which the solvers apply."""
    req = step._dynamic_request
    app = _dynamic_application(reader, step)
    integ = req.get("time_integrator") or _APPLICATION_INTEGRATOR[app]
    alpha = req["alpha"] if req.get("alpha") is not None else _HHT_ALPHA[integ]
    beta = req["beta"] if req.get("beta") is not None else (1.0 - alpha) ** 2 / 4.0
    gamma = req["gamma"] if req.get("gamma") is not None else 0.5 - alpha
    step.time_integration = {
        "name": integ.lower().replace(" ", "-"),          # hht-tf / hht-md
        "application": app,
        "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
        # INITIAL=NO: "initial accelerations are zero (first dynamic step) or carried
        # over from the previous dynamic step"; omitted: computed at the step start
        "initial_acceleration": bool(req.get("initial_acceleration", True)),
    }


@applies(ProcedureRecord)
def _apply_procedure(reader: AbaqusInputReader, rec: ProcedureRecord):
    """Increment data line -> step. An omitted or zero item takes the default the
    Keywords Guide documents, never the model's own Step defaults (those stay
    for models built through the Python API)."""
    step = reader.model.steps[rec.step]
    step.procedure = rec.procedure
    _check_step_amplitude(reader, rec.procedure, "RAMP" if rec.procedure == "STATIC" else "STEP")
    # time period: "If this entry is zero or is not specified, a default value of 1.0 is assumed."
    # (*STATIC; *DYNAMIC has no default and refuses an omitted one while parsing)
    time_period = float(rec.time_period) if rec.time_period else 1.0
    # initial increment: "If this entry is zero or is not specified, a default value that is
    # equal to the total time period of the step is assumed." (*STATIC)
    dt_init = float(rec.dt_init) if rec.dt_init else time_period
    # minimum increment: "If this entry is zero, a default value of the smaller of the suggested
    # initial time increment or 10^-5 times the total time period is assumed."
    dt_min = float(rec.dt_min) if rec.dt_min else min(dt_init, 1e-5 * time_period)
    step._dt_max_by_application = False
    if rec.procedure == "DYNAMIC":
        step._dynamic_request = {
            "application": rec.application, "time_integrator": rec.time_integrator,
            "alpha": rec.alpha, "beta": rec.beta, "gamma": rec.gamma,
            "initial_acceleration": rec.initial_acceleration,
        }
        _resolve_time_integration(reader, step)   # re-decided at *END STEP when APPLICATION is defaulted
    else:
        step.time_integration = {"name": "quasistatic"}
        if getattr(rec, "stabilize", False):
            step.stabilize = {
                "enabled": True,
                "factor": rec.stabilize_factor if rec.stabilize_factor is not None else 2e-4,
                "allsdvtol": rec.allsdvtol if rec.allsdvtol is not None else 0.05,
            }
        else:
            step.stabilize = None
    if rec.dt_max:
        dt_max = float(rec.dt_max)
    elif rec.procedure == "DYNAMIC":
        dt_max = _dynamic_dt_max_default(reader, step, time_period)
        step._dt_max_by_application = True        # re-decided at *END STEP, once contact is known
    else:
        # *STATIC: "If this value is not specified, no upper limit is imposed." The step itself
        # is the largest increment there can be.
        dt_max = time_period
    step.time_period, step.dt_init, step.dt_min, step.dt_max = time_period, dt_init, dt_min, dt_max


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
    step = reader.current_step
    if step is not None and getattr(step, "_dt_max_by_application", False):
        # contact is part of "the model" however late in the step it was defined
        step.dt_max = _dynamic_dt_max_default(reader, step, step.time_period)
    if step is not None and getattr(step, "_dynamic_request", None) is not None:
        _resolve_time_integration(reader, step)   # the defaulted APPLICATION depends on contact
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
                # one condition per contiguous dof run (YASYMM -> dofs 1 and 3), so
                # a later "N, 1, 1, v" replaces the dof-1 condition instead of
                # sitting next to it with a conflicting value
                runs, cur = [], [_NAMED_BC[named][0]]
                for d in _NAMED_BC[named][1:]:
                    if d == cur[-1] + 1:
                        cur.append(d)
                    else:
                        runs.append(cur)
                        cur = [d]
                runs.append(cur)
                for run in runs:
                    yield BoundaryRecord(ctx.step, r.region, tuple(run), 0.0, p.amplitude, p.op,
                                         f"DOF{run[0]}_{run[-1]}", loc=r.loc)
                continue
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

    # TIME INTERVAL / TIME POINTS (name of a *TIME POINTS table) / TIME MARKS (default YES: increments end
    # exactly on the requested times). FREQUENCY, NUMBER INTERVAL, TIME INTERVAL and TIME POINTS are
    # alternatives, so a time-based request does not also fire every increment.
    t_int = block.params.get("time interval", block.params.get("time_interval"))
    t_int = float(t_int) if t_int not in (None, True, "") else None
    tp_name = block.params.get("time points", block.params.get("time_points"))
    t_points = None
    if tp_name not in (None, True, ""):
        tables = getattr(reader, "time_point_tables", {})
        key = str(tp_name).strip().upper()
        if key not in tables:
            reader.diagnostics.add("error", "unknown-time-points",
                                   f"*OUTPUT, TIME POINTS={tp_name}: no *TIME POINTS table of that name", None)
        else:
            t_points = list(tables[key])
    marks = str(block.params.get("time marks", block.params.get("time_marks", "YES"))).strip().upper()
    if t_int is not None or t_points is not None or num_intervals is not None:
        frequency = None

    reader.current_step._current_output_type = out_type
    reader.current_step._current_output_freq = frequency
    reader.current_step._current_output_intervals = num_intervals
    reader.current_step._current_output_time = dict(
        time_interval=t_int, time_points=t_points, exact_time_points=(marks != "NO"))


@GLOBAL_KEYWORD_REGISTRY.register(
    "TIME POINTS",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP},
    aliases=["TIMEPOINTS"],
    support="FULL",
    params={
        "NAME": ParamSpec("NAME", required=True),
        "GENERATE": ParamSpec("GENERATE", type=bool, default=False),
    },
    rows=RowSpec(min_columns=0, max_columns=8),
    description="Table of times for output requests (*OUTPUT, TIME POINTS=name)",
)
def _handle_time_points(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    import numpy as np
    name = str(block.params.get("name", "")).strip().upper()
    rows = _parse_data_rows(block.data_lines, dtype=float, default=None)
    times = []
    if block.params.get("generate", False):
        for r in rows:
            vals = [v for v in r if v is not None]
            if len(vals) < 2:
                continue
            start, end = vals[0], vals[1]
            inc = vals[2] if len(vals) > 2 else (end - start)
            n = int(np.floor((end - start) / inc + 1e-9))
            times.extend(start + inc * np.arange(n + 1))
    else:
        for r in rows:
            times.extend(v for v in r if v is not None)
    if not hasattr(reader, "time_point_tables"):
        reader.time_point_tables = {}
    reader.time_point_tables[name] = sorted(float(t) for t in times)


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
    timing = getattr(reader.current_step, "_current_output_time", {})

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
                **timing,
            )
    else:
        req_name = f"H-Output-Node-{reader.current_step.name}-{len(reader.current_step.history_output_requests)}"
        reader.current_step.HistoryOutputRequest(
            name=req_name,
            frequency=freq,
            variables=variables,
            region=nset,
            time_interval=timing.get("time_interval"),
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
    timing = getattr(reader.current_step, "_current_output_time", {})

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
                **timing,
            )
    else:
        req_name = f"H-Output-Elem-{reader.current_step.name}-{len(reader.current_step.history_output_requests)}"
        reader.current_step.HistoryOutputRequest(
            name=req_name,
            frequency=freq,
            variables=variables,
            region=elset,
            time_interval=timing.get("time_interval"),
        )




# ─────────────────────────────────────────────────────────────────────────────

