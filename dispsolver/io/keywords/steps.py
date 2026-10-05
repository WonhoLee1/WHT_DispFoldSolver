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
from dispsolver.io.schema import InputError, Keyword, Param, Record, RowSchema, Schema, applies, keyword
from dispsolver.model.step import PredefinedField

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── STEPS, BCS & LOADS ──────────────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register(
    "STEP",
    valid_contexts={KeywordContext.GLOBAL},
    support="FULL",
    params={
        "NAME": ParamSpec("NAME"),
        "NLGEOM": ParamSpec("NLGEOM", default="NO", choices=["YES", "NO"]),
        "INC": ParamSpec("INC", type=int, default=100),
        "AMPLITUDE": ParamSpec("AMPLITUDE", default="STEP", choices=["STEP", "RAMP"], aliases=["AMP"]),
        "PERTURBATION": ParamSpec("PERTURBATION", supported=False, aliases=["PERT"]),
        "UNSYMM": ParamSpec("UNSYMM", supported=False, control=True, aliases=["UNSYM"]),
        "SOLVER": ParamSpec("SOLVER", supported=False),
    },
    description="Begin an analysis step",
)
def _handle_step(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    step_name = block.params.get("name", f"Step-{len(reader.model.steps)}")
    prev_nlgeom = getattr(reader.current_step, "nlgeom", False) if reader.current_step is not None else False
    if "nlgeom" in block.params:
        nlgeom = block.params["nlgeom"].lower() in ("yes", "true", "1")
    else:
        nlgeom = prev_nlgeom  # Abaqus standard: default NO, or inherit YES once engaged
    time_period = float(block.params.get("time period", block.params.get("time_period", 1.0)))

    # Previous step
    prev_step = reader.current_step.name if reader.current_step is not None else None
    reader.current_step = reader.model.Step(
        name=step_name,
        previous=prev_step,
        procedure="STATIC",
        time_period=time_period,
        nlgeom=nlgeom
    )
    reader.current_step._amplitude_param = block.params.get("amplitude", block.params.get("amp", None))
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


@GLOBAL_KEYWORD_REGISTRY.register(
    "STATIC",
    valid_contexts={KeywordContext.STEP},
    support="FULL",
    params={
        "DIRECT": ParamSpec("DIRECT", type=bool, default=False, supported=False, control=True),
        "ADIABATIC": ParamSpec("ADIABATIC", supported=False),
        "RIKS": ParamSpec("RIKS", supported=False),
        "STABILIZE": ParamSpec("STABILIZE", supported=False),
        "LONG TERM": ParamSpec("LONG TERM", supported=False, aliases=["LONG_TERM"]),
    },
    rows=RowSpec(column_types=[float, float, float, float], min_columns=1, max_columns=4, repeating=False),
    description="Static stress/displacement analysis procedure",
)
def _handle_static(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is not None:
        reader.current_step.procedure = "STATIC"
        _check_step_amplitude(reader, "STATIC", "RAMP")
        if block.data_lines:
            rows = _parse_data_rows(block.data_lines, dtype=float)
            if rows and len(rows[0]) >= 1:
                r = rows[0]
                if r[0] is not None and r[0] != "":
                    reader.current_step.dt_init = float(r[0])
                if len(r) >= 2 and r[1] is not None and r[1] != "":
                    reader.current_step.time_period = float(r[1])
                if len(r) >= 3 and r[2] is not None and r[2] != "":
                    reader.current_step.dt_min = float(r[2])
                if len(r) >= 4 and r[3] is not None and r[3] != "":
                    reader.current_step.dt_max = float(r[3])


@GLOBAL_KEYWORD_REGISTRY.register(
    "DYNAMIC",
    valid_contexts={KeywordContext.STEP},
    support="FULL",
    params={
        "DIRECT": ParamSpec("DIRECT", supported=False, control=True),
        "ALPHA": ParamSpec("ALPHA", supported=False),
        "HAFT": ParamSpec("HAFT", supported=False, control=True, aliases=["HAFTOL"]),
        "APPLICATION": ParamSpec("APPLICATION", supported=False),
        "EXPLICIT": ParamSpec("EXPLICIT", supported=False),
        "ELEMENT BY ELEMENT": ParamSpec("ELEMENT BY ELEMENT", supported=False),
        "NOHAF": ParamSpec("NOHAF", supported=False),
        "IMPROVED DT METHOD": ParamSpec("IMPROVED DT METHOD", supported=False),
    },
    rows=RowSpec(column_types=[float, float], min_columns=1, max_columns=2, repeating=False),
    description="Implicit/explicit dynamic analysis procedure",
)
def _handle_dynamic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is not None:
        reader.current_step.procedure = "DYNAMIC"
        _check_step_amplitude(reader, "DYNAMIC", "STEP")
        if block.data_lines:
            rows = _parse_data_rows(block.data_lines, dtype=float)
            if rows and len(rows[0]) >= 1:
                r = rows[0]
                if r[0] is not None and r[0] != "":
                    reader.current_step.dt_init = float(r[0])
                if len(r) >= 2 and r[1] is not None and r[1] != "":
                    reader.current_step.time_period = float(r[1])


@GLOBAL_KEYWORD_REGISTRY.register(
    "END STEP",
    valid_contexts={KeywordContext.STEP},
    aliases=["ENDSTEP"],
    support="FULL",
    params={},
    description="End step definition",
)
def _handle_end_step(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
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
    "CLOAD",
    valid_contexts={KeywordContext.STEP},
    support="FULL",
    params={
        "OP": ParamSpec("OP", default="MOD", choices=["MOD", "NEW"]),
        "AMPLITUDE": ParamSpec("AMPLITUDE", aliases=["AMP"]),
        "FOLLOWER": ParamSpec("FOLLOWER", supported=False),
        "LOAD CASE": ParamSpec("LOAD CASE", supported=False),
    },
    rows=RowSpec(column_types=[str, int, float], min_columns=3, max_columns=3),
    description="Concentrated force applied at nodes",
)
def _handle_cload(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is None:
        return
    amp = block.params.get("amplitude", None)
    op = block.params.get("op", "MOD").upper()
    if op == "NEW":
        for load_name in list(reader.current_step.loads.keys()):
            reader.current_step.deactivate_load(load_name)
    rows = _parse_data_rows(block.data_lines, dtype=str)
    for r in rows:
        if len(r) < 3:
            raise ValueError(f"*CLOAD line {r!r}: expected 'region, dof, magnitude'")
        if len(r) >= 3:
            region = str(r[0])
            dof = int(r[1])
            mag = float(r[2])
            if dof not in (1, 2, 3):
                # moments etc. used to be accepted and then dropped
                raise NotImplementedError(f"*CLOAD {region}: degree of freedom {dof} is not implemented "
                                          f"(continuum nodes carry dofs 1-3)")
            load_name = f"CLOAD_{region}_DOF{dof}"
            cf_kwargs = {}
            if dof == 1:
                cf_kwargs["cf1"] = mag
            elif dof == 2:
                cf_kwargs["cf2"] = mag
            elif dof == 3:
                cf_kwargs["cf3"] = mag
            reader.model.ConcentratedForce(name=load_name, createStepName=reader.current_step.name, region=region, amplitude=amp, **cf_kwargs)
            reader.current_step.loads[load_name].modified_params["op"] = op
            if region in reader.current_step.step_nsets:
                msg = (f"*CLOAD {load_name}: step-overlay region; solver load path "
                       f"resolves global sets only (P2)")
                reader.report("info", "step-overlay-region", msg, block)


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

