from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    ParamSpec,
    RowSpec,
)
from dispsolver.io.keyword_base import _parse_data_rows
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
        "TIME PERIOD": ParamSpec("TIME PERIOD", type=float, default=1.0, aliases=["TIME_PERIOD"]),
        "PERTURBATION": ParamSpec("PERTURBATION", supported=False, aliases=["PERT"]),
        "UNSYMM": ParamSpec("UNSYMM", supported=False, aliases=["UNSYM"]),
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
    reader.context = KeywordContext.STEP
    reader.current_interaction = None


@GLOBAL_KEYWORD_REGISTRY.register(
    "STATIC",
    valid_contexts={KeywordContext.STEP},
    support="FULL",
    params={
        "DIRECT": ParamSpec("DIRECT", type=bool, default=False, supported=False),
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
        "DIRECT": ParamSpec("DIRECT", supported=False),
        "ALPHA": ParamSpec("ALPHA", supported=False),
        "HAFT": ParamSpec("HAFT", supported=False, aliases=["HAFTOL"]),
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


@GLOBAL_KEYWORD_REGISTRY.register(
    "BOUNDARY",
    valid_contexts={KeywordContext.STEP, KeywordContext.GLOBAL},
    support="FULL",
    params={
        "TYPE": ParamSpec("TYPE", default="DISPLACEMENT", choices=["DISPLACEMENT", "VELOCITY", "ACCELERATION"]),
        "OP": ParamSpec("OP", default="MOD", choices=["MOD", "NEW"]),
        "AMPLITUDE": ParamSpec("AMPLITUDE", aliases=["AMP"]),
        "FIXED": ParamSpec("FIXED", supported=False),
        "USER": ParamSpec("USER", supported=False),
        "LOAD CASE": ParamSpec("LOAD CASE", supported=False),
        "SUBMODEL": ParamSpec("SUBMODEL", supported=False),
    },
    rows=RowSpec(column_types=[str, int, int, float], min_columns=2, max_columns=4),
    description="Prescribe boundary condition on nodes or sets",
)
def _handle_boundary(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    target_step = reader.current_step if reader.current_step is not None else reader.model.initial_step
    amp = block.params.get("amplitude", None)
    op = block.params.get("op", "MOD").upper()
    bc_type = block.params.get("type", "DISPLACEMENT").upper()
    if bc_type != "DISPLACEMENT":
        # A VELOCITY/ACCELERATION history used to be imposed as a displacement.
        raise NotImplementedError(f"*BOUNDARY, TYPE={bc_type} is not implemented (displacement only)")
    if "user" in block.params or "fixed" in block.params:
        raise NotImplementedError(f"*BOUNDARY parameters {sorted(k for k in block.params if k in ('user', 'fixed'))} "
                                  f"are not implemented")
    if op == "NEW":
        for bc_name in list(target_step.boundary_conditions.keys()):
            target_step.deactivate_bc(bc_name)

    for line in block.data_lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("**"):
            continue
        parts = [p.strip() for p in line_clean.split(",")]
        while len(parts) > 1 and parts[-1] == "":
            parts.pop()
        if len(parts) < 2:
            raise ValueError(f"*BOUNDARY line {line_clean!r}: missing degree of freedom")
        region = str(parts[0])
        type_str = str(parts[1]).strip().upper()
        if type_str in ("YASYMM", "XASYMM", "ZASYMM"):
            raise NotImplementedError(f"*BOUNDARY {type_str} is not implemented")
        if type_str == "ENCASTRE":
            first_dof, last_dof, val = 1, 6, 0.0
        elif type_str == "PINNED":
            first_dof, last_dof, val = 1, 3, 0.0
        elif type_str == "XSYMM":
            first_dof, last_dof, val = 1, 1, 0.0
        elif type_str == "YSYMM":
            first_dof, last_dof, val = 2, 2, 0.0
        elif type_str == "ZSYMM":
            first_dof, last_dof, val = 3, 3, 0.0
        else:
            try:
                first_dof = int(parts[1])
                last_dof = int(parts[2]) if len(parts) >= 3 and parts[2] != "" else first_dof
                val = float(parts[3]) if len(parts) >= 4 and parts[3] != "" else 0.0
            except (ValueError, TypeError):
                # used to `continue`, i.e. drop the BC line silently
                raise ValueError(f"*BOUNDARY line {line_clean!r}: expected 'region, first dof[, last dof"
                                 f"[, value]]' or a named type (ENCASTRE, PINNED, XSYMM, ...)")

        kwargs = {}
        for dof in range(first_dof, last_dof + 1):
            if dof == 1:
                kwargs["u1"] = val
            elif dof == 2:
                kwargs["u2"] = val
            elif dof == 3:
                kwargs["u3"] = val
            elif dof == 4:
                kwargs["ur1"] = val
            elif dof == 5:
                kwargs["ur2"] = val
            elif dof == 6:
                kwargs["ur3"] = val

        bc_name = f"BC_{region}_DOF{first_dof}_{last_dof}"
        reader.model.DisplacementBC(name=bc_name, createStepName=target_step.name, region=region, amplitude=amp, **kwargs)
        target_step.boundary_conditions[bc_name].modified_params["op"] = op
        if any(kwargs.get(k) is not None for k in ("ur1", "ur2", "ur3")):
            msg = (f"*BOUNDARY {bc_name}: rotational DOFs stored but continuum nodes "
                   f"have no rotation target in the solver; translational part applied")
            warnings.warn(msg)
            reader.warnings.append(msg)


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
                warnings.warn(msg)
                reader.warnings.append(msg)


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
        "NUMBER INTERVALS": ParamSpec("NUMBER INTERVALS", aliases=["NUMBER_INTERVALS", "NUMBER INTERVAL", "NUM", "NUMBER INT", "NUMBER"]),
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
    num_intervals = block.params.get("number intervals", block.params.get("number_intervals", None))
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

