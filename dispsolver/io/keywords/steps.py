from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext
from dispsolver.io.keyword_base import _parse_data_rows
from dispsolver.model.step import PredefinedField

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── STEPS, BCS & LOADS ──────────────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("STEP", valid_contexts={KeywordContext.GLOBAL})
def _handle_step(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    step_name = block.params.get("name", f"Step-{len(reader.model.steps)}")
    nlgeom = block.params.get("nlgeom", "yes").lower() == "yes"
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


@GLOBAL_KEYWORD_REGISTRY.register("STATIC", valid_contexts={KeywordContext.STEP})
def _handle_static(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is not None:
        reader.current_step.procedure = "STATIC"
        if block.data_lines:
            rows = _parse_data_rows(block.data_lines, dtype=float)
            if rows and len(rows[0]) >= 1:
                r = rows[0]
                reader.current_step.dt_init = float(r[0])
                if len(r) >= 2:
                    reader.current_step.time_period = float(r[1])
                if len(r) >= 3:
                    reader.current_step.dt_min = float(r[2])
                if len(r) >= 4:
                    reader.current_step.dt_max = float(r[3])


@GLOBAL_KEYWORD_REGISTRY.register("DYNAMIC", valid_contexts={KeywordContext.STEP})
def _handle_dynamic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is not None:
        reader.current_step.procedure = "DYNAMIC"
        if block.data_lines:
            rows = _parse_data_rows(block.data_lines, dtype=float)
            if rows and len(rows[0]) >= 1:
                r = rows[0]
                reader.current_step.dt_init = float(r[0])
                if len(r) >= 2:
                    reader.current_step.time_period = float(r[1])


@GLOBAL_KEYWORD_REGISTRY.register("END STEP", valid_contexts={KeywordContext.STEP}, aliases=["ENDSTEP"])
def _handle_end_step(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    reader.context = KeywordContext.GLOBAL


@GLOBAL_KEYWORD_REGISTRY.register("BOUNDARY", valid_contexts={KeywordContext.STEP, KeywordContext.GLOBAL})
def _handle_boundary(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    rows = _parse_data_rows(block.data_lines, dtype=str)
    target_step = reader.current_step if reader.current_step is not None else reader.model.initial_step
    amp = block.params.get("amplitude", None)
    op = block.params.get("op", "MOD").upper()
    if op == "NEW":
        for bc_name in list(target_step.boundary_conditions.keys()):
            target_step.deactivate_bc(bc_name)

    for r in rows:
        if len(r) < 2:
            continue
        region = str(r[0])
        first_dof = int(r[1])
        last_dof = int(r[2]) if len(r) >= 3 and r[2] != "" else first_dof
        val = float(r[3]) if len(r) >= 4 else 0.0

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


@GLOBAL_KEYWORD_REGISTRY.register("CLOAD", valid_contexts={KeywordContext.STEP})
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
        if len(r) >= 3:
            region = str(r[0])
            dof = int(r[1])
            mag = float(r[2])
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


@GLOBAL_KEYWORD_REGISTRY.register("TEMPERATURE", valid_contexts={KeywordContext.STEP, KeywordContext.GLOBAL})
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


# ─────────────────────────────────────────────────────────────────────────────

