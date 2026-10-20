"""Solution control and contact stabilization keywords.

Implements Abaqus keywords:
- *CONTROLS: Solution controls (time incrementation, field convergence, line search) -> Model.SolutionControl
- *CONTACT CONTROLS: Contact controls and stabilization -> Model.ContactControl
- *CONTACT STABILIZATION: Contact stabilization properties -> Model.ContactStabilization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_base import _parse_data_rows
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    ParamSpec,
    RowSpec,
)

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader


@GLOBAL_KEYWORD_REGISTRY.register(
    "CONTROLS",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP},
    support="FULL",
    params={
        "PARAMETERS": ParamSpec(
            "PARAMETERS",
            default="TIME INCREMENTATION",
            choices=["TIME INCREMENTATION", "FIELD", "LINE SEARCH", "CONSTRAINTS", "STEP"],
            aliases=["PARAMETER"],
            control=True,
        ),
        "ANALYSIS": ParamSpec(
            "ANALYSIS",
            default="DEFAULT",
            choices=["DEFAULT", "DISCONTINUOUS"],
            control=True,
        ),
        "RESET": ParamSpec("RESET", type=bool, default=False, control=True),
        # Keywords Guide: FIELD defaults to GLOBAL (all fields), not DISPLACEMENT. The handler does not
        # read it (CONVERGENCE CHECK and the data lines apply to the displacement solve).
        "FIELD": ParamSpec("FIELD", default="GLOBAL", control=True),
        "TYPE": ParamSpec("TYPE", supported=False, control=True),
        "CONVERGENCE CHECK": ParamSpec(
            "CONVERGENCE CHECK",
            default="STRICT",
            choices=["STRICT", "MODERATE", "RELAXED", "CONCEPT DESIGN", "STANDARD", "OFF"],
            aliases=["CONVERGENCE_CHECK"],
            control=True,
        ),
    },
    rows=RowSpec(min_columns=0, max_columns=8),
    description="Specify solution control parameters and convergence criteria",
)
def _handle_controls(reader: AbaqusInputReader, block: AbaqusKeywordBlock) -> None:
    # 1. Resolve target SolutionControl
    step = getattr(reader, "current_step", None)
    if step is not None:
        ctrl = getattr(step, "control", None)
        if ctrl is None:
            ctrl_name = f"{step.name}_CONTROL"
            ctrl = reader.model.SolutionControl(name=ctrl_name)
            step.control = ctrl
        reader.model.solution_controls[ctrl.name] = ctrl
    else:
        ctrl = reader.model.solution_controls.get("DEFAULT")
        if ctrl is None:
            ctrl = reader.model.SolutionControl(name="DEFAULT")
            reader.model.solution_controls["DEFAULT"] = ctrl


    # 2. Check RESET
    if block.params.get("reset", False) or "reset" in block.params:
        ctrl.apply_preset("STANDARD")
        ctrl.set_convergence_check("STRICT")

    # 3. Check ANALYSIS=DISCONTINUOUS
    analysis = str(block.params.get("analysis", "DEFAULT")).strip().upper()
    if analysis == "DISCONTINUOUS":
        ctrl.apply_preset("RELAXED")

    # 4. Check CONVERGENCE CHECK
    conv_check = block.params.get("convergence check", block.params.get("convergence_check"))
    if conv_check:
        # Keywords Guide semantics (which tests apply, how they are joined) -- not a tolerance
        # preset. The tolerances stay whatever the FIELD data line / ANALYSIS= set.
        if str(conv_check).strip().upper() != "OFF":
            ctrl.set_convergence_check(conv_check)

    # 5. Parse data lines according to PARAMETERS type
    param_type = str(block.params.get("parameters", "TIME INCREMENTATION")).strip().upper()
    rows = _parse_data_rows(block.data_lines, dtype=float, default=None)

    if param_type == "TIME INCREMENTATION" and rows:
        row0 = rows[0]
        t = ctrl.time_incrementation
        if len(row0) > 0 and row0[0] is not None:
            t.I_0 = int(row0[0])
        if len(row0) > 1 and row0[1] is not None:
            t.I_R = int(row0[1])
        if len(row0) > 2 and row0[2] is not None:
            t.I_P = int(row0[2])
        if len(row0) > 3 and row0[3] is not None:
            t.I_C = int(row0[3])
        if len(row0) > 4 and row0[4] is not None:
            t.I_L = int(row0[4])
        if len(row0) > 5 and row0[5] is not None:
            t.I_G = int(row0[5])
        if len(row0) > 6 and row0[6] is not None:
            t.D_G = float(row0[6])
        if len(row0) > 7 and row0[7] is not None:
            t.D_C = float(row0[7])

    elif param_type == "FIELD" and rows:
        row0 = rows[0]
        f = ctrl.field
        if len(row0) > 0 and row0[0] is not None:
            f.R_n = float(row0[0])
        if len(row0) > 1 and row0[1] is not None:
            f.C_n = float(row0[1])
        if len(row0) > 2 and row0[2] is not None:
            f.q_0 = float(row0[2])
        if len(row0) > 3 and row0[3] is not None:
            f.c_0 = float(row0[3])
        if len(row0) > 4 and row0[4] is not None:
            f.R_p = float(row0[4])
        if len(row0) > 5 and row0[5] is not None:
            f.C_a = float(row0[5])
        ctrl.tolerance_engine.R_n_base = f.R_n
        ctrl.tolerance_engine.C_n_base = f.C_n

    elif param_type == "LINE SEARCH" and rows:
        row0 = rows[0]
        ls = ctrl.line_search
        if len(row0) > 0 and row0[0] is not None:
            ls.N_ls = int(row0[0])
        if len(row0) > 1 and row0[1] is not None:
            ls.s_ls = float(row0[1])
        if len(row0) > 2 and row0[2] is not None:
            ls.f_ls = float(row0[2])
        if len(row0) > 3 and row0[3] is not None:
            ls.eta_ls = float(row0[3])


@GLOBAL_KEYWORD_REGISTRY.register(
    "CONTACT CONTROLS",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.STEP},
    aliases=["CONTACTCONTROLS"],
    support="FULL",
    params={
        "NAME": ParamSpec("NAME", default=None),
        "MASTER": ParamSpec("MASTER", default=None, aliases=["MAIN"]),
        "SLAVE": ParamSpec("SLAVE", default=None, aliases=["SECONDARY"]),
        "STABILIZE": ParamSpec("STABILIZE", default=None),
        "APPROACH": ParamSpec("APPROACH", type=bool, default=False),
        "RESET": ParamSpec("RESET", type=bool, default=False),
        "ABSOLUTE PENETRATION TOLERANCE": ParamSpec(
            "ABSOLUTE PENETRATION TOLERANCE",
            type=float,
            default=None,
            aliases=["ABSOLUTE_PENETRATION_TOLERANCE"],
        ),
        "RELATIVE PENETRATION TOLERANCE": ParamSpec(
            "RELATIVE PENETRATION TOLERANCE",
            type=float,
            default=None,
            aliases=["RELATIVE_PENETRATION_TOLERANCE"],
        ),
        "TANGENT FRACTION": ParamSpec(
            "TANGENT FRACTION",
            type=float,
            default=0.0,
            aliases=["TANGENT_FRACTION"],
        ),
        "STIFFNESS SCALE FACTOR": ParamSpec(
            "STIFFNESS SCALE FACTOR",
            type=float,
            default=1.0,
            aliases=["STIFFNESS_SCALE_FACTOR"],
        ),
        "CPSET": ParamSpec("CPSET", default=None),
        # Abaqus 2026: contact-related convergence criteria (STRICT default). The 2026 keyword
        # reference was not available when this was written -- placement on *CONTACT CONTROLS
        # follows the 2026 release summary and is unconfirmed.
        "CONVERGENCE CHECK": ParamSpec(
            "CONVERGENCE CHECK",
            default="STRICT",
            choices=["STRICT", "MODERATE", "RELAXED", "CONCEPT DESIGN"],
            aliases=["CONVERGENCE_CHECK"],
        ),
    },
    rows=RowSpec(min_columns=0, max_columns=4),
    description="Define contact controls and stabilization parameters",
)
def _handle_contact_controls(reader: AbaqusInputReader, block: AbaqusKeywordBlock) -> None:
    ctrl_name = block.params.get("name")
    if not ctrl_name:
        ctrl_name = f"ContCtrl-{len(reader.model.contact_controls) + 1}"

    damping_coeff = None
    tangent_fraction = float(block.params.get("tangent fraction", block.params.get("tangent_fraction", 0.0)))
    clearance = None
    ramp_down = 0.0

    rows = _parse_data_rows(block.data_lines, dtype=float, default=None)
    if rows:
        row0 = rows[0]
        if len(row0) > 0 and row0[0] is not None:
            damping_coeff = float(row0[0])
        if len(row0) > 1 and row0[1] is not None:
            tangent_fraction = float(row0[1])
        if len(row0) > 2 and row0[2] is not None:
            clearance = float(row0[2])
        if len(row0) > 3 and row0[3] is not None:
            ramp_down = float(row0[3])

    stab_val = block.params.get("stabilize")
    factor = float(block.params.get("stiffness scale factor", block.params.get("stiffness_scale_factor", 1.0)))
    if stab_val is not None:
        try:
            factor = float(stab_val)
            stabilize = "AUTOMATIC"
        except (ValueError, TypeError):
            stabilize = str(stab_val).upper()
    else:
        stabilize = "AUTOMATIC" if damping_coeff is not None else None

    abs_pen = block.params.get("absolute penetration tolerance", block.params.get("absolute_penetration_tolerance"))
    rel_pen = block.params.get("relative penetration tolerance", block.params.get("relative_penetration_tolerance"))
    pen_tol = float(abs_pen) if abs_pen is not None else (float(rel_pen) if rel_pen is not None else None)

    reader.model.ContactControl(
        name=ctrl_name,
        stabilize=stabilize,
        factor=factor,
        dampingCoefficient=damping_coeff,
        tangentFraction=tangent_fraction,
        rampDownFactor=ramp_down,
        clearance=clearance,
        penetrationTolerance=pen_tol,
        approach=bool(block.params.get("approach", False)),
        convergenceCheck=block.params.get("convergence check", block.params.get("convergence_check")),
    )


@GLOBAL_KEYWORD_REGISTRY.register(
    "CONTACT STABILIZATION",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.STEP},
    aliases=["CONTACTSTABILIZATION"],
    support="FULL",
    params={
        "NAME": ParamSpec("NAME", required=True),
        "DEFINITION": ParamSpec(
            "DEFINITION",
            default="AUTOMATIC",
            choices=["AUTOMATIC", "DAMPING COEFFICIENT", "SCALE FACTOR"],
            aliases=["DEF"],
        ),
        "RESET": ParamSpec("RESET", type=bool, default=False),
        "SCALE FACTOR": ParamSpec("SCALE FACTOR", type=float, default=1.0, aliases=["SCALE_FACTOR"]),
        "TANGENT FRACTION": ParamSpec("TANGENT FRACTION", type=float, default=0.0, aliases=["TANGENT_FRACTION"]),
        "AMPLITUDE": ParamSpec("AMPLITUDE", default=None),
        "RANGE": ParamSpec("RANGE", type=float, default=None),
        "REDUCTION PER INCREMENT": ParamSpec(
            "REDUCTION PER INCREMENT",
            type=float,
            default=None,
            aliases=["REDUCTION_PER_INCREMENT"],
        ),
    },
    rows=RowSpec(min_columns=0, max_columns=4),
    description="Define contact stabilization property",
)
def _handle_contact_stabilization(reader: AbaqusInputReader, block: AbaqusKeywordBlock) -> None:
    name = block.params.get("name")
    if not name:
        name = f"ContStab-{len(reader.model.contact_stabilizations) + 1}"

    definition = str(block.params.get("definition", "AUTOMATIC")).strip().upper()
    factor = float(block.params.get("scale factor", block.params.get("scale_factor", 1.0)))
    tangent_fraction = float(block.params.get("tangent fraction", block.params.get("tangent_fraction", 0.0)))
    damping_coeff = None
    clearance = None
    ramp_down = 0.0

    rows = _parse_data_rows(block.data_lines, dtype=float, default=None)
    if rows:
        row0 = rows[0]
        if len(row0) > 0 and row0[0] is not None:
            if definition == "DAMPING COEFFICIENT":
                damping_coeff = float(row0[0])
            else:
                factor = float(row0[0])
        if len(row0) > 1 and row0[1] is not None:
            tangent_fraction = float(row0[1])
        if len(row0) > 2 and row0[2] is not None:
            clearance = float(row0[2])
        if len(row0) > 3 and row0[3] is not None:
            ramp_down = float(row0[3])

    range_val = block.params.get("range")
    if range_val is not None and clearance is None:
        clearance = float(range_val)

    reader.model.ContactStabilization(
        name=name,
        definition=definition,
        dampingFactor=factor,
        factor=factor,
        dampingCoefficient=damping_coeff,
        tangentFraction=tangent_fraction,
        rampDownFactor=ramp_down,
        clearance=clearance,
    )
