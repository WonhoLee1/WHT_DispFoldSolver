"""*RESTART: write restart records during a step, or continue from one.

    *RESTART, WRITE [, FREQUENCY=n] [, OVERLAY]
    *RESTART, READ, STEP=s [, INC=i | END STEP]

WRITE in the model data applies to every step; in a step it applies from that step on. FREQUENCY defaults
to 1 (Keywords Guide). READ belongs in the model data. ``STEP`` is the step NUMBER (1 = first step after
the Initial one) and ``INC`` the number of converged increments; with neither ``INC`` nor ``END STEP`` the
last record written for that step is used. The old job is not a parameter of the keyword: as in Abaqus it
is named outside the deck (``execute_inp(..., oldjob=...)``).

NUMBER INTERVAL, TIME MARKS, CYCLE and ITERATION are not supported and are reported, not skipped.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext, ParamSpec, RowSpec

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader


@GLOBAL_KEYWORD_REGISTRY.register(
    "RESTART",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP},
    support="PARTIAL",
    params={
        "WRITE": ParamSpec("WRITE", type=bool, default=False),
        "READ": ParamSpec("READ", type=bool, default=False),
        "FREQUENCY": ParamSpec("FREQUENCY", type=int, default=1),
        "OVERLAY": ParamSpec("OVERLAY", type=bool, default=False),
        "STEP": ParamSpec("STEP", type=int),
        "INC": ParamSpec("INC", type=int),
        "END STEP": ParamSpec("END STEP", type=bool, default=False, aliases=["ENDSTEP"]),
        "NUMBER INTERVAL": ParamSpec("NUMBER INTERVAL", type=int, supported=False),
        "TIME MARKS": ParamSpec("TIME MARKS", type=bool, supported=False, aliases=["TIMEMARKS"]),
        "CYCLE": ParamSpec("CYCLE", type=int, supported=False),
        "ITERATION": ParamSpec("ITERATION", type=int, supported=False),
    },
    rows=RowSpec(min_columns=0, max_columns=0),
    description="Write restart records or continue an analysis from one",
)
def _handle_restart(reader: AbaqusInputReader, block: AbaqusKeywordBlock) -> None:
    p = block.params
    write, read = "write" in p, "read" in p
    if not (write or read):
        raise ValueError("*RESTART needs READ, WRITE or both")
    if write:
        freq = 1 if p.get("frequency") is None else int(p["frequency"])
        if freq < 0:
            raise ValueError(f"*RESTART, WRITE: FREQUENCY={freq} is negative")
        # FREQUENCY=0 writes no restart data (Keywords Guide); in a step it switches an inherited request off
        request = {"frequency": freq, "overlay": "overlay" in p}
        step = getattr(reader, "current_step", None)
        if step is not None:
            step.restart_write = request
        else:
            reader.model.restart_write = request if freq > 0 else None
    if read:
        if getattr(reader, "current_step", None) is not None:
            raise ValueError("*RESTART, READ belongs in the model data, not inside a step")
        if p.get("step") is None:
            raise ValueError("*RESTART, READ needs STEP=")
        end_step = "end step" in p or "endstep" in p
        if end_step and p.get("inc") is not None:
            raise ValueError("*RESTART, READ takes INC= or END STEP, not both")
        reader.model.restart_read = {"step": int(p["step"]),
                                     "inc": int(p["inc"]) if p.get("inc") is not None else None,
                                     "end_step": bool(end_step)}
