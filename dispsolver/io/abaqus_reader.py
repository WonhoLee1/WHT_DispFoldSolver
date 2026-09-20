"""
abaqus_reader.py
================
Robust, production-grade Abaqus .inp reader and executor engine.
Builds dispsolver.model.Model objects directly from parsed keyword blocks,
supporting both Part/Assembly hierarchies and Non-Assembly (flat) decks.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import numpy as np

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock, tokenize, tokenize_string
from dispsolver.io.keyword_registry import (
    KeywordContext,
    KeywordSpec,
    KeywordRegistry,
    GLOBAL_KEYWORD_REGISTRY,
)
from dispsolver.io.keyword_base import _AbortParse, _parse_data_rows
from dispsolver.model.model import Model
from dispsolver.model.part import Part
from dispsolver.model.instance import Instance
from dispsolver.model.material import Material
from dispsolver.model.section import SolidSection, ShellSection
from dispsolver.model.step import Step, PredefinedField
from dispsolver.model.set import GeneralSet, SetScope


class AbaqusInputReader:
    """Production reader converting Abaqus keyword blocks into a complete dispsolver.model.Model."""

    def __init__(self, dim: int = 3, model_name: str = "AbaqusModel", strict: bool = True):
        self.dim = int(dim)
        self.model = Model(name=model_name, dim=self.dim)
        self.registry = GLOBAL_KEYWORD_REGISTRY
        self.strict = bool(strict)

        # Parsing states
        self.context: KeywordContext = KeywordContext.GLOBAL
        self.current_part: Optional[Part] = None
        self.current_instance: Optional[Instance] = None
        self.current_step: Optional[Step] = None
        self.current_material: Optional[Material] = None
        self.current_interaction: Optional[str] = None
        self.current_coupling: Optional[Dict[str, Any]] = None

        # Flat / Non-assembly tracking
        self.has_assembly: bool = False
        self._default_part: Optional[Part] = None
        self._default_instance: Optional[Instance] = None

        # Statistics & warnings
        self.unsupported_keywords: List[str] = []
        self.warnings: List[str] = []

    def get_active_part(self) -> Part:
        """Return currently active part (or create/return default part for non-assembly models)."""
        if self.current_part is not None:
            return self.current_part
        if self._default_part is None:
            self._default_part = self.model.Part("PART_DEFAULT", dim=self.dim)
        return self._default_part

    def ensure_default_instance(self):
        """Ensure default instance exists if model has flat/non-assembly elements."""
        if not self.has_assembly and self._default_instance is None:
            part = self.get_active_part()
            self._default_instance = self.model.root_assembly.Instance("INST_DEFAULT", part=part)

    _STRUCTURAL_KEYWORDS = frozenset({
        "PART", "END PART", "ASSEMBLY", "END ASSEMBLY",
        "INSTANCE", "END INSTANCE",
    })
    _MATERIAL_SUBOPTIONS = frozenset({
        "MATERIAL", "ELASTIC", "EXPANSION", "PLASTIC",
        "CREEP", "HYPERELASTIC", "VISCOELASTIC", "DENSITY", "TRS",
    })

    def parse_blocks(self, blocks: List[AbaqusKeywordBlock]) -> Model:
        """Parse all keyword blocks into the internal Model."""
        for block in blocks:
            kw = block.keyword
            if kw in ("STEP", "MATERIAL"):
                self.context = KeywordContext.GLOBAL
            if self.context == KeywordContext.MATERIAL and kw not in self._MATERIAL_SUBOPTIONS:
                self.context = KeywordContext.GLOBAL
            spec = self.registry.get_handler(kw)
            if spec is not None:
                if self.context not in spec.valid_contexts:
                    if kw in self._STRUCTURAL_KEYWORDS:
                        raise ValueError(
                            f"Keyword *{kw} illegal in context {self.context} "
                            f"at line {block.line_number}")
                    msg = (f"Keyword *{kw} outside valid context {self.context} "
                           f"at line {block.line_number} - skipped")
                    if self.strict:
                        raise ValueError(msg)
                    warnings.warn(msg)
                    self.warnings.append(msg)
                    continue
                try:
                    spec.handler(self, block)
                except _AbortParse:
                    raise
                except Exception as e:
                    msg = f"Error processing keyword *{kw} at line {block.line_number}: {e}"
                    warnings.warn(msg)
                    self.warnings.append(msg)
            else:
                if kw not in self.unsupported_keywords:
                    self.unsupported_keywords.append(kw)

        # If non-assembly deck was loaded, bind the default instance
        if not self.has_assembly and self._default_part is not None:
            self.ensure_default_instance()

        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_source(source: Union[str, os.PathLike], dim: int = 3,
                 model_name: str = "AbaqusModel", strict: bool = True) -> Tuple[AbaqusInputReader, Model]:
    """Tokenize and parse, retaining the reader for warnings/unsupported report."""
    source_str = str(source)
    if os.path.exists(source_str):
        blocks = tokenize(source_str)
    else:
        blocks = tokenize_string(source_str)

    reader = AbaqusInputReader(dim=dim, model_name=model_name, strict=strict)
    model = reader.parse_blocks(blocks)
    return reader, model


def read_inp(source: Union[str, os.PathLike], dim: int = 3, model_name: str = "AbaqusModel", strict: bool = True) -> Model:
    """Read an Abaqus .inp file or raw string and return a populated Model instance.
    
    Parameters
    ----------
    source : str or PathLike
        Path to .inp file on disk, or raw INP text string containing '*HEADING' or '*NODE'.
    dim : int
        Spatial dimensionality (2 or 3).
    model_name : str
        Name for the constructed Model.
        
    Returns
    -------
    Model
        Populated Model container ready for solver creation and inspection.
    """
    _, model = _parse_source(source, dim=dim, model_name=model_name, strict=strict)
    return model


def _execute_single_step(model: Model, target_step: Optional[str],
                         dt: Optional[float], verbose: bool, dim: int,
                         reader: Optional[AbaqusInputReader] = None) -> Any:
    """Run one step with a single solve_step call (legacy smoke path)."""
    notices = reader.warnings if reader is not None else None
    step_obj = model.steps.get(target_step) if target_step else None
    if step_obj is not None and (step_obj.get_active_loads() or step_obj.get_active_interactions()):
        msg = (f"Smoke path ignores step loads/interactions on Step '{target_step}'; "
               f"use execute_inp (full path) for load-driven steps")
        warnings.warn(msg)
        if reader is not None:
            reader.warnings.append(msg)
    if dim == 3:
        solver, sys = model.create_solver(step_name=target_step, notices=notices)
    else:
        solver, sys = model.create_solver2d(step_name=target_step, notices=notices)

    step_dt = dt if dt is not None else (step_obj.dt_init if step_obj else 1.0)

    if verbose:
        print(f"Executing Abaqus Model '{model.name}' [{dim}D] on Step '{target_step}' (dt={step_dt})...")

    conv, iters = solver.solve_step(dt=step_dt)

    if verbose:
        print(f"Execution completed: Converged={conv}, Newton Iterations={iters}")

    return {
        "model": model,
        "solver": solver,
        "system": sys,
        "converged": conv,
        "iterations": iters,
        "u": solver.u.copy() if hasattr(solver, "u") else None,
        "unsupported_keywords": list(reader.unsupported_keywords) if reader is not None else [],
        "warnings": list(reader.warnings) if reader is not None else [],
    }


def execute_inp_smoke(
    source: Union[str, os.PathLike],
    dim: int = 3,
    step_name: Optional[str] = None,
    dt: Optional[float] = None,
    verbose: bool = True,
    strict: bool = True
) -> Any:
    """Single-step smoke run: first (or named) non-Initial step only."""
    reader, model = _parse_source(source, dim=dim, strict=strict)
    if step_name is None:
        steps_list = [s for s in model.steps.keys() if s != "Initial"]
        target_step = steps_list[0] if steps_list else None
    else:
        target_step = step_name
    return _execute_single_step(model, target_step, dt, verbose, dim,
                                reader=reader)


def execute_inp(
    source: Union[str, os.PathLike],
    dim: int = 3,
    step_name: Optional[str] = None,
    dt: Optional[float] = None,
    verbose: bool = True,
    strict: bool = True
) -> Any:
    """Read an Abaqus .inp file, compile to solver system, and execute analysis steps.

    Decks whose steps carry no loads or interactions use the legacy smoke
    path (identical results plus iteration counts). Steps with loads or
    interactions run through MultiStepExecutor so they are actually applied.
    Multi-step decks run every non-Initial step in order with
    parent-checkpoint restore.
    
    Parameters
    ----------
    source : str or PathLike
        Path to .inp file or raw string.
    dim : int
        Spatial dimensionality (2 or 3).
    step_name : Optional[str]
        Specific step to run (defaults to all non-initial steps in order).
    dt : Optional[float]
        Time increment override (single-step path, or per-step dt_init).
    verbose : bool
        If True, prints execution progress.
        
    Returns
    -------
    dict
        Execution summary with solver state, displacements, and iteration
        count. Multi-step runs add results_by_step and steps_executed;
        iterations is None there (per-step solve does not report counts).
    """
    from dispsolver.solver.step_executor import MultiStepExecutor

    reader, model = _parse_source(source, dim=dim, strict=strict)

    # Determine steps to run
    if step_name is None:
        targets = [s for s in model.steps.keys() if s != "Initial"]
    else:
        targets = [step_name]

    if len(targets) <= 1:
        single = model.steps.get(targets[0]) if targets else None
        if single is not None and (single.get_active_loads() or single.get_active_interactions()):
            targets = [single.name]
        else:
            return _execute_single_step(model, targets[0] if targets else None,
                                        dt, verbose, dim, reader=reader)

    if dt is not None:
        for sname in targets:
            if sname in model.steps:
                model.steps[sname].dt_init = float(dt)

    executor = MultiStepExecutor(model, verbose=verbose)
    executor.initialize_solver(notices=reader.warnings)
    ok_all = True
    for sname in targets:
        ok = executor.execute_step(model.steps[sname],
                                   from_parent_checkpoint=True)
        ok_all = ok_all and ok

    solver, sys = executor.solver, executor.sys
    if verbose:
        print(f"Execution completed: Converged={ok_all}, Steps={targets}")

    return {
        "model": model,
        "solver": solver,
        "system": sys,
        "converged": ok_all,
        "iterations": None,
        "u": solver.u.copy() if hasattr(solver, "u") and solver is not None else None,
        "results_by_step": executor.results_by_step,
        "steps_executed": targets,
        "unsupported_keywords": list(reader.unsupported_keywords),
        "warnings": list(reader.warnings),
    }
