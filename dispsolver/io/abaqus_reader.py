"""
abaqus_reader.py
================
Robust, production-grade Abaqus .inp reader and executor engine.
Builds dispsolver.model.Model objects directly from parsed keyword blocks,
supporting both Part/Assembly hierarchies and Non-Assembly (flat) decks.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock, tokenize, tokenize_string
from dispsolver.io.diagnostics import Diagnostics, Loc
from dispsolver.io.keyword_index import AmbiguousName, abaqus_index, resolve_name
from dispsolver.io.keyword_registry import (
    KeywordContext,
    GLOBAL_KEYWORD_REGISTRY,
    normalize_param_name,
)
from dispsolver.io.keyword_base import _AbortParse
from dispsolver.model.bind import bind_model
from dispsolver.model.model import Model
from dispsolver.model.part import Part
from dispsolver.model.instance import Instance
from dispsolver.model.material import Material
from dispsolver.model.step import Step


# Keywords that only request printed/file output, restart files or solver
# diagnostics: skipping them cannot change the computed solution.
OUTPUT_ONLY_KEYWORDS = frozenset({
    "EL PRINT", "NODE PRINT", "PREPRINT", "EL FILE", "NODE FILE", "MODAL OUTPUT", "MODAL FILE",
    "RESTART", "ENERGY OUTPUT", "NODE OUT", "ENERGY PRINT", "FILE OUTPUT", "ENERGY FILE", "PRINT",
    "DIAGNOSTICS", "CONTACT PRINT", "CONTACT FILE", "CONTACT OUTPUT", "SECTION PRINT", "FILE FORMAT",
    "MONITOR", "MODAL PRINT", "SECTION FILE", "INTEGRATED OUTPUT", "INCREMENTATION OUTPUT",
    "SURFACE OUTPUT", "RADIATION OUTPUT", "RADIATION PRINT", "RADIATION FILE", "ELEMENT MATRIX OUTPUT",
    "NODAL ENERGY RATE", "PRINT CONTROLS", "OUTPUT CONTROLS",
})
# Registered output-request keywords (no effect on the solution).
OUTPUT_REQUEST_KEYWORDS = frozenset({"OUTPUT", "NODE OUTPUT", "ELEMENT OUTPUT", "RESTART"})
# Solution-technique controls: they change how the answer is reached
# (tolerances, line search, cutback policy), not the model being solved.
SOLUTION_CONTROL_KEYWORDS = frozenset({"CONTROLS", "SOLVER CONTROLS"})


class AbaqusInputReader:
    """Production reader converting Abaqus keyword blocks into a complete dispsolver.model.Model."""

    # ------------------------------------------------------------------ diagnostics
    # Every problem the reader finds goes into self.diagnostics (io/diagnostics.py):
    # error = the deck is wrong / unreadable, gap = valid input this solver does
    # not represent, warning = represented with a caveat, info = no effect.
    # `warnings`, `physics_gaps()` and `require_complete()` are views of it.

    @property
    def warnings(self) -> List[str]:
        """All diagnostic messages, in input order (legacy view)."""
        return [d.message for d in self.diagnostics]

    def report(self, level: str, code: str, message: str, block: Optional[AbaqusKeywordBlock] = None) -> None:
        self.diagnostics.add(level, code, message, Loc.of(block))

    def _keyword_is_inert(self, kw: str) -> bool:
        base = kw.split(" table rejected")[0].strip()
        spec = self.registry.get_handler(base)
        return (base in OUTPUT_ONLY_KEYWORDS | SOLUTION_CONTROL_KEYWORDS
                or (spec is not None and spec.support in ("IGNORED", "STUB")))

    def unsupported_keyword(self, label: str, block: Optional[AbaqusKeywordBlock] = None) -> None:
        """A keyword (or a keyword option, e.g. 'CONCRETE ... table rejected')
        the reader has no landing for."""
        if label not in self.unsupported_keywords:
            self.unsupported_keywords.append(label)
            loc = block.location if block is not None else None
            if loc:
                self.unsupported_locations[label] = loc
            text = f"*{label} (unsupported" + (f", {loc})" if loc else ")")
            if self._keyword_is_inert(label):
                self.report("info", "ignored-keyword", text, block)
            else:
                self.report("gap", "unsupported-keyword", text, block)

    def unsupported_param(self, kw: str, param: str, kind: str, block: Optional[AbaqusKeywordBlock] = None) -> None:
        """kind: 'unknown' (not in the schema), 'unsupported' (in the schema,
        not implemented) or 'control' (solution control, not implemented)."""
        entry = (kw, param, kind)
        if entry in self.unsupported_params:
            return
        self.unsupported_params.append(entry)
        loc = block.location if block is not None else None
        text = f"*{kw}, {param.upper()} ({kind} parameter" + (f", {loc})" if loc else ")")
        if kind == "control" or self._keyword_is_inert(kw):
            self.report("info", f"{kind}-param-ignored", text, block)
        else:
            # "unknown" (not in the Keywords Guide) is a gap too: official
            # verification decks use undocumented parameters Abaqus accepts
            # (*VISCOELASTIC, FORMULATION=), so it is not proof the deck is wrong
            self.report("gap", f"{kind}-param", text, block)

    def physics_gaps(self) -> List[str]:
        """Everything that makes the solved model differ from the deck: errors
        (the deck is wrong or unreadable) and gaps (valid input this solver
        does not represent). Output-only keywords and solution controls are
        recorded as info and are not listed."""
        return [d.message for d in self.diagnostics.of("error", "gap")]

    def require_complete(self) -> None:
        """Raise UnsupportedDeckError if any error or gap was recorded (strict run)."""
        self.diagnostics.raise_if_blocking(
            True, UnsupportedDeckError,
            "deck uses features this solver does not implement or could not read; refusing to "
            "solve a different model than the deck describes:")

    def __init__(self, dim: int = 3, model_name: str = "AbaqusModel", strict: bool = True):
        self.dim = int(dim)
        self.model = Model(name=model_name, dim=self.dim)
        self.model.from_deck = True
        self.registry = GLOBAL_KEYWORD_REGISTRY
        self.strict = bool(strict)

        # Parsing states
        self.context: KeywordContext = KeywordContext.GLOBAL
        self._skipping: Optional[dict] = None      # body of a refused scope opener
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
        self.unsupported_params: List[Tuple[str, str, str]] = []
        self.unsupported_locations: Dict[str, str] = {}   # keyword -> first file:line
        self.diagnostics = Diagnostics()
        self.symbols = None                    # SymbolTable, built by bind_model at the end of parse_blocks

    def _stamp_locations(self, location: str) -> None:
        """Give every model object created by the keyword just handled the
        deck location it came from (`source_location`), so errors raised much
        later -- while the solver resolves a BC region, an amplitude, a
        section's material -- can point at the deck line. Objects that already
        carry a location keep it (copies propagated into later steps too)."""
        m = self.model
        groups = [m.materials, m.sections, m.amplitudes, m.constraints, m.interactions,
                  m.contact_properties]
        step = self.current_step if self.current_step is not None else m.initial_step
        for d in (step.boundary_conditions, step.loads, step.interactions):
            groups.append({k: getattr(e, "entity", e) for k, e in d.items()})
        groups.append(getattr(step, "predefined_fields", {}))
        for g in groups:
            for obj in g.values():
                if getattr(obj, "source_location", None) is None:
                    try:
                        obj.source_location = location
                    except (AttributeError, TypeError):
                        pass

    def get_active_part(self) -> Part:
        """Return currently active part (or create/return default part for non-assembly models).

        Inside *INSTANCE ... *END INSTANCE, model data (*NODE, *ELEMENT, sets,
        sections) belongs to that instance: it goes to a part of its own when
        the instance's part is empty, else to that part provided no other
        instance shares it. This used
        to fall through to PART_DEFAULT, which no instance references, so an
        instance-level mesh (Abaqus CAE writes them for independent instances)
        vanished from the assembly."""
        if self.current_part is not None:
            return self.current_part
        if self.context == KeywordContext.INSTANCE and self.current_instance is not None:
            inst = self.current_instance
            part = inst.part
            if getattr(inst, "_private_part", False):
                return part
            if not part.nodes and not part.elements:
                # Independent instance of an empty part (Abaqus CAE writes its
                # mesh inside *INSTANCE): the instance gets a part of its own,
                # so further instances of the same empty part keep theirs.
                private = self.model.Part(f"{part.name}__{inst.name}", dim=self.dim)
                inst.part = private
                inst._private_part = True
                return private
            sharing = [i.name for i in self.model.root_assembly.instances.values()
                       if i.part is part and i is not inst]
            if sharing:
                raise NotImplementedError(
                    f"*INSTANCE {inst.name}: instance-level model data on part {part.name!r}, "
                    f"which already has a mesh and is also used by instances {sharing}")
            return part
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
        "RATE DEPENDENT", "DAMAGE INITIATION", "DAMAGE EVOLUTION",
        "DRUCKER PRAGER", "DRUCKER PRAGER HARDENING",
        "CONCRETE DAMAGED PLASTICITY", "CONCRETE COMPRESSION HARDENING",
        "CONCRETE TENSION STIFFENING", "CONCRETE COMPRESSION DAMAGE",
        "CONCRETE TENSION DAMAGE", "MULLINS EFFECT",
    })

    def skip_scope(self, opener: str, closer: str, block: AbaqusKeywordBlock) -> None:
        """A scope-opening keyword was refused: skip its body up to `closer`."""
        self._skipping = {"opener": opener, "closer": closer, "loc": block.location, "count": 0}

    def _skip_block(self, block: AbaqusKeywordBlock, index) -> bool:
        """True while inside the body of a refused scope (that block is skipped)."""
        kw = block.keyword
        if self.registry.get_handler(kw) is None:
            try:
                kw = index.resolve_keyword(kw) or kw
            except AmbiguousName:
                pass
        sk = self._skipping
        if kw != sk["closer"]:
            sk["count"] += 1
            return True
        self._skipping = None
        self.report("error", "refused-scope-skipped",
                    f"*{sk['opener']} at {sk['loc']} was refused; its {sk['count']} keyword(s) up to "
                    f"*{sk['closer']} at {block.location} were not read", block)
        return True

    def _resolve_keyword(self, block: AbaqusKeywordBlock, index):
        """(spec, canonical keyword) for a block, or (None, kw) when it was
        reported instead. Abbreviated keywords are legal Abaqus input and are
        resolved against the Abaqus keyword index; a keyword Abaqus defines
        that we do not handle is a gap, one Abaqus does not define is an error."""
        kw = block.keyword
        spec = self.registry.get_handler(kw)
        if spec is not None:
            return spec, kw
        try:
            canon = index.resolve_keyword(kw)
        except AmbiguousName as e:
            self.report("error", "ambiguous-keyword", f"*{kw} at {block.location}: keyword {e}", block)
            return None, kw
        if canon is not None:
            spec = self.registry.get_handler(canon)
            if spec is not None:
                block.keyword = canon
                return spec, canon
            self.unsupported_keyword(canon, block)
            return None, canon
        # Not in the Keywords Guide: a gap, like an undocumented parameter --
        # Abaqus accepts some undocumented input, so this is not proof the deck
        # is wrong (it still blocks a strict run).
        self.report("gap", "unknown-keyword",
                    f"*{kw} at {block.location}: not a documented Abaqus keyword", block)
        if kw not in self.unsupported_keywords:
            self.unsupported_keywords.append(kw)
        return None, kw

    @staticmethod
    def _expand_param_abbreviations(block: AbaqusKeywordBlock, spec, index) -> None:
        """Legacy handlers read full lower-case parameter names
        (params.get("amplitude")); Abaqus accepts any unambiguous prefix
        (AMP=). Add the full name next to each abbreviation, and record each
        given key's full Abaqus name for the validation pass."""
        block.expanded_params = set()
        block.param_full_names = {}
        entry = index.get(spec.name)
        names = set(entry.params) if entry is not None else set()
        names |= {normalize_param_name(n) for n in spec.params}
        for key in list(block.params):
            try:
                full = resolve_name(key, names)
            except AmbiguousName:
                continue
            if full is None:
                continue
            if entry is not None and full in entry.params:
                block.param_full_names[key] = full
            low = full.lower()
            if low != key and low not in block.params:
                block.params[low] = block.params[key]
                block.expanded_params.add(low)

    def parse_blocks(self, blocks: List[AbaqusKeywordBlock]) -> Model:
        """Parse all keyword blocks into the internal Model."""
        index = abaqus_index()
        for block in blocks:
            if self._skipping is not None and self._skip_block(block, index):
                continue
            spec, kw = self._resolve_keyword(block, index)
            if spec is None:
                continue
            if kw in ("STEP", "MATERIAL"):
                self.context = KeywordContext.GLOBAL
            if self.context == KeywordContext.MATERIAL and kw not in self._MATERIAL_SUBOPTIONS:
                self.context = KeywordContext.GLOBAL
            if spec is not None:
                if self.context not in spec.valid_contexts:
                    if kw in self._STRUCTURAL_KEYWORDS:
                        raise ValueError(
                            f"Keyword *{kw} illegal in context {self.context} "
                            f"at {block.location}")
                    msg = (f"Keyword *{kw} outside valid context {self.context} "
                           f"at {block.location} - skipped")
                    if self.strict:
                        raise ValueError(msg)
                    # an output request in the wrong place changes no physics
                    inert = kw in OUTPUT_REQUEST_KEYWORDS or self._keyword_is_inert(kw)
                    self.report("info" if inert else "error", "bad-context", msg, block)
                    continue

                # Parameter validation against spec (a schema keyword validates
                # its own parameters and data, see dispsolver/io/schema.py)
                if spec.schema is None:
                    self._expand_param_abbreviations(block, spec, index)
                if spec.schema is None and spec.params:
                    # 1. Check for required parameters
                    seen_spec_names = set()
                    for p_spec in spec.params.values():
                        if p_spec.name in seen_spec_names:
                            continue
                        seen_spec_names.add(p_spec.name)
                        if p_spec.required:
                            has_p = any(
                                normalize_param_name(k) == normalize_param_name(p_spec.name)
                                or any(normalize_param_name(k) == normalize_param_name(a) for a in p_spec.aliases)
                                for k in block.params.keys()
                            )
                            if not has_p:
                                self.report("error", "missing-param",
                                            f"Error processing keyword *{kw} at {block.location}: "
                                            f"missing required parameter '{p_spec.name}'", block)

                    # 2. Check for unknown or unsupported parameters. Keys added by
                    # abbreviation expansion are not reported a second time.
                    for p_key in list(block.params.keys()):
                        if p_key in block.expanded_params:
                            continue
                        p_norm = normalize_param_name(p_key)
                        full = block.param_full_names.get(p_key)
                        if p_norm not in spec.params and full is not None:
                            p_norm = normalize_param_name(full)
                        if p_norm not in spec.params:
                            # Abaqus defines it (index) -> not implemented; else unknown
                            self.unsupported_param(kw, p_key, "unsupported" if full else "unknown", block)
                        else:
                            p_spec = spec.params[p_norm]
                            if not p_spec.supported:
                                self.unsupported_param(kw, p_key, "control" if p_spec.control else "unsupported", block)

                try:
                    spec.handler(self, block)
                    self._stamp_locations(block.location)
                except _AbortParse:
                    raise
                except Exception as e:
                    # NotImplementedError: valid input this solver does not
                    # represent (gap); anything else: the deck is wrong (error).
                    self.report("gap" if isinstance(e, NotImplementedError) else "error",
                                "unsupported-feature" if isinstance(e, NotImplementedError) else "bad-input",
                                f"Error processing keyword *{kw} at {block.location}: {e}", block)

        if self._skipping is not None:
            # the refused scope was never closed: the rest of the deck was skipped
            sk, self._skipping = self._skipping, None
            self.report("error", "refused-scope-skipped",
                        f"*{sk['opener']} at {sk['loc']} was refused and *{sk['closer']} never came; "
                        f"the remaining {sk['count']} keyword(s) of the deck were not read")

        # If non-assembly deck was loaded, bind the default instance
        if not self.has_assembly and self._default_part is not None:
            self.ensure_default_instance()

        # BIND: every name the deck refers to must resolve (forward references are legal, so
        # this runs once, after the last block). An internal failure here is a bug: not swallowed.
        self.symbols = bind_model(self.model, self.diagnostics)

        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_dim(source: Union[str, os.PathLike], default: int = 3) -> int:
    """Spatial dimension from the deck's *ELEMENT TYPE= names: 2 for plane /
    axisymmetric continuum (CPE*, CPS*, CAX*, CGAX*, CGPE*, R2D*, RAX*), 3 for
    everything else that is recognisably 3D. A deck with no elements returns
    `default`. Mixed 2D and 3D element types are refused."""
    # From the tokenized deck, not the raw text: *INCLUDE files and indented or
    # continued *ELEMENT lines count too.
    src = str(source)
    blocks = tokenize(src) if os.path.exists(src) else tokenize_string(src)
    types = {str(b.params.get("type", "")).upper() for b in blocks if b.keyword == "ELEMENT"} - {""}
    two = {t for t in types if t.startswith(("CPE", "CPS", "CAX", "CGAX", "CGPE", "R2D", "RAX"))}
    three = types - two
    if two and three:
        raise ValueError(f"deck mixes 2D element types {sorted(two)} with 3D/other types {sorted(three)}")
    if two:
        return 2
    return 3 if three else default


class UnsupportedDeckError(ValueError):
    """The deck contains physics this solver cannot represent (strict run)."""


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
    notices = reader.diagnostics.notice_sink() if reader is not None else None
    step_obj = model.steps.get(target_step) if target_step else None
    if step_obj is not None and (step_obj.get_active_loads() or step_obj.get_active_interactions()):
        msg = (f"Smoke path ignores step loads/interactions on Step '{target_step}'; "
               f"use execute_inp (full path) for load-driven steps")
        if reader is not None:
            reader.report("gap", "smoke-path-ignores-loads", msg)
        else:
            warnings.warn(msg)
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
    dim: Optional[int] = None,
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

    # The dimension comes from the deck unless the caller insists. It used to
    # default to 3, which pushed plane-strain decks into the 3D solver (a
    # 4-node CPE4 landed on the solid-shell kernel and crashed in numba).
    if dim is None:
        dim = detect_dim(source)
    reader, model = _parse_source(source, dim=dim, strict=strict)
    if strict:
        reader.require_complete()

    # Determine steps to run
    if step_name is None:
        targets = [s for s in model.steps.keys() if s != "Initial"]
    else:
        targets = [step_name]

    if not targets:
        return _execute_single_step(model, None, dt, verbose, dim, reader=reader)
    # Every step runs through the step engine, so its BCs ramp over the step
    # with its amplitudes and increments. Steps without loads used to take a
    # one-shot solve_step "smoke" shortcut instead: the whole prescribed
    # displacement in a single increment, ignoring *STATIC incrementation --
    # a different (path-dependent) answer for plasticity and contact.

    if dt is not None:
        for sname in targets:
            if sname in model.steps:
                model.steps[sname].dt_init = float(dt)

    executor = MultiStepExecutor(model, verbose=verbose, dim=dim)
    executor.initialize_solver(notices=reader.diagnostics.notice_sink())
    ok_all = True
    n_iter, n_inc, max_inc_iter = 0, 0, 0
    for sname in targets:
        ok = executor.execute_step(model.steps[sname],
                                   from_parent_checkpoint=True)
        ok_all = ok_all and ok
        st = getattr(executor.solver, "last_solve_stats", None) or {}
        n_iter += int(st.get("iterations", 0))
        n_inc += int(st.get("increments", 0))
        max_inc_iter = max(max_inc_iter, int(st.get("max_increment_iterations", 0)))

    solver, sys = executor.solver, executor.sys
    if verbose:
        print(f"Execution completed: Converged={ok_all}, Steps={targets}")

    return {
        "model": model,
        "solver": solver,
        "system": sys,
        "converged": ok_all,
        "iterations": n_iter,               # total Newton iterations, all steps
        "increments": n_inc,                # converged increments, all steps
        "max_increment_iterations": max_inc_iter,
        "u": solver.u.copy() if hasattr(solver, "u") and solver is not None else None,
        "results_by_step": executor.results_by_step,
        "steps_executed": targets,
        "unsupported_keywords": list(reader.unsupported_keywords),
        "warnings": list(reader.warnings),
    }
