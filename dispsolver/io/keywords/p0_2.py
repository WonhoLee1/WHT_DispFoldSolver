from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Tuple

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    ParamSpec,
    RowSpec,
)
from dispsolver.io.keyword_base import _AbortParse, _parse_data_rows
from dispsolver.model.bind import AmbiguousRef, BindError, SymbolTable, UnresolvedRef
from dispsolver.io.schema import Flat, Keyword, Param, Record, Schema, applies, keyword

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader


# ── *AMPLITUDE: schema reference migration (see dispsolver/io/schema.py) ──

@dataclass(frozen=True)
class AmplitudeRecord(Record):
    name: str
    definition: str                       # TABULAR or SMOOTH STEP
    points: Tuple[Tuple[float, float], ...]


@keyword("AMPLITUDE", support="PARTIAL")
class Amplitude(Keyword):
    """Time-dependent amplitude curve. TABULAR and SMOOTH STEP, step time only."""

    class Params(Schema):
        name: str
        definition: Literal["TABULAR", "EQUALLY SPACED", "PERIODIC", "MODULATED", "DECAY",
                            "SMOOTH STEP", "SOLUTION DEPENDENT", "BUBBLE", "USER", "ACTUATOR"] = Param(
            "TABULAR", supported=("TABULAR", "SMOOTH STEP"))
        # evaluated against step time; TOTAL TIME would be silently shifted
        time: Literal["STEP TIME", "TOTAL TIME"] = Param("STEP TIME", supported=("STEP TIME",))
        value: Literal["RELATIVE", "ABSOLUTE"] = Param("RELATIVE", supported=("RELATIVE",))

    # up to four (time, value) pairs per data line; the old handler read only
    # the first pair of each line, turning "0,0, 1,1" into a constant 0
    Data = Flat(float, group=2)

    def to_records(self, p, rows, ctx):
        yield AmplitudeRecord(p.name, p.definition, tuple(rows), loc=ctx.loc)


@applies(AmplitudeRecord)
def _apply_amplitude(reader: AbaqusInputReader, rec: AmplitudeRecord):
    data = list(rec.points)
    if rec.definition == "SMOOTH STEP":
        reader.model.SmoothStepAmplitude(rec.name, data)
    else:
        reader.model.TabularAmplitude(rec.name, data)


@GLOBAL_KEYWORD_REGISTRY.register(
    "MPC",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        "MODE": ParamSpec("MODE", supported=False),
        "USER": ParamSpec("USER", supported=False),
    },
    rows=RowSpec(min_columns=2, max_columns=10),
    description="Define multi-point constraints between nodes",
)
def _handle_mpc(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    # Data line: "MPC type, node or node set, node or node set, ...". The type
    # is the first data item, not a parameter (*MPC has only MODE and USER);
    # the old handler read a TYPE parameter and so stored every MPC as BEAM.
    for line in block.data_lines:
        tokens = [t.strip() for t in line.split(",") if t.strip()]
        if not tokens:
            continue
        mtype = tokens[0].upper()
        nodes = [int(t) for t in tokens[1:] if t.lstrip("-").isdigit()]
        name = f"MPC_{mtype}_{len(reader.model.constraints) + 1}"
        reader.model.MPC(name, mpcType=mtype,
                         firstPoint=nodes[0] if nodes else None,
                         secondPoint=nodes[1] if len(nodes) > 1 else None)
        msg = f"*MPC {name} parsed and stored; solver assembly not connected"
        reader.report("gap", "mpc-not-connected", msg, block)


@GLOBAL_KEYWORD_REGISTRY.register(
    "INITIAL CONDITIONS",
    valid_contexts={KeywordContext.GLOBAL},
    aliases=["INITIALCONDITIONS"],
    support="PARTIAL",
    params={
        "TYPE": ParamSpec("TYPE", default="TEMPERATURE", aliases=["TYP"]),
    },
    rows=RowSpec(min_columns=2, max_columns=10),
    description="Specify initial values of temperature, field variables, stress",
)
def _handle_initial_conditions(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    ictype = block.params.get("type", "TEMPERATURE").upper()
    if ictype == "TEMP":
        ictype = "TEMPERATURE"
    rows = _parse_data_rows(block.data_lines, dtype=str)
    if ictype == "TEMPERATURE":
        for r in rows:
            if len(r) >= 2:
                region = str(r[0])
                reader.model.PredefinedField(name=f"IC_TEMP_{region}", field_type="TEMPERATURE",
                                             region=region, values=float(r[1]))
    elif ictype == "STRESS":
        for r in rows:
            if len(r) >= 2:
                region = str(r[0])
                reader.model.PredefinedField(name=f"IC_STRESS_{region}", field_type="STRESS",
                                             region=region, values=tuple(float(x) for x in r[1:]))
    elif ictype in ("VELOCITY", "FIELD", "RATIO", "HARDENING", "PLASTIC STRAIN",
                    "DAMAGE INITIATION", "RELATIVE DENSITY", "SOLUTION"):
        for r in rows:
            if len(r) >= 2:
                region = str(r[0])
                try:
                    vals = tuple(float(x) for x in r[1:])
                except ValueError:
                    continue
                reader.model.PredefinedField(name=f"IC_{ictype.replace(' ', '_')}_{region}",
                                             field_type=ictype, region=region,
                                             values=vals[0] if len(vals) == 1 else vals)
        msg = (f"*INITIAL CONDITIONS, TYPE={ictype}: parsed and stored; "
               "solver consumption not connected")
        reader.report("gap", "initial-condition-not-connected", msg, block)
    else:
        raise _AbortParse(f"*INITIAL CONDITIONS, TYPE={ictype} has no internal landing")


@GLOBAL_KEYWORD_REGISTRY.register(
    "COUPLING",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        "CONSTRAINT NAME": ParamSpec("CONSTRAINT NAME", aliases=["CONSTRAINT_NAME"]),
        "REF NODE": ParamSpec("REF NODE", aliases=["REF_NODE", "REFNODE"]),
        "SURFACE": ParamSpec("SURFACE", aliases=["SURF"]),
    },
    rows=RowSpec(min_columns=0, max_columns=0),
    description="Define coupling constraint between reference node and surface",
)
def _handle_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("constraint name", block.params.get("name", f"CPL-{len(reader.model.constraints) + 1}"))
    ref = block.params.get("ref node", block.params.get("ref_node", None))
    surf = block.params.get("surface", None)
    reader.model.KinematicCoupling(name=name, refPoint=ref, surface=surf)
    reader.current_coupling = {"name": name}


def _finalize_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock, kind: str):
    name = block.params.get("constraint name", block.params.get("name", None))
    if name is None and reader.current_coupling is not None:
        name = reader.current_coupling.get("name")
    if name is None or name not in reader.model.constraints:
        msg = f"*{kind} without a preceding *COUPLING — skipped"
        reader.report("error", "bad-context", msg, block)
        return
    prev = reader.model.constraints[name]
    dofs = []
    for line in block.data_lines:
        for p in line.replace(",", " ").split():
            p = p.strip()
            if p.isdigit() and 1 <= int(p) <= 6:
                dofs.append(int(p))
    flags = {f"u{i}": (i in dofs) for i in (1, 2, 3)}
    flags.update({f"ur{i}": (i in dofs) for i in (1, 2, 3)})
    if not dofs:
        flags = {k: True for k in flags}
    if kind == "KINEMATIC":
        reader.model.KinematicCoupling(name=name, refPoint=prev.ref_point, surface=prev.surface, **flags)
    else:
        reader.model.DistributingCoupling(name=name, refPoint=prev.ref_point, surface=prev.surface, **flags)


@GLOBAL_KEYWORD_REGISTRY.register(
    "KINEMATIC",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        # a sub-option of the preceding *COUPLING; it takes no name of its own
        "ALPHA": ParamSpec("ALPHA", supported=False),
    },
    rows=RowSpec(min_columns=0, max_columns=6),
    description="Sub-option of *COUPLING to constrain degrees of freedom rigidly",
)
def _handle_kinematic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _finalize_coupling(reader, block, "KINEMATIC")


@GLOBAL_KEYWORD_REGISTRY.register(
    "KINEMATIC COUPLING",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        "REF NODE": ParamSpec("REF NODE", aliases=["REF_NODE", "REFNODE"]),
        "ORIENTATION": ParamSpec("ORIENTATION", supported=False),
    },
    rows=RowSpec(min_columns=0, max_columns=6),
    description="Constrain degrees of freedom on a surface to rigid body motion of reference node",
)
def _handle_kinematic_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _direct_coupling(reader, block, "KINEMATIC")


@GLOBAL_KEYWORD_REGISTRY.register(
    "DISTRIBUTING",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        # a sub-option of the preceding *COUPLING; it takes no name of its own
        "WEIGHTING METHOD": ParamSpec("WEIGHTING METHOD", supported=False),
        "COUPLING": ParamSpec("COUPLING", supported=False),
    },
    rows=RowSpec(min_columns=0, max_columns=6),
    description="Sub-option of *COUPLING to constrain degrees of freedom in an average sense",
)
def _handle_distributing(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _finalize_coupling(reader, block, "DISTRIBUTING")


@GLOBAL_KEYWORD_REGISTRY.register(
    "DISTRIBUTING COUPLING",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        "ELSET": ParamSpec("ELSET"),
        "FREE NODES": ParamSpec("FREE NODES", supported=False),
    },
    rows=RowSpec(min_columns=0, max_columns=6),
    description="Constrain degrees of freedom on a surface to average motion of reference node",
)
def _handle_distributing_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _direct_coupling(reader, block, "DISTRIBUTING")


def _direct_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock, kind: str):
    name = block.params.get("constraint name", block.params.get("name", f"{kind}_CPL_{len(reader.model.constraints) + 1}"))
    ref = block.params.get("ref node", block.params.get("ref_node", None))
    surf = block.params.get("surface", None)
    
    elset_param = block.params.get("elset", None)
    if ref is None and elset_param is not None:
        ref = elset_param
        
    if surf is None and block.data_lines:
        first_line = block.data_lines[0].strip()
        if first_line:
            tokens = [t.strip() for t in first_line.replace(",", " ").split() if t.strip()]
            if tokens:
                surf = tokens[0]

    if kind == "KINEMATIC":
        reader.model.KinematicCoupling(name=name, refPoint=ref, surface=surf)
    else:
        reader.model.DistributingCoupling(name=name, refPoint=ref, surface=surf)
    reader.current_coupling = {"name": name}
    msg = f"*{kind} COUPLING {name} parsed and stored (ref={ref}, surf={surf})"
    reader.report("info", "coupling-parsed", msg, block)


@GLOBAL_KEYWORD_REGISTRY.register(
    "FRICTION",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP},
    support="FULL",
    params={
        "ROUGH": ParamSpec("ROUGH", supported=False),
        "EXPONENTIAL DECAY": ParamSpec("EXPONENTIAL DECAY", aliases=["EXPONENTIAL_DECAY"], supported=False),
        "SLIP TOLERANCE": ParamSpec("SLIP TOLERANCE", aliases=["SLIP_TOLERANCE"], supported=False),
        "USER": ParamSpec("USER", supported=False),
    },
    rows=RowSpec(column_types=[float], min_columns=1, max_columns=5),
    description="Define friction properties for surface contact interaction",
)
def _handle_friction(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if not rows or len(rows[0]) < 1 or rows[0][0] is None:
        return
    target = reader.current_interaction
    if target is None or target not in reader.model.contact_properties:
        msg = "*FRICTION outside *SURFACE INTERACTION scope — skipped"
        reader.report("error", "bad-context", msg, block)
        return
    reader.model.contact_properties[target].friction = float(rows[0][0])


_ALL_CONTEXTS = frozenset({KeywordContext.GLOBAL, KeywordContext.PART, KeywordContext.ASSEMBLY,
                           KeywordContext.INSTANCE, KeywordContext.STEP, KeywordContext.MATERIAL})


@GLOBAL_KEYWORD_REGISTRY.register(
    "INCLUDE",
    valid_contexts=set(_ALL_CONTEXTS),
    support="STUB",
    params={
        "INPUT": ParamSpec("INPUT", required=True),
    },
    rows=RowSpec(min_columns=0, max_columns=0),
    description="Include an external file into the input deck",
)
def _handle_include(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    raise _AbortParse("*INCLUDE file insertion is not implemented; split the deck first")


@GLOBAL_KEYWORD_REGISTRY.register(
    "TRANSFORM",
    valid_contexts=set(_ALL_CONTEXTS),
    support="FULL",
    params={
        "NSET": ParamSpec("NSET", required=True),
        "TYPE": ParamSpec("TYPE", default="R", choices=["R", "C", "S"]),
    },
    rows=RowSpec(column_types=[float], min_columns=6, max_columns=7),
    description="Specify a local coordinate system at nodes",
)
def _handle_transform(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    nset = block.params.get("nset", "")
    ttype = block.params.get("type", "R").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    pts = tuple(tuple(float(x) for x in r if x is not None) for r in rows) if rows else ()
    name = f"TRF_{nset}_{len(reader.model.transforms) + 1}" if nset else f"TRF_{len(reader.model.transforms) + 1}"
    reader.model.Transform(name, nset=nset, transform_type=ttype, points=pts)


@GLOBAL_KEYWORD_REGISTRY.register(
    "EQUATION",
    valid_contexts=set(_ALL_CONTEXTS),
    support="FULL",
    params={},
    rows=RowSpec(min_columns=3, max_columns=12, repeating=True),
    description="Define linear multi-point constraint equation between degree of freedoms",
)
def _handle_equation(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    """*EQUATION: one or more linear constraint equations per block.

    Data (Abaqus Keywords Reference, *EQUATION): a line with the number of
    terms N, then the N terms "node or node set, dof, coefficient", at most
    four per line, continued on as many lines as needed; then the next
    equation's N, and so on.

    A node-set term generates one equation per node of the set (nodes paired
    by position, in the set's order); all multi-node sets in an equation must
    have the same size. A node number or a single-node set is used in every
    generated equation -- an assumption (reported as info), not taken from a
    local copy of the manual.

    The previous handler merged all lines into one list, read only the FIRST
    equation of a block, and put every node of a set into that one equation.
    """
    lines = [(t, block.data_location(i)) for i, t in enumerate(block.data_lines)]
    i = 0
    n_eq = 0
    while i < len(lines):
        text, loc = lines[i]
        head = [c.strip() for c in text.split(",") if c.strip()]
        i += 1
        if not head:
            continue
        try:
            nterms = int(float(head[0]))
        except ValueError:
            reader.report("error", "bad-row", f"*EQUATION at {loc}: expected the number of terms, got {text!r}", block)
            return
        if len(head) != 1 or nterms < 1:
            reader.report("error", "bad-row", f"*EQUATION at {loc}: the term-count line must hold one positive integer, got {text!r}", block)
            return
        vals: List[str] = []
        while len(vals) < 3 * nterms and i < len(lines):
            vals += [c.strip() for c in lines[i][0].split(",") if c.strip()]
            i += 1
        if len(vals) != 3 * nterms:
            reader.report("error", "bad-row", f"*EQUATION at {loc}: {nterms} terms declared, "
                                              f"{len(vals) / 3:g} given", block)
            return
        terms = []          # (list of node ids, dof, coeff)
        for t in range(nterms):
            ref, dof_raw, coeff_raw = vals[3 * t: 3 * t + 3]
            try:
                dof = int(float(dof_raw)) - 1
                coeff = float(coeff_raw)
            except ValueError:
                reader.report("error", "bad-row", f"*EQUATION at {loc}: bad term {ref}, {dof_raw}, {coeff_raw}", block)
                return
            if not 0 <= dof <= 5:
                reader.report("error", "bad-row", f"*EQUATION at {loc}: dof {dof + 1} out of range 1..6", block)
                return
            if ref.lstrip("-").isdigit():
                nodes = [int(ref)]
            else:
                try:
                    nodes = _equation_set_size(reader, ref)
                except BindError as e:
                    reader.report("error", "ambiguous-ref" if isinstance(e, AmbiguousRef) else "unresolved-ref",
                                  f"*EQUATION at {loc}: {e}", block)
                    return
                if not nodes:
                    reader.report("error", "unresolved-ref", f"*EQUATION at {loc}: node set {ref!r} is empty", block)
                    return
            terms.append((nodes, dof, coeff))
        sizes = {len(nodes) for nodes, _, _ in terms if len(nodes) > 1}
        if len(sizes) > 1:
            reader.report("error", "bad-row", f"*EQUATION at {loc}: node sets of different sizes {sorted(sizes)}", block)
            return
        count = sizes.pop() if sizes else 1
        if count > 1 and any(len(nodes) == 1 for nodes, _, _ in terms):
            reader.report("info", "equation-set-broadcast",
                          f"*EQUATION at {loc}: single nodes used in each of the {count} equations generated "
                          f"from the node sets", block)
        for k in range(count):
            # a set term keeps its NAME: EquationConstraint.bind resolves it through
            # model.regions (instance-qualified, ambiguity-checked) and takes the
            # k-th node of the set; a node number stays a label
            triples = [(nodes[0], dof, coeff) for nodes, dof, coeff in terms]
            name = f"EQ_{len(reader.model.constraints) + 1}"
            reader.model.Equation(name, terms=triples, pair_index=k if count > 1 else None)
            n_eq += 1
    reader.report("info", "equation-parsed", f"*EQUATION at {block.location}: {n_eq} equation(s)", block)


def _equation_set_size(reader: AbaqusInputReader, ref: str) -> list:
    """Resolve an EQUATION term's node set name against the model with the
    BC/load rules (``model/bind.py``: qualified names, unique bare names,
    ambiguity error) and return a list holding the set's node *count* worth of
    placeholders ([ref]*n). Only the size is needed at parse time -- the nodes
    themselves are resolved at bind time. Raises BindError."""
    asm = getattr(reader.model, "root_assembly", None)
    if not getattr(asm, "instances", None):
        # no *ASSEMBLY: the deck's sets live in its part(s), which become the
        # (single) instance at build time -- resolve there, refusing a name
        # several parts define instead of taking the first
        hits = []
        for part in reader.model.parts.values():
            for table in (part.node_sets, part.sets):
                key = next((k for k in table if k.upper() == ref.split(".")[-1].upper()), None)
                if key is not None and getattr(table[key], "node_ids", None):
                    hits.append((part.name, table[key]))
                    break
        if len(hits) > 1:
            raise AmbiguousRef("node_region", ref, [f"{n}.{ref}" for n, _ in hits])
        if not hits:
            raise UnresolvedRef("node_region", ref, f"node set {ref!r} is not defined in this model")
        return [ref] * len({int(i) for i in hits[0][1].node_ids})
    got = SymbolTable(reader.model).resolve("node_region", ref)
    if isinstance(got, tuple):                       # ("node", inst, label)
        return [ref]
    if isinstance(got, str):                         # "ALL"
        raise UnresolvedRef("node_region", ref, f"*EQUATION term {ref!r}: ALL is not a valid equation node set")
    ids = None
    if hasattr(got, "get_nodes"):
        try:
            ids = got.get_nodes(include_elements=True)
        except TypeError:
            ids = None
    if ids is None:
        ids = getattr(got, "node_ids", None)
    if ids is None:
        raise UnresolvedRef("node_region", ref, f"*EQUATION term {ref!r}: cannot count the nodes of this region")
    return [ref] * len(set(int(i) for i in ids))
