from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    ParamSpec,
    RowSpec,
)
from dispsolver.io.keyword_base import _AbortParse, _parse_data_rows
from dispsolver.model.step import PredefinedField

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── P0-2 MATERIAL SUBOPTIONS, LOADS, CONSTRAINTS ───────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register(
    "DENSITY",
    valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL},
    support="FULL",
    params={
        "DEPENDENCIES": ParamSpec("DEPENDENCIES", type=int, default=0, aliases=["DEP", "DEPE"], supported=False),
    },
    rows=RowSpec(column_types=[float], min_columns=1, max_columns=2),
    description="Define material mass density",
)
def _handle_density(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        msg = "*DENSITY outside *MATERIAL scope — skipped"
        warnings.warn(msg)
        reader.warnings.append(msg)
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows and len(rows[0]) >= 1 and rows[0][0] is not None:
        mat.density = float(rows[0][0])


@GLOBAL_KEYWORD_REGISTRY.register(
    "TRS",
    valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL},
    support="PARTIAL",
    params={
        "DEFINITION": ParamSpec("DEFINITION", default="WLF", choices=["WLF", "ARRHENIUS", "USER"], aliases=["DEF"]),
    },
    rows=RowSpec(column_types=[float], min_columns=3, max_columns=3),
    description="Define temperature-time shift for viscoelastic materials",
)
def _handle_trs(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        msg = "*TRS outside *MATERIAL scope — skipped"
        warnings.warn(msg)
        reader.warnings.append(msg)
        return
    defn = block.params.get("definition", "WLF").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    vals = rows[0] if rows else []
    if len(vals) >= 3 and vals[0] is not None and vals[1] is not None and vals[2] is not None:
        mat.extra_params["trs"] = {"definition": defn, "T_ref": float(vals[0]),
                                   "C1": float(vals[1]), "C2": float(vals[2])}
    else:
        mat.extra_params["trs"] = {"definition": defn, "values": tuple(float(v) for v in vals if v is not None)}


@GLOBAL_KEYWORD_REGISTRY.register(
    "AMPLITUDE",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP},
    support="PARTIAL",
    params={
        "NAME": ParamSpec("NAME", required=True),
        "DEFINITION": ParamSpec("DEFINITION", default="TABULAR", choices=["TABULAR", "SMOOTH STEP", "SMOOTHSTEP"], aliases=["DEF"]),
        "TIME": ParamSpec("TIME", default="STEP TIME", choices=["STEP TIME", "STEPTIME", "TOTAL TIME"], aliases=["STEP_TIME"]),
        "VALUE": ParamSpec("VALUE", choices=["RELATIVE", "ABSOLUTE"], supported=False),
        "SHIFT": ParamSpec("SHIFT", supported=False),
        "SCALEX": ParamSpec("SCALEX", supported=False),
        "SCALEY": ParamSpec("SCALEY", supported=False),
    },
    rows=RowSpec(column_types=[float], min_columns=2, max_columns=8, repeating=True),
    description="Define time-dependent load or boundary condition amplitude curve",
)
def _handle_amplitude(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", "")
    if not name:
        msg = "*AMPLITUDE without NAME — skipped"
        warnings.warn(msg)
        reader.warnings.append(msg)
        return
    defn = block.params.get("definition", "TABULAR").upper().replace(" ", "")
    if defn not in ("TABULAR", "SMOOTHSTEP"):
        raise NotImplementedError(f"*AMPLITUDE {name}: DEFINITION={defn} is not implemented")
    time_ref = block.params.get("time", "STEP TIME").upper().replace(" ", "")
    if time_ref != "STEPTIME":
        # evaluated against step time; TOTAL TIME would be silently shifted
        raise NotImplementedError(f"*AMPLITUDE {name}: TIME={block.params.get('time')} is not implemented")
    if any(k in block.params for k in ("value", "shift", "scalex", "scaley")):
        raise NotImplementedError(f"*AMPLITUDE {name}: VALUE/SHIFTX/SCALE parameters are not implemented")
    # Up to four (time, value) pairs per data line. Only the first pair of
    # each line used to be read, so a typical "0,0, 1,1" table became a
    # single point and the amplitude a constant 0.
    flat = [float(v) for r in _parse_data_rows(block.data_lines, dtype=float) for v in r if v is not None]
    if len(flat) % 2:
        raise ValueError(f"*AMPLITUDE {name}: odd number of values ({len(flat)}); expected time/value pairs")
    data = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    if "SMOOTHSTEP" in defn:
        reader.model.SmoothStepAmplitude(name, data)
    else:
        reader.model.TabularAmplitude(name, data)


@GLOBAL_KEYWORD_REGISTRY.register(
    "DLOAD",
    valid_contexts={KeywordContext.STEP},
    support="PARTIAL",
    params={
        "AMPLITUDE": ParamSpec("AMPLITUDE", aliases=["AMP"]),
        "OP": ParamSpec("OP", default="MOD", choices=["MOD", "NEW"]),
    },
    rows=RowSpec(min_columns=3, max_columns=7),
    description="Define distributed body loads (gravity, body force, pressure)",
)
def _handle_dload(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    if reader.current_step is None:
        return
    step_name = reader.current_step.name
    amp = block.params.get("amplitude", None)
    op = block.params.get("op", "MOD").upper()
    if op == "NEW":
        for load_name in list(reader.current_step.loads.keys()):
            reader.current_step.deactivate_load(load_name)
    dload_names = []
    faced = []
    # Split the data lines here rather than with _parse_data_rows, which drops
    # empty fields: Abaqus allows a blank region (data line ", GRAV, 9.81, ...")
    # meaning the whole model, and dropping it shifted every column left.
    rows = [[p.strip() for p in ln.split(",")] for ln in block.data_lines if ln.strip()]
    for r in rows:
        while r and r[-1] == "":
            r.pop()
        if len(r) < 3:
            raise ValueError(f"*DLOAD line {','.join(r)!r}: expected 'region, load type, magnitude'")
        elset = str(r[0])
        ftype = str(r[1]).upper()
        mag = float(r[2])
        if ftype == "GRAV":
            dirs = [float(x) if x else 0.0 for x in r[3:6]]
            if len(dirs) < 3:
                raise ValueError(f"*DLOAD GRAV on {elset!r}: direction vector missing")
            reader.model.Gravity(name=f"DLOAD_GRAV_{elset}", createStepName=step_name,
                                 region=elset, comp1=mag * dirs[0], comp2=mag * dirs[1],
                                 comp3=mag * dirs[2], amplitude=amp)
            dload_names.append(f"DLOAD_GRAV_{elset}")
        elif ftype in ("BX", "BY", "BZ"):
            comps = {"BX": (mag, 0.0, 0.0), "BY": (0.0, mag, 0.0), "BZ": (0.0, 0.0, mag)}[ftype]
            reader.model.BodyForce(name=f"DLOAD_{elset}_{ftype}", createStepName=step_name,
                                   region=elset, b1=comps[0], b2=comps[1], b3=comps[2],
                                   amplitude=amp)
            dload_names.append(f"DLOAD_{elset}_{ftype}")
        elif re.fullmatch(r"P[1-6]?", ftype):
            faced.append(ftype)
            reader.model.Pressure(name=f"DLOAD_{elset}_{ftype}", createStepName=step_name,
                                  region=elset, magnitude=mag, amplitude=amp, face=ftype)
            dload_names.append(f"DLOAD_{elset}_{ftype}")
        else:
            raise NotImplementedError(f"*DLOAD load type {ftype} on {elset!r} is not implemented")
    for lname in dload_names:
        entry = reader.current_step.loads.get(lname)
        if entry is not None and entry.entity.region in reader.current_step.step_nsets:
            msg = (f"*DLOAD {lname}: step-overlay region; solver load path "
                   f"resolves global sets only (P2)")
            warnings.warn(msg)
            reader.warnings.append(msg)


@GLOBAL_KEYWORD_REGISTRY.register(
    "DSLOAD",
    valid_contexts={KeywordContext.STEP},
    support="FULL",
    params={
        "AMPLITUDE": ParamSpec("AMPLITUDE", aliases=["AMP"]),
        "OP": ParamSpec("OP", default="MOD", choices=["MOD", "NEW"]),
    },
    rows=RowSpec(column_types=[str, str, float], min_columns=3, max_columns=3),
    description="Define distributed surface loads (pressure on surfaces)",
)
def _handle_dsload(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    """Handle Abaqus *DSLOAD keyword (distributed surface loads, pressure)."""
    if reader.current_step is None:
        return
    step_name = reader.current_step.name
    amp = block.params.get("amplitude", None)
    op = block.params.get("op", "MOD").upper()
    if op == "NEW":
        for load_name in list(reader.current_step.loads.keys()):
            reader.current_step.deactivate_load(load_name)
    rows = _parse_data_rows(block.data_lines, dtype=str)
    for r in rows:
        if len(r) < 3:
            continue
        surf_name = str(r[0])
        ltype = str(r[1]).upper()
        mag = float(r[2])
        if ltype in ("P", "HP", "PN"):
            reader.model.Pressure(
                name=f"DSLOAD_{surf_name}_{ltype}",
                createStepName=step_name,
                region=surf_name,
                magnitude=mag,
                amplitude=amp,
            )


@GLOBAL_KEYWORD_REGISTRY.register(
    "MPC",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        "TYPE": ParamSpec("TYPE", default="BEAM", choices=["BEAM", "TIE", "LINK", "PIN", "SLIDER"]),
    },
    rows=RowSpec(min_columns=2, max_columns=10),
    description="Define multi-point constraints between nodes",
)
def _handle_mpc(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mtype = block.params.get("type", "BEAM").upper()
    nodes = []
    for line in block.data_lines:
        for p in line.replace(",", " ").split():
            p = p.strip()
            if p.lstrip("-").isdigit():
                nodes.append(int(p))
    name = f"MPC_{mtype}_{len(reader.model.constraints) + 1}"
    reader.model.MPC(name, mpcType=mtype,
                     firstPoint=nodes[0] if nodes else None,
                     secondPoint=nodes[1] if len(nodes) > 1 else None)
    msg = f"*MPC {name} parsed and stored; solver assembly not connected"
    warnings.warn(msg)
    reader.warnings.append(msg)


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
        warnings.warn(msg)
        reader.warnings.append(msg)
    else:
        raise _AbortParse(f"*INITIAL CONDITIONS, TYPE={ictype} has no internal landing")


@GLOBAL_KEYWORD_REGISTRY.register(
    "COUPLING",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY},
    support="PARTIAL",
    params={
        "CONSTRAINT NAME": ParamSpec("CONSTRAINT NAME", aliases=["CONSTRAINT_NAME", "NAME"]),
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
        warnings.warn(msg)
        reader.warnings.append(msg)
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
        "CONSTRAINT NAME": ParamSpec("CONSTRAINT NAME", aliases=["CONSTRAINT_NAME", "NAME"]),
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
        "CONSTRAINT NAME": ParamSpec("CONSTRAINT NAME", aliases=["CONSTRAINT_NAME", "NAME"]),
        "REF NODE": ParamSpec("REF NODE", aliases=["REF_NODE", "REFNODE"]),
        "SURFACE": ParamSpec("SURFACE", aliases=["SURF"]),
        "ELSET": ParamSpec("ELSET"),
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
        "CONSTRAINT NAME": ParamSpec("CONSTRAINT NAME", aliases=["CONSTRAINT_NAME", "NAME"]),
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
        "CONSTRAINT NAME": ParamSpec("CONSTRAINT NAME", aliases=["CONSTRAINT_NAME", "NAME"]),
        "REF NODE": ParamSpec("REF NODE", aliases=["REF_NODE", "REFNODE"]),
        "SURFACE": ParamSpec("SURFACE", aliases=["SURF"]),
        "ELSET": ParamSpec("ELSET"),
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
    warnings.warn(msg)
    reader.warnings.append(msg)


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
        warnings.warn(msg)
        reader.warnings.append(msg)
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
    msg = f"*TRANSFORM {name}: parsed and stored; connected to solver"


@GLOBAL_KEYWORD_REGISTRY.register(
    "EQUATION",
    valid_contexts=set(_ALL_CONTEXTS),
    support="FULL",
    params={},
    rows=RowSpec(min_columns=3, max_columns=12, repeating=True),
    description="Define linear multi-point constraint equation between degree of freedoms",
)
def _handle_equation(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    rows = _parse_data_rows(block.data_lines, dtype=str)
    cells: List[str] = []
    for r in rows:
        cells.extend([str(c).strip() for c in r if str(c).strip() != ""])
    if not cells:
        return
    try:
        nterms = int(float(cells[0]))
    except ValueError:
        msg = f"*EQUATION at line {block.line_number}: bad term count — skipped"
        warnings.warn(msg)
        reader.warnings.append(msg)
        return
    triples: List[Tuple[Any, int, float]] = []
    pos = 1
    while pos + 2 < len(cells) + 1 and len(triples) < nterms:
        if pos + 2 >= len(cells):
            break
        ref_raw, dof_raw, coeff_raw = cells[pos], cells[pos + 1], cells[pos + 2]
        pos += 3
        try:
            dof = int(float(dof_raw)) - 1
            coeff = float(coeff_raw)
        except ValueError:
            continue
        if dof < 0 or dof > 5:
            msg = (f"*EQUATION at line {block.line_number}: dof {dof + 1} out of "
                   "range 1..6 — equation skipped")
            warnings.warn(msg)
            reader.warnings.append(msg)
            triples = []
            break
        if ref_raw.lstrip("-").isdigit():
            triples.append((int(ref_raw), dof, coeff))
        else:
            members = _resolve_equation_nodes(reader, ref_raw)
            if not members:
                msg = (f"*EQUATION at line {block.line_number}: set '{ref_raw}' "
                       "unresolvable — equation skipped")
                warnings.warn(msg)
                reader.warnings.append(msg)
                if "EQUATION" not in reader.unsupported_keywords:
                    reader.unsupported_keywords.append(f"EQUATION set {ref_raw}")
                triples = []
                break
            for nid in members:
                triples.append((int(nid), dof, coeff))
    if not triples:
        return
    if len(triples) != nterms:
        msg = (f"*EQUATION at line {block.line_number}: term count {len(triples)} "
               f"!= declared {nterms} — kept as parsed")
        warnings.warn(msg)
        reader.warnings.append(msg)
    name = f"EQ_{len(reader.model.constraints) + 1}"
    reader.model.Equation(name, terms=triples)
    msg = (f"*EQUATION {name}: {len(triples)} terms parsed; "
           "penalty assembly connected")
    warnings.warn(msg)
    reader.warnings.append(msg)


def _resolve_equation_nodes(reader: AbaqusInputReader, ref: str) -> List[int]:
    """Resolve an EQUATION term reference to part-local node ids."""
    names = [ref, ref.split(".")[-1]]
    for part in reader.model.parts.values():
        for key in names:
            if key in part.node_sets:
                return sorted(int(n) for n in part.node_sets[key].node_ids)
            if key in part.sets:
                gset = part.sets[key]
                ids = getattr(gset, "node_ids", None)
                if ids:
                    return sorted(int(n) for n in ids)
    return []



