from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext
from dispsolver.io.keyword_base import _AbortParse, _parse_data_rows
from dispsolver.model.step import PredefinedField

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── P0-2 MATERIAL SUBOPTIONS, LOADS, CONSTRAINTS ───────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("DENSITY", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_density(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        msg = "*DENSITY outside *MATERIAL scope — skipped"
        warnings.warn(msg)
        reader.warnings.append(msg)
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows and len(rows[0]) >= 1:
        mat.density = float(rows[0][0])


@GLOBAL_KEYWORD_REGISTRY.register("TRS", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
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
    if len(vals) >= 3:
        mat.extra_params["trs"] = {"definition": defn, "T_ref": float(vals[0]),
                                   "C1": float(vals[1]), "C2": float(vals[2])}
    else:
        mat.extra_params["trs"] = {"definition": defn, "values": tuple(float(v) for v in vals)}


@GLOBAL_KEYWORD_REGISTRY.register("AMPLITUDE", valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP})
def _handle_amplitude(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", "")
    if not name:
        msg = "*AMPLITUDE without NAME — skipped"
        warnings.warn(msg)
        reader.warnings.append(msg)
        return
    defn = block.params.get("definition", "TABULAR").upper().replace(" ", "")
    rows = _parse_data_rows(block.data_lines, dtype=float)
    data = [(float(r[0]), float(r[1])) for r in rows if len(r) >= 2]
    if "SMOOTHSTEP" in defn:
        reader.model.SmoothStepAmplitude(name, data)
    else:
        reader.model.TabularAmplitude(name, data)


@GLOBAL_KEYWORD_REGISTRY.register("DLOAD", valid_contexts={KeywordContext.STEP})
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
    rows = _parse_data_rows(block.data_lines, dtype=str)
    for r in rows:
        if len(r) < 3:
            continue
        elset = str(r[0])
        ftype = str(r[1]).upper()
        mag = float(r[2])
        if ftype == "GRAV":
            dirs = [float(x) for x in r[3:6]]
            if len(dirs) < 3:
                msg = f"*DLOAD GRAV on {elset} missing direction — skipped"
                warnings.warn(msg)
                reader.warnings.append(msg)
                continue
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
        elif ftype.startswith("P") or ftype.startswith("S"):
            faced.append(ftype)
            reader.model.Pressure(name=f"DLOAD_{elset}_{ftype}", createStepName=step_name,
                                  region=elset, magnitude=mag, amplitude=amp)
            dload_names.append(f"DLOAD_{elset}_{ftype}")
        else:
            msg = f"*DLOAD face {ftype} on {elset} unsupported — skipped"
            warnings.warn(msg)
            reader.warnings.append(msg)
    if faced:
        msg = (f"*DLOAD faces {sorted(set(faced))} kept in load names; "
               f"face-selective solver mapping is P1")
        warnings.warn(msg)
        reader.warnings.append(msg)
    for lname in dload_names:
        entry = reader.current_step.loads.get(lname)
        if entry is not None and entry.entity.region in reader.current_step.step_nsets:
            msg = (f"*DLOAD {lname}: step-overlay region; solver load path "
                   f"resolves global sets only (P2)")
            warnings.warn(msg)
            reader.warnings.append(msg)


@GLOBAL_KEYWORD_REGISTRY.register("MPC", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY})
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


@GLOBAL_KEYWORD_REGISTRY.register("INITIAL CONDITIONS", valid_contexts={KeywordContext.GLOBAL}, aliases=["INITIALCONDITIONS"])
def _handle_initial_conditions(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    ictype = block.params.get("type", "TEMPERATURE").upper()
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
    else:
        raise _AbortParse(f"*INITIAL CONDITIONS, TYPE={ictype} has no internal landing")


@GLOBAL_KEYWORD_REGISTRY.register("COUPLING", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY})
def _handle_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("constraint name", block.params.get("name", f"CPL-{len(reader.model.constraints) + 1}"))
    ref = block.params.get("ref node", block.params.get("ref_node", None))
    surf = block.params.get("surface", None)
    reader.model.KinematicCoupling(name=name, refPoint=ref, surface=surf)
    reader.current_coupling = {"name": name}
    msg = f"*COUPLING {name} parsed and stored; solver assembly not connected"
    warnings.warn(msg)
    reader.warnings.append(msg)


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
    msg = f"*{kind} {name} parsed and stored; solver assembly not connected"
    warnings.warn(msg)
    reader.warnings.append(msg)


@GLOBAL_KEYWORD_REGISTRY.register("KINEMATIC", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY})
def _handle_kinematic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _finalize_coupling(reader, block, "KINEMATIC")


@GLOBAL_KEYWORD_REGISTRY.register("KINEMATIC COUPLING", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY})
def _handle_kinematic_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _direct_coupling(reader, block, "KINEMATIC")


@GLOBAL_KEYWORD_REGISTRY.register("DISTRIBUTING", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY})
def _handle_distributing(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _finalize_coupling(reader, block, "DISTRIBUTING")


@GLOBAL_KEYWORD_REGISTRY.register("DISTRIBUTING COUPLING", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY})
def _handle_distributing_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    _direct_coupling(reader, block, "DISTRIBUTING")


def _direct_coupling(reader: AbaqusInputReader, block: AbaqusKeywordBlock, kind: str):
    name = block.params.get("constraint name", block.params.get("name", f"{kind}_CPL_{len(reader.model.constraints) + 1}"))
    ref = block.params.get("ref node", block.params.get("ref_node", None))
    surf = block.params.get("surface", block.params.get("elset", None))
    if kind == "KINEMATIC":
        reader.model.KinematicCoupling(name=name, refPoint=ref, surface=surf)
    else:
        reader.model.DistributingCoupling(name=name, refPoint=ref, surface=surf)
    reader.current_coupling = {"name": name}
    msg = f"*{kind} COUPLING {name} parsed and stored; solver assembly not connected"
    warnings.warn(msg)
    reader.warnings.append(msg)


@GLOBAL_KEYWORD_REGISTRY.register("FRICTION", valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP})
def _handle_friction(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if not rows or len(rows[0]) < 1:
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


@GLOBAL_KEYWORD_REGISTRY.register("INCLUDE", valid_contexts=set(_ALL_CONTEXTS))
def _handle_include(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    raise _AbortParse("*INCLUDE file insertion is not implemented; split the deck first")


@GLOBAL_KEYWORD_REGISTRY.register("ORIENTATION", valid_contexts=set(_ALL_CONTEXTS))
def _handle_orientation(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    raise _AbortParse("*ORIENTATION has no internal landing; local material axes would be misread")


@GLOBAL_KEYWORD_REGISTRY.register("TRANSFORM", valid_contexts=set(_ALL_CONTEXTS))
def _handle_transform(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    raise _AbortParse("*TRANSFORM has no internal landing; nodal transforms would be misread")


@GLOBAL_KEYWORD_REGISTRY.register("EQUATION", valid_contexts=set(_ALL_CONTEXTS))
def _handle_equation(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    raise _AbortParse("*EQUATION has no internal landing; linear multi-point constraints would be dropped")



