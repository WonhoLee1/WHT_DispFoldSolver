from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext
from dispsolver.io.keyword_base import _parse_data_rows

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── MATERIAL & SECTION DEFINITIONS ─────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("MATERIAL", valid_contexts={KeywordContext.GLOBAL})
def _handle_material(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"MAT-{len(reader.model.materials) + 1}")
    mat = reader.model.Material(name)
    reader.current_material = mat
    reader.context = KeywordContext.MATERIAL


@GLOBAL_KEYWORD_REGISTRY.register("ELASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_elastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        mat = reader.model.Material("DEFAULT_ELASTIC")
        reader.current_material = mat

    exp_type = block.params.get("type", "ISOTROPIC").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        flat_tokens = [v for r in rows for v in r]
        table = [tuple(r) for r in rows]
        if "dependencies" in block.params:
            mat.Elastic(field_table=tuple(table), type=exp_type, flat_tokens=flat_tokens)
        else:
            mat.Elastic(table=tuple(table), type=exp_type, flat_tokens=flat_tokens)


@GLOBAL_KEYWORD_REGISTRY.register("EXPANSION", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_expansion(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    exp_type = block.params.get("type", "ISO").upper()
    zero_val = float(block.params.get("zero", 0.0))
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        table = [tuple(r) for r in rows]
        mat.Expansion(type=exp_type, table=tuple(table), zero=zero_val)


@GLOBAL_KEYWORD_REGISTRY.register("PLASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_plastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    hardening = block.params.get("hardening", "ISOTROPIC").upper().replace(" ", "_").replace("-", "_")
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        table = [tuple(r) for r in rows]
        if hardening == "JOHNSON_COOK":
            # Abaqus row: A, B, n, m, melting-temp, transition-temp.
            r0 = table[0]
            mat.JohnsonCook(A=float(r0[0]), B=float(r0[1]), n=float(r0[2]),
                            m=float(r0[3]) if len(r0) > 3 else 0.0,
                            T_melt=float(r0[4]) if len(r0) > 4 else 1793.15,
                            T_room=float(r0[5]) if len(r0) > 5 else 293.15)
            msg = (f"*PLASTIC HARDENING=JOHNSON_COOK {mat.name}: quasi-static "
                   "isothermal landing (C=0, dispatcher carries no rate)")
            warnings.warn(msg)
            reader.warnings.append(msg)
        elif "dependencies" in block.params:
            mat.Plastic(table=tuple(table), hardening=hardening,
                        field_table=tuple(table))
        else:
            mat.Plastic(table=tuple(table), hardening=hardening)


@GLOBAL_KEYWORD_REGISTRY.register("CREEP", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_creep(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    law = block.params.get("law", "STRAIN").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        table = [tuple(r) for r in rows]
        mat.Creep(law=law, table=tuple(table))


@GLOBAL_KEYWORD_REGISTRY.register("HYPERELASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_hyperelastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    model_type = "NEO_HOOKEAN"
    bare = {str(k).lower().replace("-", "").replace("_", "").replace(" ", "")
            for k in block.params}
    for known in ["ARRUDA_BOYCE", "YEOH", "MOONEY_RIVLIN", "OGDEN", "POLYNOMIAL",
                  "NEO_HOOKE", "NEO_HOOKEAN", "MARLOW", "REDUCED_POLYNOMIAL"]:
        if known.lower().replace("_", "") in bare:
            model_type = known
            break
    if "type" in block.params:
        model_type = block.params["type"].upper().replace(" ", "_").replace("-", "_")
    if model_type == "NEO_HOOKE":
        model_type = "NEO_HOOKEAN"
    n_terms = int(float(block.params.get("n", 1)))
    rows = _parse_data_rows(block.data_lines, dtype=float)
    table = [tuple(r) for r in rows] if rows else ()
    mat.Hyperelastic(model=model_type, n=n_terms, table=table)


@GLOBAL_KEYWORD_REGISTRY.register("VISCOELASTIC", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_viscoelastic(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    time_domain = block.params.get("time", "PRONY").upper()
    rows = _parse_data_rows(block.data_lines, dtype=float)
    table = [tuple(r) for r in rows] if rows else ()
    mat.Viscoelastic(time=time_domain, table=table)


@GLOBAL_KEYWORD_REGISTRY.register("RATE DEPENDENT", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_rate_dependent(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    rd_type = block.params.get("type", "POWER LAW").upper().replace(" ", "_").replace("-", "_")
    rows = _parse_data_rows(block.data_lines, dtype=float)
    table = tuple(tuple(r) for r in rows) if rows else None
    mat.RateDependent(type=rd_type, table=table)


@GLOBAL_KEYWORD_REGISTRY.register("DAMAGE INITIATION", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_damage_initiation(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    crit = block.params.get("criterion", "DUCTILE").upper().replace(" ", "_").replace("-", "_")
    if crit not in ("DUCTILE", "JOHNSON_COOK", "JOHNSONCOOK", "JC"):
        _record_unsupported(reader, block, f"DAMAGE INITIATION CRITERION={crit}")
        return
    if crit in ("JOHNSONCOOK", "JC"):
        crit = "JOHNSON_COOK"
    rows = _parse_data_rows(block.data_lines, dtype=float)
    table = tuple(tuple(r) for r in rows) if rows else None
    mat.DamageInitiation(type=crit, table=table)


@GLOBAL_KEYWORD_REGISTRY.register("DAMAGE EVOLUTION", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_damage_evolution(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    evo_type = block.params.get("type", "ENERGY").upper()
    soft = block.params.get("softening", "LINEAR").upper()
    if evo_type != "ENERGY" or soft not in ("LINEAR", "EXPONENTIAL"):
        _record_unsupported(reader, block, f"DAMAGE EVOLUTION TYPE={evo_type} SOFTENING={soft}")
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    gf = float(rows[0][0]) if rows and rows[0] else 1.0
    mat.DamageEvolution(type=f"ENERGY_{soft}", Gf=gf)


@GLOBAL_KEYWORD_REGISTRY.register("DRUCKER PRAGER", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_drucker_prager(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    beta = float(block.params.get("angle", 20.0))
    psi = float(block.params.get("dilation angle", block.params.get("dilation_angle", 20.0)))
    rows = _parse_data_rows(block.data_lines, dtype=float)
    d0 = float(rows[0][0]) if rows and rows[0] else 100.0
    mat.DruckerPrager(beta=beta, psi=psi, d0=d0)


@GLOBAL_KEYWORD_REGISTRY.register("DRUCKER PRAGER HARDENING", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_drucker_prager_hardening(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None or "drucker_prager" not in mat.extra_params:
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if len(rows) >= 2 and len(rows[0]) >= 1 and len(rows[1]) >= 1:
        d0, e0 = float(rows[0][0]), float(rows[0][1]) if len(rows[0]) > 1 else 0.0
        d1, e1 = float(rows[1][0]), float(rows[1][1]) if len(rows[1]) > 1 else 0.0
        if e1 > e0:
            mat.extra_params["drucker_prager"]["H"] = (d1 - d0) / (e1 - e0)


def _try_finalize_cdp(reader: AbaqusInputReader, mat) -> None:
    ep = mat.extra_params
    if "concrete_dp" in ep or "cdp_params" not in ep:
        return
    if "cdp_compression" not in ep or "cdp_tension" not in ep:
        return
    if "cdp_dc" not in ep or "cdp_dt" not in ep:
        _record_unsupported(reader, None, "CONCRETE_DAMAGED_PLASTICITY without damage tables")
        return
    p = ep["cdp_params"]
    premature = "CONCRETE_DAMAGED_PLASTICITY without damage tables"
    if premature in reader.unsupported_keywords:
        reader.unsupported_keywords.remove(premature)
    try:
        mat.ConcreteDP(beta=p.get("beta", 20.0), psi=p.get("psi", 15.0),
                       compression=ep["cdp_compression"], tension=ep["cdp_tension"],
                       dt_table=ep["cdp_dt"], dc_table=ep["cdp_dc"])
    except ValueError as e:
        _record_unsupported(reader, None, f"CONCRETE_DAMAGED_PLASTICITY table rejected ({e})")
        return


@GLOBAL_KEYWORD_REGISTRY.register("CONCRETE DAMAGED PLASTICITY", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_concrete_dp(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    for ignored in ("eccentricity", "fb0/fc0", "fb0_fc0", "k", "viscosity"):
        if ignored in block.params:
            _record_unsupported(reader, block, f"CONCRETE DAMAGED PLASTICITY option {ignored}")
    mat.extra_params["cdp_params"] = {
        "psi": float(block.params.get("angle", 15.0)),
        "beta": 20.0,
    }
    _try_finalize_cdp(reader, mat)


@GLOBAL_KEYWORD_REGISTRY.register("CONCRETE COMPRESSION HARDENING", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_cdp_compression(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        mat.extra_params["cdp_compression"] = [tuple(r[:2]) for r in rows]
        _try_finalize_cdp(reader, mat)


@GLOBAL_KEYWORD_REGISTRY.register("CONCRETE TENSION STIFFENING", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_cdp_tension(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    stype = block.params.get("type", "STRAIN").upper()
    if stype != "STRAIN":
        _record_unsupported(reader, block, f"CONCRETE TENSION STIFFENING TYPE={stype}")
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        mat.extra_params["cdp_tension"] = [tuple(r[:2]) for r in rows]
        _try_finalize_cdp(reader, mat)


@GLOBAL_KEYWORD_REGISTRY.register("CONCRETE COMPRESSION DAMAGE", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_cdp_dc(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        mat.extra_params["cdp_dc"] = [tuple(r[:2]) for r in rows]
        _try_finalize_cdp(reader, mat)


@GLOBAL_KEYWORD_REGISTRY.register("CONCRETE TENSION DAMAGE", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_cdp_dt(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    stype = block.params.get("type", "STRAIN").upper()
    if stype != "STRAIN":
        _record_unsupported(reader, block, f"CONCRETE TENSION DAMAGE TYPE={stype}")
        return
    rows = _parse_data_rows(block.data_lines, dtype=float)
    if rows:
        mat.extra_params["cdp_dt"] = [tuple(r[:2]) for r in rows]
        _try_finalize_cdp(reader, mat)


@GLOBAL_KEYWORD_REGISTRY.register("MULLINS EFFECT", valid_contexts={KeywordContext.MATERIAL, KeywordContext.GLOBAL})
def _handle_mullins(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    mat = reader.current_material
    if mat is None:
        return
    r = float(block.params.get("r", 2.0))
    m = float(block.params.get("m", 0.5))
    beta = float(block.params.get("beta", 0.1))
    rows = _parse_data_rows(block.data_lines, dtype=float)
    table = tuple(tuple(x) for x in rows) if rows else None
    mat.MullinsEffect(r=r, m=m, beta=beta, table=table)


def _record_unsupported(reader: AbaqusInputReader, block, label: str) -> None:
    if label not in reader.unsupported_keywords:
        reader.unsupported_keywords.append(label)
    line = getattr(block, "line_number", "?") if block is not None else "?"
    msg = f"*{label} at line {line} - no internal landing, recorded as unsupported"
    warnings.warn(msg)
    reader.warnings.append(msg)


@GLOBAL_KEYWORD_REGISTRY.register("ORIENTATION", valid_contexts={KeywordContext.GLOBAL, KeywordContext.PART})
def _handle_orientation(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"ORI-{len(reader.model.orientations) + 1}")
    system = block.params.get("system", "RECTANGULAR").upper()
    definition = block.params.get("definition", block.params.get("def", "COORDINATES")).upper()
    local_dirs = int(float(block.params.get("local directions",
                                            block.params.get("local_directions", 1))))
    float_rows = _parse_data_rows(block.data_lines, dtype=float)
    raw_rows = tuple(tuple(r) for r in float_rows) if float_rows else ()
    frame = None
    node_ids: tuple = ()
    if definition == "NODES":
        ids: list = []
        for line in block.data_lines:
            for p in line.replace(",", " ").split():
                p = p.strip()
                if p.lstrip("-").isdigit():
                    ids.append(int(p))
        node_ids = tuple(ids)
    elif system == "RECTANGULAR" and raw_rows and len(raw_rows[0]) >= 6:
        import numpy as np
        a = np.array(raw_rows[0][:3], dtype=np.float64)
        b = np.array(raw_rows[0][3:6], dtype=np.float64)
        na = float(np.linalg.norm(a))
        if na > 1e-12:
            a = a / na
        nb = float(np.linalg.norm(b - a * float(np.dot(b, a))))
        if nb > 1e-12:
            b = (b - a * float(np.dot(b, a))) / nb
        c = np.cross(a, b)
        frame = np.array([a, b, c], dtype=np.float64)
    reader.model.Orientation(name=name, system=system, definition=definition,
                             frame=frame, raw_rows=raw_rows, node_ids=node_ids,
                             local_directions=local_dirs)
    if frame is None or system != "RECTANGULAR" or definition == "NODES":
        msg = (f"*ORIENTATION {name}: parsed and stored; material-kernel "
               "consumption not connected")
        warnings.warn(msg)
        reader.warnings.append(msg)


def _section_part(reader, elset: str, kind: str):
    """Part a *SOLID/*SHELL SECTION applies to. Inside a *PART: that part,
    which must define the ELSET. Outside any part (written after *END PART):
    the one part that defines the ELSET; several candidates are ambiguous.
    The old fallback took the FIRST part with that ELSET name."""
    if reader.context == KeywordContext.PART or elset.upper() == "ALL":
        part = reader.get_active_part()
        if elset.upper() != "ALL" and elset not in part.element_sets and elset not in part.sets:
            raise KeyError(f"*{kind} SECTION: ELSET {elset!r} is not defined in part {part.name!r}")
        return part
    owners = [p for p in reader.model.parts.values() if elset in p.element_sets or elset in p.sets]
    if len(owners) == 1:
        return owners[0]
    if not owners:
        raise KeyError(f"*{kind} SECTION: ELSET {elset!r} is not defined in any part")
    raise KeyError(f"*{kind} SECTION: ELSET {elset!r} is defined in parts {[p.name for p in owners]}; "
                   f"put the section inside the *PART it belongs to")


@GLOBAL_KEYWORD_REGISTRY.register("SOLID SECTION", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE}, aliases=["SOLIDSECTION"])
def _handle_solid_section(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    elset = block.params.get("elset", "ALL")
    mat_name = block.params.get("material", "DEFAULT")
    # Section names are per part: Abaqus CAE writes the same ELSET name
    # (e.g. _PickedSet2) in every part, and a model-wide "SEC_SOLID_<elset>"
    # made the last part's section overwrite every earlier part's material.
    part = _section_part(reader, elset, "SOLID")
    sec_name = f"SEC_SOLID_{part.name}_{elset}"
    orientation = block.params.get("orientation", None)

    thickness = 1.0
    is_composite = any("COMPOSITE" in str(k).upper() for k in block.params.keys())
    if block.data_lines:
        if is_composite:
            # Composite data line: thickness, num_int_pts, mat_name, [orientation, ply_name]
            first_line = [tok.strip() for tok in block.data_lines[0].split(',') if tok.strip()]
            if len(first_line) >= 1:
                try:
                    thickness = float(first_line[0])
                except ValueError:
                    pass
            if len(first_line) >= 3:
                mat_name = first_line[2]
            if len(first_line) >= 4 and first_line[3]:
                orientation = first_line[3]
        else:
            rows = _parse_data_rows(block.data_lines, dtype=float)
            if rows and len(rows[0]) >= 1:
                thickness = float(rows[0][0])

    sec = reader.model.HomogeneousSolidSection(name=sec_name, material=mat_name,
                                               thickness=thickness,
                                               orientation=orientation)
    
    part.SectionAssignment(region=elset, sectionName=sec_name)


@GLOBAL_KEYWORD_REGISTRY.register("SHELL SECTION", valid_contexts={KeywordContext.PART, KeywordContext.GLOBAL, KeywordContext.INSTANCE}, aliases=["SHELLSECTION"])
def _handle_shell_section(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    elset = block.params.get("elset", "ALL")
    mat_name = block.params.get("material", "DEFAULT")
    # Section names are per part: Abaqus CAE writes the same ELSET name
    # (e.g. _PickedSet2) in every part, and a model-wide "SEC_SHELL_<elset>"
    # made the last part's section overwrite every earlier part's material.
    part = _section_part(reader, elset, "SHELL")
    sec_name = f"SEC_SHELL_{part.name}_{elset}"

    thickness = 1.0
    if block.data_lines:
        rows = _parse_data_rows(block.data_lines, dtype=float)
        if rows and len(rows[0]) >= 1:
            thickness = float(rows[0][0])

    sec = reader.model.ShellSection(name=sec_name, material=mat_name, thickness=thickness)
    
    # Locate part containing this elset
    part.SectionAssignment(region=elset, sectionName=sec_name)



