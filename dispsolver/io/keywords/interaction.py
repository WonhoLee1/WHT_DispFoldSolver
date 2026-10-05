from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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

# ─── CONSTRAINTS & INTERACTIONS ──────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register(
    "TIE",
    valid_contexts={KeywordContext.ASSEMBLY, KeywordContext.GLOBAL},
    support="PARTIAL",
    params={
        "NAME": ParamSpec("NAME"),
        "POSITION TOLERANCE": ParamSpec("POSITION TOLERANCE", type=float, default=0.0, aliases=["POSITION_TOLERANCE"]),
        "ADJUST": ParamSpec("ADJUST", default="YES", choices=["YES", "NO"]),
        "TYPE": ParamSpec("TYPE", supported=False),
    },
    rows=RowSpec(column_types=[str, str], min_columns=2, max_columns=2),
    description="Tie constraint between two surfaces",
)
def _handle_tie(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"TIE-{len(reader.model.root_assembly.ties) + 1}")
    pos_tol = float(block.params.get("position tolerance", block.params.get("position_tolerance", 0.0)))
    adjust = block.params.get("adjust", "yes").lower() == "yes"

    master = None
    slave = None
    pairs = []
    for line in block.data_lines:
        tokens = [t.strip() for t in line.split(",") if t.strip()]
        if len(tokens) >= 2:
            pairs.append((tokens[0], tokens[1]))

    for i, (slave, master) in enumerate(pairs):
        tie_name = name if len(pairs) == 1 else f"{name}_{i + 1}"
        reader.model.Tie(name=tie_name, master=master, slave=slave, positionTolerance=pos_tol, adjust=adjust)


@GLOBAL_KEYWORD_REGISTRY.register(
    "RIGID BODY",
    valid_contexts={KeywordContext.PART, KeywordContext.ASSEMBLY, KeywordContext.GLOBAL},
    aliases=["RIGIDBODY"],
    support="PARTIAL",
    params={
        "REF NODE": ParamSpec("REF NODE", aliases=["REF_NODE", "REFNODE"]),
        "ELSET": ParamSpec("ELSET"),
        "TIE NSET": ParamSpec("TIE NSET", aliases=["TIE_NSET"]),
        "PIN NSET": ParamSpec("PIN NSET", aliases=["PIN_NSET"]),
        "ANALYTICAL SURFACE": ParamSpec("ANALYTICAL SURFACE", aliases=["ANALYTICAL_SURFACE"]),
        "ISOTHERMAL": ParamSpec("ISOTHERMAL", supported=False),
        "POSITION": ParamSpec("POSITION", supported=False),
    },
    description="Rigid body constraint definition",
)
def _handle_rigid_body(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"RB_{len(reader.model.constraints) + 1}")
    rp = block.params.get("ref node", block.params.get("ref_node", block.params.get("refnode", None)))
    elset = block.params.get("elset", None)
    tie_nset = block.params.get("tie nset", block.params.get("tie_nset",
                 block.params.get("nset", None)))
    pin_nset = block.params.get("pin nset", block.params.get("pin_nset", None))

    analytical_surf = block.params.get("analytical surface", block.params.get("analytical_surface", None))
    reader.model.RigidBody(name=name, refPoint=rp, elset=elset, tieNset=tie_nset, pinNset=pin_nset)

    if analytical_surf:
        # Link reference node to analytical rigid surface if present
        ref_nid = int(rp) if rp is not None and str(rp).isdigit() else None
        if not hasattr(reader, "pending_analytical_surfaces"):
            reader.pending_analytical_surfaces = {}
        reader.pending_analytical_surfaces[analytical_surf] = ref_nid

        if hasattr(reader.model, "analytical_rigid_surfaces") and analytical_surf in reader.model.analytical_rigid_surfaces:
            reader.model.analytical_rigid_surfaces[analytical_surf].reference_node = ref_nid
        part = reader.get_active_part()
        if part is not None and hasattr(part, "surfaces") and analytical_surf in part.surfaces:
            surf = part.surfaces[analytical_surf]
            if hasattr(surf, "reference_node"):
                surf.reference_node = ref_nid


@GLOBAL_KEYWORD_REGISTRY.register(
    "SURFACE INTERACTION",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP},
    aliases=["SURFACEINTERACTION"],
    support="FULL",
    params={"NAME": ParamSpec("NAME")},
    description="Contact property / surface interaction container",
)
def _handle_surface_interaction(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"INTERACTION-{len(reader.model.contact_properties) + 1}")
    reader.model.ContactProperty(name=name)
    reader.current_interaction = name


@GLOBAL_KEYWORD_REGISTRY.register(
    "CONTACT PAIR",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.STEP},
    aliases=["CONTACTPAIR"],
    support="FULL",
    params={
        "INTERACTION": ParamSpec("INTERACTION", default="DEFAULT"),
        "SMALL SLIDING": ParamSpec("SMALL SLIDING", type=bool, default=False, supported=True, aliases=["SMALL_SLIDING"]),
        "ADJUST": ParamSpec("ADJUST", supported=False),
        "TYPE": ParamSpec("TYPE", supported=False),
        "WEIGHT": ParamSpec("WEIGHT", supported=False),
        "TIED": ParamSpec("TIED", supported=False),
        "CPSET": ParamSpec("CPSET", supported=False),
        "TRACKING": ParamSpec("TRACKING", supported=False),
    },
    rows=RowSpec(column_types=[str, str], min_columns=2, max_columns=2),
    description="Contact pair between master and slave surfaces",
)
def _handle_contact_pair(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    interaction = block.params.get("interaction", "DEFAULT")
    # Abaqus: *CONTACT PAIR is FINITE sliding unless SMALL SLIDING is given.
    sliding = "SMALL" if "small sliding" in block.params else "FINITE"
    if block.data_lines:
        for line in block.data_lines:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                slave, master = parts[0], parts[1]
                name = f"CP_{slave}_{master}"
                reader.model.ContactPair(name=name, main=master, secondary=slave, interactionProperty=interaction,
                                         sliding=sliding)


@GLOBAL_KEYWORD_REGISTRY.register(
    "SURFACE BEHAVIOR",
    valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP},
    aliases=["SURFACEBEHAVIOR"],
    support="FULL",
    params={
        "PRESSURE OVERCLOSURE": ParamSpec(
            "PRESSURE OVERCLOSURE",
            default="HARD",
            choices=["HARD", "LINEAR", "SOFT_LINEAR", "EXPONENTIAL", "SOFT_EXPONENTIAL", "TABULAR", "SOFT_TABULAR"],
            aliases=["PRESSURE_OVERCLOSURE", "PRESSURE-OVERCLOSURE"],
        ),
        "NO SEPARATION": ParamSpec("NO SEPARATION", supported=False),
    },
    description="Normal contact surface behavior and pressure-overclosure relationship",
)
def _handle_surface_behavior(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    target = reader.current_interaction
    if target is None or target not in reader.model.contact_properties:
        raise ValueError("*SURFACE BEHAVIOR outside a *SURFACE INTERACTION")
    prop = reader.model.contact_properties[target]
    po = str(block.params.get("pressure-overclosure", block.params.get("pressure_overclosure", "HARD"))).upper()
    po = po.replace("SOFT_", "")
    if po not in ("HARD", "LINEAR", "EXPONENTIAL", "TABULAR"):
        raise NotImplementedError(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE={po} is not implemented")
    # Data lines (Abaqus): LINEAR k | EXPONENTIAL c0, p0 | TABULAR (p_i, h_i)
    # pairs -- the same layout ContactProperty.NormalBehavior's table takes.
    # They used to be ignored (and the PRESSURE-OVERCLOSURE parameter itself
    # was lost by the lexer, so every deck contact ran HARD).
    rows = [[v for v in r if v is not None] for r in _parse_data_rows(block.data_lines, dtype=float)]
    rows = [r for r in rows if r]
    if po != "HARD" and not rows:
        raise ValueError(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE={po}: data line missing")
    if po == "EXPONENTIAL" and len(rows[0]) < 2:
        raise ValueError("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=EXPONENTIAL: expected 'c0, p0'")
    if po == "TABULAR" and any(len(r) < 2 for r in rows):
        raise ValueError("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TABULAR: expected 'pressure, overclosure' pairs")
    prop.NormalBehavior(pressureOverclosure=po, table=[tuple(r) for r in rows] if rows else None)





