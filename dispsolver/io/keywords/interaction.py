from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext

if TYPE_CHECKING:
    from dispsolver.io.abaqus_reader import AbaqusInputReader

# ─── CONSTRAINTS & INTERACTIONS ──────────────────────────────────────────────

@GLOBAL_KEYWORD_REGISTRY.register("TIE", valid_contexts={KeywordContext.ASSEMBLY, KeywordContext.GLOBAL})
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


@GLOBAL_KEYWORD_REGISTRY.register("RIGID BODY", valid_contexts={KeywordContext.PART, KeywordContext.ASSEMBLY, KeywordContext.GLOBAL}, aliases=["RIGIDBODY"])
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


@GLOBAL_KEYWORD_REGISTRY.register("SURFACE INTERACTION", valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP}, aliases=["SURFACEINTERACTION"])
def _handle_surface_interaction(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    name = block.params.get("name", f"INTERACTION-{len(reader.model.contact_properties) + 1}")
    reader.model.ContactProperty(name=name)
    reader.current_interaction = name


@GLOBAL_KEYWORD_REGISTRY.register("CONTACT PAIR", valid_contexts={KeywordContext.GLOBAL, KeywordContext.ASSEMBLY, KeywordContext.STEP}, aliases=["CONTACTPAIR"])
def _handle_contact_pair(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    interaction = block.params.get("interaction", "DEFAULT")
    if block.data_lines:
        for line in block.data_lines:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                slave, master = parts[0], parts[1]
                name = f"CP_{slave}_{master}"
                reader.model.ContactPair(name=name, main=master, secondary=slave, interactionProperty=interaction)


@GLOBAL_KEYWORD_REGISTRY.register("SURFACE BEHAVIOR", valid_contexts={KeywordContext.GLOBAL, KeywordContext.STEP}, aliases=["SURFACEBEHAVIOR"])
def _handle_surface_behavior(reader: AbaqusInputReader, block: AbaqusKeywordBlock):
    target = reader.current_interaction
    if target is None or target not in reader.model.contact_properties:
        return
    prop = reader.model.contact_properties[target]
    po = block.params.get("pressure-overclosure", block.params.get("pressure_overclosure", "HARD")).upper()
    if po == "HARD":
        prop.normal_behavior = "HARD"
    elif po in ("LINEAR", "SOFT_LINEAR"):
        prop.normal_behavior = "SOFT_LINEAR"
    elif po in ("EXPONENTIAL", "SOFT_EXPONENTIAL"):
        prop.normal_behavior = "SOFT_EXPONENTIAL"
    elif po in ("TABULAR", "SOFT_TABULAR"):
        prop.normal_behavior = "SOFT_TABULAR"




