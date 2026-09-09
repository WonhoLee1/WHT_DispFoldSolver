import numpy as np
import os
import sys
from dispsolver.mesh.plate_builder import create_folding_plate_parts
from dispsolver.mesh.display_builder import build_display_grid, graded_display_x

sys.path.insert(0, os.path.dirname(__file__))
from dispsolver.fold_model_config import FoldModelConfig, DEFAULT_CONFIG, make_teardrop_config
from dispsolver.material.factory import emit_abaqus_material_block

OUTPUT = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")

# Backward-compatible alias -- the graded-x and node/element generation
# moved to dispsolver/mesh/display_builder.py so the .inp writer and
# ex13's pure-Python builder share one implementation (they had already
# drifted apart, and void regions change node numbering in both).
_graded_display_x = graded_display_x


def generate(config: FoldModelConfig = None):
    if config is None:
        config = make_teardrop_config()
    lines = []
    _w = lines.append

    _w("**")
    _w("** ex12_rigid_plate_display_fold.inp")
    _w("** 90-degree folding with rigid plate + surface tie (High-Fidelity Mesh)")
    _w("**")
    _w("*HEADING")
    _w("Display folding with rigid plate and tie constraints")
    _w("**")

    # 1. Display Mesh (graded x -- see _graded_display_x() docstring)
    # 14 REAL PHYSICAL LAYERS (not just mesh rows): 7 repeats of
    # [PET substrate, 0.05mm, 3 element rows] + [PSA adhesive, 0.03mm,
    # 1 element row] -- see AGENTS.md 1.4, still a simplified alternating
    # layup, not a validated real stackup. Total thickness
    # 7*(0.05+0.03) = 0.56mm (was 0.5mm with the old uniform 14-row model).
    #
    # Materials are defined ONCE per distinct name (PET/PSA/STEEL, via
    # config.materials.definitions) -- no more duplicating identical
    # properties under 14 unique per-layer names. Each physical layer
    # still gets its own pid: dispsolver/io/model_builder.py assigns one
    # pid per *SOLID SECTION statement (not per unique *MATERIAL name
    # anymore), so grouping this layer's element rows into one ELSET
    # per physical layer (layer_row_spans below) is what makes each
    # layer independently selectable/inspectable (e.g. in the Qt
    # viewer's Part/Layer list), independent of material-name reuse.
    grid = build_display_grid(config)
    geo = config.geometry

    _w("*NODE")
    for nid, x_val, y_val in grid.nodes:
        _w(f"{nid}, {x_val:.6f}, {y_val:.6f}")

    # 2. Plate Meshes
    # base_node_id/base_elem_id = 100000 is a load-bearing convention several
    # OTHER files also hardcode (plotter.py's plate-detection fallback,
    # dynamic.py's max_displacement_corr plate-DOF exclusion, interlayer.py,
    # check_interlayer_shear.py, ex12's own diagnostics). Raised from 10000
    # to 100000 on 2026-09-08 across every one of those files together (same
    # commit) to give the display mesh 10x the node-id headroom -- a finer
    # PSA row count or tighter free-span dx can grow the display's own node
    # numbering past whatever this boundary is, at which point display
    # elements silently reuse plate node ids and connectivity corrupts
    # without any error (found 2026-09-08 at the old 10000 boundary: PSA at
    # 2 rows + dx=0.05 across the whole untied span produced 721 elements in
    # pid 7/8 spanning up to 44mm -- display nodes stitched to the real LEFT
    # PLATE's nodes near x=-40). The assert below still fails loudly if the
    # display mesh ever grows past the new boundary too -- if it fires,
    # either coarsen the display mesh or raise base_node_id here AND in
    # every file listed above together, in the same change (do not raise it
    # in only one file).
    _max_display_nid = max(nid for nid, _, _ in grid.nodes)
    assert _max_display_nid < 100000, (
        f"display mesh uses node id {_max_display_nid} >= 100000 (the "
        "reserved plate-node base) -- element connectivity would silently "
        "corrupt by stitching to plate nodes. Coarsen the display mesh "
        "(fewer PSA rows / larger free-span dx) or see this assert's "
        "comment before raising base_node_id.")
    plates = create_folding_plate_parts(
        left_x_range=(-geo.display_half_length, -geo.hinge_half_gap),
        right_x_range=(geo.hinge_half_gap, geo.display_half_length),
        y_range=(-geo.plate_thickness, 0.0),
        left_pivot=(-geo.hinge_pivot_x, geo.hinge_pivot_y),
        right_pivot=(geo.hinge_pivot_x, geo.hinge_pivot_y),
        nx=geo.plate_mesh_nx,
        ny=geo.plate_mesh_ny,
        base_node_id=100000,
        base_elem_id=100000,
    )
    left_plate = plates["left"]
    right_plate = plates["right"]

    for p_name, p_data in plates.items():
        for nid_p, coord in p_data["nodes_dict"].items():
            _w(f"{nid_p}, {coord[0]:.6f}, {coord[1]:.6f}")

    _w("**")

    # 3. Elements -- one ELSET per PHYSICAL LAYER (spanning that layer's
    # n_rows element rows) so each layer carries its own *SOLID SECTION/
    # pid below, independent of how many rows it has or whether its
    # material name repeats across other layers.
    display_elsets = []
    elset_material = []
    for layer_idx, (_r0, _r1, mat_name) in enumerate(grid.layer_row_spans):
        layer_cells = grid.cells_of_layer(layer_idx)
        if not layer_cells:
            # Entirely voided layer -- emit no ELSET, and (below) no
            # *SOLID SECTION referencing it, since Abaqus decks cannot
            # carry an empty element set.
            continue
        elset_name = f"DISP_LAYER{layer_idx + 1:02d}"
        display_elsets.append(elset_name)
        elset_material.append(mat_name)
        _w(f"*ELEMENT, TYPE=CPE4, ELSET={elset_name}")
        for cell in layer_cells:
            n1, n2, n3, n4 = cell.conn
            _w(f"{cell.eid}, {n1}, {n2}, {n3}, {n4}")

    for p_name, p_data in plates.items():
        _w(f"*ELEMENT, TYPE=CPE4, ELSET=PLATE_{p_name.upper()}")
        for eid_p, conn in p_data["elements_dict"].items():
            _w(f"{eid_p}, {conn[0]}, {conn[1]}, {conn[2]}, {conn[3]}")

    _w("**")

    # 4. Materials -- one *MATERIAL block per DISTINCT material (PET,
    # PSA, STEEL -- 3 total, was 15), sourced from config.materials.definitions.
    for mdef in config.materials.definitions.values():
        for line in emit_abaqus_material_block(mdef):
            _w(line)

    # Sections -- one per physical layer (elset_material repeats e.g.
    # "PET" seven times across the 7 pairs, referencing the ONE PET
    # *MATERIAL block above).
    for elset_name, mat_name in zip(display_elsets, elset_material):
        _w(f"*SOLID SECTION, ELSET={elset_name}, MATERIAL={mat_name}")
        _w("1.0,")
    _w("*SOLID SECTION, ELSET=PLATE_LEFT, MATERIAL=STEEL")
    _w("1.0,")
    _w("*SOLID SECTION, ELSET=PLATE_RIGHT, MATERIAL=STEEL")
    _w("1.0,")
    _w("**")

    # 5. Node sets and Surfaces
    # Display bottom left (x <= -10) and right (x >= 10)
    # x looked up per node id rather than by position -- the old
    # `xs[(nid - 1) % (nx + 1)]` arithmetic assumed a full dense node grid
    # and returns the wrong x for every node past the first void.
    tie_start = geo.tie_attach_start if geo.tie_attach_start is not None else geo.hinge_half_gap
    left_disp_bot = [n for n in grid.bottom_surface_nids
                     if grid.x_of_node(n) <= -tie_start]
    right_disp_bot = [n for n in grid.bottom_surface_nids
                      if grid.x_of_node(n) >= tie_start]

    _w("*NSET, NSET=DISP_BOT_L_NODES")
    _w(", ".join(str(nid) for nid in left_disp_bot))
    _w("*SURFACE, NAME=DISPLAY_BOT_L, TYPE=NODE")
    _w("DISP_BOT_L_NODES,")

    _w("*NSET, NSET=DISP_BOT_R_NODES")
    _w(", ".join(str(nid) for nid in right_disp_bot))
    _w("*SURFACE, NAME=DISPLAY_BOT_R, TYPE=NODE")
    _w("DISP_BOT_R_NODES,")

    # Plate tops
    _w("*NSET, NSET=PLATE_L_TOP_NODES")
    _w(", ".join(str(nid) for nid in left_plate["top_surface_nids"]))
    _w("*SURFACE, NAME=PLATE_L_TOP, TYPE=NODE")
    _w("PLATE_L_TOP_NODES,")

    _w("*NSET, NSET=PLATE_R_TOP_NODES")
    _w(", ".join(str(nid) for nid in right_plate["top_surface_nids"]))
    _w("*SURFACE, NAME=PLATE_R_TOP, TYPE=NODE")
    _w("PLATE_R_TOP_NODES,")

    # Plate slaves for rigid body definitions
    _w("*NSET, NSET=PLATE_L_SLAVES")
    _w(", ".join(str(nid) for nid in left_plate["slave_nids"]))
    _w("*NSET, NSET=PLATE_R_SLAVES")
    _w(", ".join(str(nid) for nid in right_plate["slave_nids"]))
    _w("**")

    # 6. Constraints
    _w(f"*RIGID BODY, NSET=PLATE_L_SLAVES, REF NODE={left_plate['master_rp_id']}")
    _w(f"*RIGID BODY, NSET=PLATE_R_SLAVES, REF NODE={right_plate['master_rp_id']}")

    _w("*TIE, NAME=TIE_LEFT")
    _w("DISPLAY_BOT_L, PLATE_L_TOP")
    _w("*TIE, NAME=TIE_RIGHT")
    _w("DISPLAY_BOT_R, PLATE_R_TOP")
    _w("**")

    # 7. Step with boundary conditions and solver controls
    drv = config.drive
    theta_rad = np.radians(drv.theta_max_deg)
    abq_cfg = config.to_abaqus_config()
    _w(abq_cfg.to_inp_snippet())
    _w("**")

    # Fix hinge translations
    _w("*BOUNDARY")
    _w(f"{left_plate['master_rp_id']}, 1, 2, 0.0")
    _w(f"{right_plate['master_rp_id']}, 1, 2, 0.0")
    _w("**")

    # Prescribe hinge rotations
    _w("*BOUNDARY")
    _w(f"{left_plate['master_rp_id']}, 6, 6, {-theta_rad:.6f}")
    _w(f"{right_plate['master_rp_id']}, 6, 6, {theta_rad:.6f}")
    _w("**")
    _w("*END STEP")

    return "\n".join(lines)

if __name__ == "__main__":
    content = generate()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated: {OUTPUT}")
