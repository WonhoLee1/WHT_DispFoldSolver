import numpy as np
import os
import sys
from dispsolver.mesh.plate_builder import create_folding_plate_parts

sys.path.insert(0, os.path.dirname(__file__))
from dispsolver.fold_model_config import FoldModelConfig, DEFAULT_CONFIG
from dispsolver.material.factory import emit_abaqus_material_block

OUTPUT = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")


def _n_pts(length: float, dx: float) -> int:
    """Point count for a `np.linspace` segment of the given length/dx.

    +1 for the linspace endpoint; `max(2,...)` guards a zero-length or
    dx-larger-than-length segment from collapsing to <2 points.
    """
    return max(2, round(length / dx) + 1)


def _graded_display_x(config: FoldModelConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Graded (clustered) x-coordinates for the display mesh.

    Uniform 1mm spacing lets shear/rotation gradient concentrate into a
    single element wherever the plate (exactly rigid, RBE2 condensation)
    meets a compliant region -- worst at the two free-edge tip elements
    (x=+-half_length, no outboard neighbor to redistribute strain into)
    and at the plate/hinge boundary (x=+-hinge_half_gap). Clustering
    columns there spreads that gradient across more, narrower elements,
    keeping det(F) > 0 to a much larger fold angle -- see AGENTS.md
    section 4.10-era plan (fix for the tip-element inversion observed
    at ~47.9 deg/side).

    Segments (mirrored about x=0): fine near the tips and the hinge/
    plate boundary, moderately fine through the free hinge span (where
    curvature must resolve smoothly per AGENTS.md 1.0), coarser spacing
    under the rigid plate body in between. Segment point counts are
    derived from `grading.*_dx` (length/dx), not hardcoded integers, so
    geometry changes don't silently change element size in some zone.
    """
    g = config.grading
    half_len = config.geometry.display_half_length
    half_gap = config.geometry.hinge_half_gap
    span = g.hinge_span_half_width
    hinge_lo = half_gap - g.hinge_edge_cluster_width  # inner edge of hinge cluster
    assert span <= hinge_lo, "hinge_span_half_width must be <= hinge_half_gap - hinge_edge_cluster_width"

    tip_lo = half_len - g.tip_cluster_width

    def seg(a, b, dx):
        return np.linspace(a, b, _n_pts(b - a, dx))

    segments = [
        seg(-half_len, -tip_lo, g.tip_dx),          # tip cluster
        seg(-tip_lo, -half_gap, g.plate_body_dx),   # under-plate, coarse
        seg(-half_gap, -hinge_lo, g.hinge_edge_dx), # hinge-edge cluster
        seg(-hinge_lo, span, g.hinge_span_dx),      # free hinge span
        seg(span, hinge_lo, g.hinge_edge_dx),       # hinge-edge cluster (dropped if span==hinge_lo)
        seg(hinge_lo, half_gap, g.hinge_edge_dx),   # hinge-edge cluster
        seg(half_gap, tip_lo, g.plate_body_dx),     # under-plate, coarse
        seg(tip_lo, half_len, g.tip_dx),            # tip cluster
    ]
    # Drop degenerate (zero-length) segments -- e.g. the middle hinge-edge
    # cluster collapses to a point whenever span == hinge_lo (the default),
    # since the free-hinge-span segment already reaches exactly that x.
    segments = [s for s in segments if s[-1] > s[0]]
    parts = [segments[0]] + [s[1:] for s in segments[1:]]
    xs = np.concatenate(parts)
    assert np.all(np.diff(xs) > 0), "graded x-coordinates must be strictly increasing"
    return xs


def generate(config: FoldModelConfig = DEFAULT_CONFIG):
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
    xs_disp = _graded_display_x(config)
    nx_disp = len(xs_disp) - 1

    geo = config.geometry
    n_pairs = geo.n_layer_pairs
    row_heights = []       # mm, one entry per mesh row
    layer_row_spans = []   # (row_start, row_end_exclusive, material_name) per physical layer
    row_cursor = 0
    for _pair in range(n_pairs):
        for layer in geo.layer_pattern:
            row_heights += [layer.thickness_mm / layer.n_rows] * layer.n_rows
            layer_row_spans.append((row_cursor, row_cursor + layer.n_rows, layer.material_name))
            row_cursor += layer.n_rows

    ny_disp = len(row_heights)
    ys_disp = np.concatenate([[0.0], np.cumsum(row_heights)])

    _w("*NODE")
    grid_nids = np.zeros((ny_disp + 1, nx_disp + 1), dtype=int)
    nid = 1
    bottom_surface_nids = []
    for j, y_val in enumerate(ys_disp):
        for i, x_val in enumerate(xs_disp):
            grid_nids[j, i] = nid
            _w(f"{nid}, {x_val:.6f}, {y_val:.6f}")
            if j == 0:
                bottom_surface_nids.append(nid)
            nid += 1

    # 2. Plate Meshes
    plates = create_folding_plate_parts(
        left_x_range=(-geo.display_half_length, -geo.hinge_half_gap),
        right_x_range=(geo.hinge_half_gap, geo.display_half_length),
        y_range=(-geo.plate_thickness, 0.0),
        left_pivot=(-geo.hinge_pivot_x, 0.0),
        right_pivot=(geo.hinge_pivot_x, 0.0),
        nx=geo.plate_mesh_nx,
        ny=geo.plate_mesh_ny,
        base_node_id=10000,
        base_elem_id=10000,
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
    eid = 1
    display_elsets = []
    elset_material = []
    for layer_idx, (r0, r1, mat_name) in enumerate(layer_row_spans):
        elset_name = f"DISP_LAYER{layer_idx + 1:02d}"
        display_elsets.append(elset_name)
        elset_material.append(mat_name)
        _w(f"*ELEMENT, TYPE=CPE4, ELSET={elset_name}")
        for j in range(r0, r1):
            for i in range(nx_disp):
                n1 = grid_nids[j, i]
                n2 = grid_nids[j, i + 1]
                n3 = grid_nids[j + 1, i + 1]
                n4 = grid_nids[j + 1, i]
                _w(f"{eid}, {n1}, {n2}, {n3}, {n4}")
                eid += 1

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
    left_disp_bot = [nid for nid in bottom_surface_nids if xs_disp[(nid - 1) % (nx_disp + 1)] <= -geo.hinge_half_gap]
    right_disp_bot = [nid for nid in bottom_surface_nids if xs_disp[(nid - 1) % (nx_disp + 1)] >= geo.hinge_half_gap]

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

    # 7. Step with boundary conditions
    drv = config.drive
    theta_rad = np.radians(drv.theta_max_deg)
    _w("*STEP")
    _w("*STATIC")
    _w(f"{drv.dt_init}, {drv.t_total}, {drv.dt_min}, {drv.dt_max}")
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
