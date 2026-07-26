import numpy as np
import os
from dispsolver.mesh.plate_builder import create_folding_plate_parts

OUTPUT = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")


def _graded_display_x() -> np.ndarray:
    """Graded (clustered) x-coordinates for the display mesh.

    Uniform 1mm spacing lets shear/rotation gradient concentrate into a
    single element wherever the plate (exactly rigid, RBE2 condensation)
    meets a compliant region -- worst at the two free-edge tip elements
    (x=+-40, no outboard neighbor to redistribute strain into) and at the
    plate/hinge boundary (x=+-10). Clustering columns there spreads that
    gradient across more, narrower elements, keeping det(F) > 0 to a much
    larger fold angle -- see AGENTS.md section 4.10-era plan (fix for the
    tip-element inversion observed at ~47.9 deg/side).

    Segments (mirrored about x=0): fine near the tips (+-40) and the
    hinge/plate boundary (+-10), moderately fine through the free hinge
    span (+-8, where curvature must resolve smoothly per AGENTS.md 1.0),
    coarser 1mm spacing under the rigid plate body in between.
    """
    def seg(a, b, n):
        return np.linspace(a, b, n)

    segments = [
        seg(-40.0, -38.0, 9),   # tip cluster (0.25mm)
        seg(-38.0, -10.0, 29),  # under-plate, coarse (1.0mm)
        seg(-10.0, -8.0, 9),    # hinge-edge cluster (0.25mm)
        seg(-8.0, 8.0, 33),     # free hinge span, moderate (0.5mm)
        seg(8.0, 10.0, 9),      # hinge-edge cluster (0.25mm)
        seg(10.0, 38.0, 29),    # under-plate, coarse (1.0mm)
        seg(38.0, 40.0, 9),     # tip cluster (0.25mm)
    ]
    parts = [segments[0]] + [s[1:] for s in segments[1:]]
    xs = np.concatenate(parts)
    assert np.all(np.diff(xs) > 0), "graded x-coordinates must be strictly increasing"
    return xs


def generate():
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
    # 14-layer multi-material stack: alternating stiff substrate rows
    # (PET, elastic-plastic) and compliant adhesive rows (PSA,
    # Prony+WLF viscoelastic pressure-sensitive-adhesive) -- see
    # AGENTS.md 1.4, still a simplified alternating layup (PET-PSA-PET-...),
    # not a validated real stackup.
    xs_disp = _graded_display_x()
    nx_disp = len(xs_disp) - 1
    ny_disp = 14
    ys_disp = np.linspace(0.0, 0.5, ny_disp + 1)
    LAYER_MATERIAL = ["PET" if j % 2 == 0 else "PSA" for j in range(ny_disp)]

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
        left_x_range=(-40.0, -10.0),
        right_x_range=(10.0, 40.0),
        y_range=(-0.5, 0.0),
        left_pivot=(-3.0, 0.0),
        right_pivot=(3.0, 0.0),
        nx=30,
        ny=2,
        base_node_id=10000,
        base_elem_id=10000,
    )
    left_plate = plates["left"]
    right_plate = plates["right"]

    for p_name, p_data in plates.items():
        for nid_p, coord in p_data["nodes_dict"].items():
            _w(f"{nid_p}, {coord[0]:.6f}, {coord[1]:.6f}")

    _w("**")

    # 3. Elements -- one ELSET per display layer (row) so each can carry
    # its own material via *SOLID SECTION below.
    eid = 1
    display_elsets = []
    for j in range(ny_disp):
        elset_name = f"DISP_L{j + 1:02d}"
        display_elsets.append(elset_name)
        _w(f"*ELEMENT, TYPE=CPE4, ELSET={elset_name}")
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

    # 4. Materials
    # PET: stiff elastic-plastic substrate layer.
    _w("*MATERIAL, NAME=PET")
    _w("*ELASTIC")
    _w("4000.0, 0.3")
    _w("*PLASTIC")
    _w("80.0, 0.0")
    _w("480.0, 1.0")
    _w("**")
    # PSA: compliant pressure-sensitive-adhesive layer, Prony-series
    # viscoelastic with WLF time-temperature shift
    # (dispsolver.material.viscoelastic.ViscoelasticMaterial).
    _w("*MATERIAL, NAME=PSA")
    _w("*ELASTIC")
    _w("50.0, 0.45")
    _w("*VISCOELASTIC, TIME=PRONY")
    _w("0.6, 0.0, 0.1")
    _w("0.3, 0.0, 1.0")
    _w("*TRS, DEFINITION=WLF")
    _w("25.0, 17.0, 51.6")
    _w("**")
    _w("*MATERIAL, NAME=STEEL")
    _w("*ELASTIC")
    _w("20000.0, 0.3")
    _w("**")

    # Sections -- one per display layer, alternating SUBSTRATE/ADHESIVE
    for elset_name, mat_name in zip(display_elsets, LAYER_MATERIAL):
        _w(f"*SOLID SECTION, ELSET={elset_name}, MATERIAL={mat_name}")
        _w("1.0,")
    _w("*SOLID SECTION, ELSET=PLATE_LEFT, MATERIAL=STEEL")
    _w("1.0,")
    _w("*SOLID SECTION, ELSET=PLATE_RIGHT, MATERIAL=STEEL")
    _w("1.0,")
    _w("**")

    # 5. Node sets and Surfaces
    # Display bottom left (x <= -10) and right (x >= 10)
    left_disp_bot = [nid for nid in bottom_surface_nids if xs_disp[(nid - 1) % (nx_disp + 1)] <= -10.0]
    right_disp_bot = [nid for nid in bottom_surface_nids if xs_disp[(nid - 1) % (nx_disp + 1)] >= 10.0]

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
    _w("*STEP")
    _w("*STATIC")
    _w("0.005, 1.0, 1e-5, 0.01")
    _w("**")

    # Fix hinge translations
    _w("*BOUNDARY")
    _w(f"{left_plate['master_rp_id']}, 1, 2, 0.0")
    _w(f"{right_plate['master_rp_id']}, 1, 2, 0.0")
    _w("**")

    # Prescribe hinge rotations (90 degrees = 1.570796 radians)
    _w("*BOUNDARY")
    _w(f"{left_plate['master_rp_id']}, 6, 6, -1.570796")
    _w(f"{right_plate['master_rp_id']}, 6, 6, 1.570796")
    _w("**")
    _w("*END STEP")

    return "\n".join(lines)

if __name__ == "__main__":
    content = generate()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated: {OUTPUT}")
