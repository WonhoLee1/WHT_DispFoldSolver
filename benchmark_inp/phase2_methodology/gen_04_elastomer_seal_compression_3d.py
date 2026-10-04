"""
gen_04_elastomer_seal_compression_3d.py
=========================================
Generates 04_elastomer_seal_compression_3d.inp:
3D Elastomer Seal Compression Benchmark.

Physics & Mechanics:
- Foundation Elastomer Seal:
  Dimensions: L = 20.0 mm (x in [-10.0, 10.0]), H = 6.0 mm (y in [0.0, 6.0]), W = 4.0 mm (z in [-2.0, 2.0])
  Material: Incompressible Hyperelastic (Mooney-Rivlin, C10 = 0.5 MPa, C01 = 0.1 MPa, D1 = 0.001 MPa^-1)
  Element Type: C3D8H (8-node hybrid u-P mixed hexahedral element to eliminate volumetric locking)
- Upper Compressive Steel Plate:
  Dimensions: L = 24.0 mm (x in [-12.0, 12.0]), H = 2.0 mm (y in [6.0, 8.0]), W = 4.0 mm (z in [-2.0, 2.0])
  Material: Steel (E = 210,000.0 MPa, nu = 0.30)
  Element Type: C3D8
- Interaction:
  3D Surface-to-Surface Hard Contact (Frictionless / pure normal compression)
- Step:
  - Step 1 (COMPRESS): Uy = -1.5 mm (25% nominal seal compression)
"""

import os
import numpy as np


def generate_deck():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    inp_path = os.path.join(out_dir, "04_elastomer_seal_compression_3d.inp")

    # -------------------------------------------------------------
    # 1. Seal Geometry (C3D8H, Mooney-Rivlin)
    # -------------------------------------------------------------
    nx_s = 20  # dx = 1.0 mm
    ny_s = 6   # dy = 1.0 mm
    nz_s = 4   # dz = 1.0 mm
    xs_s = np.linspace(-10.0, 10.0, nx_s + 1)
    ys_s = np.linspace(0.0, 6.0, ny_s + 1)
    zs_s = np.linspace(-2.0, 2.0, nz_s + 1)

    # -------------------------------------------------------------
    # 2. Plate Geometry (C3D8, Steel)
    # -------------------------------------------------------------
    nx_p = 12  # dx = 2.0 mm
    ny_p = 2   # dy = 1.0 mm
    nz_p = 4   # dz = 1.0 mm
    xs_p = np.linspace(-12.0, 12.0, nx_p + 1)
    ys_p = np.linspace(6.0, 8.0, ny_p + 1)
    zs_p = np.linspace(-2.0, 2.0, nz_p + 1)

    lines = []
    lines.append("*HEADING")
    lines.append("04_elastomer_seal_compression_3d: 3D Elastomer Seal Compression (C3D8H + Mooney-Rivlin)")

    # -------------------------------------------------------------
    # Part 1: SEAL
    # -------------------------------------------------------------
    lines.append("*PART, NAME=SEAL")
    lines.append("*NODE")
    s_grid = {}
    node_id_s = 1
    for k in range(nz_s + 1):
        for j in range(ny_s + 1):
            for i in range(nx_s + 1):
                x = xs_s[i]
                y = ys_s[j]
                z = zs_s[k]
                lines.append(f"{node_id_s}, {x:.6f}, {y:.6f}, {z:.6f}")
                s_grid[(i, j, k)] = node_id_s
                node_id_s += 1

    lines.append("*ELEMENT, TYPE=C3D8H, ELSET=SEAL_ELEMS")
    elem_id_s = 1
    top_seal_faces = []
    for k in range(nz_s):
        for j in range(ny_s):
            for i in range(nx_s):
                n1 = s_grid[(i, j, k)]
                n2 = s_grid[(i + 1, j, k)]
                n3 = s_grid[(i + 1, j + 1, k)]
                n4 = s_grid[(i, j + 1, k)]
                n5 = s_grid[(i, j, k + 1)]
                n6 = s_grid[(i + 1, j, k + 1)]
                n7 = s_grid[(i + 1, j + 1, k + 1)]
                n8 = s_grid[(i, j + 1, k + 1)]
                lines.append(f"{elem_id_s}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
                if j == ny_s - 1:
                    # Abaqus C3D8 face S5 is y-max face: (3, 7, 8, 4)
                    top_seal_faces.append(f"{elem_id_s}, S5")
                elem_id_s += 1

    # Node sets for Seal bottom (y = 0.0)
    bot_seal_nodes = []
    for k in range(nz_s + 1):
        for i in range(nx_s + 1):
            bot_seal_nodes.append(s_grid[(i, 0, k)])

    lines.append("*NSET, NSET=SEAL_BOTTOM")
    for chunk in [bot_seal_nodes[m:m+16] for m in range(0, len(bot_seal_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Surface for Seal top
    lines.append("*SURFACE, NAME=SEAL_TOP, TYPE=ELEMENT")
    for face in top_seal_faces:
        lines.append(face)

    lines.append("*SOLID SECTION, ELSET=SEAL_ELEMS, MATERIAL=RUBBER_MOONEY")
    lines.append("1.0,")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Part 2: PLATE
    # -------------------------------------------------------------
    lines.append("*PART, NAME=PLATE")
    lines.append("*NODE")
    p_grid = {}
    node_id_p = 1
    for k in range(nz_p + 1):
        for j in range(ny_p + 1):
            for i in range(nx_p + 1):
                x = xs_p[i]
                y = ys_p[j]
                z = zs_p[k]
                lines.append(f"{node_id_p}, {x:.6f}, {y:.6f}, {z:.6f}")
                p_grid[(i, j, k)] = node_id_p
                node_id_p += 1

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=PLATE_ELEMS")
    elem_id_p = 1
    bot_plate_faces = []
    for k in range(nz_p):
        for j in range(ny_p):
            for i in range(nx_p):
                n1 = p_grid[(i, j, k)]
                n2 = p_grid[(i + 1, j, k)]
                n3 = p_grid[(i + 1, j + 1, k)]
                n4 = p_grid[(i, j + 1, k)]
                n5 = p_grid[(i, j, k + 1)]
                n6 = p_grid[(i + 1, j, k + 1)]
                n7 = p_grid[(i + 1, j + 1, k + 1)]
                n8 = p_grid[(i, j + 1, k + 1)]
                lines.append(f"{elem_id_p}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
                if j == 0:
                    # Abaqus C3D8 face S3 is y-min face: (1, 5, 6, 2)
                    bot_plate_faces.append(f"{elem_id_p}, S3")
                elem_id_p += 1

    # Node sets for Plate top (y = 8.0, for driving)
    top_plate_nodes = []
    for k in range(nz_p + 1):
        for i in range(nx_p + 1):
            top_plate_nodes.append(p_grid[(i, ny_p, k)])

    lines.append("*NSET, NSET=PLATE_TOP")
    for chunk in [top_plate_nodes[m:m+16] for m in range(0, len(top_plate_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Surface for Plate bottom
    lines.append("*SURFACE, NAME=PLATE_BOTTOM, TYPE=ELEMENT")
    for face in bot_plate_faces:
        lines.append(face)

    lines.append("*SOLID SECTION, ELSET=PLATE_ELEMS, MATERIAL=STEEL_PLATE")
    lines.append("1.0,")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------
    lines.append("*ASSEMBLY, NAME=ASSEMBLY")
    lines.append("*INSTANCE, NAME=SEAL_1, PART=SEAL")
    lines.append("*END INSTANCE")
    lines.append("*INSTANCE, NAME=PLATE_1, PART=PLATE")
    lines.append("*END INSTANCE")

    # Assembly surfaces
    lines.append("*SURFACE, NAME=SURF_SEAL, COMBINE=UNION")
    lines.append("SEAL_1.SEAL_TOP")
    lines.append("*SURFACE, NAME=SURF_PLATE, COMBINE=UNION")
    lines.append("PLATE_1.PLATE_BOTTOM")
    lines.append("*END ASSEMBLY")

    # -------------------------------------------------------------
    # Materials
    # -------------------------------------------------------------
    lines.append("*MATERIAL, NAME=RUBBER_MOONEY")
    lines.append("*HYPERELASTIC, MOONEY-RIVLIN")
    lines.append("0.5, 0.1, 0.001")

    lines.append("*MATERIAL, NAME=STEEL_PLATE")
    lines.append("*ELASTIC")
    lines.append("210000.0, 0.30")

    # -------------------------------------------------------------
    # Contact Interaction
    # -------------------------------------------------------------
    lines.append("*SURFACE INTERACTION, NAME=INT_PROP")
    lines.append("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD")

    lines.append("*CONTACT PAIR, INTERACTION=INT_PROP")
    lines.append("SURF_SEAL, SURF_PLATE")

    # -------------------------------------------------------------
    # Boundary Conditions & Step
    # -------------------------------------------------------------
    lines.append("*BOUNDARY")
    lines.append("SEAL_1.SEAL_BOTTOM, 1, 3, 0.0")

    lines.append("*STEP, NAME=COMPRESS, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.05, 1.0, 1e-5, 0.1")
    lines.append("*BOUNDARY")
    lines.append("PLATE_1.PLATE_TOP, 1, 1, 0.0")
    lines.append("PLATE_1.PLATE_TOP, 3, 3, 0.0")
    lines.append("PLATE_1.PLATE_TOP, 2, 2, -1.50")
    lines.append("*END STEP")

    content = "\n".join(lines) + "\n"
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Deck written successfully to {inp_path}")
    return inp_path


if __name__ == "__main__":
    generate_deck()
