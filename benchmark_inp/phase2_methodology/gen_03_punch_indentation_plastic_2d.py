"""
gen_03_punch_indentation_plastic_2d.py
======================================
Generates 03_punch_indentation_plastic_2d.inp:
2D Elasto-Plastic Punch Indentation Press (CPE4I elements, J2 Plasticity).

Physical setup:
- Foundation Elastic-Plastic Metal Block:
  W = 40.0 mm (x in [-20.0, 20.0]), H = 15.0 mm (y in [-15.0, 0.0])
  Material: Aluminum alloy with J2 isotropic hardening
  E = 70,000.0 MPa, nu = 0.33, sigma_y0 = 250.0 MPa, H = 1,000.0 MPa
  Mesh: Graded CPE4I elements (fine mesh around contact zone x in [-6, 6])
- Upper Rigid-like Steel Punch:
  B = 8.0 mm (x in [-4.0, 4.0]), H_p = 4.0 mm (y in [0.0, 4.0])
  Material: Steel (E = 210,000.0 MPa, nu = 0.30)
- Step:
  - Step 1: Vertical Indentation Press (Uy = -0.20 mm)
- Interaction: Hard normal contact (Frictionless / pure indentation press)
"""

import os
import numpy as np


def generate_deck():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    inp_path = os.path.join(out_dir, "03_punch_indentation_plastic_2d.inp")

    # -------------------------------------------------------------
    # 1. Block Geometry & Mesh
    # -------------------------------------------------------------
    # Graded grid in X:
    # Left zone: x in [-20.0, -6.0], dx = 1.0 (14 elements)
    # Center zone: x in [-6.0, 6.0], dx = 0.5 (24 elements)
    # Right zone: x in [6.0, 20.0], dx = 1.0 (14 elements)
    xs_left = np.linspace(-20.0, -6.0, 14, endpoint=False)
    xs_center = np.linspace(-6.0, 6.0, 25, endpoint=True)
    xs_right = np.linspace(7.0, 20.0, 14, endpoint=True)
    xs_b = np.concatenate([xs_left, xs_center, xs_right])
    nx_b = len(xs_b) - 1  # 51 elements

    # Graded grid in Y:
    # Bottom zone: y in [-15.0, -5.0], dy = 1.0 (10 elements)
    # Top zone: y in [-5.0, 0.0], dy = 0.5 (10 elements)
    ys_bot = np.linspace(-15.0, -5.0, 10, endpoint=False)
    ys_top = np.linspace(-5.0, 0.0, 11, endpoint=True)
    ys_b = np.concatenate([ys_bot, ys_top])
    ny_b = len(ys_b) - 1  # 20 elements

    # -------------------------------------------------------------
    # 2. Punch Geometry & Mesh
    # -------------------------------------------------------------
    # x in [-4.0, 4.0], dx = 0.5 (16 elements)
    # y in [0.0, 4.0], dy = 0.5 (8 elements)
    nx_p = 16
    ny_p = 8
    xs_p = np.linspace(-4.0, 4.0, nx_p + 1)
    ys_p = np.linspace(0.0, 4.0, ny_p + 1)

    lines = []
    lines.append("*HEADING")
    lines.append("03_punch_indentation_plastic_2d: 2D Elasto-Plastic Punch Indentation Press (CPE4I + J2)")

    # -------------------------------------------------------------
    # Part 1: BLOCK (Foundation Plate)
    # -------------------------------------------------------------
    lines.append("*PART, NAME=BLOCK")
    lines.append("*NODE")
    b_grid = {}
    node_id_b = 1
    for j in range(ny_b + 1):
        for i in range(nx_b + 1):
            x = xs_b[i]
            y = ys_b[j]
            lines.append(f"{node_id_b}, {x:.6f}, {y:.6f}")
            b_grid[(i, j)] = node_id_b
            node_id_b += 1

    lines.append("*ELEMENT, TYPE=CPE4I, ELSET=BLOCK_ELEMS")
    elem_id_b = 1
    top_block_edges = []
    for j in range(ny_b):
        for i in range(nx_b):
            n1 = b_grid[(i, j)]
            n2 = b_grid[(i + 1, j)]
            n3 = b_grid[(i + 1, j + 1)]
            n4 = b_grid[(i, j + 1)]
            lines.append(f"{elem_id_b}, {n1}, {n2}, {n3}, {n4}")
            if j == ny_b - 1:
                # Top edge is n3-n4 (Abaqus 2D CPE4 face S3)
                top_block_edges.append(f"{elem_id_b}, S3")
            elem_id_b += 1

    # Node sets for Block
    bot_nodes = [b_grid[(i, 0)] for i in range(nx_b + 1)]
    lines.append("*NSET, NSET=BLOCK_BOTTOM")
    for chunk in [bot_nodes[m:m+16] for m in range(0, len(bot_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Block Top Surface for contact
    lines.append("*SURFACE, NAME=BLOCK_TOP, TYPE=ELEMENT")
    for te in top_block_edges:
        lines.append(te)

    lines.append("*SOLID SECTION, ELSET=BLOCK_ELEMS, MATERIAL=ALUMINUM_PLASTIC")
    lines.append("1.0,")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Part 2: PUNCH
    # -------------------------------------------------------------
    lines.append("*PART, NAME=PUNCH")
    lines.append("*NODE")
    p_grid = {}
    node_id_p = 1
    for j in range(ny_p + 1):
        for i in range(nx_p + 1):
            x = xs_p[i]
            y = ys_p[j]
            lines.append(f"{node_id_p}, {x:.6f}, {y:.6f}")
            p_grid[(i, j)] = node_id_p
            node_id_p += 1

    lines.append("*ELEMENT, TYPE=CPE4I, ELSET=PUNCH_ELEMS")
    elem_id_p = 1
    bot_punch_edges = []
    for j in range(ny_p):
        for i in range(nx_p):
            n1 = p_grid[(i, j)]
            n2 = p_grid[(i + 1, j)]
            n3 = p_grid[(i + 1, j + 1)]
            n4 = p_grid[(i, j + 1)]
            lines.append(f"{elem_id_p}, {n1}, {n2}, {n3}, {n4}")
            if j == 0:
                # Bottom edge is n1-n2 (Abaqus 2D CPE4 face S1)
                bot_punch_edges.append(f"{elem_id_p}, S1")
            elem_id_p += 1

    # Punch top nodes (y = 4.0, for driving)
    punch_top_nodes = [p_grid[(i, ny_p)] for i in range(nx_p + 1)]
    lines.append("*NSET, NSET=PUNCH_TOP")
    for chunk in [punch_top_nodes[m:m+16] for m in range(0, len(punch_top_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    lines.append("*SURFACE, NAME=PUNCH_BOTTOM, TYPE=ELEMENT")
    for pe_edge in bot_punch_edges:
        lines.append(pe_edge)

    lines.append("*SOLID SECTION, ELSET=PUNCH_ELEMS, MATERIAL=STEEL_PUNCH")
    lines.append("1.0,")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------
    lines.append("*ASSEMBLY, NAME=ASSEMBLY")
    lines.append("*INSTANCE, NAME=BLOCK_1, PART=BLOCK")
    lines.append("*END INSTANCE")
    lines.append("*INSTANCE, NAME=PUNCH_1, PART=PUNCH")
    lines.append("*END INSTANCE")

    # Assembly surfaces
    lines.append("*SURFACE, NAME=SURF_BLOCK, COMBINE=UNION")
    lines.append("BLOCK_1.BLOCK_TOP")
    lines.append("*SURFACE, NAME=SURF_PUNCH, COMBINE=UNION")
    lines.append("PUNCH_1.PUNCH_BOTTOM")
    lines.append("*END ASSEMBLY")

    # -------------------------------------------------------------
    # Materials
    # -------------------------------------------------------------
    lines.append("*MATERIAL, NAME=ALUMINUM_PLASTIC")
    lines.append("*ELASTIC")
    lines.append("70000.0, 0.33")
    lines.append("*PLASTIC, HARDENING=ISOTROPIC")
    lines.append("250.0, 0.0")
    lines.append("350.0, 0.1")

    lines.append("*MATERIAL, NAME=STEEL_PUNCH")
    lines.append("*ELASTIC")
    lines.append("210000.0, 0.30")

    # -------------------------------------------------------------
    # Contact Interaction (Hard Normal Contact)
    # -------------------------------------------------------------
    lines.append("*SURFACE INTERACTION, NAME=INT_PROP")
    lines.append("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD")

    lines.append("*CONTACT PAIR, INTERACTION=INT_PROP")
    lines.append("SURF_PUNCH, SURF_BLOCK")

    # -------------------------------------------------------------
    # Initial Boundary Conditions
    # -------------------------------------------------------------
    lines.append("*BOUNDARY")
    lines.append("BLOCK_1.BLOCK_BOTTOM, 1, 2, 0.0")

    # -------------------------------------------------------------
    # Step 1: Vertical Indentation Press
    # -------------------------------------------------------------
    lines.append("*STEP, NAME=PRESS, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.02, 1.0, 1e-5, 0.05")
    lines.append("*BOUNDARY")
    lines.append("PUNCH_1.PUNCH_TOP, 1, 1, 0.0")
    lines.append("PUNCH_1.PUNCH_TOP, 2, 2, -0.18")
    lines.append("*END STEP")

    content = "\n".join(lines) + "\n"
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Deck written successfully to {inp_path}")
    return inp_path


if __name__ == "__main__":
    generate_deck()
