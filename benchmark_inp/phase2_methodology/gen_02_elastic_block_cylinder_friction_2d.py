"""
gen_02_elastic_block_cylinder_friction_2d.py
============================================
Generates 02_elastic_block_cylinder_friction_2d.inp:
2D Elastic Block - Roller Contact and Friction Sliding (CPE4I_CR elements, mu = 0.20).

Physical setup:
- Foundation Elastic Block: 60 mm x 10 mm (x in [-30, 30], y in [-10, 0])
  E = 1000.0 MPa, nu = 0.30
- Upper Roller Indenter: R = 10 mm, center at (0, 10), contacting at (0, 0)
  Sector mesh: r in [7.0, 10.0], theta in [-3pi/4, -pi/4]
  E = 200000.0 MPa, nu = 0.30 (Steel-like rigid/elastic roller)
- Multi-Step:
  - Step 1: Vertical Indentation (Uy = -0.10 mm)
  - Step 2: Horizontal Sliding (Ux = +1.0 mm, Uy held at -0.10 mm)
- Interaction: Hard normal contact with Coulomb friction mu = 0.20
"""

import os
import numpy as np


def generate_deck():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    inp_path = os.path.join(out_dir, "02_elastic_block_cylinder_friction_2d.inp")

    # -------------------------------------------------------------
    # Geometry Parameters
    # -------------------------------------------------------------
    # Block: x in [-30.0, 30.0], y in [-10.0, 0.0]
    nx_b = 60  # dx = 1.0 mm
    ny_b = 10  # dy = 1.0 mm
    xs_b = np.linspace(-30.0, 30.0, nx_b + 1)
    ys_b = np.linspace(-10.0, 0.0, ny_b + 1)

    # Roller: R = 10 mm, center (0, 10)
    nr_r = 4
    nth_r = 32
    rs_r = np.linspace(7.0, 10.0, nr_r + 1)
    thetas_r = np.linspace(-np.pi / 2.0 - np.pi / 4.0, -np.pi / 2.0 + np.pi / 4.0, nth_r + 1)
    center_y = 10.0

    lines = []
    lines.append("*HEADING")
    lines.append("02_elastic_block_cylinder_friction_2d: 2D Elastic Block - Cylinder Friction Sliding")

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
            # Counter-clockwise 2D quad: (x, y) -> (+x, y) -> (+x, +y) -> (x, +y)
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

    lines.append("*SOLID SECTION, ELSET=BLOCK_ELEMS, MATERIAL=ELASTIC_BLOCK")
    lines.append("1.0,")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Part 2: ROLLER (Cylindrical Indenter)
    # -------------------------------------------------------------
    lines.append("*PART, NAME=ROLLER")
    lines.append("*NODE")
    r_grid = {}
    node_id_r = 1
    for j in range(nr_r + 1):
        r = rs_r[j]
        for i in range(nth_r + 1):
            th = thetas_r[i]
            x = r * np.cos(th)
            y = center_y + r * np.sin(th)
            lines.append(f"{node_id_r}, {x:.6f}, {y:.6f}")
            r_grid[(i, j)] = node_id_r
            node_id_r += 1

    lines.append("*ELEMENT, TYPE=CPE4I, ELSET=ROLLER_ELEMS")
    elem_id_r = 1
    bot_roller_edges = []
    for j in range(nr_r):
        for i in range(nth_r):
            # Verified orientation for theta-r 2D sector:
            # For bottom sector theta in [-3pi/4, -pi/4]:
            # Increasing theta (+i) moves in +x direction.
            # Increasing r (+j) moves in -y direction (since sin(th) < 0).
            # To maintain positive Jacobian det(J) > 0 (counter-clockwise):
            # n1: (i, j+1)     [outer, smaller y]
            # n2: (i+1, j+1)   [outer, larger x]
            # n3: (i+1, j)     [inner, larger y]
            # n4: (i, j)       [inner, smaller x]
            # Vector 1->2 is (+dx, 0); Vector 1->4 is (0, +dy) -> Cross product > 0!
            n1 = r_grid[(i, j + 1)]
            n2 = r_grid[(i + 1, j + 1)]
            n3 = r_grid[(i + 1, j)]
            n4 = r_grid[(i, j)]
            lines.append(f"{elem_id_r}, {n1}, {n2}, {n3}, {n4}")

            if j == nr_r - 1:
                # Outer contacting surface is edge n1-n2 (Abaqus 2D CPE4 face S1)
                bot_roller_edges.append(f"{elem_id_r}, S1")
            elem_id_r += 1

    # Roller top nodes (inner radius r=7, for kinematic driving)
    roller_top_nodes = [r_grid[(i, 0)] for i in range(nth_r + 1)]
    lines.append("*NSET, NSET=ROLLER_TOP")
    for chunk in [roller_top_nodes[m:m+16] for m in range(0, len(roller_top_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    lines.append("*SURFACE, NAME=ROLLER_OUTER, TYPE=ELEMENT")
    for re_edge in bot_roller_edges:
        lines.append(re_edge)

    lines.append("*SOLID SECTION, ELSET=ROLLER_ELEMS, MATERIAL=STEEL_ROLLER")
    lines.append("1.0,")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------
    lines.append("*ASSEMBLY, NAME=ASSEMBLY")
    lines.append("*INSTANCE, NAME=BLOCK_1, PART=BLOCK")
    lines.append("*END INSTANCE")
    lines.append("*INSTANCE, NAME=ROLLER_1, PART=ROLLER")
    lines.append("*END INSTANCE")

    # Assembly surfaces
    lines.append("*SURFACE, NAME=SURF_BLOCK, COMBINE=UNION")
    lines.append("BLOCK_1.BLOCK_TOP")
    lines.append("*SURFACE, NAME=SURF_ROLLER, COMBINE=UNION")
    lines.append("ROLLER_1.ROLLER_OUTER")
    lines.append("*END ASSEMBLY")

    # -------------------------------------------------------------
    # Materials
    # -------------------------------------------------------------
    lines.append("*MATERIAL, NAME=ELASTIC_BLOCK")
    lines.append("*ELASTIC")
    lines.append("1000.0, 0.30")

    lines.append("*MATERIAL, NAME=STEEL_ROLLER")
    lines.append("*ELASTIC")
    lines.append("200000.0, 0.30")

    # -------------------------------------------------------------
    # Contact Interaction (Hard + Coulomb Friction mu = 0.20)
    # -------------------------------------------------------------
    lines.append("*SURFACE INTERACTION, NAME=INT_PROP")
    lines.append("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD")
    lines.append("*FRICTION")
    lines.append("0.20,")

    lines.append("*CONTACT PAIR, INTERACTION=INT_PROP")
    lines.append("SURF_ROLLER, SURF_BLOCK")

    # -------------------------------------------------------------
    # Initial Boundary Conditions
    # -------------------------------------------------------------
    lines.append("*BOUNDARY")
    lines.append("BLOCK_1.BLOCK_BOTTOM, 1, 2, 0.0")

    # -------------------------------------------------------------
    # Step 1: Vertical Indentation (Establish Contact)
    # -------------------------------------------------------------
    lines.append("*STEP, NAME=INDENT, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.02, 1.0, 1e-5, 0.05")
    lines.append("*BOUNDARY")
    lines.append("ROLLER_1.ROLLER_TOP, 1, 1, 0.0")
    lines.append("ROLLER_1.ROLLER_TOP, 2, 2, -0.10")
    lines.append("*END STEP")

    # -------------------------------------------------------------
    # Step 2: Horizontal Sliding (Stick -> Steady Slip)
    # -------------------------------------------------------------
    lines.append("*STEP, NAME=SLIDE, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.02, 1.0, 1e-5, 0.05")
    lines.append("*BOUNDARY")
    lines.append("ROLLER_1.ROLLER_TOP, 1, 1, 1.0")
    lines.append("ROLLER_1.ROLLER_TOP, 2, 2, -0.10")
    lines.append("*END STEP")

    content = "\n".join(lines) + "\n"
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Deck written successfully to {inp_path}")
    return inp_path


if __name__ == "__main__":
    generate_deck()
