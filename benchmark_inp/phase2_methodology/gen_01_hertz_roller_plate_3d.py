"""
gen_01_hertz_roller_plate_3d.py
===============================
Generates 01_hertz_roller_plate_3d.inp:
3D Roller on Flat Foundation Plate with Hertz Contact & Coulomb Friction (mu = 0.2).

Multi-Step:
- Step-1: Vertical Indentation (Uy = -0.05 mm) to establish Hertz contact.
- Step-2: Horizontal Sliding (Ux = +0.20 mm, Uy held) to verify friction stick-slip transition.
"""

import numpy as np
import os


def generate_deck():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    inp_path = os.path.join(out_dir, "01_hertz_roller_plate_3d.inp")

    # Geometry parameters
    # Plate: x in [-15, 15], y in [-4, 0], z in [-2.5, 2.5]
    nx_p = 30  # dx = 1.0 mm
    ny_p = 4   # dy = 1.0 mm
    nz_p = 5   # dz = 1.0 mm
    xs_p = np.linspace(-15.0, 15.0, nx_p + 1)
    ys_p = np.linspace(-4.0, 0.0, ny_p + 1)
    zs_p = np.linspace(-2.5, 2.5, nz_p + 1)

    # Roller: half cylinder centered at x=0, y=10, radius R=10, bottom at y=0
    # angles from -pi/2 (bottom) to +pi/2 (top) or structured sector
    # Let's mesh a structured block representing a 90-degree sector or full cylindrical slice
    # For clean contact, a cylinder of radius R=10, x in [-6, 6] contact zone
    # Sector: r in [7.0, 10.0], theta in [-np.pi/3, np.pi/3] centered around bottom (theta = -pi/2)
    # Bottom point: x = 0, y = 0 (R=10, center at x=0, y=10)
    nr_r = 3
    nth_r = 16  # angles around bottom
    nz_r = 5
    rs_r = np.linspace(7.0, 10.0, nr_r + 1)
    # theta from -pi/2 - pi/4 to -pi/2 + pi/4 (i.e. -3pi/4 to -pi/4)
    thetas_r = np.linspace(-np.pi/2 - np.pi/4, -np.pi/2 + np.pi/4, nth_r + 1)
    zs_r = np.linspace(-2.5, 2.5, nz_r + 1)

    lines = []
    lines.append("*HEADING")
    lines.append("01_hertz_roller_plate_3d: 3D Roller Contact and Friction Sliding")

    # Part 1: Plate
    lines.append("*PART, NAME=PLATE")
    lines.append("*NODE")
    node_id = 1
    p_grid = {}
    for k in range(nz_p + 1):
        for j in range(ny_p + 1):
            for i in range(nx_p + 1):
                x = xs_p[i]
                y = ys_p[j]
                z = zs_p[k]
                lines.append(f"{node_id}, {x:.6f}, {y:.6f}, {z:.6f}")
                p_grid[(i, j, k)] = node_id
                node_id += 1

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=PLATE_ELEMS")
    elem_id = 1
    top_faces = []
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
                lines.append(f"{elem_id}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
                if j == ny_p - 1:  # top element row (y=0)
                    # For C3D8: Face S3 is (3,4,8,7)? No:
                    # Abaqus C3D8 face numbering:
                    # S1: (1,2,3,4) (z-min)
                    # S2: (5,8,7,6) (z-max)
                    # S3: (1,5,6,2) (y-min)
                    # S4: (2,6,7,3) (x-max)
                    # S5: (3,7,8,4) (y-max)
                    # S6: (1,4,8,5) (x-min)
                    top_faces.append(f"{elem_id}, S5")
                elem_id += 1

    # Node sets for Plate
    bot_nodes = [p_grid[(i, 0, k)] for k in range(nz_p + 1) for i in range(nx_p + 1)]
    lines.append("*NSET, NSET=PLATE_BOTTOM")
    for chunk in [bot_nodes[m:m+16] for m in range(0, len(bot_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Plate Top Surface for contact
    lines.append("*SURFACE, NAME=PLATE_TOP, TYPE=ELEMENT")
    for tf in top_faces:
        lines.append(tf)

    lines.append("*SOLID SECTION, ELSET=PLATE_ELEMS, MATERIAL=STEEL_PLATE")
    lines.append("1.0,")
    lines.append("*END PART")

    # Part 2: Roller
    lines.append("*PART, NAME=ROLLER")
    lines.append("*NODE")
    node_id_r = 1
    r_grid = {}
    center_y = 10.0
    for k in range(nz_r + 1):
        for j in range(nr_r + 1):
            r = rs_r[j]
            for i in range(nth_r + 1):
                th = thetas_r[i]
                x = r * np.cos(th)
                y = center_y + r * np.sin(th)
                z = zs_r[k]
                lines.append(f"{node_id_r}, {x:.6f}, {y:.6f}, {z:.6f}")
                r_grid[(i, j, k)] = node_id_r
                node_id_r += 1

    lines.append("*ELEMENT, TYPE=C3D8I, ELSET=ROLLER_ELEMS")
    elem_id_r = 1
    bot_roller_faces = []
    for k in range(nz_r):
        for j in range(nr_r):
            for i in range(nth_r):
                # radial slice (i: theta, j: r, k: z)
                # Near the roller bottom, +theta ~ +x and +r ~ -y, so the
                # (theta, r, z) node order is LEFT-handed (det J < 0). The
                # bottom face (n1-n4) is therefore taken at k+1 and the top
                # face (n5-n8) at k, which gives (theta, r, -z) and det J > 0.
                # Face S5 (n3, n7, n8, n4) is still the j+1 (outer, r = 10) face.
                n1 = r_grid[(i, j, k + 1)]
                n2 = r_grid[(i + 1, j, k + 1)]
                n3 = r_grid[(i + 1, j + 1, k + 1)]
                n4 = r_grid[(i, j + 1, k + 1)]
                n5 = r_grid[(i, j, k)]
                n6 = r_grid[(i + 1, j, k)]
                n7 = r_grid[(i + 1, j + 1, k)]
                n8 = r_grid[(i, j + 1, k)]
                lines.append(f"{elem_id_r}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
                if j == nr_r - 1:  # outer radius r = 10 (contacting surface)
                    # Outer surface is j=nr_r (y-max in local r direction: n3, n7, n8, n4 -> S5)
                    bot_roller_faces.append(f"{elem_id_r}, S5")
                elem_id_r += 1

    # Roller top nodes (inner radius r=7 or top surface for driving)
    roller_top_nodes = [r_grid[(i, 0, k)] for k in range(nz_r + 1) for i in range(nth_r + 1)]
    lines.append("*NSET, NSET=ROLLER_TOP")
    for chunk in [roller_top_nodes[m:m+16] for m in range(0, len(roller_top_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    lines.append("*SURFACE, NAME=ROLLER_OUTER, TYPE=ELEMENT")
    for rf in bot_roller_faces:
        lines.append(rf)

    lines.append("*SOLID SECTION, ELSET=ROLLER_ELEMS, MATERIAL=STEEL_ROLLER")
    lines.append("1.0,")
    lines.append("*END PART")

    # Assembly
    lines.append("*ASSEMBLY, NAME=ASSEMBLY")
    lines.append("*INSTANCE, NAME=PLATE_1, PART=PLATE")
    lines.append("*END INSTANCE")
    lines.append("*INSTANCE, NAME=ROLLER_1, PART=ROLLER")
    lines.append("*END INSTANCE")

    # Assembly sets/surfaces
    lines.append("*SURFACE, NAME=SURF_PLATE, COMBINE=UNION")
    lines.append("PLATE_1.PLATE_TOP")
    lines.append("*SURFACE, NAME=SURF_ROLLER, COMBINE=UNION")
    lines.append("ROLLER_1.ROLLER_OUTER")

    lines.append("*END ASSEMBLY")

    # Materials
    lines.append("*MATERIAL, NAME=STEEL_PLATE")
    lines.append("*ELASTIC")
    lines.append("200000.0, 0.3")

    lines.append("*MATERIAL, NAME=STEEL_ROLLER")
    lines.append("*ELASTIC")
    lines.append("200000.0, 0.3")

    # Contact Interaction
    lines.append("*SURFACE INTERACTION, NAME=INT_PROP")
    lines.append("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD")
    lines.append("*FRICTION")
    lines.append("0.2,")

    lines.append("*CONTACT PAIR, INTERACTION=INT_PROP")
    lines.append("SURF_ROLLER, SURF_PLATE")

    # Boundary conditions in Initial Step
    lines.append("*BOUNDARY")
    lines.append("PLATE_1.PLATE_BOTTOM, 1, 3, 0.0")

    # Step 1: Vertical Indentation
    lines.append("*STEP, NAME=INDENT, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.02, 1.0, 1e-5, 0.05")
    lines.append("*BOUNDARY")
    lines.append("ROLLER_1.ROLLER_TOP, 1, 1, 0.0")
    lines.append("ROLLER_1.ROLLER_TOP, 2, 2, -0.05")
    lines.append("ROLLER_1.ROLLER_TOP, 3, 3, 0.0")
    lines.append("*END STEP")

    # Step 2: Horizontal Sliding
    lines.append("*STEP, NAME=SLIDE, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.02, 1.0, 1e-5, 0.05")
    lines.append("*BOUNDARY")
    lines.append("ROLLER_1.ROLLER_TOP, 1, 1, 0.20")
    lines.append("ROLLER_1.ROLLER_TOP, 2, 2, -0.05")
    lines.append("ROLLER_1.ROLLER_TOP, 3, 3, 0.0")
    lines.append("*END STEP")

    content = "\n".join(lines) + "\n"
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Deck written to {inp_path}")
    return inp_path


if __name__ == "__main__":
    generate_deck()
