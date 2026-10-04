"""
gen_05_gear_tooth_finite_sliding_2d.py
======================================
Generates 05_gear_tooth_finite_sliding_2d.inp:
2D Gear Tooth Finite Sliding & Meshing Contact Benchmark.

Physics & Mechanics:
- Driver Gear Tooth (DRIVER):
  Curved involute contact flank, driven in 2 multi-step increments:
  - Step 1 (ENGAGE): Uy = -0.10 mm (Initial engagement onto follower tooth)
  - Step 2 (DRIVE_SLIDE): Ux = +0.50 mm (Transverse drive causing sliding contact)
  Material: Steel (E = 210,000 MPa, nu = 0.30)
  Element Type: CPE4I (2D Plane Strain 4-node Incompatible Modes EAS)
- Follower Gear Tooth (FOLLOWER):
  Meshing curved flank, bottom fixed in All DOFs (Ux=0, Uy=0).
  Material: Steel (E = 210,000 MPa, nu = 0.30)
  Element Type: CPE4I
- Contact Interaction:
  Surface-to-Surface Hard Contact with Coulomb Friction (mu = 0.15)
"""

import os
import numpy as np


def generate_deck():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    inp_path = os.path.join(out_dir, "05_gear_tooth_finite_sliding_2d.inp")

    # Driver Tooth Mesh Dimensions
    # Base block 10mm wide x 10mm high, with a curved involute top surface
    nx_d = 10
    ny_d = 8
    xs_d = np.linspace(-5.0, 5.0, nx_d + 1)
    ys_base_d = np.linspace(12.0, 20.0, ny_d + 1)

    # Follower Tooth Mesh Dimensions
    # Base block 10mm wide x 10mm high, sitting below
    nx_f = 10
    ny_f = 8
    xs_f = np.linspace(-5.0, 5.0, nx_f + 1)
    ys_base_f = np.linspace(0.0, 8.0, ny_f + 1)

    lines = []
    lines.append("*HEADING")
    lines.append("05_gear_tooth_finite_sliding_2d: 2D Gear Tooth Finite Sliding Contact (CPE4I + Friction mu=0.15)")

    # -------------------------------------------------------------
    # Part 1: DRIVER_TOOTH
    # -------------------------------------------------------------
    lines.append("*PART, NAME=DRIVER_TOOTH")
    lines.append("*NODE")
    d_grid = {}
    node_id_d = 1

    # Add involute curvature to bottom contact profile of driver
    for j in range(ny_d + 1):
        for i in range(nx_d + 1):
            x = xs_d[i]
            y_base = ys_base_d[j]
            # Curved bottom flank profile: smooth parabolic arc y_offset = 0.015 * x^2
            if j == 0:
                y = y_base + 0.015 * (x ** 2)
            else:
                frac = j / ny_d
                y = (1.0 - frac) * (ys_base_d[0] + 0.015 * (x ** 2)) + frac * ys_base_d[-1]

            lines.append(f"{node_id_d}, {x:.6f}, {y:.6f}")
            d_grid[(i, j)] = node_id_d
            node_id_d += 1

    lines.append("*ELEMENT, TYPE=CPE4I, ELSET=DRIVER_ELEMS")
    elem_id_d = 1
    for j in range(ny_d):
        for i in range(nx_d):
            n1 = d_grid[(i, j)]
            n2 = d_grid[(i + 1, j)]
            n3 = d_grid[(i + 1, j + 1)]
            n4 = d_grid[(i, j + 1)]
            lines.append(f"{elem_id_d}, {n1}, {n2}, {n3}, {n4}")
            elem_id_d += 1

    # Nsets and Surfaces for DRIVER
    lines.append("*NSET, NSET=NSET_TOP_DRIVER")
    top_nodes_d = [d_grid[(i, ny_d)] for i in range(nx_d + 1)]
    for idx in range(0, len(top_nodes_d), 10):
        lines.append(", ".join(map(str, top_nodes_d[idx:idx + 10])))

    lines.append("*SURFACE, NAME=SURF_DRIVER_FLANK")
    for i in range(nx_d):
        # Bottom edge face (S1 = nodes n1-n2)
        eid = i + 1
        lines.append(f"{eid}, S1")

    lines.append("*SOLID SECTION, ELSET=DRIVER_ELEMS, MATERIAL=STEEL")
    lines.append("1.0")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Part 2: FOLLOWER_TOOTH
    # -------------------------------------------------------------
    lines.append("*PART, NAME=FOLLOWER_TOOTH")
    lines.append("*NODE")
    f_grid = {}
    node_id_f = 1

    for j in range(ny_f + 1):
        for i in range(nx_f + 1):
            x = xs_f[i]
            y_base = ys_base_f[j]
            # Curved top flank profile: matching convex curvature
            if j == ny_f:
                y = y_base + 0.015 * (x ** 2)
            else:
                frac = j / ny_f
                y = (1.0 - frac) * ys_base_f[0] + frac * (ys_base_f[-1] + 0.015 * (x ** 2))

            lines.append(f"{node_id_f}, {x:.6f}, {y:.6f}")
            f_grid[(i, j)] = node_id_f
            node_id_f += 1

    lines.append("*ELEMENT, TYPE=CPE4I, ELSET=FOLLOWER_ELEMS")
    elem_id_f = 1
    for j in range(ny_f):
        for i in range(nx_f):
            n1 = f_grid[(i, j)]
            n2 = f_grid[(i + 1, j)]
            n3 = f_grid[(i + 1, j + 1)]
            n4 = f_grid[(i, j + 1)]
            lines.append(f"{elem_id_f}, {n1}, {n2}, {n3}, {n4}")
            elem_id_f += 1

    # Nsets and Surfaces for FOLLOWER
    lines.append("*NSET, NSET=NSET_BOT_FOLLOWER")
    bot_nodes_f = [f_grid[(i, 0)] for i in range(nx_f + 1)]
    for idx in range(0, len(bot_nodes_f), 10):
        lines.append(", ".join(map(str, bot_nodes_f[idx:idx + 10])))

    lines.append("*SURFACE, NAME=SURF_FOLLOWER_FLANK")
    for i in range(nx_f):
        # Top edge face (S3 = nodes n3-n4)
        eid = (ny_f - 1) * nx_f + i + 1
        lines.append(f"{eid}, S3")

    lines.append("*SOLID SECTION, ELSET=FOLLOWER_ELEMS, MATERIAL=STEEL")
    lines.append("1.0")
    lines.append("*END PART")

    # -------------------------------------------------------------
    # Assembly & Materials
    # -------------------------------------------------------------
    lines.append("*ASSEMBLY, NAME=ASSEMBLY")
    lines.append("*INSTANCE, NAME=INST_DRIVER, PART=DRIVER_TOOTH")
    # Position driver slightly above follower gap
    lines.append("0.0, -3.95, 0.0")
    lines.append("*END INSTANCE")

    lines.append("*INSTANCE, NAME=INST_FOLLOWER, PART=FOLLOWER_TOOTH")
    lines.append("0.0, 0.0, 0.0")
    lines.append("*END INSTANCE")

    lines.append("*SURFACE, NAME=SURF_DRIVER, COMBINE=UNION")
    lines.append("INST_DRIVER.SURF_DRIVER_FLANK")
    lines.append("*SURFACE, NAME=SURF_FOLLOWER, COMBINE=UNION")
    lines.append("INST_FOLLOWER.SURF_FOLLOWER_FLANK")
    lines.append("*END ASSEMBLY")

    lines.append("*MATERIAL, NAME=STEEL")
    lines.append("*ELASTIC")
    lines.append("210000.0, 0.30")

    lines.append("*SURFACE INTERACTION, NAME=GEAR_CONTACT_PROP")
    lines.append("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD")
    lines.append("*FRICTION")
    lines.append("0.15")

    lines.append("*CONTACT PAIR, INTERACTION=GEAR_CONTACT_PROP, TYPE=SURFACE TO SURFACE")
    lines.append("SURF_DRIVER, SURF_FOLLOWER")

    # -------------------------------------------------------------
    # Step 1: ENGAGE
    # -------------------------------------------------------------
    lines.append("*STEP, NAME=ENGAGE, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.1, 1.0, 1.0e-5, 0.1")

    lines.append("*BOUNDARY")
    lines.append("INST_FOLLOWER.NSET_BOT_FOLLOWER, 1, 2, 0.0")
    lines.append("INST_DRIVER.NSET_TOP_DRIVER, 1, 1, 0.0")
    lines.append("INST_DRIVER.NSET_TOP_DRIVER, 2, 2, -0.05")

    lines.append("*END STEP")

    # -------------------------------------------------------------
    # Step 2: DRIVE_SLIDE
    # -------------------------------------------------------------
    lines.append("*STEP, NAME=DRIVE_SLIDE, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.1, 1.0, 1.0e-5, 0.1")

    lines.append("*BOUNDARY")
    lines.append("INST_DRIVER.NSET_TOP_DRIVER, 1, 1, 0.20")
    lines.append("INST_DRIVER.NSET_TOP_DRIVER, 2, 2, -0.05")

    lines.append("*END STEP")



    with open(inp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[+] Successfully generated Deck #05: {inp_path}")


if __name__ == "__main__":
    generate_deck()
