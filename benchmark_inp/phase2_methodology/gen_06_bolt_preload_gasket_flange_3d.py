"""
gen_06_bolt_preload_gasket_flange_3d.py
=========================================
Generates 06_bolt_preload_gasket_flange_3d.inp:
3D Pipe Flange & Bolted Gasket Sealing Contact Benchmark (1/4 Symmetric Sector).

Physics & Multi-Step Mechanics:
- 1/4 Symmetric Sector (Theta in [0, 90 deg], X >= 0, Y >= 0).
- Soft Gasket:
  R_in = 16.0 mm, R_out = 24.0 mm, Height Z in [0.0, 1.5 mm] (half-thickness).
  Material: Gasket Elastomer/PTFE (E = 1,000 MPa, nu = 0.38).
  Element: C3D8.
- Steel Flange:
  R_in = 15.0 mm, R_out = 38.0 mm, Height Z in [1.5, 8.0 mm].
  Material: Structural Steel (E = 200,000 MPa, nu = 0.30).
  Element: C3D8.
- Multi-Step Analysis:
  - Step 1 (BOLT_PRELOAD): Clamping bolt displacement Uz = -0.06 mm on the bolt seat.
  - Step 2 (INTERNAL_PRESSURE): Internal pressure P = 5.0 MPa on the pipe inner wall.
- Keywords:
  Embedded *OUTPUT, FIELD and *OUTPUT, HISTORY for direct streaming to VTKHDF.
"""

import os
import numpy as np


def generate_deck():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    inp_path = os.path.join(out_dir, "06_bolt_preload_gasket_flange_3d.inp")

    # -------------------------------------------------------------
    # Discretization parameters
    # -------------------------------------------------------------
    # 1. Gasket: R in [16.0, 24.0], Theta in [0, pi/2], Z in [0.0, 1.5]
    nr_g = 4
    ntheta_g = 6
    nz_g = 2
    r_g = np.linspace(16.0, 24.0, nr_g + 1)
    theta_g = np.linspace(0.0, np.pi / 2.0, ntheta_g + 1)
    z_g = np.linspace(0.0, 1.5, nz_g + 1)

    # 2. Flange: R in [15.0, 38.0], Theta in [0, pi/2], Z in [1.5, 8.0]
    # R split into two zones: R in [15.0, 24.0] (matching gasket) and [24.0, 38.0] (bolt seat)
    nr_f1 = 4
    nr_f2 = 4
    r_f1 = np.linspace(15.0, 24.0, nr_f1 + 1)
    r_f2 = np.linspace(24.0, 38.0, nr_f2 + 1)[1:]
    r_f = np.concatenate([r_f1, r_f2])
    nr_f = len(r_f) - 1  # 8
    ntheta_f = 6
    nz_f = 3
    theta_f = np.linspace(0.0, np.pi / 2.0, ntheta_f + 1)
    z_f = np.linspace(1.5, 8.0, nz_f + 1)

    lines = []
    lines.append("*HEADING")
    lines.append("06_bolt_preload_gasket_flange_3d: 3D Pipe Flange & Gasket Sealing Contact Benchmark")

    # =============================================================
    # PART 1: GASKET
    # =============================================================
    lines.append("*PART, NAME=GASKET")
    lines.append("*NODE")
    g_grid = {}
    node_id_g = 1
    for k in range(nz_g + 1):
        for j in range(ntheta_g + 1):
            th = theta_g[j]
            cos_th = np.cos(th)
            sin_th = np.sin(th)
            for i in range(nr_g + 1):
                r = r_g[i]
                x = r * cos_th
                y = r * sin_th
                z = z_g[k]
                lines.append(f"{node_id_g}, {x:.6f}, {y:.6f}, {z:.6f}")
                g_grid[(i, j, k)] = node_id_g
                node_id_g += 1

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=GASKET_ELEMS")
    elem_id_g = 1
    top_gasket_faces = []
    for k in range(nz_g):
        for j in range(ntheta_g):
            for i in range(nr_g):
                # Standard counter-clockwise orientation
                # Bottom face: k
                n1 = g_grid[(i, j, k)]
                n2 = g_grid[(i + 1, j, k)]
                n3 = g_grid[(i + 1, j + 1, k)]
                n4 = g_grid[(i, j + 1, k)]
                # Top face: k + 1
                n5 = g_grid[(i, j, k + 1)]
                n6 = g_grid[(i + 1, j, k + 1)]
                n7 = g_grid[(i + 1, j + 1, k + 1)]
                n8 = g_grid[(i, j + 1, k + 1)]
                lines.append(f"{elem_id_g}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
                if k == nz_g - 1:
                    # Top face in Abaqus C3D8 is S2 (5, 6, 7, 8 or 5, 8, 7, 6)
                    top_gasket_faces.append(f"{elem_id_g}, S2")
                elem_id_g += 1

    # Symmetry NSETs for Gasket
    # Z-symmetry at Z = 0.0 (k = 0)
    z_sym_g = [g_grid[(i, j, 0)] for j in range(ntheta_g + 1) for i in range(nr_g + 1)]
    lines.append("*NSET, NSET=GASKET_Z_SYM")
    for chunk in [z_sym_g[m:m+16] for m in range(0, len(z_sym_g), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # X-symmetry at Y = 0.0 (theta = 0, j = 0) -> Uy = 0
    y_sym_g = [g_grid[(i, 0, k)] for k in range(nz_g + 1) for i in range(nr_g + 1)]
    lines.append("*NSET, NSET=GASKET_YSYM")
    for chunk in [y_sym_g[m:m+16] for m in range(0, len(y_sym_g), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Y-symmetry at X = 0.0 (theta = pi/2, j = ntheta_g) -> Ux = 0
    x_sym_g = [g_grid[(i, ntheta_g, k)] for k in range(nz_g + 1) for i in range(nr_g + 1)]
    lines.append("*NSET, NSET=GASKET_XSYM")
    for chunk in [x_sym_g[m:m+16] for m in range(0, len(x_sym_g), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Surface for Gasket Top
    lines.append("*SURFACE, NAME=GASKET_TOP, TYPE=ELEMENT")
    for face in top_gasket_faces:
        lines.append(face)

    lines.append("*SOLID SECTION, ELSET=GASKET_ELEMS, MATERIAL=GASKET_MAT")
    lines.append("1.0,")
    lines.append("*END PART")

    # =============================================================
    # PART 2: FLANGE
    # =============================================================
    lines.append("*PART, NAME=FLANGE")
    lines.append("*NODE")
    f_grid = {}
    node_id_f = 1
    for k in range(nz_f + 1):
        for j in range(ntheta_f + 1):
            th = theta_f[j]
            cos_th = np.cos(th)
            sin_th = np.sin(th)
            for i in range(nr_f + 1):
                r = r_f[i]
                x = r * cos_th
                y = r * sin_th
                z = z_f[k]
                lines.append(f"{node_id_f}, {x:.6f}, {y:.6f}, {z:.6f}")
                f_grid[(i, j, k)] = node_id_f
                node_id_f += 1

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=FLANGE_ELEMS")
    elem_id_f = 1
    bot_flange_faces = []
    inner_pipe_faces = []
    for k in range(nz_f):
        for j in range(ntheta_f):
            for i in range(nr_f):
                n1 = f_grid[(i, j, k)]
                n2 = f_grid[(i + 1, j, k)]
                n3 = f_grid[(i + 1, j + 1, k)]
                n4 = f_grid[(i, j + 1, k)]
                n5 = f_grid[(i, j, k + 1)]
                n6 = f_grid[(i + 1, j, k + 1)]
                n7 = f_grid[(i + 1, j + 1, k + 1)]
                n8 = f_grid[(i, j + 1, k + 1)]
                lines.append(f"{elem_id_f}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
                if k == 0:
                    # Bottom face in Abaqus C3D8 is S1 (1, 2, 3, 4)
                    bot_flange_faces.append(f"{elem_id_f}, S1")
                if i == 0:
                    # Inner radial face (i = 0): face 4-8-5-1 or 1-5-8-4 -> S6 in C3D8
                    inner_pipe_faces.append(f"{elem_id_f}, S6")
                elem_id_f += 1

    # Symmetry NSETs for Flange
    # X-symmetry at Y = 0.0 (j = 0) -> Uy = 0
    y_sym_f = [f_grid[(i, 0, k)] for k in range(nz_f + 1) for i in range(nr_f + 1)]
    lines.append("*NSET, NSET=FLANGE_YSYM")
    for chunk in [y_sym_f[m:m+16] for m in range(0, len(y_sym_f), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Y-symmetry at X = 0.0 (j = ntheta_f) -> Ux = 0
    x_sym_f = [f_grid[(i, ntheta_f, k)] for k in range(nz_f + 1) for i in range(nr_f + 1)]
    lines.append("*NSET, NSET=FLANGE_XSYM")
    for chunk in [x_sym_f[m:m+16] for m in range(0, len(x_sym_f), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Bolt seat NSET on top surface (k = nz_f, outer radius i >= 5, mid angles j in [2, 4])
    bolt_seat_nodes = []
    for j in range(2, 5):  # 30 deg to 60 deg (center at 45 deg)
        for i in range(5, nr_f + 1):  # R >= 27.5 mm
            bolt_seat_nodes.append(f_grid[(i, j, nz_f)])
    lines.append("*NSET, NSET=FLANGE_BOLT_SEAT")
    for chunk in [bolt_seat_nodes[m:m+16] for m in range(0, len(bolt_seat_nodes), 16)]:
        lines.append(", ".join(map(str, chunk)))

    # Surface for Flange Bottom (Master contact surface)
    lines.append("*SURFACE, NAME=FLANGE_BOTTOM, TYPE=ELEMENT")
    for face in bot_flange_faces:
        lines.append(face)

    # Surface for Pipe Inner Wall (Pressure surface)
    lines.append("*SURFACE, NAME=PIPE_INNER, TYPE=ELEMENT")
    for face in inner_pipe_faces:
        lines.append(face)

    lines.append("*SOLID SECTION, ELSET=FLANGE_ELEMS, MATERIAL=STEEL_FLANGE")
    lines.append("1.0,")
    lines.append("*END PART")

    # =============================================================
    # ASSEMBLY
    # =============================================================
    lines.append("*ASSEMBLY, NAME=ASSEMBLY")
    lines.append("*INSTANCE, NAME=GASKET_1, PART=GASKET")
    lines.append("*END INSTANCE")
    lines.append("*INSTANCE, NAME=FLANGE_1, PART=FLANGE")
    lines.append("*END INSTANCE")

    # Assembly-level contact surfaces
    lines.append("*SURFACE, NAME=SURF_GASKET, COMBINE=UNION")
    lines.append("GASKET_1.GASKET_TOP")
    lines.append("*SURFACE, NAME=SURF_FLANGE, COMBINE=UNION")
    lines.append("FLANGE_1.FLANGE_BOTTOM")
    lines.append("*END ASSEMBLY")

    # =============================================================
    # MATERIALS
    # =============================================================
    lines.append("*MATERIAL, NAME=GASKET_MAT")
    lines.append("*ELASTIC")
    lines.append("1000.0, 0.38")

    lines.append("*MATERIAL, NAME=STEEL_FLANGE")
    lines.append("*ELASTIC")
    lines.append("200000.0, 0.30")

    # =============================================================
    # CONTACT INTERACTION
    # =============================================================
    lines.append("*SURFACE INTERACTION, NAME=INT_SEALING")
    lines.append("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD")

    lines.append("*CONTACT PAIR, INTERACTION=INT_SEALING")
    lines.append("SURF_GASKET, SURF_FLANGE")

    # =============================================================
    # BOUNDARY CONDITIONS (Symmetry)
    # =============================================================
    lines.append("*BOUNDARY")
    lines.append("GASKET_1.GASKET_Z_SYM, 3, 3, 0.0")
    lines.append("GASKET_1.GASKET_YSYM, 2, 2, 0.0")
    lines.append("GASKET_1.GASKET_XSYM, 1, 1, 0.0")
    lines.append("FLANGE_1.FLANGE_YSYM, 2, 2, 0.0")
    lines.append("FLANGE_1.FLANGE_XSYM, 1, 1, 0.0")

    # =============================================================
    # STEP 1: BOLT PRELOAD (Clamping)
    # =============================================================
    lines.append("*STEP, NAME=BOLT_PRELOAD, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.25, 1.0, 1e-5, 0.5")
    lines.append("*BOUNDARY")
    lines.append("FLANGE_1.FLANGE_BOLT_SEAT, 3, 3, -0.060")
    lines.append("*OUTPUT, FIELD, FREQUENCY=1")
    lines.append("*NODE OUTPUT")
    lines.append("U, RF")
    lines.append("*ELEMENT OUTPUT")
    lines.append("S, E")
    lines.append("*OUTPUT, HISTORY, FREQUENCY=1")
    lines.append("*NODE OUTPUT, NSET=FLANGE_1.FLANGE_BOLT_SEAT")
    lines.append("U3, RF3")
    lines.append("*END STEP")

    # =============================================================
    # STEP 2: INTERNAL PRESSURE (P = 5.0 MPa)
    # =============================================================
    lines.append("*STEP, NAME=INTERNAL_PRESSURE, NLGEOM=YES")
    lines.append("*STATIC")
    lines.append("0.5, 1.0, 1e-5, 1.0")
    lines.append("*DSLOAD")
    lines.append("FLANGE_1.PIPE_INNER, P, 5.0")
    lines.append("*OUTPUT, FIELD, FREQUENCY=1")
    lines.append("*NODE OUTPUT")
    lines.append("U, RF")
    lines.append("*ELEMENT OUTPUT")
    lines.append("S, E")
    lines.append("*OUTPUT, HISTORY, FREQUENCY=1")
    lines.append("*NODE OUTPUT, NSET=FLANGE_1.FLANGE_BOLT_SEAT")
    lines.append("U3, RF3")
    lines.append("*END STEP")

    content = "\n".join(lines) + "\n"
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Deck written successfully to {inp_path}")
    return inp_path


if __name__ == "__main__":
    generate_deck()
