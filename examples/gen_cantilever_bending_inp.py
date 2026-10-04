"""
gen_cantilever_bending_inp.py
=============================
Generates a pure Abaqus .inp input deck for a cantilever beam undergoing
90-degree large-rotation bending (Abaqus Benchmark Manual test problem).
"""

import os
import numpy as np


def generate_cantilever_bending_inp(output_path: str = "examples/cantilever_90deg_bending.inp"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Geometry: Length L = 10.0 mm, Height H = 0.5 mm
    L = 10.0
    H = 0.5
    nx = 20  # 20 elements along length
    ny = 2   # 2 elements through height

    x_coords = np.linspace(0, L, nx + 1)
    y_coords = np.linspace(0, H, ny + 1)

    nodes = []
    node_id_map = {}
    nid = 1

    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            nodes.append((nid, x, y))
            node_id_map[(i, j)] = nid
            nid += 1

    # Elements: CPE4 plane strain quad elements
    elements = []
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_id_map[(i, j)]
            n2 = node_id_map[(i + 1, j)]
            n3 = node_id_map[(i + 1, j + 1)]
            n4 = node_id_map[(i, j + 1)]
            elements.append((eid, n1, n2, n3, n4))
            eid += 1

    # Node sets
    fixed_nodes = [node_id_map[(0, j)] for j in range(ny + 1)]
    tip_nodes = [node_id_map[(nx, j)] for j in range(ny + 1)]

    lines = []
    lines.append("*HEADING")
    lines.append(" 90-DEGREE LARGE ROTATION CANTILEVER BEAM BENDING BENCHMARK")
    lines.append("**")
    lines.append("*NODE")
    for nid, x, y in nodes:
        lines.append(f"{nid:5d}, {x:12.6f}, {y:12.6f}")

    lines.append("**")
    lines.append("*ELEMENT, TYPE=CPE4, ELSET=BEAM")
    for eid, n1, n2, n3, n4 in elements:
        lines.append(f"{eid:5d}, {n1:5d}, {n2:5d}, {n3:5d}, {n4:5d}")

    lines.append("**")
    lines.append("*SOLID SECTION, ELSET=BEAM, MATERIAL=STEEL")
    lines.append("1.0,")

    lines.append("**")
    lines.append("*MATERIAL, NAME=STEEL")
    lines.append("*ELASTIC")
    lines.append("200000.0, 0.3")

    lines.append("** Node Sets")
    lines.append("*NSET, NSET=FIXED_ROOT")
    for idx, nid in enumerate(fixed_nodes):
        end_char = "\n" if (idx + 1) % 10 == 0 or idx == len(fixed_nodes) - 1 else ", "
        lines.append(f"{nid}" + end_char)

    lines.append("*NSET, NSET=TIP_NODES")
    for idx, nid in enumerate(tip_nodes):
        end_char = "\n" if (idx + 1) % 10 == 0 or idx == len(tip_nodes) - 1 else ", "
        lines.append(f"{nid}" + end_char)

    lines.append("**")
    lines.append("*BOUNDARY")
    for nid in fixed_nodes:
        lines.append(f"{nid}, 1, 2, 0.0")

    lines.append("**")
    lines.append("*STEP, NLGEOM")
    lines.append(" 90_DEGREE_BENDING_STEP")
    lines.append("*STATIC")
    lines.append("0.05, 1.0, 1e-5, 0.1")

    # Apply prescribed tip bending displacement (bending tip upwards by 6.0 mm)
    lines.append("*BOUNDARY")
    for nid in tip_nodes:
        lines.append(f"{nid}, 2, 2, 6.0")

    lines.append("*END STEP")

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Generated] Abaqus .inp bending benchmark written to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_cantilever_bending_inp()
