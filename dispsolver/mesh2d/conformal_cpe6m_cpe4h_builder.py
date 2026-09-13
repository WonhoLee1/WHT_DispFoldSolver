"""
conformal_cpe6m_cpe4h_builder.py
================================
Conformal Multi-Layer Mesh Generator for Flexible Display stackups:
- Upper Substrate Layer (PET/PI): CPE6M (6-node modified quadratic triangles)
- Lower Adhesive Interlayer (PSA): CPE4H (4-node linear hybrid quadrilaterals)

Features 2:1 horizontal refinement on the PSA layer, ensuring that all CPE6M
midside nodes along the interface coincide 1:1 with PSA corner nodes, achieving
100% conformal direct node sharing without hanging nodes.
"""

from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np
from dispsolver.mesh2d.mesh2d import Mesh2D


def create_cpe6m_cpe4h_conformal_stack(
    length: float = 10.0,
    t_pet: float = 0.1,
    t_psa: float = 0.05,
    nx: int = 10,
    ny_pet: int = 1,
    ny_psa: int = 1,
    pid_pet: int = 1,
    pid_psa: int = 2
) -> Tuple[Mesh2D, Dict[str, List[int]]]:
    """
    Generate a 100% conformal 2D multi-layer mesh with shared interface nodes:
    - PET layer: CPE6M quadratic triangle elements (nx blocks x ny_pet blocks, 2 triangles per block)
    - PSA layer: CPE4H linear quadrilateral elements (2*nx blocks x ny_psa blocks)

    Parameters
    ----------
    length : float
        Total span length in X direction (e.g. mm or m).
    t_pet : float
        Thickness of upper PET/PI layer in Y direction.
    t_psa : float
        Thickness of lower PSA adhesive layer in Y direction.
    nx : int
        Number of primary grid divisions in X direction for the PET block.
        The PSA layer will automatically have 2*nx horizontal divisions.
    ny_pet : int
        Number of block divisions in Y direction for PET layer (default 1).
    ny_psa : int
        Number of layer divisions in Y direction for PSA layer (default 1).
    pid_pet : int
        Property/Part ID for PET elements.
    pid_psa : int
        Property/Part ID for PSA elements.

    Returns
    -------
    mesh : Mesh2D
        Constructed 2D finite element mesh with shared interface nodes.
    sets : Dict[str, List[int]]
        Dictionary containing node sets:
        - "BOTTOM_NODES": Node IDs on y = 0
        - "INTERFACE_NODES": Shared node IDs on y = t_psa
        - "TOP_NODES": Node IDs on y = t_psa + t_pet
        - "LEFT_NODES": Node IDs on x = 0
        - "RIGHT_NODES": Node IDs on x = length
        - "PET_ELEMENTS": Element IDs in PET layer
        - "PSA_ELEMENTS": Element IDs in PSA layer
    """
    mesh = Mesh2D()
    
    # Node coordinate cache to prevent duplicates: (round(x, 8), round(y, 8)) -> node_id
    node_registry: Dict[Tuple[float, float], int] = {}
    current_node_id = 1

    def get_or_add_node(x: float, y: float) -> int:
        nonlocal current_node_id
        key = (round(float(x), 8), round(float(y), 8))
        if key not in node_registry:
            mesh.add_node(current_node_id, x, y)
            node_registry[key] = current_node_id
            current_node_id += 1
        return node_registry[key]

    elem_id = 1
    pet_elements: List[int] = []
    psa_elements: List[int] = []

    # -------------------------------------------------------------
    # 1. Build Lower PSA Layer (CPE4H with 2*nx horizontal divisions)
    #    Span: x in [0, length], y in [0, t_psa]
    # -------------------------------------------------------------
    nx_psa = 2 * nx
    dx_psa = length / nx_psa
    dy_psa = t_psa / ny_psa

    for j in range(ny_psa):
        y0 = j * dy_psa
        y1 = (j + 1) * dy_psa
        for i in range(nx_psa):
            x0 = i * dx_psa
            x1 = (i + 1) * dx_psa

            # 4 Corner nodes of CPE4H element in counter-clockwise order:
            # 4 (x0, y1) --- 3 (x1, y1)
            #   |              |
            # 1 (x0, y0) --- 2 (x1, y0)
            n1 = get_or_add_node(x0, y0)
            n2 = get_or_add_node(x1, y0)
            n3 = get_or_add_node(x1, y1)
            n4 = get_or_add_node(x0, y1)

            mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type="CPE4H", pid=pid_psa)
            psa_elements.append(elem_id)
            elem_id += 1

    # -------------------------------------------------------------
    # 2. Build Upper PET Layer (CPE6M with nx horizontal divisions)
    #    Span: x in [0, length], y in [t_psa, t_psa + t_pet]
    # -------------------------------------------------------------
    dx_pet = length / nx
    dy_pet = t_pet / ny_pet

    for j in range(ny_pet):
        y_bot = t_psa + j * dy_pet
        y_mid = t_psa + (j + 0.5) * dy_pet
        y_top = t_psa + (j + 1) * dy_pet

        for i in range(nx):
            x_left = i * dx_pet
            x_mid = (i + 0.5) * dx_pet
            x_right = (i + 1) * dx_pet

            # Block vertices and edge midpoints:
            # V4 (x_left, y_top) --- M3 (x_mid, y_top) --- V3 (x_right, y_top)
            #   |                     \                      |
            # M4 (x_left, y_mid)       Md (x_mid, y_mid)   M2 (x_right, y_mid)
            #   |                         \                  |
            # V1 (x_left, y_bot) --- M1 (x_mid, y_bot) --- V2 (x_right, y_bot)

            # Note: For j=0, (x_left, y_bot), (x_mid, y_bot), (x_right, y_bot)
            # are on y = t_psa, which EXACTLY match PSA top nodes!
            v1 = get_or_add_node(x_left, y_bot)
            m1 = get_or_add_node(x_mid, y_bot)
            v2 = get_or_add_node(x_right, y_bot)

            m4 = get_or_add_node(x_left, y_mid)
            md = get_or_add_node(x_mid, y_mid)
            m2 = get_or_add_node(x_right, y_mid)

            v4 = get_or_add_node(x_left, y_top)
            m3 = get_or_add_node(x_mid, y_top)
            v3 = get_or_add_node(x_right, y_top)

            # Divide rectangular block into 2 quadratic triangles with diagonal V1-V3:
            # Tri 1: Vertices (V1, V2, V3), Midsides (M1, M2, Md)
            # Abaqus CPE6/CPE6M node ordering: [Vertex 1, Vertex 2, Vertex 3, Mid 1-2, Mid 2-3, Mid 3-1]
            tri1_nodes = [v1, v2, v3, m1, m2, md]
            mesh.add_element(elem_id, tri1_nodes, elem_type="CPE6M", pid=pid_pet)
            pet_elements.append(elem_id)
            elem_id += 1

            # Tri 2: Vertices (V1, V3, V4), Midsides (Md, M3, M4)
            tri2_nodes = [v1, v3, v4, md, m3, m4]
            mesh.add_element(elem_id, tri2_nodes, elem_type="CPE6M", pid=pid_pet)
            pet_elements.append(elem_id)
            elem_id += 1

    # -------------------------------------------------------------
    # 3. Categorize Node Sets
    # -------------------------------------------------------------
    bottom_nodes: List[int] = []
    interface_nodes: List[int] = []
    top_nodes: List[int] = []
    left_nodes: List[int] = []
    right_nodes: List[int] = []

    tol = 1e-7
    y_interface = t_psa
    y_top_val = t_psa + t_pet

    for nid, node in mesh.nodes.items():
        if abs(node.y - 0.0) < tol:
            bottom_nodes.append(nid)
        if abs(node.y - y_interface) < tol:
            interface_nodes.append(nid)
        if abs(node.y - y_top_val) < tol:
            top_nodes.append(nid)
        if abs(node.x - 0.0) < tol:
            left_nodes.append(nid)
        if abs(node.x - length) < tol:
            right_nodes.append(nid)

    node_sets = {
        "BOTTOM_NODES": sorted(bottom_nodes),
        "INTERFACE_NODES": sorted(interface_nodes),
        "TOP_NODES": sorted(top_nodes),
        "LEFT_NODES": sorted(left_nodes),
        "RIGHT_NODES": sorted(right_nodes),
        "PET_ELEMENTS": sorted(pet_elements),
        "PSA_ELEMENTS": sorted(psa_elements)
    }

    mesh.node_sets = node_sets
    return mesh, node_sets


def create_5layer_cpe6m_cpe4h_conformal_mesh(
    length: float = 40.0,
    t_pet: float = 0.05,
    t_psa: float = 0.03,
    nx: int = 40,
    ny_pet: int = 1,
    ny_psa: int = 1,
    pid_pet: int = 1,
    pid_psa: int = 2
) -> Tuple[Mesh2D, Dict[str, List[int]]]:
    """
    Generate a 100% conformal 5-layer sandwich mesh (PET-PSA-PET-PSA-PET):
    - Layer 0 (Bottom PET): CPE6M (nx blocks x ny_pet)
    - Layer 1 (PSA 1):      CPE4H (2*nx blocks x ny_psa)
    - Layer 2 (Middle PET): CPE6M (nx blocks x ny_pet)
    - Layer 3 (PSA 2):      CPE4H (2*nx blocks x ny_psa)
    - Layer 4 (Top PET):    CPE6M (nx blocks x ny_pet)

    Every PET midside node on all 4 interfaces coincides with a PSA corner node,
    achieving 100% conformal node sharing with zero hanging nodes and zero duplicate nodes.
    """
    mesh = Mesh2D()
    node_registry: Dict[Tuple[float, float], int] = {}
    current_node_id = 1

    def get_or_add_node(x: float, y: float) -> int:
        nonlocal current_node_id
        key = (round(float(x), 8), round(float(y), 8))
        if key not in node_registry:
            mesh.add_node(current_node_id, x, y)
            node_registry[key] = current_node_id
            current_node_id += 1
        return node_registry[key]

    elem_id = 1
    pet_elements: List[int] = []
    psa_elements: List[int] = []

    layers = [
        ("PET", t_pet, ny_pet, pid_pet),
        ("PSA", t_psa, ny_psa, pid_psa),
        ("PET", t_pet, ny_pet, pid_pet),
        ("PSA", t_psa, ny_psa, pid_psa),
        ("PET", t_pet, ny_pet, pid_pet),
    ]

    y_current = 0.0
    dx_pet = length / nx
    dx_psa = length / (2 * nx)

    for l_idx, (l_type, l_thick, ny_l, pid_l) in enumerate(layers):
        dy = l_thick / ny_l
        for j in range(ny_l):
            y_bot = y_current + j * dy
            y_top = y_current + (j + 1) * dy

            if l_type == "PET":
                y_mid = y_current + (j + 0.5) * dy
                for i in range(nx):
                    x_left = i * dx_pet
                    x_mid = (i + 0.5) * dx_pet
                    x_right = (i + 1) * dx_pet

                    v1 = get_or_add_node(x_left, y_bot)
                    m1 = get_or_add_node(x_mid, y_bot)
                    v2 = get_or_add_node(x_right, y_bot)

                    m4 = get_or_add_node(x_left, y_mid)
                    md = get_or_add_node(x_mid, y_mid)
                    m2 = get_or_add_node(x_right, y_mid)

                    v4 = get_or_add_node(x_left, y_top)
                    m3 = get_or_add_node(x_mid, y_top)
                    v3 = get_or_add_node(x_right, y_top)

                    # Tri 1: [V1, V2, V3, M1, M2, Md]
                    mesh.add_element(elem_id, [v1, v2, v3, m1, m2, md], elem_type="CPE6M", pid=pid_l)
                    pet_elements.append(elem_id)
                    elem_id += 1

                    # Tri 2: [V1, V3, V4, Md, M3, M4]
                    mesh.add_element(elem_id, [v1, v3, v4, md, m3, m4], elem_type="CPE6M", pid=pid_l)
                    pet_elements.append(elem_id)
                    elem_id += 1

            elif l_type == "PSA":
                for i in range(2 * nx):
                    x0 = i * dx_psa
                    x1 = (i + 1) * dx_psa

                    n1 = get_or_add_node(x0, y_bot)
                    n2 = get_or_add_node(x1, y_bot)
                    n3 = get_or_add_node(x1, y_top)
                    n4 = get_or_add_node(x0, y_top)

                    mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type="CPE4H", pid=pid_l)
                    psa_elements.append(elem_id)
                    elem_id += 1

        y_current += l_thick

    total_h = y_current
    z_mid = total_h / 2.0
    tol = 1e-7

    bottom_nodes: List[int] = []
    top_nodes: List[int] = []
    left_nodes: List[int] = []
    right_nodes: List[int] = []
    centerline_nodes: List[int] = []
    interface_nodes: List[int] = []

    interface_ys = [t_pet, t_pet + t_psa, 2 * t_pet + t_psa, 2 * t_pet + 2 * t_psa]

    for nid, node in mesh.nodes.items():
        if abs(node.y - 0.0) < tol:
            bottom_nodes.append(nid)
        if abs(node.y - total_h) < tol:
            top_nodes.append(nid)
        if abs(node.x - 0.0) < tol:
            left_nodes.append(nid)
        if abs(node.x - length) < tol:
            right_nodes.append(nid)
        if abs(node.y - z_mid) < tol:
            centerline_nodes.append(nid)
        for iy in interface_ys:
            if abs(node.y - iy) < tol:
                interface_nodes.append(nid)
                break

    node_sets = {
        "BOTTOM_NODES": sorted(bottom_nodes),
        "TOP_NODES": sorted(top_nodes),
        "LEFT_NODES": sorted(left_nodes),
        "RIGHT_NODES": sorted(right_nodes),
        "CENTERLINE_NODES": sorted(centerline_nodes),
        "INTERFACE_NODES": sorted(list(set(interface_nodes))),
        "PET_ELEMENTS": sorted(pet_elements),
        "PSA_ELEMENTS": sorted(psa_elements)
    }

    mesh.node_sets = node_sets
    return mesh, node_sets
