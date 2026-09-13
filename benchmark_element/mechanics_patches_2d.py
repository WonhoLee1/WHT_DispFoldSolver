"""
mechanics_patches_2d.py
=======================
Parametric Mesh Generators for 2D Solid Mechanics Benchmarks & Patch Tests.
Generates canonical 2D meshes for all 10 Abaqus-compatible 2D plane-strain elements:
  - Quadrilaterals: CPE4, CPE4I, CPE4R, CPE4H, CPE4_FBAR, CPE4_CR, CPE8
  - Triangles: CPE3, CPE6, CPE6M
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from dispsolver.mesh2d.mesh2d import Mesh2D


def make_distorted_patch_mesh_2d(
    elem_type: str,
    distort_amplitude: float = 0.08
) -> Tuple[Mesh2D, List[int], List[int]]:
    """Create a 2D multi-element patch on [-1, 1]^2 with at least one strictly internal node.
    
    A 3x3 node grid defines 4 quadrants.
    Node (1,1) at center is strictly interior.
    
    Returns:
        (mesh, boundary_node_ids, interior_node_ids)
    """
    elem_type = elem_type.upper()
    mesh = Mesh2D()

    xs = np.linspace(-1.0, 1.0, 3)
    ys = np.linspace(-1.0, 1.0, 3)

    nid = 1
    node_grid = {}
    node_coords = {}
    boundary_nodes = []
    interior_nodes = []

    for j in range(3):
        for i in range(3):
            x = xs[i]
            y = ys[j]
            is_boundary = (i == 0 or i == 2 or j == 0 or j == 2)
            if not is_boundary:
                # Strictly interior node: perturb
                x += distort_amplitude * 0.5
                y -= distort_amplitude * 0.6
                interior_nodes.append(nid)
            else:
                # Boundary node: slightly perturb intermediate nodes along the edges
                if distort_amplitude > 0.0:
                    if (i == 1 and (j == 0 or j == 2)):
                        x += distort_amplitude * 0.2
                    elif (j == 1 and (i == 0 or i == 2)):
                        y -= distort_amplitude * 0.2
                boundary_nodes.append(nid)

            node_grid[(i, j)] = nid
            node_coords[nid] = np.array([x, y], dtype=np.float64)
            mesh.add_node(nid, x, y)
            nid += 1

    eid = 1
    if elem_type in ["CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE4_FBAR", "CPE4_CR"]:
        # 4 Quad elements
        for j in range(2):
            for i in range(2):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3, n4], elem_type=elem_type)
                eid += 1
        return mesh, boundary_nodes, interior_nodes

    elif elem_type == "CPE3":
        # 8 Linear triangular elements (split each quad diagonally)
        for j in range(2):
            for i in range(2):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                # 2 triangles
                mesh.add_element(eid, [n1, n2, n3], elem_type=elem_type)
                eid += 1
                mesh.add_element(eid, [n1, n3, n4], elem_type=elem_type)
                eid += 1
        return mesh, boundary_nodes, interior_nodes

    elif elem_type in ["CPE6", "CPE6M"]:
        # 6-node quadratic triangle patch built on the 3x3 quad grid (split into 8 triangles)
        # Adding mid-edge nodes
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        def _edge_is_on_patch_boundary(ga: Tuple[int, int], gb: Tuple[int, int]) -> bool:
            return (ga[0] == gb[0] and ga[0] in (0, 2)) or (ga[1] == gb[1] and ga[1] in (0, 2))

        for j in range(2):
            for i in range(2):
                c_ijk = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                n1, n2, n3, n4 = [node_grid[g] for g in c_ijk]
                g_of = dict(zip([n1, n2, n3, n4], c_ijk))

                tri_corners = [
                    [n1, n2, n3],
                    [n1, n3, n4]
                ]
                for c1, c2, c3 in tri_corners:
                    edges = [
                        (min(c1, c2), max(c1, c2)),
                        (min(c2, c3), max(c2, c3)),
                        (min(c3, c1), max(c3, c1)),
                    ]
                    mids = []
                    for ea, eb in edges:
                        if (ea, eb) not in edge_nodes:
                            mid_coord = 0.5 * (node_coords[ea] + node_coords[eb])
                            mesh.add_node(next_nid, mid_coord[0], mid_coord[1])
                            node_coords[next_nid] = mid_coord
                            if _edge_is_on_patch_boundary(g_of.get(ea, (-1,-1)), g_of.get(eb, (-1,-1))):
                                boundary_nodes.append(next_nid)
                            else:
                                interior_nodes.append(next_nid)
                            edge_nodes[(ea, eb)] = next_nid
                            next_nid += 1
                        mids.append(edge_nodes[(ea, eb)])

                    full_conn = [c1, c2, c3, mids[0], mids[1], mids[2]]
                    mesh.add_element(eid, full_conn, elem_type=elem_type)
                    eid += 1
        return mesh, boundary_nodes, interior_nodes

    elif elem_type in ["CPE8", "CPE8R"]:
        # 8-node serendipity quad patch
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        def _edge_is_on_patch_boundary(ga: Tuple[int, int], gb: Tuple[int, int]) -> bool:
            return (ga[0] == gb[0] and ga[0] in (0, 2)) or (ga[1] == gb[1] and ga[1] in (0, 2))

        for j in range(2):
            for i in range(2):
                c_ijk = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                n1, n2, n3, n4 = [node_grid[g] for g in c_ijk]
                g_of = dict(zip([n1, n2, n3, n4], c_ijk))

                edges = [
                    (min(n1, n2), max(n1, n2)),
                    (min(n2, n3), max(n2, n3)),
                    (min(n3, n4), max(n3, n4)),
                    (min(n4, n1), max(n4, n1)),
                ]
                mids = []
                for ea, eb in edges:
                    if (ea, eb) not in edge_nodes:
                        mid_coord = 0.5 * (node_coords[ea] + node_coords[eb])
                        mesh.add_node(next_nid, mid_coord[0], mid_coord[1])
                        node_coords[next_nid] = mid_coord
                        if _edge_is_on_patch_boundary(g_of[ea], g_of[eb]):
                            boundary_nodes.append(next_nid)
                        else:
                            interior_nodes.append(next_nid)
                        edge_nodes[(ea, eb)] = next_nid
                        next_nid += 1
                    mids.append(edge_nodes[(ea, eb)])

                full_conn = [n1, n2, n3, n4, mids[0], mids[1], mids[2], mids[3]]
                mesh.add_element(eid, full_conn, elem_type=elem_type)
                eid += 1
        return mesh, boundary_nodes, interior_nodes

    else:
        raise ValueError(f"Unknown element type: {elem_type}")


def make_cantilever_beam_mesh_2d(
    elem_type: str,
    L: float = 10.0,
    h: float = 1.0,
    nx: int = 10,
    ny: int = 2
) -> Tuple[Mesh2D, List[int], List[int]]:
    """Generate structured 2D cantilever beam mesh [0, L] x [-h/2, h/2].
    
    Returns:
        (mesh, root_nodes_at_x0, tip_nodes_at_xL)
    """
    elem_type = elem_type.upper()
    mesh = Mesh2D()

    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(-h / 2.0, h / 2.0, ny + 1)

    node_grid = {}
    nid = 1
    root_nodes = []
    tip_nodes = []

    for j in range(ny + 1):
        for i in range(nx + 1):
            x = xs[i]
            y = ys[j]
            mesh.add_node(nid, x, y)
            node_grid[(i, j)] = nid
            if i == 0:
                root_nodes.append(nid)
            elif i == nx:
                tip_nodes.append(nid)
            nid += 1

    eid = 1
    if elem_type in ["CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE4_FBAR", "CPE4_CR"]:
        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3, n4], elem_type=elem_type)
                eid += 1

    elif elem_type == "CPE3":
        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3], elem_type=elem_type)
                eid += 1
                mesh.add_element(eid, [n1, n3, n4], elem_type=elem_type)
                eid += 1

    elif elem_type in ["CPE6", "CPE6M"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                tri_corners = [
                    [n1, n2, n3],
                    [n1, n3, n4]
                ]
                for c1, c2, c3 in tri_corners:
                    edges = [
                        (min(c1, c2), max(c1, c2)),
                        (min(c2, c3), max(c2, c3)),
                        (min(c3, c1), max(c3, c1)),
                    ]
                    mids = []
                    for ea, eb in edges:
                        if (ea, eb) not in edge_nodes:
                            c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                            c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                            mid_c = 0.5 * (c_a + c_b)
                            mesh.add_node(next_nid, mid_c[0], mid_c[1])
                            if np.isclose(mid_c[0], 0.0):
                                root_nodes.append(next_nid)
                            elif np.isclose(mid_c[0], L):
                                tip_nodes.append(next_nid)
                            edge_nodes[(ea, eb)] = next_nid
                            next_nid += 1
                        mids.append(edge_nodes[(ea, eb)])

                    mesh.add_element(eid, [c1, c2, c3, mids[0], mids[1], mids[2]], elem_type=elem_type)
                    eid += 1

    elif elem_type in ["CPE8", "CPE8R"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                edges = [
                    (min(n1, n2), max(n1, n2)),
                    (min(n2, n3), max(n2, n3)),
                    (min(n3, n4), max(n3, n4)),
                    (min(n4, n1), max(n4, n1)),
                ]
                mids = []
                for ea, eb in edges:
                    if (ea, eb) not in edge_nodes:
                        c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                        c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                        mid_c = 0.5 * (c_a + c_b)
                        mesh.add_node(next_nid, mid_c[0], mid_c[1])
                        if np.isclose(mid_c[0], 0.0):
                            root_nodes.append(next_nid)
                        elif np.isclose(mid_c[0], L):
                            tip_nodes.append(next_nid)
                        edge_nodes[(ea, eb)] = next_nid
                        next_nid += 1
                    mids.append(edge_nodes[(ea, eb)])

                mesh.add_element(eid, [n1, n2, n3, n4, mids[0], mids[1], mids[2], mids[3]], elem_type=elem_type)
                eid += 1
    else:
        raise ValueError(f"Unknown element type: {elem_type}")

    return mesh, root_nodes, tip_nodes


def make_cooks_membrane_mesh_2d(
    elem_type: str,
    nx: int = 8,
    ny: int = 8
) -> Tuple[Mesh2D, List[int], List[int]]:
    """Cook's Membrane benchmark: tapered swept panel under shear loading.
    
    Geometry:
      - Left edge: x = 0, y in [0, 44]
      - Right edge: x = 48, y in [44, 60]
      - Bottom edge: line connecting (0, 0) and (48, 44)
      - Top edge: line connecting (0, 44) and (48, 60)
    
    Returns:
        (mesh, root_nodes_at_x0, tip_nodes_at_x48)
    """
    elem_type = elem_type.upper()
    mesh = Mesh2D()

    xis = np.linspace(0.0, 1.0, nx + 1)
    etas = np.linspace(0.0, 1.0, ny + 1)

    node_grid = {}
    nid = 1
    root_nodes = []
    tip_nodes = []

    for j in range(ny + 1):
        for i in range(nx + 1):
            xi = xis[i]
            eta = etas[j]
            x = 48.0 * xi
            y = 44.0 * xi + eta * (44.0 - 28.0 * xi)
            mesh.add_node(nid, x, y)
            node_grid[(i, j)] = nid
            if i == 0:
                root_nodes.append(nid)
            elif i == nx:
                tip_nodes.append(nid)
            nid += 1

    eid = 1
    if elem_type in ["CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE4_FBAR", "CPE4_CR"]:
        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3, n4], elem_type=elem_type)
                eid += 1

    elif elem_type == "CPE3":
        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3], elem_type=elem_type)
                eid += 1
                mesh.add_element(eid, [n1, n3, n4], elem_type=elem_type)
                eid += 1

    elif elem_type in ["CPE6", "CPE6M"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                tri_corners = [
                    [n1, n2, n3],
                    [n1, n3, n4]
                ]
                for c1, c2, c3 in tri_corners:
                    edges = [
                        (min(c1, c2), max(c1, c2)),
                        (min(c2, c3), max(c2, c3)),
                        (min(c3, c1), max(c3, c1)),
                    ]
                    mids = []
                    for ea, eb in edges:
                        if (ea, eb) not in edge_nodes:
                            c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                            c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                            mid_c = 0.5 * (c_a + c_b)
                            mesh.add_node(next_nid, mid_c[0], mid_c[1])
                            if np.isclose(mid_c[0], 0.0):
                                root_nodes.append(next_nid)
                            elif np.isclose(mid_c[0], 48.0):
                                tip_nodes.append(next_nid)
                            edge_nodes[(ea, eb)] = next_nid
                            next_nid += 1
                        mids.append(edge_nodes[(ea, eb)])

                    mesh.add_element(eid, [c1, c2, c3, mids[0], mids[1], mids[2]], elem_type=elem_type)
                    eid += 1

    elif elem_type in ["CPE8", "CPE8R"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                edges = [
                    (min(n1, n2), max(n1, n2)),
                    (min(n2, n3), max(n2, n3)),
                    (min(n3, n4), max(n3, n4)),
                    (min(n4, n1), max(n4, n1)),
                ]
                mids = []
                for ea, eb in edges:
                    if (ea, eb) not in edge_nodes:
                        c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                        c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                        mid_c = 0.5 * (c_a + c_b)
                        mesh.add_node(next_nid, mid_c[0], mid_c[1])
                        if np.isclose(mid_c[0], 0.0):
                            root_nodes.append(next_nid)
                        elif np.isclose(mid_c[0], 48.0):
                            tip_nodes.append(next_nid)
                        edge_nodes[(ea, eb)] = next_nid
                        next_nid += 1
                    mids.append(edge_nodes[(ea, eb)])

                mesh.add_element(eid, [n1, n2, n3, n4, mids[0], mids[1], mids[2], mids[3]], elem_type=elem_type)
                eid += 1
    else:
        raise ValueError(f"Unknown element type: {elem_type}")

    return mesh, root_nodes, tip_nodes


def make_near_incompressible_block_mesh_2d(
    elem_type: str,
    L: float = 1.0,
    n: int = 4
) -> Tuple[Mesh2D, Dict[str, List[int]]]:
    """Square block [0, L] x [0, L] for confined compression / volumetric locking benchmark.
    
    Returns:
        (mesh, boundaries) where boundaries = {'bottom': [...], 'top': [...], 'left': [...], 'right': [...]}
    """
    elem_type = elem_type.upper()
    mesh = Mesh2D()

    xs = np.linspace(0.0, L, n + 1)
    ys = np.linspace(0.0, L, n + 1)

    node_grid = {}
    nid = 1
    boundaries: Dict[str, List[int]] = {"bottom": [], "top": [], "left": [], "right": []}

    for j in range(n + 1):
        for i in range(n + 1):
            x = xs[i]
            y = ys[j]
            mesh.add_node(nid, x, y)
            node_grid[(i, j)] = nid
            if j == 0:
                boundaries["bottom"].append(nid)
            if j == n:
                boundaries["top"].append(nid)
            if i == 0:
                boundaries["left"].append(nid)
            if i == n:
                boundaries["right"].append(nid)
            nid += 1

    eid = 1
    if elem_type in ["CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE4_FBAR", "CPE4_CR"]:
        for j in range(n):
            for i in range(n):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3, n4], elem_type=elem_type)
                eid += 1

    elif elem_type == "CPE3":
        for j in range(n):
            for i in range(n):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3], elem_type=elem_type)
                eid += 1
                mesh.add_element(eid, [n1, n3, n4], elem_type=elem_type)
                eid += 1

    elif elem_type in ["CPE6", "CPE6M"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(n):
            for i in range(n):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                tri_corners = [
                    [n1, n2, n3],
                    [n1, n3, n4]
                ]
                for c1, c2, c3 in tri_corners:
                    edges = [
                        (min(c1, c2), max(c1, c2)),
                        (min(c2, c3), max(c2, c3)),
                        (min(c3, c1), max(c3, c1)),
                    ]
                    mids = []
                    for ea, eb in edges:
                        if (ea, eb) not in edge_nodes:
                            c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                            c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                            mid_c = 0.5 * (c_a + c_b)
                            mesh.add_node(next_nid, mid_c[0], mid_c[1])
                            if np.isclose(mid_c[1], 0.0):
                                boundaries["bottom"].append(next_nid)
                            if np.isclose(mid_c[1], L):
                                boundaries["top"].append(next_nid)
                            if np.isclose(mid_c[0], 0.0):
                                boundaries["left"].append(next_nid)
                            if np.isclose(mid_c[0], L):
                                boundaries["right"].append(next_nid)
                            edge_nodes[(ea, eb)] = next_nid
                            next_nid += 1
                        mids.append(edge_nodes[(ea, eb)])

                    mesh.add_element(eid, [c1, c2, c3, mids[0], mids[1], mids[2]], elem_type=elem_type)
                    eid += 1

    elif elem_type in ["CPE8", "CPE8R"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(n):
            for i in range(n):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                edges = [
                    (min(n1, n2), max(n1, n2)),
                    (min(n2, n3), max(n2, n3)),
                    (min(n3, n4), max(n3, n4)),
                    (min(n4, n1), max(n4, n1)),
                ]
                mids = []
                for ea, eb in edges:
                    if (ea, eb) not in edge_nodes:
                        c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                        c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                        mid_c = 0.5 * (c_a + c_b)
                        mesh.add_node(next_nid, mid_c[0], mid_c[1])
                        if np.isclose(mid_c[1], 0.0):
                            boundaries["bottom"].append(next_nid)
                        if np.isclose(mid_c[1], L):
                            boundaries["top"].append(next_nid)
                        if np.isclose(mid_c[0], 0.0):
                            boundaries["left"].append(next_nid)
                        if np.isclose(mid_c[0], L):
                            boundaries["right"].append(next_nid)
                        edge_nodes[(ea, eb)] = next_nid
                        next_nid += 1
                    mids.append(edge_nodes[(ea, eb)])

                mesh.add_element(eid, [n1, n2, n3, n4, mids[0], mids[1], mids[2], mids[3]], elem_type=elem_type)
                eid += 1

    return mesh, boundaries


def make_multilayer_two_point_bending_mesh_2d(
    elem_type: str,
    layers: Optional[List[Any]] = None,
    L: float = 80.0,
    nx: int = 60,
    ny_per_layer: int = 2
) -> Tuple[Mesh2D, Dict[str, List[int]], Dict[int, int]]:
    """Generate structured 2D multilayer composite beam mesh for 2-point bending.
    
    Domain: [-L/2, L/2] x [-t_total/2, t_total/2]
    
    Parameters:
        elem_type: Element formulation (CPE4, CPE4I, CPE4R, CPE4H, CPE4_FBAR, CPE4_CR, CPE8, etc.)
        layers: List of LayerSpec or dicts with 'thickness' and 'pid'
        L: Total beam length in mm (default: 80 mm)
        nx: Element subdivisions along length
        ny_per_layer: Element subdivisions through thickness per layer
        
    Returns:
        (mesh, groups, elem_pids)
        groups = {'left': [...], 'right': [...], 'bottom': [...], 'top': [...], 'apex_nodes': [...]}
    """
    elem_type = elem_type.upper()
    mesh = Mesh2D()

    if layers is None:
        from benchmark_element.two_point_bending_multilayer_theory import get_standard_display_stackup
        layers = get_standard_display_stackup("3layer")

    def _get_thick(layer_obj):
        if hasattr(layer_obj, "t"):
            return float(layer_obj.t)
        elif isinstance(layer_obj, dict):
            return float(layer_obj.get("thickness", layer_obj.get("t", 0.05)))
        return 0.05

    t_total = sum(_get_thick(l) for l in layers)
    xs = np.linspace(-L / 2.0, L / 2.0, nx + 1)

    # Build composite y-coordinates with refined division per layer
    y_coords_list = []
    layer_elem_ranges = []  # (start_row, end_row, pid)
    cur_y = -0.5 * t_total
    current_row = 0

    for idx, l in enumerate(layers):
        l_thick = _get_thick(l)
        pid = idx + 1
        y_sub = np.linspace(cur_y, cur_y + l_thick, ny_per_layer + 1)
        if idx == 0:
            y_coords_list.extend(y_sub)
        else:
            y_coords_list.extend(y_sub[1:])
        layer_elem_ranges.append((current_row, current_row + ny_per_layer, pid))
        current_row += ny_per_layer
        cur_y += l_thick

    ys_all = np.array(y_coords_list, dtype=np.float64)
    ny_total = len(ys_all) - 1

    node_grid = {}
    nid = 1
    groups: Dict[str, List[int]] = {
        "left": [],
        "right": [],
        "bottom": [],
        "top": [],
        "apex_nodes": [],
    }

    mid_i = nx // 2

    for j in range(ny_total + 1):
        for i in range(nx + 1):
            x = xs[i]
            y = ys_all[j]
            mesh.add_node(nid, x, y)
            node_grid[(i, j)] = nid

            if i == 0:
                groups["left"].append(nid)
            elif i == nx:
                groups["right"].append(nid)

            if j == 0:
                groups["bottom"].append(nid)
            elif j == ny_total:
                groups["top"].append(nid)

            if i == mid_i:
                groups["apex_nodes"].append(nid)

            nid += 1

    eid = 1
    elem_pids: Dict[int, int] = {}

    def get_pid_for_row(row_idx: int) -> int:
        for r_start, r_end, p in layer_elem_ranges:
            if r_start <= row_idx < r_end:
                return p
        return 1

    if elem_type in ["CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE4_FBAR", "CPE4_CR"]:
        for j in range(ny_total):
            row_pid = get_pid_for_row(j)
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3, n4], elem_type=elem_type, pid=row_pid)
                elem_pids[eid] = row_pid
                eid += 1

    elif elem_type == "CPE3":
        for j in range(ny_total):
            row_pid = get_pid_for_row(j)
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3], elem_type=elem_type, pid=row_pid)
                elem_pids[eid] = row_pid
                eid += 1
                mesh.add_element(eid, [n1, n3, n4], elem_type=elem_type, pid=row_pid)
                elem_pids[eid] = row_pid
                eid += 1
    elif elem_type in ["CPE6", "CPE6M"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(ny_total):
            row_pid = get_pid_for_row(j)
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                tris = [
                    (n1, n2, n3),
                    (n1, n3, n4)
                ]
                for c1, c2, c3 in tris:
                    edges = [
                        (min(c1, c2), max(c1, c2)),
                        (min(c2, c3), max(c2, c3)),
                        (min(c3, c1), max(c3, c1)),
                    ]
                    mids = []
                    for ea, eb in edges:
                        if (ea, eb) not in edge_nodes:
                            c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                            c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                            mid_c = 0.5 * (c_a + c_b)
                            mesh.add_node(next_nid, mid_c[0], mid_c[1])
                            if np.isclose(mid_c[0], -L / 2.0):
                                groups["left"].append(next_nid)
                            elif np.isclose(mid_c[0], L / 2.0):
                                groups["right"].append(next_nid)
                            if np.isclose(mid_c[1], -0.5 * t_total):
                                groups["bottom"].append(next_nid)
                            elif np.isclose(mid_c[1], 0.5 * t_total):
                                groups["top"].append(next_nid)
                            if np.isclose(mid_c[0], xs[mid_i]):
                                groups["apex_nodes"].append(next_nid)
                            edge_nodes[(ea, eb)] = next_nid
                            next_nid += 1
                        mids.append(edge_nodes[(ea, eb)])

                    mesh.add_element(eid, [c1, c2, c3, mids[0], mids[1], mids[2]], elem_type=elem_type, pid=row_pid)
                    elem_pids[eid] = row_pid
                    eid += 1

    elif elem_type in ["CPE8", "CPE8R"]:
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid

        for j in range(ny_total):
            row_pid = get_pid_for_row(j)
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]

                edges = [
                    (min(n1, n2), max(n1, n2)),
                    (min(n2, n3), max(n2, n3)),
                    (min(n3, n4), max(n3, n4)),
                    (min(n4, n1), max(n4, n1)),
                ]
                mids = []
                for ea, eb in edges:
                    if (ea, eb) not in edge_nodes:
                        c_a = np.array([mesh.nodes[ea].x, mesh.nodes[ea].y])
                        c_b = np.array([mesh.nodes[eb].x, mesh.nodes[eb].y])
                        mid_c = 0.5 * (c_a + c_b)
                        mesh.add_node(next_nid, mid_c[0], mid_c[1])
                        if np.isclose(mid_c[0], -L / 2.0):
                            groups["left"].append(next_nid)
                        elif np.isclose(mid_c[0], L / 2.0):
                            groups["right"].append(next_nid)
                        if np.isclose(mid_c[1], -0.5 * t_total):
                            groups["bottom"].append(next_nid)
                        elif np.isclose(mid_c[1], 0.5 * t_total):
                            groups["top"].append(next_nid)
                        if np.isclose(mid_c[0], xs[mid_i]):
                            groups["apex_nodes"].append(next_nid)
                        edge_nodes[(ea, eb)] = next_nid
                        next_nid += 1
                    mids.append(edge_nodes[(ea, eb)])

                mesh.add_element(eid, [n1, n2, n3, n4, mids[0], mids[1], mids[2], mids[3]], elem_type=elem_type, pid=row_pid)
                elem_pids[eid] = row_pid
                eid += 1
    else:
        # Fallback to CPE4
        for j in range(ny_total):
            row_pid = get_pid_for_row(j)
            for i in range(nx):
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i + 1, j)]
                n3 = node_grid[(i + 1, j + 1)]
                n4 = node_grid[(i, j + 1)]
                mesh.add_element(eid, [n1, n2, n3, n4], elem_type=elem_type, pid=row_pid)
                elem_pids[eid] = row_pid
                eid += 1

    return mesh, groups, elem_pids

