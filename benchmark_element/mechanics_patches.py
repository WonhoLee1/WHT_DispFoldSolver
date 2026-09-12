"""
mechanics_patches.py
====================
Parametric Mesh Generators for 3D Solid Mechanics Benchmarks & Patch Tests.
Generates canonical meshes for all 11 solid elements:
  - Hexahedra: C3D8, C3D8I, C3D8_FBAR, C3D8_CR, C3D8H, C3D8R
  - Tetrahedra: C3D4, C3D4_ANP, C3D10, C3D10M
  - Wedges: C3D6
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from dispsolver.mesh3d.mesh3d import Mesh3D


def make_distorted_patch_mesh(
    elem_type: str,
    distort_amplitude: float = 0.08
) -> Tuple[Mesh3D, List[int], List[int]]:
    """Create a 3D multi-element patch with at least one strictly internal node.
    
    Returns:
        (mesh, boundary_node_ids, interior_node_ids)
    """
    elem_type = elem_type.upper()
    mesh = Mesh3D()

    # Base: 2x2x2 grid on [-1, 1]^3 with 27 nodes and 8 hexes
    xs = np.linspace(-1.0, 1.0, 3)
    ys = np.linspace(-1.0, 1.0, 3)
    zs = np.linspace(-1.0, 1.0, 3)

    nid = 1
    node_grid = {}
    node_coords = {}
    boundary_nodes = []
    interior_nodes = []

    # Deterministic perturbation vectors for internal distortion
    np.random.seed(42)

    for k in range(3):
        for j in range(3):
            for i in range(3):
                x = xs[i]
                y = ys[j]
                z = zs[k]
                is_boundary = (i == 0 or i == 2 or j == 0 or j == 2 or k == 0 or k == 2)
                if not is_boundary:
                    # Interior node: apply deterministic distortion
                    x += distort_amplitude * 0.5
                    y -= distort_amplitude * 0.6
                    z += distort_amplitude * 0.4
                    interior_nodes.append(nid)
                else:
                    # Boundary nodes: slightly perturb intermediate surface nodes to avoid trivial rectangular shapes
                    if distort_amplitude > 0.0 and (i == 1 or j == 1 or k == 1):
                        # perturb within surface plane
                        if i == 0 or i == 2:
                            y += distort_amplitude * 0.2 * (1 if j == 1 else 0)
                            z += distort_amplitude * 0.2 * (1 if k == 1 else 0)
                        elif j == 0 or j == 2:
                            x += distort_amplitude * 0.2 * (1 if i == 1 else 0)
                            z += distort_amplitude * 0.2 * (1 if k == 1 else 0)
                        elif k == 0 or k == 2:
                            x += distort_amplitude * 0.2 * (1 if i == 1 else 0)
                            y += distort_amplitude * 0.2 * (1 if j == 1 else 0)
                    boundary_nodes.append(nid)

                node_grid[(i, j, k)] = nid
                node_coords[nid] = np.array([x, y, z], dtype=np.float64)
                mesh.add_node(nid, x, y, z)
                nid += 1

    eid = 1
    if elem_type in ["C3D8", "C3D8I", "C3D8_FBAR", "C3D8_CR", "C3D8H", "C3D8R"]:
        # 8 Hexahedral elements
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    conn = [
                        node_grid[(i,     j,     k    )],
                        node_grid[(i + 1, j,     k    )],
                        node_grid[(i + 1, j + 1, k    )],
                        node_grid[(i,     j + 1, k    )],
                        node_grid[(i,     j,     k + 1)],
                        node_grid[(i + 1, j,     k + 1)],
                        node_grid[(i + 1, j + 1, k + 1)],
                        node_grid[(i,     j + 1, k + 1)]
                    ]
                    mesh.add_element(eid, conn, elem_type=elem_type)
                    eid += 1

    elif elem_type in ["C3D6", "C3D6_WEDGE", "WEDGE6"]:
        # 8 Hexes each split into 2 Wedges -> 16 wedges
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    n1 = node_grid[(i,     j,     k    )]
                    n2 = node_grid[(i + 1, j,     k    )]
                    n3 = node_grid[(i + 1, j + 1, k    )]
                    n4 = node_grid[(i,     j + 1, k    )]
                    n5 = node_grid[(i,     j,     k + 1)]
                    n6 = node_grid[(i + 1, j,     k + 1)]
                    n7 = node_grid[(i + 1, j + 1, k + 1)]
                    n8 = node_grid[(i,     j + 1, k + 1)]

                    # Wedge 1: (n1, n2, n4) bottom, (n5, n6, n8) top
                    mesh.add_element(eid, [n1, n2, n4, n5, n6, n8], elem_type="C3D6")
                    eid += 1
                    # Wedge 2: (n2, n3, n4) bottom, (n6, n7, n8) top
                    mesh.add_element(eid, [n2, n3, n4, n6, n7, n8], elem_type="C3D6")
                    eid += 1

    elif elem_type in ["C3D4", "C3D4_ANP", "ANP"]:
        # 8 Hexes each split into 6 Tetrahedra -> 48 tets
        # Standard Kuhn triangulation of a cube into 6 tets
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    n1 = node_grid[(i,     j,     k    )]
                    n2 = node_grid[(i + 1, j,     k    )]
                    n3 = node_grid[(i + 1, j + 1, k    )]
                    n4 = node_grid[(i,     j + 1, k    )]
                    n5 = node_grid[(i,     j,     k + 1)]
                    n6 = node_grid[(i + 1, j,     k + 1)]
                    n7 = node_grid[(i + 1, j + 1, k + 1)]
                    n8 = node_grid[(i,     j + 1, k + 1)]

                    tets = [
                        [n1, n2, n3, n7],
                        [n1, n3, n4, n7],
                        [n1, n4, n8, n7],
                        [n1, n8, n5, n7],
                        [n1, n5, n6, n7],
                        [n1, n6, n2, n7]
                    ]
                    for t in tets:
                        mesh.add_element(eid, t, elem_type=elem_type)
                        eid += 1

    elif elem_type in ["C3D10", "C3D10M", "C3D10_MODIFIED"]:
        # 10-node tetrahedra patch, built on the SAME 3x3x3-grid / 8-subcube
        # 6-tet-per-cube topology already proven correct for C3D4 above (that
        # branch passes the patch test to ~1e-19/1e-20) -- add quadratic
        # mid-edge nodes on top of it.
        #
        # 2026-09-13 FIX: the previous version of this branch built a SINGLE
        # distorted cube split into 5 tets, with the 6 "principal" edges of
        # the alternating-corner interior tet ([2,4,5,7]) hand-labeled
        # "interior". That is wrong: corners 2,4,5,7 are 4 of the cube's own
        # 8 corners, and each of the 6 edges connecting them is a FACE
        # DIAGONAL of one of the cube's 6 outer faces (e.g. edge (2,4) has
        # both endpoints at z=-1, so it lies IN the z=-1 face) -- i.e. every
        # node in that construction sits on the cube's outer surface; there
        # was no genuinely interior node at all. Leaving those "interior"
        # nodes unconstrained (no Dirichlet BC) then left them as free
        # surface DOFs implicitly subject to a natural (zero-traction)
        # condition inconsistent with the imposed nonzero-stress affine
        # field -- producing a real equilibrium mismatch of exactly the
        # measured ~1e-4 magnitude, unrelated to the C3D10/C3D10M element
        # formulation itself. Confirmed geometrically (all 6 "interior"
        # edges found to be face diagonals of the unit cube) before writing
        # this fix -- do not revert to the single-cube/5-tet construction
        # without re-deriving that it has an actual interior node.
        edge_nodes: Dict[Tuple[int, int], int] = {}
        next_nid = nid  # continue the running counter from the 3x3x3 corner grid above

        def _edge_is_on_patch_boundary(ga: Tuple[int, int, int], gb: Tuple[int, int, int]) -> bool:
            """An edge lies on the whole patch's outer surface iff its two
            endpoints share a boundary grid-index plane (both i=0, both
            i=2, both j=0, ...). Otherwise its midpoint is genuinely
            interior to the 3x3x3 domain."""
            for axis in range(3):
                if ga[axis] == gb[axis] and ga[axis] in (0, 2):
                    return True
            return False

        eid = 1
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    corners_ijk = [
                        (i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
                        (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1),
                    ]
                    n1, n2, n3, n4, n5, n6, n7, n8 = [node_grid[g] for g in corners_ijk]
                    g_of = dict(zip([n1, n2, n3, n4, n5, n6, n7, n8], corners_ijk))

                    tets = [
                        [n1, n2, n3, n7],
                        [n1, n3, n4, n7],
                        [n1, n4, n8, n7],
                        [n1, n8, n5, n7],
                        [n1, n5, n6, n7],
                        [n1, n6, n2, n7],
                    ]
                    for c1, c2, c3, c4 in tets:
                        edges = [
                            (min(c1, c2), max(c1, c2)),
                            (min(c2, c3), max(c2, c3)),
                            (min(c3, c1), max(c3, c1)),
                            (min(c1, c4), max(c1, c4)),
                            (min(c2, c4), max(c2, c4)),
                            (min(c3, c4), max(c3, c4)),
                        ]
                        mids = []
                        for ea, eb in edges:
                            if (ea, eb) not in edge_nodes:
                                mid_coord = 0.5 * (node_coords[ea] + node_coords[eb])
                                mesh.add_node(next_nid, mid_coord[0], mid_coord[1], mid_coord[2])
                                node_coords[next_nid] = mid_coord
                                if _edge_is_on_patch_boundary(g_of[ea], g_of[eb]):
                                    boundary_nodes.append(next_nid)
                                else:
                                    interior_nodes.append(next_nid)
                                edge_nodes[(ea, eb)] = next_nid
                                next_nid += 1
                            mids.append(edge_nodes[(ea, eb)])

                        full_conn = [c1, c2, c3, c4, mids[0], mids[1], mids[2], mids[3], mids[4], mids[5]]
                        mesh.add_element(eid, full_conn, elem_type=elem_type)
                        eid += 1

        return mesh, boundary_nodes, interior_nodes

    return mesh, boundary_nodes, interior_nodes


def make_cantilever_beam_mesh(
    elem_type: str,
    L: float = 10.0,
    h: float = 1.0,
    b: float = 1.0,
    nx: int = 10,
    ny: int = 1,
    nz: int = 2
) -> Tuple[Mesh3D, List[int], List[int]]:
    """Generate a structured cantilever beam mesh of dimension L x b x h.
    
    Beam extends along X in [0, L], Y in [-b/2, b/2], Z in [-h/2, h/2].
    
    Returns:
        (mesh, root_nodes_at_x0, tip_nodes_at_xL)
    """
    elem_type = elem_type.upper()
    mesh = Mesh3D()

    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(-b / 2.0, b / 2.0, ny + 1)
    zs = np.linspace(-h / 2.0, h / 2.0, nz + 1)

    nid = 1
    node_grid = {}
    root_nodes = []
    tip_nodes = []

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = xs[i]
                y = ys[j]
                z = zs[k]
                mesh.add_node(nid, x, y, z)
                node_grid[(i, j, k)] = nid
                if i == 0:
                    root_nodes.append(nid)
                elif i == nx:
                    tip_nodes.append(nid)
                nid += 1

    eid = 1
    if elem_type in ["C3D8", "C3D8I", "C3D8_FBAR", "C3D8_CR", "C3D8H", "C3D8R"]:
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    conn = [
                        node_grid[(i,     j,     k    )],
                        node_grid[(i + 1, j,     k    )],
                        node_grid[(i + 1, j + 1, k    )],
                        node_grid[(i,     j + 1, k    )],
                        node_grid[(i,     j,     k + 1)],
                        node_grid[(i + 1, j,     k + 1)],
                        node_grid[(i + 1, j + 1, k + 1)],
                        node_grid[(i,     j + 1, k + 1)]
                    ]
                    mesh.add_element(eid, conn, elem_type=elem_type)
                    eid += 1

    elif elem_type in ["C3D6", "C3D6_WEDGE", "WEDGE6"]:
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n1 = node_grid[(i,     j,     k    )]
                    n2 = node_grid[(i + 1, j,     k    )]
                    n3 = node_grid[(i + 1, j + 1, k    )]
                    n4 = node_grid[(i,     j + 1, k    )]
                    n5 = node_grid[(i,     j,     k + 1)]
                    n6 = node_grid[(i + 1, j,     k + 1)]
                    n7 = node_grid[(i + 1, j + 1, k + 1)]
                    n8 = node_grid[(i,     j + 1, k + 1)]

                    mesh.add_element(eid, [n1, n2, n4, n5, n6, n8], elem_type="C3D6")
                    eid += 1
                    mesh.add_element(eid, [n2, n3, n4, n6, n7, n8], elem_type="C3D6")
                    eid += 1

    elif elem_type in ["C3D4", "C3D4_ANP", "ANP"]:
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n1 = node_grid[(i,     j,     k    )]
                    n2 = node_grid[(i + 1, j,     k    )]
                    n3 = node_grid[(i + 1, j + 1, k    )]
                    n4 = node_grid[(i,     j + 1, k    )]
                    n5 = node_grid[(i,     j,     k + 1)]
                    n6 = node_grid[(i + 1, j,     k + 1)]
                    n7 = node_grid[(i + 1, j + 1, k + 1)]
                    n8 = node_grid[(i,     j + 1, k + 1)]

                    tets = [
                        [n1, n2, n3, n7],
                        [n1, n3, n4, n7],
                        [n1, n4, n8, n7],
                        [n1, n8, n5, n7],
                        [n1, n5, n6, n7],
                        [n1, n6, n2, n7]
                    ]
                    for t in tets:
                        mesh.add_element(eid, t, elem_type=elem_type)
                        eid += 1

    elif elem_type in ["C3D10", "C3D10M", "C3D10_MODIFIED"]:
        # Quadratic tet beam: split each hex into 5 tets with mid-edge nodes
        c_dict = {nid: np.array([node.x, node.y, node.z]) for nid, node in mesh.nodes.items()}
        edge_nodes = {}
        next_nid = len(mesh.nodes) + 1

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n1 = node_grid[(i,     j,     k    )]
                    n2 = node_grid[(i + 1, j,     k    )]
                    n3 = node_grid[(i + 1, j + 1, k    )]
                    n4 = node_grid[(i,     j + 1, k    )]
                    n5 = node_grid[(i,     j,     k + 1)]
                    n6 = node_grid[(i + 1, j,     k + 1)]
                    n7 = node_grid[(i + 1, j + 1, k + 1)]
                    n8 = node_grid[(i,     j + 1, k + 1)]

                    tets = [
                        [n1, n2, n4, n5],
                        [n2, n3, n4, n7],
                        [n2, n5, n6, n7],
                        [n4, n5, n7, n8],
                        [n2, n4, n5, n7]
                    ]
                    for corners in tets:
                        c1, c2, c3, c4 = corners
                        edges = [
                            (min(c1, c2), max(c1, c2)),
                            (min(c2, c3), max(c2, c3)),
                            (min(c3, c1), max(c3, c1)),
                            (min(c1, c4), max(c1, c4)),
                            (min(c2, c4), max(c2, c4)),
                            (min(c3, c4), max(c3, c4)),
                        ]
                        mids = []
                        for ea, eb in edges:
                            if (ea, eb) not in edge_nodes:
                                mid_coord = 0.5 * (c_dict[ea] + c_dict[eb])
                                mesh.add_node(next_nid, mid_coord[0], mid_coord[1], mid_coord[2])
                                c_dict[next_nid] = mid_coord
                                if abs(mid_coord[0] - 0.0) < 1e-6:
                                    root_nodes.append(next_nid)
                                elif abs(mid_coord[0] - L) < 1e-6:
                                    tip_nodes.append(next_nid)
                                edge_nodes[(ea, eb)] = next_nid
                                next_nid += 1
                            mids.append(edge_nodes[(ea, eb)])

                        full_conn = [c1, c2, c3, c4, mids[0], mids[1], mids[2], mids[3], mids[4], mids[5]]
                        mesh.add_element(eid, full_conn, elem_type=elem_type)
                        eid += 1

    return mesh, root_nodes, tip_nodes


def make_confined_compression_mesh(
    elem_type: str,
    n: int = 4,
    L: float = 1.0,
    distort_amplitude: float = 0.06,
    seed: int = 7,
) -> Tuple[Mesh3D, Dict[str, List[int]]]:
    """Cube of side L, subdivided into an n x n x n grid of subcubes, each
    split into 6 tets (same Kuhn scheme as make_distorted_patch_mesh's C3D4
    branch). ONLY interior nodes (not on any of the 6 outer faces) are
    randomly perturbed -- boundary faces stay perfectly flat/aligned so
    BOUNDARY-condition faces (bottom/top/four sides) can be identified and
    constrained cleanly.

    Purpose: a genuine confined/constrained compression test -- rollers on
    all four sides, fixed bottom, prescribed downward displacement on top
    -- is the classic setup for demonstrating element-to-element pressure
    "checkerboarding" in linear tets under near-incompressible material,
    unlike a smooth cantilever-bending load (which this project's own
    benchmark showed does NOT reliably trigger it -- see
    dev_log/solve_step_false_convergence_20260913.md's C3D4_ANP section).
    The random interior-node perturbation matters: a perfectly regular
    structured grid can hide checkerboarding through accidental symmetry
    (also observed directly in that same investigation).

    Only the C3D4/C3D4_ANP (4-node tet) connectivity is implemented --
    this generator exists specifically to verify ANP's actual effect, not
    as a general-purpose mesh builder for every element type.

    Returns:
        (mesh, faces) where faces = {
            'bottom': [node ids at z=0], 'top': [node ids at z=L],
            'x0': [...], 'xL': [...], 'y0': [...], 'yL': [...],
        } -- a node on an edge/corner appears in every face it touches.
    """
    elem_type = elem_type.upper()
    if elem_type not in ("C3D4", "C3D4_ANP", "ANP"):
        raise NotImplementedError(
            f"make_confined_compression_mesh only supports C3D4/C3D4_ANP, got {elem_type}"
        )

    mesh = Mesh3D()
    rng = np.random.RandomState(seed)

    xs = np.linspace(0.0, L, n + 1)
    node_grid: Dict[Tuple[int, int, int], int] = {}
    nid = 1
    h = L / n
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                x, y, z = xs[i], xs[j], xs[k]
                is_boundary = (i == 0 or i == n or j == 0 or j == n or k == 0 or k == n)
                if not is_boundary:
                    # Perturb only interior nodes, bounded well within the
                    # local cell so the mesh never tangles/inverts.
                    dx, dy, dz = (rng.rand(3) - 0.5) * 2.0 * distort_amplitude * h
                    x, y, z = x + dx, y + dy, z + dz
                mesh.add_node(nid, x, y, z)
                node_grid[(i, j, k)] = nid
                nid += 1

    eid = 1
    for k in range(n):
        for j in range(n):
            for i in range(n):
                n1 = node_grid[(i, j, k)]
                n2 = node_grid[(i + 1, j, k)]
                n3 = node_grid[(i + 1, j + 1, k)]
                n4 = node_grid[(i, j + 1, k)]
                n5 = node_grid[(i, j, k + 1)]
                n6 = node_grid[(i + 1, j, k + 1)]
                n7 = node_grid[(i + 1, j + 1, k + 1)]
                n8 = node_grid[(i, j + 1, k + 1)]
                tets = [
                    [n1, n2, n3, n7],
                    [n1, n3, n4, n7],
                    [n1, n4, n8, n7],
                    [n1, n8, n5, n7],
                    [n1, n5, n6, n7],
                    [n1, n6, n2, n7],
                ]
                for t in tets:
                    mesh.add_element(eid, t, elem_type=elem_type)
                    eid += 1

    faces: Dict[str, List[int]] = {"bottom": [], "top": [], "x0": [], "xL": [], "y0": [], "yL": []}
    for (i, j, k), the_nid in node_grid.items():
        if k == 0:
            faces["bottom"].append(the_nid)
        if k == n:
            faces["top"].append(the_nid)
        if i == 0:
            faces["x0"].append(the_nid)
        if i == n:
            faces["xL"].append(the_nid)
        if j == 0:
            faces["y0"].append(the_nid)
        if j == n:
            faces["yL"].append(the_nid)

    return mesh, faces


def compute_per_element_dilatation(mesh: Mesh3D, u: np.ndarray, nid_to_idx: Dict[int, int]) -> np.ndarray:
    """Post-processing only, independent of which kernel solved the
    problem: for every 4-node-tet element in `mesh`, compute det(F) at its
    single integration point from the CURRENT global displacement field
    `u`. Used to measure element-to-element pressure/dilatation
    "checkerboarding" -- the actual phenomenon ANP-style average-nodal-
    pressure elements exist to relieve -- directly, rather than inferring
    it indirectly from a tip-deflection ratio (which
    dev_log/solve_step_false_convergence_20260913.md found can fail to
    show any effect at all even when the fix is real).

    Plain NumPy, not Numba -- this runs once after convergence, not in the
    assembly hot path, so JIT compilation overhead isn't worth it here.
    """
    eids = sorted(mesh.elements.keys())
    detF = np.zeros(len(eids), dtype=np.float64)
    dN_dxi = np.array([
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    for idx, eid in enumerate(eids):
        el = mesh.elements[eid]
        coords = np.array([[mesh.nodes[nid].x, mesh.nodes[nid].y, mesh.nodes[nid].z] for nid in el.node_ids])
        J0 = dN_dxi @ coords  # (3,3)
        invJ0 = np.linalg.inv(J0)
        # Same convention as c3d4_anp_numba.py's own construction:
        # dN_dX[i,a] = sum_j invJ0[i,j] * dN_dxi[j,a].
        dN_dX = invJ0 @ dN_dxi  # (3,4)

        u_elem = np.zeros((4, 3), dtype=np.float64)
        for a, nid in enumerate(el.node_ids):
            i3 = nid_to_idx[nid]
            u_elem[a] = u[3 * i3: 3 * i3 + 3]

        F = np.eye(3, dtype=np.float64)
        for a in range(4):
            F += np.outer(u_elem[a], dN_dX[:, a])

        detF[idx] = np.linalg.det(F)

    return detF
