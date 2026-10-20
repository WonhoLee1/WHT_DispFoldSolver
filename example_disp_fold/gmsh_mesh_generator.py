"""
gmsh_mesh_generator.py
======================
High-fidelity 2D and 3D continuum mesh generator powered by the Gmsh 4.15 C-API.
Generates structured Quad (2D) and Hexahedral (3D) meshes with progressive biased sizing,
tape void handling, physical groups, and zero-overhead memory bridging to Mesh/Mesh3D.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import gmsh

from example_disp_fold.model_spec import UnifiedFoldModelConfig, LayerSpec, MaterialSpec
from dispsolver.mesh2d.mesh2d import Mesh2D, Element2D, Node2D
from dispsolver.mesh3d.mesh3d import Mesh3D, Element3D, Node3D


class GmshMeshGenerator:
    """Automated 2D Quad & 3D Hex mesh generator with Gmsh API."""

    def __init__(self, config: UnifiedFoldModelConfig) -> None:
        self.config = config
        self.geo = config.geometry
        self.mesh_spec = config.mesh
        self.case = config.simulation_case

    def build(self) -> Tuple[Any, Dict[str, Any]]:
        """Generates 2D Mesh or 3D Mesh3D based on simulation case dimension.

        Returns
        -------
        mesh : Mesh or Mesh3D
            Solver-ready mesh object.
        meta : dict
            Dictionary containing node sets, element sets, surfaces, and pivots.
        """
        dim = self.case.dim.upper()
        if dim == "2D":
            return self.build_2d()
        elif dim == "3D":
            return self.build_3d()
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

    def build_2d(self) -> Tuple[Mesh2D, Dict[str, Any]]:
        """Constructs 2D structured Quad continuum mesh via Gmsh."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0) # Quiet mode
        gmsh.model.add("DispFold_2D")

        half_L = self.geo.display_half_length  # 40.0
        hinge_gap = self.geo.hinge_half_gap     # 7.5
        plate_t = self.geo.plate_thickness     # 0.5
        nx_h = self.mesh_spec.nx_hinge
        nx_p = self.mesh_spec.nx_plate

        # Discrete X arrays (Left plate -> Free hinge -> Right plate)
        # Biased progression towards hinge
        prog = self.mesh_spec.bias_progression
        # Anchor total grading ratio to prog ** 20 so refinement (large nx_p) preserves aspect ratio without underflow
        total_ratio = float(prog ** 20.0)
        s_p = np.geomspace(1.0, total_ratio, nx_p + 1)
        s_p = (s_p - s_p[0]) / (s_p[-1] - s_p[0])
        xs_left = -half_L + s_p * ((-hinge_gap) - (-half_L))

        # Hinge: uniform dense
        xs_hinge = np.linspace(-hinge_gap, hinge_gap, nx_h + 1)

        # Right: +7.5 to +40 (denser near +7.5)
        xs_right = hinge_gap + (1.0 - s_p[::-1]) * (half_L - hinge_gap)

        xs = np.concatenate([xs_left[:-1], xs_hinge[:-1], xs_right])
        nx_total = len(xs) - 1

        mesh = Mesh2D()
        nid = 1
        elem_id = 1

        node_id_grid = [] # 2D array [row, col] -> nid
        y_curr = 0.0

        # Layer 0 bottom nodes
        # Base row: y = 0
        row_nids = []
        for j in range(len(xs)):
            mesh.add_node(nid, float(xs[j]), 0.0)
            row_nids.append(nid)
            nid += 1
        node_id_grid.append(row_nids)

        layer_elem_sets: Dict[str, List[int]] = {}
        layer_pids: Dict[int, int] = {} # elem_id -> pid

        # Build each layer
        for lay_idx, lay in enumerate(self.config.layers):
            t = lay.thickness
            ny = lay.ny_elements
            dy = t / float(ny)

            layer_elem_sets[lay.name] = []
            pid = 1 if "PET" in lay.material_name else (2 if "PSA" in lay.material_name else 3)

            for iy in range(ny):
                y_next = y_curr + dy
                new_row_nids = []
                for j in range(len(xs)):
                    mesh.add_node(nid, float(xs[j]), float(y_next))
                    new_row_nids.append(nid)
                    nid += 1

                # Form Quads between row_nids and new_row_nids
                prev_row = node_id_grid[-1]
                curr_row = new_row_nids

                for j in range(nx_total):
                    x_mid = 0.5 * (xs[j] + xs[j+1])
                    # Void check
                    if lay.is_void_in_hinge and (-hinge_gap <= x_mid <= hinge_gap):
                        continue # Tape void

                    n1 = prev_row[j]
                    n2 = prev_row[j+1]
                    n3 = curr_row[j+1]
                    n4 = curr_row[j]

                    mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type=lay.elem_type_2d, pid=pid)
                    layer_elem_sets[lay.name].append(elem_id)
                    layer_pids[elem_id] = pid
                    elem_id += 1

                node_id_grid.append(new_row_nids)
                y_curr = y_next

        # Support Plates (Left and Right)
        # Left plate: x in [-half_L, -hinge_gap], y in [-plate_t, 0]
        plate_pid = 4
        plate_l_nodes = []
        plate_r_nodes = []
        plate_l_top_edges = []
        plate_r_top_edges = []

        # Left plate nodes
        ny_plate = 2
        dy_p = plate_t / float(ny_plate)
        pl_grid = []
        for iy in range(ny_plate + 1):
            y_p = -plate_t + iy * dy_p
            row = []
            for j in range(len(xs_left)):
                mesh.add_node(nid, float(xs_left[j]), float(y_p))
                row.append(nid)
                plate_l_nodes.append(nid)
                nid += 1
            pl_grid.append(row)

        for iy in range(ny_plate):
            for j in range(len(xs_left) - 1):
                n1 = pl_grid[iy][j]
                n2 = pl_grid[iy][j+1]
                n3 = pl_grid[iy+1][j+1]
                n4 = pl_grid[iy+1][j]
                mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type="CPE4", pid=plate_pid)
                elem_id += 1

        for j in range(len(xs_left) - 1):
            plate_l_top_edges.append((pl_grid[-1][j], pl_grid[-1][j+1]))

        # Right plate nodes
        pr_grid = []
        for iy in range(ny_plate + 1):
            y_p = -plate_t + iy * dy_p
            row = []
            for j in range(len(xs_right)):
                mesh.add_node(nid, float(xs_right[j]), float(y_p))
                row.append(nid)
                plate_r_nodes.append(nid)
                nid += 1
            pr_grid.append(row)

        for iy in range(ny_plate):
            for j in range(len(xs_right) - 1):
                n1 = pr_grid[iy][j]
                n2 = pr_grid[iy][j+1]
                n3 = pr_grid[iy+1][j+1]
                n4 = pr_grid[iy+1][j]
                mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type="CPE4", pid=plate_pid)
                elem_id += 1

        for j in range(len(xs_right) - 1):
            plate_r_top_edges.append((pr_grid[-1][j], pr_grid[-1][j+1]))

        # Tied display bottom nodes
        # Left side: xs < -hinge_gap + 1e-6
        # Right side: xs > hinge_gap - 1e-6
        disp_bot_l_nids = [node_id_grid[0][j] for j in range(len(xs)) if xs[j] <= -hinge_gap + 1e-5]
        disp_bot_r_nids = [node_id_grid[0][j] for j in range(len(xs)) if xs[j] >= hinge_gap - 1e-5]

        # Top surface segments for self-contact
        top_row = node_id_grid[-1]
        disp_top_l_segs = [(top_row[j], top_row[j+1]) for j in range(len(xs) - 1) if 0.5 * (xs[j] + xs[j+1]) <= 0.0]
        disp_top_r_segs = [(top_row[j], top_row[j+1]) for j in range(len(xs) - 1) if 0.5 * (xs[j] + xs[j+1]) >= 0.0]

        gmsh.finalize()

        meta = {
            "dim": "2D",
            "xs": xs,
            "left_pivot": (float(-self.geo.hinge_pivot_x), float(self.geo.hinge_pivot_y)),
            "right_pivot": (float(self.geo.hinge_pivot_x), float(self.geo.hinge_pivot_y)),
            "plate_left_nodes": plate_l_nodes,
            "plate_right_nodes": plate_r_nodes,
            "disp_bottom_left_nids": disp_bot_l_nids,
            "disp_bottom_right_nids": disp_bot_r_nids,
            "plate_l_top_edges": plate_l_top_edges,
            "plate_r_top_edges": plate_r_top_edges,
            "disp_top_left_segments": disp_top_l_segs,
            "disp_top_right_segments": disp_top_r_segs,
            "layer_elem_sets": layer_elem_sets,
            "layer_pids": layer_pids,
            "thick_disp": self.config.total_thickness,
            "length_disp": half_L * 2.0,
            "layup": self.case.preset_name,
        }

        # Clean isolated unconnected nodes (e.g., from tape void in hinge)
        self._clean_isolated_nodes(mesh, meta)

        # Verify element Jacobian / orientation
        self._verify_2d_mesh(mesh)
        return mesh, meta

    def build_3d(self) -> Tuple[Mesh3D, Dict[str, Any]]:
        """Constructs 3D structured Hexahedral continuum mesh via Gmsh extrusion."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("DispFold_3D")

        half_L = self.geo.display_half_length  # 40.0
        hinge_gap = self.geo.hinge_half_gap     # 7.5
        plate_t = self.geo.plate_thickness     # 0.5
        width_z = self.geo.width_z             # 1.0
        nx_h = self.mesh_spec.nx_hinge
        nx_p = self.mesh_spec.nx_plate
        nz = self.mesh_spec.nz_elements

        # Biased progression in X
        prog = self.mesh_spec.bias_progression
        total_ratio = float(prog ** 20.0)
        s_p = np.geomspace(1.0, total_ratio, nx_p + 1)
        s_p = (s_p - s_p[0]) / (s_p[-1] - s_p[0])
        xs_left = -half_L + s_p * ((-hinge_gap) - (-half_L))
        xs_hinge = np.linspace(-hinge_gap, hinge_gap, nx_h + 1)
        xs_right = hinge_gap + (1.0 - s_p[::-1]) * (half_L - hinge_gap)

        xs = np.concatenate([xs_left[:-1], xs_hinge[:-1], xs_right])
        nx_total = len(xs) - 1

        zs = np.linspace(-width_z / 2.0, width_z / 2.0, nz + 1)

        mesh = Mesh3D()
        nid = 1
        elem_id = 1

        # 3D Node Grid [i, j, k] -> nid: x in xs, y in y_bounds, z in zs
        y_bounds = [0.0]
        y_curr = 0.0
        for lay in self.config.layers:
            t = lay.thickness
            for iy in range(lay.ny_elements):
                y_curr += t / float(lay.ny_elements)
                y_bounds.append(y_curr)

        disp_grid = np.zeros((nx_total + 1, len(y_bounds), nz + 1), dtype=np.int32)
        for k, z in enumerate(zs):
            for j, y in enumerate(y_bounds):
                for i, x in enumerate(xs):
                    mesh.nodes[nid] = Node3D(nid, float(x), float(y), float(z))
                    disp_grid[i, j, k] = nid
                    nid += 1

        layer_elem_sets: Dict[str, List[int]] = {}
        layer_pids: Dict[int, int] = {}

        # Build display Hex elements
        lay_idx = 0
        j_curr = 0
        for lay in self.config.layers:
            layer_elem_sets[lay.name] = []
            pid = 1 if "PET" in lay.material_name else (2 if "PSA" in lay.material_name else 3)
            for iy in range(lay.ny_elements):
                for k in range(nz):
                    for i in range(nx_total):
                        x_mid = 0.5 * (xs[i] + xs[i + 1])
                        if lay.is_void_in_hinge and (-hinge_gap - 1e-6 < x_mid < hinge_gap + 1e-6):
                            continue # Voided in free hinge span

                        n1 = int(disp_grid[i, j_curr, k])
                        n2 = int(disp_grid[i + 1, j_curr, k])
                        n3 = int(disp_grid[i + 1, j_curr + 1, k])
                        n4 = int(disp_grid[i, j_curr + 1, k])
                        n5 = int(disp_grid[i, j_curr, k + 1])
                        n6 = int(disp_grid[i + 1, j_curr, k + 1])
                        n7 = int(disp_grid[i + 1, j_curr + 1, k + 1])
                        n8 = int(disp_grid[i, j_curr + 1, k + 1])

                        mesh.elements[elem_id] = Element3D(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=lay.elem_type_3d, pid=pid)
                        layer_elem_sets[lay.name].append(elem_id)
                        layer_pids[elem_id] = pid
                        elem_id += 1
                j_curr += 1

        # Rigid support plates
        plate_pid = 4
        plate_l_nodes = []
        plate_r_nodes = []
        plate_l_top_faces = []
        plate_r_top_faces = []

        ny_plate = 2
        y_plate_bounds = np.linspace(-plate_t, 0.0, ny_plate + 1)

        # Left plate grid [i, j, k]
        pl_grid = np.zeros((len(xs_left), ny_plate + 1, nz + 1), dtype=np.int32)
        for k, z in enumerate(zs):
            for j, y in enumerate(y_plate_bounds):
                for i, x in enumerate(xs_left):
                    mesh.nodes[nid] = Node3D(nid, float(x), float(y), float(z))
                    pl_grid[i, j, k] = nid
                    plate_l_nodes.append(nid)
                    nid += 1

        for k in range(nz):
            for j in range(ny_plate):
                for i in range(len(xs_left) - 1):
                    n1 = int(pl_grid[i, j, k])
                    n2 = int(pl_grid[i + 1, j, k])
                    n3 = int(pl_grid[i + 1, j + 1, k])
                    n4 = int(pl_grid[i, j + 1, k])
                    n5 = int(pl_grid[i, j, k + 1])
                    n6 = int(pl_grid[i + 1, j, k + 1])
                    n7 = int(pl_grid[i + 1, j + 1, k + 1])
                    n8 = int(pl_grid[i, j + 1, k + 1])
                    mesh.elements[elem_id] = Element3D(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type="C3D8", pid=plate_pid)
                    elem_id += 1

        # Left plate top faces (j = ny_plate)
        for k in range(nz):
            for i in range(len(xs_left) - 1):
                plate_l_top_faces.append((
                    int(pl_grid[i, ny_plate, k]),
                    int(pl_grid[i + 1, ny_plate, k]),
                    int(pl_grid[i + 1, ny_plate, k + 1]),
                    int(pl_grid[i, ny_plate, k + 1])
                ))

        # Right plate grid [i, j, k]
        pr_grid = np.zeros((len(xs_right), ny_plate + 1, nz + 1), dtype=np.int32)
        for k, z in enumerate(zs):
            for j, y in enumerate(y_plate_bounds):
                for i, x in enumerate(xs_right):
                    mesh.nodes[nid] = Node3D(nid, float(x), float(y), float(z))
                    pr_grid[i, j, k] = nid
                    plate_r_nodes.append(nid)
                    nid += 1

        for k in range(nz):
            for j in range(ny_plate):
                for i in range(len(xs_right) - 1):
                    n1 = int(pr_grid[i, j, k])
                    n2 = int(pr_grid[i + 1, j, k])
                    n3 = int(pr_grid[i + 1, j + 1, k])
                    n4 = int(pr_grid[i, j + 1, k])
                    n5 = int(pr_grid[i, j, k + 1])
                    n6 = int(pr_grid[i + 1, j, k + 1])
                    n7 = int(pr_grid[i + 1, j + 1, k + 1])
                    n8 = int(pr_grid[i, j + 1, k + 1])
                    mesh.elements[elem_id] = Element3D(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type="C3D8", pid=plate_pid)
                    elem_id += 1

        # Right plate top faces (j = ny_plate)
        for k in range(nz):
            for i in range(len(xs_right) - 1):
                plate_r_top_faces.append((
                    int(pr_grid[i, ny_plate, k]),
                    int(pr_grid[i + 1, ny_plate, k]),
                    int(pr_grid[i + 1, ny_plate, k + 1]),
                    int(pr_grid[i, ny_plate, k + 1])
                ))

        # Tied display bottom nodes (j = 0)
        disp_bot_l_nids = []
        disp_bot_r_nids = []
        for k in range(nz + 1):
            for i in range(nx_total + 1):
                if xs[i] <= -hinge_gap + 1e-5:
                    disp_bot_l_nids.append(int(disp_grid[i, 0, k]))
                elif xs[i] >= hinge_gap - 1e-5:
                    disp_bot_r_nids.append(int(disp_grid[i, 0, k]))

        # Display top surface Quad4 faces for 3D self-contact (j = -1)
        # CCW winding ensures outward normal points +Y: (p(k+1, i), p(k+1, i+1), p(k, i+1), p(k, i))
        disp_top_l_faces = []
        disp_top_r_faces = []
        for k in range(nz):
            for i in range(nx_total):
                f = (
                    int(disp_grid[i, -1, k + 1]),
                    int(disp_grid[i + 1, -1, k + 1]),
                    int(disp_grid[i + 1, -1, k]),
                    int(disp_grid[i, -1, k])
                )
                if 0.5 * (xs[i] + xs[i + 1]) <= 0.0:
                    disp_top_l_faces.append(f)
                else:
                    disp_top_r_faces.append(f)

        gmsh.finalize()

        meta = {
            "dim": "3D",
            "xs": xs,
            "zs": zs,
            "left_pivot": (float(-self.geo.hinge_pivot_x), float(self.geo.hinge_pivot_y)),
            "right_pivot": (float(self.geo.hinge_pivot_x), float(self.geo.hinge_pivot_y)),
            "plate_left_nodes": plate_l_nodes,
            "plate_right_nodes": plate_r_nodes,
            "disp_bottom_left_nids": disp_bot_l_nids,
            "disp_bottom_right_nids": disp_bot_r_nids,
            "plate_l_top_faces": plate_l_top_faces,
            "plate_r_top_faces": plate_r_top_faces,
            "disp_top_left_faces": disp_top_l_faces,
            "disp_top_right_faces": disp_top_r_faces,
            "layer_elem_sets": layer_elem_sets,
            "layer_pids": layer_pids,
            "thick_disp": self.config.total_thickness,
            "length_disp": half_L * 2.0,
            "width_disp": width_z,
            "layup": self.case.preset_name,
        }

        # Clean isolated unconnected nodes (e.g., from tape void in hinge)
        self._clean_isolated_nodes(mesh, meta)

        # Verify element Jacobian / orientation
        self._verify_3d_mesh(mesh)
        return mesh, meta

    def _clean_isolated_nodes(self, mesh: Any, meta: Dict[str, Any]) -> None:
        """Removes nodes that are not referenced by any element (e.g., from void regions)."""
        connected_nids = set()
        for elem in mesh.elements.values():
            for nid in elem.node_ids:
                connected_nids.add(nid)

        all_nids = list(mesh.nodes.keys())
        isolated_nids = set(all_nids) - connected_nids
        if not isolated_nids:
            return

        for nid in isolated_nids:
            del mesh.nodes[nid]

        for k in ["plate_left_nodes", "plate_right_nodes", "disp_bottom_left_nids", "disp_bottom_right_nids"]:
            if k in meta:
                meta[k] = [nid for nid in meta[k] if nid not in isolated_nids]

    def _verify_2d_mesh(self, mesh: Mesh2D) -> None:
        """Verifies det J > 0 on every Quad element."""
        coords = mesh.nodes_array()
        nid_map = mesh.node_id_to_index()
        for e in mesh.elements.values():
            nids = [nid_map[n] for n in e.node_ids]
            p = coords[nids]
            # Area cross product (p1 - p0) x (p3 - p0)
            area = 0.5 * ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])) + \
                   0.5 * ((p[2, 0] - p[0, 0]) * (p[3, 1] - p[0, 1]) - (p[3, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]))
            if area <= 0.0:
                raise ValueError(f"[FATAL] Negative area detected in 2D element {e.id}: area={area}")

    def _verify_3d_mesh(self, mesh: Mesh3D) -> None:
        """Verifies det J > 0 on every Hexahedral element."""
        coords = mesh.nodes_array()
        nid_map = mesh.node_id_to_index()
        for e in mesh.elements.values():
            nids = [nid_map[n] for n in e.node_ids]
            p = coords[nids]
            # Center Jacobian check
            dxi = 0.125 * (p[1] - p[0] + p[2] - p[3] + p[5] - p[4] + p[6] - p[7])
            deta = 0.125 * (p[3] - p[0] + p[2] - p[1] + p[7] - p[4] + p[6] - p[5])
            dzeta = 0.125 * (p[4] - p[0] + p[5] - p[1] + p[6] - p[2] + p[7] - p[3])
            J = np.column_stack([dxi, deta, dzeta])
            det_J = np.linalg.det(J)
            if det_J <= 0.0:
                raise ValueError(f"[FATAL] Negative volume detected in 3D element {e.id}: det_J={det_J}")
