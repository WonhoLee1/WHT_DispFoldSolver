"""
plate_builder.py
================
Automated geometry and mesh generator for rigid folding plates.

Generates 2D Q4 quad meshes for left and right folding plates with defined
dimensions, thickness, pivot reference points (RP), and top surface node sets
for surface-to-surface tie constraints.
"""

from typing import Dict, List, Tuple
import numpy as np
from .mesh import Mesh


def create_folding_plate_parts(
    left_x_range: Tuple[float, float] = (-40.0, -10.0),
    right_x_range: Tuple[float, float] = (10.0, 40.0),
    y_range: Tuple[float, float] = (-0.5, 0.0),
    left_pivot: Tuple[float, float] = (-3.0, 0.0),
    right_pivot: Tuple[float, float] = (3.0, 0.0),
    nx: int = 30,
    ny: int = 4,
    base_node_id: int = 100000,
    base_elem_id: int = 100000,
) -> Dict[str, Dict]:
    """Generate structured Q4 meshes and node sets for left and right folding plates.

    Parameters
    ----------
    left_x_range : (x_min, x_max)
        X bounds for left plate [mm].
    right_x_range : (x_min, x_max)
        X bounds for right plate [mm].
    y_range : (y_bottom, y_top)
        Y bounds for plates [mm].
    left_pivot : (x, y)
        Pivot center for left plate rotation [mm].
    right_pivot : (x, y)
        Pivot center for right plate rotation [mm].
    nx : int
        Number of element divisions in X direction per plate.
    ny : int
        Number of element divisions in Y direction.
    base_node_id : int
        Starting node ID offset to avoid collisions with display mesh.
    base_elem_id : int
        Starting element ID offset.

    Returns
    -------
    plate_data : dict
        Dictionary containing 'left' and 'right' plate details:
        - 'mesh': Mesh object
        - 'master_rp_id': Node ID of master RP
        - 'master_rp_coord': Coordinates of master RP
        - 'top_surface_nids': List of node IDs on top surface
        - 'slave_nids': List of all plate node IDs for rigid body definition
    """
    results = {}
    curr_nid = base_node_id
    curr_eid = base_elem_id

    plates_config = [
        ("left", left_x_range, left_pivot),
        ("right", right_x_range, right_pivot),
    ]

    for name, x_rng, pivot in plates_config:
        nodes_dict = {}
        elements_dict = {}

        # 1. Master Reference Point (RP)
        master_rp_id = curr_nid
        curr_nid += 1
        nodes_dict[master_rp_id] = np.array(pivot, dtype=np.float64)

        # 2. Grid nodes: (nx + 1) x (ny + 1)
        x_coords = np.linspace(x_rng[0], x_rng[1], nx + 1)
        y_coords = np.linspace(y_range[0], y_range[1], ny + 1)

        grid_nids = np.zeros((ny + 1, nx + 1), dtype=int)
        top_surface_nids = []
        slave_nids = []

        for j, y_val in enumerate(y_coords):
            for i, x_val in enumerate(x_coords):
                nid = curr_nid
                curr_nid += 1
                grid_nids[j, i] = nid
                nodes_dict[nid] = np.array([x_val, y_val], dtype=np.float64)
                slave_nids.append(nid)

                # Top surface nodes (y = y_top)
                if j == ny:
                    top_surface_nids.append(nid)

        # 3. Elements: nx x ny
        for j in range(ny):
            for i in range(nx):
                eid = curr_eid
                curr_eid += 1
                n1 = grid_nids[j, i]
                n2 = grid_nids[j, i + 1]
                n3 = grid_nids[j + 1, i + 1]
                n4 = grid_nids[j + 1, i]
                elements_dict[eid] = [n1, n2, n3, n4]

        # 4. Build Mesh object
        mesh = Mesh()
        for nid, coord in nodes_dict.items():
            mesh.add_node(nid, coord[0], coord[1])

        for eid, conn in elements_dict.items():
            mesh.add_element(eid, conn, "Q4")

        mesh.add_nodeset(f"{name.upper()}_RP", set([master_rp_id]))
        mesh.add_nodeset(f"{name.upper()}_SLAVES", set(slave_nids))
        mesh.add_nodeset(f"{name.upper()}_TOP", set(top_surface_nids))

        results[name] = {
            "mesh": mesh,
            "master_rp_id": master_rp_id,
            "master_rp_coord": np.array(pivot, dtype=np.float64),
            "top_surface_nids": top_surface_nids,
            "slave_nids": slave_nids,
            "nodes_dict": nodes_dict,
            "elements_dict": elements_dict,
        }

    return results
