"""
vtu_exporter.py
===============
Export mesh and state fields to VTU format using meshio.

Supports outputting:
  - Displacement (point data)
  - Element State (cell data):
    - Cauchy Stress
    - Von Mises Stress
    - Equivalent Plastic Strain (eqps)
    - Temperature
    - Prony Series Norm (h_i norm)
"""

from typing import Dict, Optional, Any
import numpy as np
import meshio
from dispsolver.mesh import Mesh

def export_vtu(
    filepath: str,
    mesh: Mesh,
    u: np.ndarray,
    element_state: Optional[Dict[str, np.ndarray]] = None,
    point_state: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Exports the current state to a VTU file.
    
    Parameters
    ----------
    filepath : str
        Output file path (e.g. 'output_step_1.vtu')
    mesh : Mesh
        The computational mesh.
    u : np.ndarray
        Displacement vector of shape (2*n_nodes,).
    element_state : Dict[str, np.ndarray], optional
        Dictionary of cell data (shape: (n_elem,) or (n_elem, N)).
    point_state : Dict[str, np.ndarray], optional
        Dictionary of point data (shape: (n_nodes,) or (n_nodes, N)).
    """
    points = mesh.nodes_array().copy()
    
    n_nodes = points.shape[0]
    u_2d = u.reshape((n_nodes, 2))
    u_3d = np.zeros((n_nodes, 3))
    u_3d[:, :2] = u_2d
    
    point_data = {"Displacement": u_3d}
    if point_state is not None:
        for k, v in point_state.items():
            point_data[k] = v
            
    conn, _, _, _ = mesh.connectivity_array()
    cells = [("quad", conn)]
    
    cell_data = {}
    if element_state is not None:
        for k, v in element_state.items():
            cell_data[k] = [v]  # meshio expects list of arrays matching cells list
            
    meshio.write_points_cells(
        filepath,
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data
    )
