"""
vtkhdf_exporter.py
==================
Export mesh and state fields to VTKHDF format using h5py.

Supports outputting:
  - Displacement (point data)
  - Element State (cell data):
    - Cauchy Stress
    - Von Mises Stress
    - Equivalent Plastic Strain (eqps)
    - Temperature
    - Prony Series Norm (h_i norm)
"""

from typing import Dict, Optional
import numpy as np
import h5py as h5
from dispsolver.mesh import Mesh

# VTK Cell Types
VTK_TRIANGLE = 5
VTK_QUAD = 9

def export_vtkhdf(
    filepath: str,
    mesh: Mesh,
    u: np.ndarray,
    element_state: Optional[Dict[str, np.ndarray]] = None,
    point_state: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Exports the current state to a VTKHDF file.
    
    Parameters
    ----------
    filepath : str
        Output file path (e.g. 'output_step_1.vtkhdf')
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
    
    # Expand 2D points to 3D for VTK
    points_3d = np.zeros((n_nodes, 3), dtype=np.float32)
    points_3d[:, :2] = points
    
    # Expand 2D displacements to 3D for VTK
    u_2d = u.reshape((n_nodes, 2))
    u_3d = np.zeros((n_nodes, 3), dtype=np.float32)
    u_3d[:, :2] = u_2d
    
    # Build Connectivity, Offsets, Types
    nid_to_idx = mesh.node_id_to_index()
    
    connectivity_list = []
    offsets_list = [0]
    types_list = []
    
    # Traverse elements in the order they are stored/assembled
    for elem in mesh.elements.values():
        elem_type = elem.elem_type
        if elem_type == "QUAD4" or elem_type == "QUAD":
            vtk_type = VTK_QUAD
        elif elem_type == "TRIA3" or elem_type == "TRIA":
            vtk_type = VTK_TRIANGLE
        else:
            # Fallback based on number of nodes
            if elem.n_nodes == 4:
                vtk_type = VTK_QUAD
            elif elem.n_nodes == 3:
                vtk_type = VTK_TRIANGLE
            else:
                raise ValueError(f"Unsupported element type: {elem_type}")
                
        types_list.append(vtk_type)
        
        # Local node indices
        for nid in elem.node_ids:
            connectivity_list.append(nid_to_idx[nid])
            
        offsets_list.append(offsets_list[-1] + elem.n_nodes)
        
    connectivity = np.array(connectivity_list, dtype=np.int32)
    offsets = np.array(offsets_list, dtype=np.int32)
    types = np.array(types_list, dtype=np.uint8)
    
    n_cells = len(types_list)
    
    # Write to HDF5
    with h5.File(filepath, "w") as f:
        g = f.create_group("VTKHDF")
        
        # Metadata
        g.attrs["Version"] = np.array([2, 2], dtype=np.int32)
        ascii_type = b"UnstructuredGrid"
        g.attrs.create("Type", ascii_type, dtype=h5.string_dtype("ascii", len(ascii_type)))
        
        # Geometry
        g.create_dataset("Points", data=points_3d)
        g.create_dataset("NumberOfPoints", data=np.array([n_nodes], dtype=np.int32))
        g.create_dataset("NumberOfCells", data=np.array([n_cells], dtype=np.int32))
        g.create_dataset("NumberOfConnectivityIds", data=np.array([connectivity.size], dtype=np.int32))
        g.create_dataset("Connectivity", data=connectivity)
        g.create_dataset("Offsets", data=offsets)
        g.create_dataset("Types", data=types)
        
        # PointData
        pd = g.create_group("PointData")
        pd.create_dataset("Displacement", data=u_3d)
        if point_state is not None:
            for k, v in point_state.items():
                pd.create_dataset(k, data=v.astype(np.float32))
                
        # CellData
        cd = g.create_group("CellData")
        if element_state is not None:
            for k, v in element_state.items():
                cd.create_dataset(k, data=v.astype(np.float32))
