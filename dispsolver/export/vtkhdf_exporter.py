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

class TransientVTKHDFExporter:
    """
    Exports transient (time-series) data to a single VTKHDF file.
    Supports fixed static geometry and changing Point/Cell data over time.
    """
    def __init__(self, filepath: str, mesh: Mesh):
        self.filepath = filepath
        self.mesh = mesh
        # NOTE: the HDF5 file is NOT opened here — opening it "w" at construction
        # holds an exclusive lock for the ENTIRE simulation, so a crashed/lingering
        # run keeps the path locked and the next run fails to create it
        # (Win32 sharing-violation). Geometry is computed in memory now and every
        # step is buffered; the file is created and written only in close()
        # (i.e. at post-processing time), minimizing the lock window.

        # Geometry (Static) — kept in memory
        points = mesh.nodes_array().copy()
        self.n_nodes = points.shape[0]
        self._points_3d = np.zeros((self.n_nodes, 3), dtype=np.float32)
        self._points_3d[:, :2] = points

        # Connectivity
        nid_to_idx = mesh.node_id_to_index()
        connectivity_list = []
        offsets_list = [0]
        types_list = []

        for elem in mesh.elements.values():
            elem_type = elem.elem_type
            if elem_type == "QUAD4" or elem_type == "QUAD" or elem_type == "QUAD4_UP" or elem_type == "Q4_UP" or elem_type == "Q4":
                vtk_type = VTK_QUAD
            elif elem_type == "TRIA3" or elem_type == "TRIA":
                vtk_type = VTK_TRIANGLE
            else:
                if elem.n_nodes == 4:
                    vtk_type = VTK_QUAD
                elif elem.n_nodes == 3:
                    vtk_type = VTK_TRIANGLE
                else:
                    raise ValueError(f"Unsupported element type: {elem_type}")
            types_list.append(vtk_type)
            for nid in elem.node_ids:
                connectivity_list.append(nid_to_idx[nid])
            offsets_list.append(offsets_list[-1] + elem.n_nodes)

        self._connectivity = np.array(connectivity_list, dtype=np.int32)
        self._offsets = np.array(offsets_list, dtype=np.int32)
        self._types = np.array(types_list, dtype=np.uint8)
        self.n_cells = len(types_list)
        self.connectivity_size = self._connectivity.size

        # In-memory step buffers (written out only in close())
        self.time_values = []
        self.step_count = 0
        self._u_buf = []                      # list of (n_nodes, 3) float32
        self._point_state_buf: Dict[str, list] = {}
        self._cell_state_buf: Dict[str, list] = {}

    def add_step(self, time_val: float, u: np.ndarray, element_state: Optional[Dict[str, np.ndarray]] = None, point_state: Optional[Dict[str, np.ndarray]] = None):
        # Buffer in memory only — no file I/O during the simulation.
        self.time_values.append(time_val)

        u_2d = u.reshape((self.n_nodes, 2))
        u_3d = np.zeros((self.n_nodes, 3), dtype=np.float32)
        u_3d[:, :2] = u_2d
        self._u_buf.append(u_3d)

        if point_state is not None:
            for k, v in point_state.items():
                self._point_state_buf.setdefault(k, []).append(v.astype(np.float32))

        if element_state is not None:
            for k, v in element_state.items():
                self._cell_state_buf.setdefault(k, []).append(v.astype(np.float32))

        self.step_count += 1

    def close(self):
        # Open the file ONLY now (post-processing): minimal lock window.
        nsteps = self.step_count
        with h5.File(self.filepath, "w") as f:
            g = f.create_group("VTKHDF")
            g.attrs["Version"] = np.array([2, 2], dtype=np.int32)
            ascii_type = b"UnstructuredGrid"
            g.attrs.create("Type", ascii_type, dtype=h5.string_dtype("ascii", len(ascii_type)))

            # Static geometry
            g.create_dataset("Points", data=self._points_3d)
            g.create_dataset("Connectivity", data=self._connectivity)
            g.create_dataset("Offsets", data=self._offsets)
            g.create_dataset("Types", data=self._types)

            # Per-step field data (concatenated across steps)
            pd = g.create_group("PointData")
            cd = g.create_group("CellData")
            if self._u_buf:
                pd.create_dataset("Displacement", data=np.concatenate(self._u_buf, axis=0))
            for k, lst in self._point_state_buf.items():
                pd.create_dataset(k, data=np.concatenate(lst, axis=0))
            for k, lst in self._cell_state_buf.items():
                cd.create_dataset(k, data=np.concatenate(lst, axis=0))

            # Topology/Geometry metadata sizes (length NSteps)
            g.create_dataset("NumberOfPoints", data=np.full(nsteps, self.n_nodes, dtype=np.int32))
            g.create_dataset("NumberOfCells", data=np.full(nsteps, self.n_cells, dtype=np.int32))
            g.create_dataset("NumberOfConnectivityIds", data=np.full(nsteps, self.connectivity_size, dtype=np.int32))

            steps = g.create_group("Steps")
            steps.create_dataset("Values", data=np.array(self.time_values, dtype=np.float32))
            steps.attrs["NSteps"] = np.array(nsteps, dtype=np.int32)
            steps.create_dataset("PartOffsets", data=np.zeros(nsteps, dtype=np.int32))
            steps.create_dataset("PointOffsets", data=np.zeros(nsteps, dtype=np.int32))
            steps.create_dataset("CellOffsets", data=np.zeros(nsteps, dtype=np.int32))
            steps.create_dataset("ConnectivityIdOffsets", data=np.zeros(nsteps, dtype=np.int32))

            p_offsets = np.arange(nsteps, dtype=np.int32) * self.n_nodes
            c_offsets = np.arange(nsteps, dtype=np.int32) * self.n_cells
            pd_offsets = steps.create_group("PointDataOffsets")
            pd_offsets.create_dataset("Displacement", data=p_offsets)
            for k in self._point_state_buf.keys():
                pd_offsets.create_dataset(k, data=p_offsets)
            cd_offsets = steps.create_group("CellDataOffsets")
            for k in self._cell_state_buf.keys():
                cd_offsets.create_dataset(k, data=c_offsets)
