"""
vtkhdf_exporter.py
===================
VTKHDF (HDF5-based ParaView 9.1+ format) field data exporter.
Provides high-performance, compact binary file export for 2D plane strain
and 3D solid finite element analyses.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, Union, List
import numpy as np
import h5py

# VTK Cell Type Constants
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_QUADRATIC_TRIANGLE = 22
VTK_QUADRATIC_QUAD = 23
VTK_QUADRATIC_TETRA = 24

ABAQUS_TO_VTK_TYPE = {
    # 2D Continuum
    "CPE4": VTK_QUAD, "CPS4": VTK_QUAD, "CPE4I": VTK_QUAD, "CPS4I": VTK_QUAD,
    "CPE4R": VTK_QUAD, "CPS4R": VTK_QUAD, "CPE4H": VTK_QUAD, "CPS4H": VTK_QUAD,
    "CPE4_CR": VTK_QUAD, "CPE4_FBAR": VTK_QUAD, "CPE4I_CR": VTK_QUAD,
    "CPE4H_CR": VTK_QUAD, "CPE4R_CR": VTK_QUAD, "Q4": VTK_QUAD,
    "CPE3": VTK_TRIANGLE, "CPS3": VTK_TRIANGLE, "T3": VTK_TRIANGLE, "CST": VTK_TRIANGLE,
    "CPE6": VTK_QUADRATIC_TRIANGLE, "CPS6": VTK_QUADRATIC_TRIANGLE,
    "CPE6M": VTK_QUADRATIC_TRIANGLE, "CPE6M_CR": VTK_QUADRATIC_TRIANGLE,
    "CPE8": VTK_QUADRATIC_QUAD, "CPS8": VTK_QUADRATIC_QUAD, "CPE8R": VTK_QUADRATIC_QUAD,
    # 3D Continuum
    "C3D8": VTK_HEXAHEDRON, "C3D8I": VTK_HEXAHEDRON, "C3D8H": VTK_HEXAHEDRON,
    "C3D8R": VTK_HEXAHEDRON, "C3D8_FBAR": VTK_HEXAHEDRON, "C3D8_CR": VTK_HEXAHEDRON,
    "C3D8I_CR": VTK_HEXAHEDRON, "C3D8H_CR": VTK_HEXAHEDRON, "C3D8R_CR": VTK_HEXAHEDRON,
    "C3D4": VTK_TETRA, "C3D4_ANP": VTK_TETRA,
    "C3D10": VTK_QUADRATIC_TETRA, "C3D10M": VTK_QUADRATIC_TETRA,
    "C3D6": VTK_WEDGE,
}


def export_vtkhdf(
    file_path: str,
    nodes: np.ndarray,
    elements: Union[List[List[int]], np.ndarray],
    u: np.ndarray,
    element_type: str = "CPE4",
    stresses: Optional[np.ndarray] = None,
    strains: Optional[np.ndarray] = None,
    sdvs: Optional[np.ndarray] = None,
    reaction_forces: Optional[np.ndarray] = None,
    time_value: float = 0.0,
) -> str:
    """Export finite element results to VTKHDF (.vtkhdf) binary format.

    Parameters
    ----------
    file_path : str
        Target file path ending in .vtkhdf
    nodes : np.ndarray
        Nodal coordinates (N_nodes, 2) or (N_nodes, 3)
    elements : list or np.ndarray
        Element node connectivity list/array (0-indexed or 1-indexed)
    u : np.ndarray
        Nodal displacement vector of length (N_nodes * dim)
    element_type : str
        Abaqus element type name (e.g. "CPE4", "C3D8", "CPE4I_CR")
    stresses : np.ndarray, optional
        Element stress tensor array (N_elems, n_components)
    strains : np.ndarray, optional
        Element strain tensor array (N_elems, n_components)
    sdvs : np.ndarray, optional
        Element state dependent variables array (N_elems, ...)
    reaction_forces : np.ndarray, optional
        Nodal reaction force vector of length (N_nodes * dim)
    time_value : float
        Current analysis time

    Returns
    -------
    file_path : str
        Path to the saved .vtkhdf file
    """
    if not file_path.endswith(".vtkhdf"):
        file_path += ".vtkhdf"

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    nodes = np.asarray(nodes, dtype=np.float64)
    n_nodes = nodes.shape[0]
    dim = nodes.shape[1]

    # Convert 2D nodes (N, 2) -> 3D (N, 3) by padding Z=0.0
    if dim == 2:
        points_3d = np.zeros((n_nodes, 3), dtype=np.float64)
        points_3d[:, :2] = nodes
    else:
        points_3d = nodes.copy()

    # Convert displacements (N*dim,) -> (N, 3)
    u = np.asarray(u, dtype=np.float64)
    u_3d = np.zeros((n_nodes, 3), dtype=np.float64)
    if dim == 2:
        u_3d[:, 0] = u[0::2]
        u_3d[:, 1] = u[1::2]
    else:
        u_3d[:, 0] = u[0::3]
        u_3d[:, 1] = u[1::3]
        u_3d[:, 2] = u[2::3]

    # Process elements connectivity
    elem_arr = np.asarray(elements, dtype=np.int64)
    # Ensure 0-indexed connectivity
    if elem_arr.min() >= 1:
        elem_arr = elem_arr - 1

    n_elems = elem_arr.shape[0]
    nodes_per_elem = elem_arr.shape[1]

    vtk_cell_type = ABAQUS_TO_VTK_TYPE.get(str(element_type).upper(), VTK_QUAD)

    types = np.full(n_elems, vtk_cell_type, dtype=np.uint8)
    offsets = np.arange(0, (n_elems + 1) * nodes_per_elem, nodes_per_elem, dtype=np.int64)
    connectivity = elem_arr.flatten().astype(np.int64)

    # Compute von Mises stress if stress tensor supplied
    von_mises = None
    if stresses is not None:
        stresses = np.asarray(stresses, dtype=np.float64)
        if stresses.ndim == 1:
            stresses = stresses.reshape(-1, 1)
        if stresses.ndim == 2 and stresses.shape[0] > 0:
            n_c = stresses.shape[1]
            if n_c == 1:
                von_mises = np.abs(stresses[:, 0])
            elif n_c == 2:
                s11, s22 = stresses[:, 0], stresses[:, 1]
                von_mises = np.sqrt(np.maximum(0.0, s11**2 - s11 * s22 + s22**2))
            elif n_c >= 3:
                s11, s22, s12 = stresses[:, 0], stresses[:, 1], stresses[:, 2]
                s33 = stresses[:, 3] if n_c >= 4 else np.zeros_like(s11)
                s23 = stresses[:, 4] if n_c >= 6 else np.zeros_like(s11)
                s31 = stresses[:, 5] if n_c >= 6 else np.zeros_like(s11)
                von_mises = np.sqrt(
                    np.maximum(0.0, 0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 6.0 * (s12**2 + s23**2 + s31**2)))
                )

    with h5py.File(file_path, "w") as f:
        vtkhdf_group = f.create_group("VTKHDF")
        vtkhdf_group.attrs["Version"] = np.array([1, 0], dtype=np.int32)
        vtkhdf_group.attrs["Type"] = "UnstructuredGrid"

        vtkhdf_group.create_dataset("Version", data=np.array([1, 0], dtype=np.int32))
        vtkhdf_group.create_dataset("Type", data=np.frombuffer(b"UnstructuredGrid", dtype=np.uint8))

        vtkhdf_group.create_dataset("NumberOfPoints", data=np.array([n_nodes], dtype=np.int64))
        vtkhdf_group.create_dataset("NumberOfCells", data=np.array([n_elems], dtype=np.int64))
        vtkhdf_group.create_dataset("NumberOfConnectivityIds", data=np.array([len(connectivity)], dtype=np.int64))

        vtkhdf_group.create_dataset("Points", data=points_3d, compression="gzip")
        vtkhdf_group.create_dataset("Types", data=types)
        vtkhdf_group.create_dataset("Offsets", data=offsets)
        vtkhdf_group.create_dataset("Connectivity", data=connectivity, compression="gzip")

        # Nodal Data Group
        point_data = vtkhdf_group.create_group("PointData")
        point_data.create_dataset("Displacement", data=u_3d, compression="gzip")

        if reaction_forces is not None:
            rf = np.asarray(reaction_forces, dtype=np.float64)
            rf_3d = np.zeros((n_nodes, 3), dtype=np.float64)
            if dim == 2:
                rf_3d[:, 0] = rf[0::2]
                rf_3d[:, 1] = rf[1::2]
            else:
                rf_3d[:, 0] = rf[0::3]
                rf_3d[:, 1] = rf[1::3]
                rf_3d[:, 2] = rf[2::3]
            point_data.create_dataset("ReactionForce", data=rf_3d, compression="gzip")

        # Cell Data Group
        cell_data = vtkhdf_group.create_group("CellData")
        if stresses is not None:
            cell_data.create_dataset("Stress", data=stresses, compression="gzip")
        if von_mises is not None:
            cell_data.create_dataset("VonMises", data=von_mises, compression="gzip")
        if strains is not None:
            cell_data.create_dataset("Strain", data=np.asarray(strains, dtype=np.float64), compression="gzip")

    return file_path
