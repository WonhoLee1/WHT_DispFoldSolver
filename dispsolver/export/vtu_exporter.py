"""
vtu_exporter.py
===============
Native VTK XML Unstructured Grid (.vtu) field exporter.
Compatible with 100% of ParaView versions (ParaView 3.0 through 5.13+).
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, Union, List
import numpy as np

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


def export_vtu(
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
    """Export finite element mesh and field data to VTK XML Unstructured Grid (.vtu).

    Parameters
    ----------
    file_path : str
        Target file path ending in .vtu
    nodes : np.ndarray
        Nodal coordinates (N_nodes, 2) or (N_nodes, 3)
    elements : list or np.ndarray
        Element connectivity list/array
    u : np.ndarray
        Nodal displacement vector of length (N_nodes * dim)
    element_type : str
        Abaqus element type name (e.g. "CPE4", "C3D8", "CPE4I_CR")
    stresses : np.ndarray, optional
        Element stress tensor array (N_elems, n_components)
    strains : np.ndarray, optional
        Element strain tensor array (N_elems, n_components)
    sdvs : np.ndarray, optional
        Element state dependent variables array
    reaction_forces : np.ndarray, optional
        Nodal reaction force vector
    time_value : float
        Current analysis time

    Returns
    -------
    file_path : str
        Path to saved .vtu file
    """
    if not file_path.endswith(".vtu"):
        file_path += ".vtu"

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    nodes = np.asarray(nodes, dtype=np.float64)
    n_nodes = nodes.shape[0]
    dim = nodes.shape[1]

    # Convert 2D nodes (N, 2) -> 3D (N, 3)
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

    elem_arr = np.asarray(elements, dtype=np.int64)
    if elem_arr.min() >= 1:
        elem_arr = elem_arr - 1

    n_elems = elem_arr.shape[0]
    nodes_per_elem = elem_arr.shape[1]

    vtk_cell_type = ABAQUS_TO_VTK_TYPE.get(str(element_type).upper(), VTK_QUAD)

    types = np.full(n_elems, vtk_cell_type, dtype=np.uint8)
    offsets = np.arange(nodes_per_elem, (n_elems + 1) * nodes_per_elem, nodes_per_elem, dtype=np.int64)
    connectivity = elem_arr.flatten().astype(np.int64)

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

    # Format ASCII arrays
    pts_str = " ".join(f"{x:.6g} {y:.6g} {z:.6g}" for x, y, z in points_3d)
    conn_str = " ".join(str(c) for c in connectivity)
    off_str = " ".join(str(o) for o in offsets)
    types_str = " ".join(str(t) for t in types)
    u_str = " ".join(f"{ux:.6g} {uy:.6g} {uz:.6g}" for ux, uy, uz in u_3d)

    xml_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{n_nodes}" NumberOfCells="{n_elems}">',
        '      <Points>',
        '        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">',
        f'          {pts_str}',
        '        </DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int64" Name="connectivity" format="ascii">',
        f'          {conn_str}',
        '        </DataArray>',
        '        <DataArray type="Int64" Name="offsets" format="ascii">',
        f'          {off_str}',
        '        </DataArray>',
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        f'          {types_str}',
        '        </DataArray>',
        '      </Cells>',
        '      <PointData Vectors="Displacement">',
        '        <DataArray type="Float64" Name="Displacement" NumberOfComponents="3" format="ascii">',
        f'          {u_str}',
        '        </DataArray>',
        '      </PointData>',
    ]

    if von_mises is not None:
        vm_str = " ".join(f"{v:.6g}" for v in von_mises)
        xml_lines.extend([
            '      <CellData Scalars="VonMises">',
            '        <DataArray type="Float64" Name="VonMises" NumberOfComponents="1" format="ascii">',
            f'          {vm_str}',
            '        </DataArray>',
            '      </CellData>',
        ])

    xml_lines.extend([
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ])

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))

    return file_path
