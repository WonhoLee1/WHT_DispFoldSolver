"""
surface_tie_unified.py
======================
Unified SurfaceTie factory function providing Abaqus Scripting Interface parity.

In Abaqus/CAE Python API, surface tie constraints are defined via a single unified
method `model.Tie(name=..., master=..., slave=...)` without distinguishing between
2D and 3D.

This module provides `SurfaceTie()`, which automatically inspects spatial dimensionality
(from the supplied mesh or coordinate array) and dispatches to either:
  - 2D: `SurfaceTieConstraint` (from `dispsolver.constraint.surface_tie`)
  - 3D: `SurfaceTieConstraint3D` (from `dispsolver.constraint3d.surface_tie3d`)

Backward compatibility:
Direct imports of `SurfaceTieConstraint` and `SurfaceTieConstraint3D` remain fully
supported.
"""

from __future__ import annotations
from typing import Union, List, Tuple, Dict, Optional, Any
import numpy as np

from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D


def SurfaceTie(
    secondary: Optional[Union[List[int], Any]] = None,
    main: Optional[Union[List[int], List[Tuple[int, ...]], Any]] = None,
    *,
    slave: Optional[Union[List[int], Any]] = None,
    master: Optional[Union[List[int], List[Tuple[int, ...]], Any]] = None,
    mesh: Optional[Any] = None,
    coords: Optional[np.ndarray] = None,
    nid_to_idx: Optional[Dict[int, int]] = None,
    penalty_stiffness: float = 1e6,
    position_tolerance: Optional[float] = None,
    name: str = "SURFACE_TIE",
) -> Union[SurfaceTieConstraint, SurfaceTieConstraint3D]:
    """Unified Surface Tie Constraint Factory (Abaqus API Parity).

    Automatically inspects spatial dimension (2D vs 3D) and returns the appropriate
    underlying constraint instance.

    Uses modern CAE standard terminology: 'main' (formerly 'master') and
    'secondary' (formerly 'slave'), with 100% backward compatibility for legacy names.

    Parameters
    ----------
    secondary : list of int or Region, optional
        Node IDs of the secondary (slave) surface.
    main : list of int, list of tuple, or Region, optional
        For 2D: List of main (master) node IDs along segment line.
        For 3D: List of main face tuples [(m1, m2, m3, m4), ...], or list of
        main node IDs (if mesh is provided, surface faces are automatically extracted).
    slave : list of int or Region, optional
        Legacy alias for `secondary`.
    master : list of int, list of tuple, or Region, optional
        Legacy alias for `main`.
    mesh : Mesh2D or Mesh3D, optional
        Mesh object from which coords and nid_to_idx can be automatically extracted.
    coords : ndarray of shape (N, 2) or (N, 3), optional
        Nodal coordinate array. Required if mesh is not provided.
    nid_to_idx : dict, optional
        Mapping from node ID to solver index. Auto-extracted if mesh is provided.
    penalty_stiffness : float
        Penalty stiffness parameter k_tie [N/mm]. Default 1e6.
    position_tolerance : float, optional
        Search radius / tolerance. Default 0.5 mm for 2D, 2.0 mm for 3D.
    name : str
        Constraint identifier.

    Returns
    -------
    SurfaceTieConstraint or SurfaceTieConstraint3D
        Instantiated constraint object matching the spatial dimension.
    """
    # Resolve main / secondary aliases
    if secondary is None:
        if slave is not None:
            secondary = slave
        else:
            raise ValueError("SurfaceTie requires 'secondary' (or legacy alias 'slave').")
    if main is None:
        if master is not None:
            main = master
        else:
            raise ValueError("SurfaceTie requires 'main' (or legacy alias 'master').")
    # 1. Resolve mesh / coords / nid_to_idx
    dim = None
    if mesh is not None:
        if coords is None:
            if hasattr(mesh, "nodes_array"):
                coords = mesh.nodes_array()
            elif hasattr(mesh, "nodes"):
                # fallback manual array extraction
                nids = sorted(mesh.nodes.keys())
                sample_n = mesh.nodes[nids[0]]
                if hasattr(sample_n, "z"):
                    coords = np.array([[mesh.nodes[n].x, mesh.nodes[n].y, mesh.nodes[n].z] for n in nids])
                else:
                    coords = np.array([[mesh.nodes[n].x, mesh.nodes[n].y] for n in nids])
        if nid_to_idx is None and hasattr(mesh, "node_id_to_index"):
            nid_to_idx = mesh.node_id_to_index()

        # Check dimension from mesh class
        cls_name = mesh.__class__.__name__
        if "3D" in cls_name or "3d" in cls_name:
            dim = 3
        elif "2D" in cls_name or "2d" in cls_name:
            dim = 2

    if coords is not None:
        coords = np.asarray(coords)
        if dim is None:
            dim = coords.shape[1]
    else:
        raise ValueError("Either 'mesh' or 'coords' must be provided to SurfaceTie.")

    if nid_to_idx is None:
        raise ValueError("Either 'mesh' or 'nid_to_idx' mapping must be provided to SurfaceTie.")

    # 2. Dispatch based on dimension
    if dim == 2:
        tol = position_tolerance if position_tolerance is not None else 0.5
        main_nodes = list(main)
        return SurfaceTieConstraint(
            slave_node_ids=list(secondary),
            master_node_ids=main_nodes,
            nid_to_idx=nid_to_idx,
            coords=coords,
            penalty_stiffness=penalty_stiffness,
            position_tolerance=tol,
            name=name,
        )
    elif dim == 3:
        tol = position_tolerance if position_tolerance is not None else 2.0
        # If main is a list of node IDs rather than face tuples, and mesh is provided,
        # extract exterior Quad faces containing those nodes.
        main_faces = []
        if len(main) > 0 and isinstance(main[0], (tuple, list)):
            main_faces = [tuple(f) for f in main]
        elif mesh is not None and hasattr(mesh, "elements"):
            main_node_set = set(main)
            for eid, elem in mesh.elements.items():
                conn = getattr(elem, "node_ids", getattr(elem, "connectivity", []))
                if len(conn) == 8:  # Hex8
                    hex_face_indices = [
                        (0, 1, 2, 3),
                        (4, 5, 6, 7),
                        (0, 1, 5, 4),
                        (1, 2, 6, 5),
                        (2, 3, 7, 6),
                        (0, 3, 7, 4),
                    ]
                    for f_idx in hex_face_indices:
                        fnodes = tuple(conn[k] for k in f_idx)
                        if all(n in main_node_set for n in fnodes):
                            main_faces.append(fnodes)
        else:
            raise ValueError(
                "For 3D SurfaceTie, 'main' must be a list of face tuples [(n1,n2,n3,n4), ...] "
                "or 'mesh' must be provided to auto-resolve main node IDs to faces."
            )

        return SurfaceTieConstraint3D(
            slave_node_ids=list(secondary),
            master_faces=main_faces,
            nid_to_idx=nid_to_idx,
            coords=coords,
            penalty_stiffness=penalty_stiffness,
            position_tolerance=tol,
            name=name,
        )
    else:
        raise ValueError(f"Unsupported spatial dimension: {dim}. Expected 2 or 3.")
