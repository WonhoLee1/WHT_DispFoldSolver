"""
unified.py
==========
Unified Finite Element Dynamic Solver Factory (Abaqus API Parity).

Provides a single, transparent entry point `DynamicSolver` that automatically
inspects the provided mesh and dispatches to:
  1. `DynamicSolver3D` for 3D meshes (Mesh3D) - high-performance Numba OpenMP DOD engine
  2. `DynamicSolver2D` for modern 2D meshes (Mesh2D) - 10-element Numba OpenMP DOD engine
  3. `LegacyDynamicSolver2D` for legacy 2D meshes (dispsolver.mesh.Mesh) - JAX corotational engine

Preserves 100% backward compatibility with existing R&D and benchmark scripts.
"""

from __future__ import annotations
from typing import Any, Optional, Dict, Union
import numpy as np


class DynamicSolver:
    """Unified Multi-Dimensional Dynamic Finite Element Solver.

    Automatically routes to the appropriate high-performance solver engine
    based on the spatial dimension and data structure of the input mesh.
    """

    def __new__(cls, mesh: Any, *args, **kwargs):
        cls_name = mesh.__class__.__name__
        module_name = mesh.__class__.__module__

        # 1. 3D Mesh -> DynamicSolver3D
        if "3D" in cls_name or "3d" in cls_name or "mesh3d" in module_name:
            from dispsolver.solver3d.dynamic3d import DynamicSolver3D
            return DynamicSolver3D(mesh, *args, **kwargs)

        # 2. Modern 2D Mesh -> DynamicSolver2D
        elif "Mesh2D" in cls_name or "mesh2d" in module_name:
            from dispsolver.solver2d.dynamic2d import DynamicSolver2D
            return DynamicSolver2D(mesh, *args, **kwargs)

        # 3. Legacy 2D Mesh (dispsolver.mesh.mesh.Mesh) -> Legacy JAX DynamicSolver
        else:
            from dispsolver.solver.dynamic import DynamicSolver as _LegacyDynamicSolver
            return _LegacyDynamicSolver(mesh, *args, **kwargs)

    @classmethod
    def from_config(cls, *args, **kwargs):
        """Forward classmethod factory to legacy DynamicSolver for config-based scripts."""
        from dispsolver.solver.dynamic import DynamicSolver as _LegacyDynamicSolver
        return _LegacyDynamicSolver.from_config(*args, **kwargs)
