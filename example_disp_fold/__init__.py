"""
example_disp_fold package
=========================
Unified 2D and 3D Multi-Layer Continuum Display Folding Simulation Framework.
"""

from example_disp_fold.model_spec import (
    LayerSpec,
    MaterialSpec,
    HingeGeometrySpec,
    MeshSpec,
    SimulationCaseSpec,
    UnifiedFoldModelConfig,
    DisplayStackConfig,
    FoldingCaseSpec,
)
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.abaqus_api_builder import AbaqusApiBuilder, AbaqusModelBuilder
from example_disp_fold.vtkhdf_unified_io import (
    UnifiedVTKHDFDatabase,
    UnifiedVTKHDFWriter,
    UnifiedVTKHDFReader,
    load_vtkhdf_summary,
)

__all__ = [
    "LayerSpec",
    "MaterialSpec",
    "HingeGeometrySpec",
    "MeshSpec",
    "SimulationCaseSpec",
    "UnifiedFoldModelConfig",
    "DisplayStackConfig",
    "FoldingCaseSpec",
    "GmshMeshGenerator",
    "AbaqusApiBuilder",
    "AbaqusModelBuilder",
    "UnifiedVTKHDFDatabase",
    "UnifiedVTKHDFWriter",
    "UnifiedVTKHDFReader",
    "load_vtkhdf_summary",
]
