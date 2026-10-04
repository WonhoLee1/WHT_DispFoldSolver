"""
dispsolver/loads
================
Catalog of loads and consistent pure-NumPy integration routines.
"""

from dispsolver.loads.integration import (
    body_force_weights,
    face_pressure_forces,
    face_area_normal,
)
from dispsolver.loads.catalog import (
    LoadTypeSpec,
    LOAD_TYPES,
    load_type,
)

__all__ = [
    "body_force_weights",
    "face_pressure_forces",
    "face_area_normal",
    "LoadTypeSpec",
    "LOAD_TYPES",
    "load_type",
]
