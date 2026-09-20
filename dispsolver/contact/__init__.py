"""Contact package: surface detection and penalty NTS contact."""

from .contact_surface import ContactSurface, ContactSurface3D, auto_detect_exterior, auto_detect_exterior_3d
from .contact_solver import ContactPair, Segment, SpatialHashGrid
from .bvh_tree3d import DynamicBVHTree3D, AABB3D, compute_morton_code_3d
from .flat_bvh_tree3d import (
    FlatBVHTree3D,
    bvh_refit_flat_numba,
    sutherland_hodgman_clip_quad4_numba,
    integrate_dual_mortar_polygon_numba,
)
from .general_contact3d import GeneralContact3D

__all__ = [
    "ContactSurface",
    "ContactSurface3D",
    "ContactPair",
    "Segment",
    "SpatialHashGrid",
    "auto_detect_exterior",
    "auto_detect_exterior_3d",
    "DynamicBVHTree3D",
    "AABB3D",
    "compute_morton_code_3d",
    "FlatBVHTree3D",
    "bvh_refit_flat_numba",
    "sutherland_hodgman_clip_quad4_numba",
    "integrate_dual_mortar_polygon_numba",
    "GeneralContact3D",
]

