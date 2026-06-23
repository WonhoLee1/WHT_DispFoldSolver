"""Contact package: surface detection and penalty NTS contact."""

from .contact_surface import ContactSurface, auto_detect_exterior
from .contact_solver import ContactPair, Segment, SpatialHashGrid

__all__ = [
    "ContactSurface",
    "ContactPair",
    "Segment",
    "SpatialHashGrid",
    "auto_detect_exterior",
]
