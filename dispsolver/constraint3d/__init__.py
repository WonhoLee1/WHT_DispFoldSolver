"""
constraint3d package
====================
3D Multi-point constraints, surface tie, and rigid body ties.
"""

from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D
from dispsolver.constraint3d.rbe3_distributing3d import RBE3DistributingConstraint3D
from dispsolver.constraint3d.surface_contact3d import SurfaceContactConstraint3D
from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D
from dispsolver.constraint3d.surface_contact3d_s2s import SurfaceToSurfaceContactConstraint3D
from dispsolver.constraint3d.dual_mortar_clipping3d import DualMortarClippedContactConstraint3D
from dispsolver.constraint3d.nitsche_contact3d import NitscheContactConstraint3D
from dispsolver.constraint3d.finite_sliding_mortar3d import (
    FiniteSlidingDualMortar3D,
    assemble_finite_sliding_mortar_numba,
)

__all__ = [
    "SurfaceTieConstraint3D",
    "RBE3DistributingConstraint3D",
    "SurfaceContactConstraint3D",
    "DeformableSurfaceContactConstraint3D",
    "SurfaceToSurfaceContactConstraint3D",
    "DualMortarClippedContactConstraint3D",
    "NitscheContactConstraint3D",
    "FiniteSlidingDualMortar3D",
    "assemble_finite_sliding_mortar_numba",
]

