"""
element3d package
"""

from .base3d import SolidElement3D, QuadraturePointState3D
from .c3d8_eas_jax import Hexa8EASElement
from .c3d8_fbar_jax import Hexa8FbarElement
from .c3d8r_jax import Hexa8ReducedElement
from .c3d4_jax import Tetra4Element
from .c3d4_anp_jax import Tetra4ANPElement
from .c3d10_jax import Tetra10Element
from .c3d10m_jax import Tetra10ModifiedElement
from .c3d6_jax import Wedge6Element

from .c3d8_numba import compute_c3d8_element_numba, assemble_mesh_c3d8_numba
from .c3d8r_numba import compute_c3d8r_element_numba, assemble_mesh_c3d8r_numba
from .c3d8_eas_numba import compute_c3d8_eas_element_numba, assemble_mesh_c3d8_eas_numba
from .c3d8_fbar_numba import compute_c3d8_fbar_element_numba, assemble_mesh_c3d8_fbar_numba
from .c3d4_numba import (
    compute_c3d4_element_numba,
    assemble_mesh_c3d4_numba,
)
from .c3d4_anp_numba import (
    compute_c3d4_anp_element_matrices,
    assemble_mesh_c3d4_anp_numba
)
from .c3d10_numba import (
    compute_c3d10_element_numba,
    assemble_mesh_c3d10_numba
)
from .c3d10m_numba import (
    compute_c3d10m_element_numba,
    assemble_mesh_c3d10m_numba,
    compute_c3d10m_face_forces
)
from .c3d6_numba import (
    compute_c3d6_element_numba,
    assemble_mesh_c3d6_numba
)

from .c3d8_corotational_numba import (
    compute_c3d8_corotational_element_umat_numba,
    assemble_mesh_c3d8_corotational_numba,
    compute_element_rotation_3d
)
from .c3d8_hybrid_numba import (
    compute_c3d8_hybrid_element_umat_numba,
    assemble_mesh_c3d8_hybrid_numba
)

__all__ = [
    "SolidElement3D",
    "QuadraturePointState3D",
    "Hexa8EASElement",
    "Hexa8FbarElement",
    "Hexa8ReducedElement",
    "Tetra4Element",
    "Tetra4ANPElement",
    "Tetra10Element",
    "Tetra10ModifiedElement",
    "Wedge6Element",
    "compute_c3d8_element_numba",
    "assemble_mesh_c3d8_numba",
    "compute_c3d8r_element_numba",
    "assemble_mesh_c3d8r_numba",
    "compute_c3d8_eas_element_numba",
    "assemble_mesh_c3d8_eas_numba",
    "compute_c3d8_fbar_element_numba",
    "assemble_mesh_c3d8_fbar_numba",
    "compute_c3d4_element_numba",
    "assemble_mesh_c3d4_numba",
    "compute_c3d4_anp_element_matrices",
    "assemble_mesh_c3d4_anp_numba",
    "compute_c3d10_element_numba",
    "assemble_mesh_c3d10_numba",
    "compute_c3d10m_element_numba",
    "assemble_mesh_c3d10m_numba",
    "compute_c3d10m_face_forces",
    "compute_c3d6_element_numba",
    "assemble_mesh_c3d6_numba",
    "compute_c3d8_corotational_element_umat_numba",
    "assemble_mesh_c3d8_corotational_numba",
    "compute_element_rotation_3d",
    "compute_c3d8_hybrid_element_umat_numba",
    "assemble_mesh_c3d8_hybrid_numba",
]
