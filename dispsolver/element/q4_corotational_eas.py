"""
q4_corotational_eas.py
======================
Co-rotational Enhanced Assumed Strain (CR-EAS) Q4 Plane Strain Element in NumPy.

Combines:
1. Co-rotational frame extraction (R_elem) to isolate element rigid rotation
   from deformational displacement -> eliminates det(F) <= 0 element inversion
   under 90-degree large global rotation.
2. Simo & Rifai (1990) Enhanced Assumed Strain (EAS-4) in the local frame ->
   eliminates artificial shear locking under high aspect ratio (AR) elements,
   guaranteeing element-size independent bending stiffness.

References
----------
Simo, J.C. & Rifai, M.S. (1990). A class of mixed assumed strain methods.
Crisfield, M.A. (1997). Non-linear Finite Element Analysis of Solids and Structures, Vol. 2.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Optional

from .q4_eas import compute_eas_j2_contributions


def compute_element_rotation(coords_init: np.ndarray, coords_curr: np.ndarray) -> np.ndarray:
    """Compute 2x2 element rotation matrix R using element edge vectors.

    Parameters
    ----------
    coords_init : (4, 2) initial reference coordinates
    coords_curr : (4, 2) current deformed coordinates

    Returns
    -------
    R_elem : (2, 2) rotation matrix mapping reference edge frame to current
    """
    # Edge vectors in deformed state
    v12 = coords_curr[1] - coords_curr[0]
    v43 = coords_curr[2] - coords_curr[3]
    e1_def = v12 + v43
    len1 = np.linalg.norm(e1_def) + 1e-15
    e1 = e1_def / len1
    e2 = np.array([-e1[1], e1[0]], dtype=np.float64)
    R_curr = np.column_stack([e1, e2])

    # Edge vectors in reference state
    v12_0 = coords_init[1] - coords_init[0]
    v43_0 = coords_init[2] - coords_init[3]
    e1_0_def = v12_0 + v43_0
    len1_0 = np.linalg.norm(e1_0_def) + 1e-15
    e1_0 = e1_0_def / len1_0
    e2_0 = np.array([-e1_0[1], e1_0[0]], dtype=np.float64)
    R_ref = np.column_stack([e1_0, e2_0])

    # Relative rotation
    return R_curr @ R_ref.T


def build_block_rotation(R_elem: np.ndarray) -> np.ndarray:
    """Build 8x8 block diagonal rotation matrix for 4-node element DOFs."""
    T8 = np.zeros((8, 8), dtype=np.float64)
    T8[0:2, 0:2] = R_elem
    T8[2:4, 2:4] = R_elem
    T8[4:6, 4:6] = R_elem
    T8[6:8, 6:8] = R_elem
    return T8


def compute_corotational_eas_j2_contributions(
    coords: np.ndarray,
    u_elem: np.ndarray,
    alpha: np.ndarray,
    state_elem: Optional[np.ndarray],
    mat_obj: Any,
    mat_params: Dict[str, Any],
    thickness: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CR-EAS internal force and consistent tangent stiffness in NumPy.

    Parameters
    ----------
    coords     : (4, 2) reference node coordinates
    u_elem     : (8,) global element displacement vector
    alpha      : (4,) warm-start EAS internal parameters
    state_elem : (4, 5) per-GP material state or None
    mat_obj    : material model instance (e.g. J2Plasticity)
    mat_params : material parameters dict
    thickness  : element thickness

    Returns
    -------
    f_global   : (8,) global internal force vector
    K_global   : (8, 8) global tangent stiffness matrix
    alpha_new  : (4,) updated EAS internal parameters
    state_new  : (4, 5) updated per-GP material state
    """
    coords_curr = coords + u_elem.reshape((4, 2))
    R_elem = compute_element_rotation(coords, coords_curr)
    T8 = build_block_rotation(R_elem)

    # Local deformational displacement (rigid body rotation subtracted)
    u_local = (coords_curr @ R_elem - coords).flatten()

    # Evaluate EAS in local co-rotational frame
    f_local, K_local, alpha_new, state_new = compute_eas_j2_contributions(
        coords=coords,
        u_elem=u_local,
        alpha=alpha,
        state_elem=state_elem,
        material=mat_obj,
        params=mat_params,
        thickness=thickness,
    )

    # Transform local force and tangent stiffness to global frame
    f_global = T8 @ f_local
    K_global = T8 @ K_local @ T8.T

    return f_global, K_global, alpha_new, state_new
