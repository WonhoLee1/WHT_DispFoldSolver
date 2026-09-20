# -*- coding: utf-8 -*-
"""anneal.py: Abaqus *ANNEAL TEMPERATURE hardening reset for J2 plasticity.

Abaqus Theory Guide 4.3.5: when nodal temperature T reaches the anneal
temperature, the material loses all hardening memory -- equivalent plastic
strain and backstress reset to zero (recrystallization wipes the dislocation
structure). Below T_anneal the state passes through bit-identically.
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def anneal_reset(alpha_old, ep_old, eq_p_old, T, T_anneal):
    """Reset J2 hardening state above the anneal temperature.

    Args:
        alpha_old: (6,) backstress, ep_old: (6,) plastic strain (Voigt,
            engineering shear), eq_p_old: scalar equivalent plastic strain.
        T: current temperature. T_anneal: anneal temperature (<=0 disables).
    Returns:
        alpha_new, ep_new, eq_p_new, reset_flag (1.0 if reset, else 0.0).
    """
    alpha_new = np.zeros(6, dtype=np.float64)
    ep_new = np.zeros(6, dtype=np.float64)
    if T_anneal > 0.0 and T >= T_anneal:
        return alpha_new, ep_new, 0.0, 1.0
    for i in range(6):
        alpha_new[i] = alpha_old[i]
        ep_new[i] = ep_old[i]
    return alpha_new, ep_new, eq_p_old, 0.0
