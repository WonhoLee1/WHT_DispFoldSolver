import sys
sys.path.insert(0, "examples")
import numpy as np
from dispsolver.solver import DynamicSolver
from dispsolver.element import q4_visco_eas_numba as M

# Monkeypatch the batch assembler: same math, but a PLAIN python loop over
# compute_single_eas_numba (still njit-compiled per call) instead of
# numba.prange -- lets any exception surface with a normal traceback and
# tells us exactly which element index/pid triggers it, instead of numba's
# prange machinery mangling it into an opaque SystemError.
_orig = M.assemble_eas_visco_batch_numba
def _diag_assemble(base_code, elem_coords, u_elems, alpha_elems, state_elems,
                    kappa, bparams, g_i, tau_i, g_inf, dt, thicknesses, F_n):
    n = elem_coords.shape[0]
    for e in range(n):
        try:
            M.compute_single_eas_numba(
                base_code, elem_coords[e], u_elems[e], alpha_elems[e],
                state_elems[e], kappa, bparams, g_i, tau_i, g_inf, dt,
                thicknesses[e], F_n[e])
        except Exception as exc:
            print(f"\n!!! CRASH at element index {e} !!!")
            print("coords:\n", elem_coords[e])
            print("u_elem:", u_elems[e])
            print("alpha:", alpha_elems[e])
            print("F_n:\n", F_n[e])
            print("thickness:", thicknesses[e])
            print("exception:", type(exc).__name__, exc)
            raise
    return _orig(base_code, elem_coords, u_elems, alpha_elems, state_elems,
                kappa, bparams, g_i, tau_i, g_inf, dt, thicknesses, F_n)
M.assemble_eas_visco_batch_numba = _diag_assemble

DynamicSolver.enable_new_numba_elements = True
from ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding
run_abaqus_inp_folding(elem_jit="numba", max_steps=8)
