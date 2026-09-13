import sys
sys.path.insert(0, "examples")
import numpy as np
from dispsolver.solver import DynamicSolver
from dispsolver.element import q4_visco_eas_numba as M

captured = {}
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
            captured['args'] = (base_code, elem_coords[e].copy(), u_elems[e].copy(),
                                 alpha_elems[e].copy(), state_elems[e].copy(),
                                 kappa, bparams.copy(), g_i.copy(), tau_i.copy(),
                                 g_inf, dt, thicknesses[e], F_n[e].copy())
            np.savez("scratch/crash_state.npz",
                     base_code=base_code, coords=elem_coords[e], u_elem=u_elems[e],
                     alpha0=alpha_elems[e], state_elem=state_elems[e], kappa=kappa,
                     bparams=bparams, g_i=g_i, tau_i=tau_i, g_inf=g_inf, dt=dt,
                     thickness=thicknesses[e], F_n=F_n[e])
            print("SAVED crash_state.npz, element", e)
            # Now call the UNDECORATED python function directly for a real
            # Python traceback with exact line numbers -- njit's compiled
            # frame gives none.
            py_fn = M.compute_single_eas_numba.py_func
            try:
                py_fn(*captured['args'])
            except Exception:
                import traceback
                traceback.print_exc()
            raise SystemExit(0)
    return _orig(base_code, elem_coords, u_elems, alpha_elems, state_elems,
                kappa, bparams, g_i, tau_i, g_inf, dt, thicknesses, F_n)
M.assemble_eas_visco_batch_numba = _diag_assemble

DynamicSolver.enable_new_numba_elements = True
from ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding
run_abaqus_inp_folding(elem_jit="numba", max_steps=8)
