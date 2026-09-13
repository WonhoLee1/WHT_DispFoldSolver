import numpy as np
from dispsolver.element import q4_visco_eas_numba as M

d = np.load("scratch/crash_state.npz")
args = (int(d["base_code"]), d["coords"], d["u_elem"], d["alpha0"], d["state_elem"],
        float(d["kappa"]), d["bparams"], d["g_i"], d["tau_i"], float(d["g_inf"]),
        float(d["dt"]), float(d["thickness"]), d["F_n"])

f_e, K_e, a_conv, state_new, F_n_new = M.compute_single_eas_numba(*args)
print("f_e finite:", np.all(np.isfinite(f_e)), "max|f_e|=", np.max(np.abs(f_e)))
print("K_e finite:", np.all(np.isfinite(K_e)), "max|K_e|=", np.max(np.abs(K_e)))
print("NO CRASH -- fix holds on the exact captured failing state")
