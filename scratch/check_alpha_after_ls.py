"""Post-`_ALPHA_MAX`-removal re-run of scratch/check_alpha_saturation.py.

Same driver, same 40 steps, same numba backend as the pre-change baseline
`scratch/alpha_sat3.log` (clamp 0.5), so the two logs are directly comparable.
Reports, per accepted increment:
  - max|alpha| (no clamp any more -- the number is now unbounded by design)
  - the element-local enhanced-mode solve status the kernels now return
  - the running count of increments abandoned because that local solve failed
"""
import sys, os
import numpy as np

sys.path.insert(0, "examples")
from dispsolver.solver import DynamicSolver

_orig = DynamicSolver.solve_step


def probe(self, dt):
    r = _orig(self, dt)
    a = np.abs(self.eas_alpha)
    nz = a[a > 0]
    st = getattr(self, "_eas_local_status", np.zeros(1))
    st_fin = st[np.isfinite(st)]
    if nz.size:
        print(f"[ALPHA] max|a|={a.max():.4e} (no clamp)  "
              f"mean|a|(nonzero)={nz.mean():.3e}  "
              f"local_status max={st_fin.max() if st_fin.size else float('nan'):.2e}  "
              f"n_nonfinite_status={int(np.count_nonzero(~np.isfinite(st)))}  "
              f"cum_local_failures={getattr(self, 'n_eas_local_failures', 0)}",
              flush=True)
    return r


DynamicSolver.solve_step = probe

from ex12_abaqus_inp_plate_fold import read_abaqus_input, run_folding_from_result

res = read_abaqus_input(os.path.join("examples", "ex12_rigid_plate_display_fold.inp"))
run_folding_from_result(res, before_png_name="dbg3_b.png", after_png_name="dbg3_a.png",
                        result_name="dbg3.pkl", max_steps=40, elem_jit="numba")
