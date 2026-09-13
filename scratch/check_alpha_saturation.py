"""Discriminator 2: is eas_alpha riding the _ALPHA_MAX clamp in the real fold?

HISTORICAL. `_ALPHA_MAX` was removed on 2026-09-09 (see
dev_log/eas_stabilization_modernization_20260909.md); this script is kept for
provenance of `scratch/alpha_sat3.log` and now reads the clamp value it was
run against as a literal. The post-change equivalent is
`scratch/check_alpha_after_ls.py`.
"""
import sys, numpy as np
sys.path.insert(0,"examples")
from dispsolver.solver import DynamicSolver
_ALPHA_MAX = 0.5  # the clamp value this probe was originally run against

_orig = DynamicSolver.solve_step
def probe(self, dt):
    r = _orig(self, dt)
    a = np.abs(self.eas_alpha)
    nz = a[a > 0]
    if nz.size:
        frac = float((a.max(axis=1) > 0.9*_ALPHA_MAX).mean())
        print(f"[ALPHA] max|a|={a.max():.4e} (clamp {_ALPHA_MAX})  "
              f"mean|a|(nonzero)={nz.mean():.3e}  frac elems >0.9*clamp={frac:.3%}", flush=True)
    return r
DynamicSolver.solve_step = probe

from ex12_abaqus_inp_plate_fold import read_abaqus_input, run_folding_from_result
import os
res = read_abaqus_input(os.path.join("examples","ex12_rigid_plate_display_fold.inp"))
run_folding_from_result(res, before_png_name="dbg2_b.png", after_png_name="dbg2_a.png",
                        result_name="dbg2.pkl", max_steps=40, elem_jit="numba")
