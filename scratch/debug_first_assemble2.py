import time, sys
sys.path.insert(0, "examples")
from dispsolver.solver import DynamicSolver

_orig_assemble = DynamicSolver._assemble
_call_count = [0]
def _timed_assemble(self, u, dt):
    _call_count[0] += 1
    n = _call_count[0]
    t0 = time.time()
    print(f"[assemble call #{n}] starting...", flush=True)
    result = _orig_assemble(self, u, dt)
    print(f"[assemble call #{n}] done in {time.time()-t0:.1f}s", flush=True)
    if n >= 3:
        print("Stopping after 3 assemble calls (debug probe).", flush=True)
        sys.exit(0)
    return result
DynamicSolver._assemble = _timed_assemble

from ex12_abaqus_inp_plate_fold import read_abaqus_input, run_folding_from_result
import os

inp_path = os.path.join("examples", "ex12_rigid_plate_display_fold.inp")
result = read_abaqus_input(inp_path)
run_folding_from_result(result, before_png_name="dbg_before.png", after_png_name="dbg_after.png",
                        result_name="dbg_result.pkl", max_steps=2, elem_jit="numba")
