import time, sys
sys.path.insert(0, "examples")
from ex12_abaqus_inp_plate_fold import read_abaqus_input
import os

inp_path = os.path.join("examples", "ex12_rigid_plate_display_fold.inp")
print("reading inp...", flush=True)
t0 = time.time()
result = read_abaqus_input(inp_path)
print(f"read done in {time.time()-t0:.1f}s", flush=True)

solver = result.solver
u0 = solver.u.copy()
print("calling solver._assemble(u0, dt=0.005) ONCE, timing it...", flush=True)
t0 = time.time()
f_int, K, state_new = solver._assemble(u0, dt=0.005)
print(f"first _assemble() done in {time.time()-t0:.1f}s, max|f_int|={abs(f_int).max():.3e}", flush=True)

print("calling it a SECOND time (should be fast, cache warm)...", flush=True)
t0 = time.time()
f_int2, K2, state_new2 = solver._assemble(u0, dt=0.005)
print(f"second _assemble() done in {time.time()-t0:.1f}s", flush=True)
