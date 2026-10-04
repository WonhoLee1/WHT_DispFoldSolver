"""
debug_03_inc22.py
=================
Diagnose why Inc 22 at t=0.9950 struggled to converge.
"""

from dispsolver.io.abaqus_reader import _parse_source
import numpy as np

inp_path = "benchmark_inp/phase2_methodology/03_punch_indentation_plastic_2d.inp"
reader, model = _parse_source(inp_path, dim=2, strict=False)
step = model.steps["PRESS"]
solver, sys2d = model.create_solver(dim=2)

total_time = 1.0
dt = 0.02
dt_max = 0.05
dt_min = 1e-6
t_curr = 0.0
inc = 0

while t_curr < total_time - 1e-9:
    inc += 1
    if t_curr + dt > total_time:
        dt = total_time - t_curr

    t_next = t_curr + dt
    solver._update_step_bcs(step, sys2d, t_next, total_time)
    f_ext = solver._compute_step_external_loads(step, sys2d, t_next, total_time)

    u_prev = solver.u.copy()
    sdvs_prev = solver.elem_sdvs.copy()

    converged, iters = solver.solve_step(dt=dt, f_ext=f_ext, max_iters=50)
    print(f"Inc {inc:2d}: t={t_next:.4f}, dt={dt:.5f}, iters={iters}, converged={converged}")
    if not converged:
        print(f"Failed at inc {inc}, t_next={t_next}, dt={dt}")
        break

    t_curr = t_next
    if iters <= 4 and dt < dt_max:
        dt = min(dt * 1.5, dt_max)
    elif iters >= 10:
        dt = max(dt * 0.85, dt_min)
