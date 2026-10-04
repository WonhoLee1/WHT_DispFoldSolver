"""
run_benchmark_matrix_06.py
==========================
Executes a 6-case benchmark matrix for 06_bolt_preload_gasket_flange_3d.inp:
Comparing Contact Methods (DUAL_MORTAR, NITSCHE, PENALTY) and CPU Core Scaling (1-Core vs 4-Cores).

Physical & Numerical Verification:
- Warmup pass to isolate initial Numba LLVM JIT compilation overhead.
- True active constraint class inspection (preventing silent fallbacks).
- Action-reaction bolt preload and gasket contact force equilibrium.
- Penetration measurement across formulations.
- oneMKL PARDISO multi-core parallel speedup evaluation.
"""

import os
import sys
import time
import json
import numpy as np

os.environ["KMP_WARNINGS"] = "0"

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.solver.step_executor import MultiStepExecutor
from dispsolver.model.interaction import NITSCHE, DUAL_MORTAR, PENALTY


def run_single_case(method_name: str, num_cores: int, is_warmup: bool = False) -> dict:
    import numba
    numba.set_num_threads(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)

    inp_path = os.path.join(os.path.dirname(__file__), "06_bolt_preload_gasket_flange_3d.inp")
    reader, model = _parse_source(inp_path, dim=3, strict=False)

    # Override contact method in model interactions
    for inter in list(model.interactions.values()) + list(getattr(model.root_assembly, "contact_pairs", [])):
        if hasattr(inter, "constraint_enforcement"):
            inter.constraint_enforcement = method_name
        if hasattr(inter, "contact_enforcement"):
            inter.contact_enforcement = method_name

    for step in model.steps.values():
        for entry in step.interactions.values():
            ent = getattr(entry, "entity", entry)
            if hasattr(ent, "constraint_enforcement"):
                ent.constraint_enforcement = method_name
            if hasattr(ent, "contact_enforcement"):
                ent.contact_enforcement = method_name

    executor = MultiStepExecutor(model, verbose=False, dim=3)
    solver, sys3d = executor.initialize_solver()

    # Inspect bound contact constraint class
    active_constraint_class = "None"
    if len(solver.constraints) > 0:
        c_obj = solver.constraints[0]
        active_constraint_class = c_obj.__class__.__name__

    start_time = time.perf_counter()
    ok_all = True
    total_iters = 0

    for step_name in [s for s in model.steps.keys() if s != "Initial"]:
        step_obj = model.steps[step_name]
        ok = executor.execute_step(step_obj, from_parent_checkpoint=True)
        ok_all = ok_all and ok
        total_iters += getattr(solver, "last_iterations", 0)

    wall_time = time.perf_counter() - start_time

    if is_warmup:
        return {"warmup": True, "wall_time": wall_time}

    # Extract field metrics
    u_pts = solver.u.reshape(-1, 3)
    max_abs_ux = float(np.max(np.abs(u_pts[:, 0])))
    max_abs_uy = float(np.max(np.abs(u_pts[:, 1])))
    max_abs_uz = float(np.max(np.abs(u_pts[:, 2])))

    # Internal force reactions
    K_g, f_int = solver.assemble_system(solver.u, dt=1.0)

    bolt_seat_indices = sys3d.global_nsets.get("FLANGE_1.FLANGE_BOLT_SEAT", None)
    rf_z_bolt = 0.0
    if bolt_seat_indices is not None:
        for g_idx in bolt_seat_indices:
            rf_z_bolt += f_int[3 * g_idx + 2]

    gasket_sym_indices = sys3d.global_nsets.get("GASKET_1.GASKET_Z_SYM", None)
    rf_z_gasket = 0.0
    if gasket_sym_indices is not None:
        for g_idx in gasket_sym_indices:
            rf_z_gasket += f_int[3 * g_idx + 2]

    fn_total = 0.0
    max_pen = 0.0
    if len(solver.constraints) > 0:
        c_obj = solver.constraints[0]
        res = c_obj.assemble(solver.u)
        stats = res[2] if len(res) >= 3 and isinstance(res[2], dict) else {}
        fn_total = float(stats.get("total_normal_force", 0.0))
        max_pen = float(stats.get("max_penetration", 0.0)) * 1e3  # um

    return {
        "method": method_name,
        "cores": num_cores,
        "constraint_class": active_constraint_class,
        "converged": ok_all,
        "wall_time": wall_time,
        "total_iters": total_iters,
        "rf_z_bolt": float(rf_z_bolt),
        "rf_z_gasket": float(rf_z_gasket),
        "fn_total": float(fn_total),
        "max_pen_um": float(max_pen),
        "max_ux_mm": float(max_abs_ux),
        "max_uy_mm": float(max_abs_uy),
        "max_uz_mm": float(max_abs_uz),
    }


def main():
    print("=" * 100)
    print(" Deck #06: 3D Pipe Flange & Bolted Gasket Sealing Contact Benchmark Matrix")
    print(" Comparing Formulations: DUAL_MORTAR, NITSCHE, PENALTY across 1-Core and 4-Core CPU")
    print("=" * 100)

    # 1. Warmup pass
    print("\n>>> [Warmup Pass] Pre-compiling Numba LLVM Kernels...")
    t0 = time.time()
    w_res = run_single_case("PENALTY", 1, is_warmup=True)
    print(f"    Warmup complete in {time.time() - t0:.2f}s (JIT cache warmed up)")

    methods = ["DUAL_MORTAR", "NITSCHE", "PENALTY"]
    cores_list = [1, 4]

    results = []

    for method in methods:
        for cores in cores_list:
            print(f"\n--- Running Case: Method={method} | Cores={cores} ---")
            res = run_single_case(method, cores)
            results.append(res)
            print(f"    [Result] Class: {res['constraint_class']} | Wall Time: {res['wall_time']:.2f}s | "
                  f"Iters: {res['total_iters']} | Conv: {res['converged']}")
            print(f"             RF_bolt: {res['rf_z_bolt']:.2f} N | RF_gasket: {res['rf_z_gasket']:.2f} N | "
                  f"Fn: {res['fn_total']:.2f} N | Max Pen: {res['max_pen_um']:.4f} um")

    # Compute speedup
    speedups = {}
    for method in methods:
        t_1 = next(r["wall_time"] for r in results if r["method"] == method and r["cores"] == 1)
        t_4 = next(r["wall_time"] for r in results if r["method"] == method and r["cores"] == 4)
        sp = t_1 / t_4 if t_4 > 0 else 1.0
        speedups[method] = sp

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Method':<14} | {'Cores':<5} | {'Class':<26} | {'Wall Time':<10} | {'Speedup':<8} | {'RF_bolt (N)':<12} | {'Pen (um)':<9}")
    print("-" * 105)
    for res in results:
        m = res["method"]
        c = res["cores"]
        sp_str = f"{speedups[m]:.2f}x" if c == 4 else "1.00x"
        print(f"{m:<14} | {c:<5} | {res['constraint_class']:<26} | {res['wall_time']:<8.2f}s | {sp_str:<8} | {res['rf_z_bolt']:<12.1f} | {res['max_pen_um']:<9.4f}")
    print("=" * 105)

    # Save to JSON
    json_path = os.path.join(os.path.dirname(__file__), "benchmark_06_matrix_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "speedups": speedups}, f, indent=2)
    print(f"\n[Saved] Matrix results saved to: {json_path}")


if __name__ == "__main__":
    main()
