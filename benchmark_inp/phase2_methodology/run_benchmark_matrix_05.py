"""
run_benchmark_matrix_05.py
==========================
Executes a 6-case benchmark matrix for 05_gear_tooth_finite_sliding_2d.inp:
Comparing Contact Enforcement Methods (NITSCHE, DUAL_MORTAR, PENALTY) and CPU Core Counts (1-Core vs 4-Cores).
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


def run_single_case(method_name: str, num_cores: int) -> dict:
    import numba
    numba.set_num_threads(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)

    inp_path = os.path.join(os.path.dirname(__file__), "05_gear_tooth_finite_sliding_2d.inp")
    reader, model = _parse_source(inp_path, dim=2, strict=False)

    # Override contact method in model interactions
    for inter_name, inter in model.interactions.items():
        if hasattr(inter, "constraint_enforcement"):
            inter.constraint_enforcement = method_name
        if hasattr(inter, "contact_enforcement"):
            inter.contact_enforcement = method_name

    for step_name, step in model.steps.items():
        for entry in step.interactions.values():
            if hasattr(entry, "entity"):
                if hasattr(entry.entity, "constraint_enforcement"):
                    entry.entity.constraint_enforcement = method_name
                if hasattr(entry.entity, "contact_enforcement"):
                    entry.entity.contact_enforcement = method_name

    executor = MultiStepExecutor(model, verbose=False, dim=2)
    solver, sys2d = executor.initialize_solver()

    c_types = [type(c).__name__ for c in solver.constraints]
    print(f"      [DISPATCH] {method_name} -> Active constraint class: {c_types}")

    start_time = time.perf_counter()

    ok_all = True
    total_iters = 0

    for step_name in [s for s in model.steps.keys() if s != "Initial"]:
        step_obj = model.steps[step_name]
        ok = executor.execute_step(step_obj, from_parent_checkpoint=True)
        ok_all = ok_all and ok
        total_iters += getattr(solver, "last_iterations", 0)

    wall_time = time.perf_counter() - start_time

    # Extract field metrics
    u_pts = solver.u.reshape(-1, 2)
    max_abs_ux = float(max(abs(np.max(u_pts[:, 0])), abs(np.min(u_pts[:, 0]))))
    max_abs_uy = float(max(abs(np.max(u_pts[:, 1])), abs(np.min(u_pts[:, 1]))))

    # Evaluate internal forces & contact stats
    K_g, f_int = solver.assemble_system(solver.u, dt=1.0)
    
    driver_top_indices = sys2d.global_nsets.get("INST_DRIVER.NSET_TOP_DRIVER", None)
    rf_y_driver = 0.0
    rf_x_driver = 0.0
    if driver_top_indices is not None:
        for g_idx in driver_top_indices:
            rf_x_driver += f_int[2 * g_idx]
            rf_y_driver += f_int[2 * g_idx + 1]

    follower_bot_indices = sys2d.global_nsets.get("INST_FOLLOWER.NSET_BOT_FOLLOWER", None)
    rf_y_follower = 0.0
    rf_x_follower = 0.0
    if follower_bot_indices is not None:
        for g_idx in follower_bot_indices:
            rf_x_follower += f_int[2 * g_idx]
            rf_y_follower += f_int[2 * g_idx + 1]

    fn_total = 0.0
    max_pen = 0.0
    if len(solver.constraints) > 0:
        c_obj = solver.constraints[0]
        res = c_obj.assemble(solver.u)
        stats = res[2] if len(res) >= 3 and isinstance(res[2], dict) else {}
        fn_total = float(stats.get("total_normal_force", 0.0))
        max_pen = float(stats.get("max_penetration", 0.0)) * 1e3  # in um

    return {
        "method": method_name,
        "cores": num_cores,
        "converged": ok_all,
        "wall_time": wall_time,
        "total_iters": total_iters,
        "rf_x_driver": float(rf_x_driver),
        "rf_y_driver": float(rf_y_driver),
        "rf_x_follower": float(rf_x_follower),
        "rf_y_follower": float(rf_y_follower),
        "fn_total": fn_total,
        "max_pen_um": max_pen,
        "max_ux_mm": max_abs_ux,
        "max_uy_mm": max_abs_uy,
    }


def main():
    cases = [
        ("NITSCHE", 1),
        ("NITSCHE", 4),
        ("DUAL_MORTAR", 1),
        ("DUAL_MORTAR", 4),
        ("PENALTY", 1),
        ("PENALTY", 4),
    ]

    results = []
    base_time = {}

    print("====================================================================================================")
    print(" DECK #05 (2D GEAR TOOTH FINITE SLIDING) BENCHMARK MATRIX (NITSCHE / DUAL_MORTAR / PENALTY x 1/4 CORES)")
    print("====================================================================================================\n")

    print("[*] Performing initial Numba JIT Warmup pass (Cold compilation)...")
    _ = run_single_case("NITSCHE", 1)
    print("[*] JIT Warmup complete! Starting precise benchmark timing runs...\n")

    for i, (method_name, num_cores) in enumerate(cases, 1):

        print(f"--- Running Case {i}/6: Method={method_name:<11} Cores={num_cores} ---")
        t0 = time.time()
        res = run_single_case(method_name, num_cores)
        elapsed = time.time() - t0
        print(f"    Done! (Converged={res['converged']}, Time={res['wall_time']:.2f}s, Iters={res['total_iters']})\n")
        results.append(res)
        if num_cores == 1:
            base_time[method_name] = res["wall_time"]

    # Save JSON summary report
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "benchmark_05_matrix_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Render structured Markdown summary table
    print("====================================================================================================")
    print(" BENCHMARK MATRIX SUMMARY REPORT (DECK #05 2D GEAR TOOTH)")
    print("====================================================================================================")
    print(f"{'Case':<6} | {'Method':<12} | {'Cores':<6} | {'Converged':<10} | {'Time (s)':<10} | {'Speedup':<10} | {'RF_y Driver (N)':<16} | {'RF_x Driver (N)':<16} | {'Fn Total (N)':<14}")
    print("-" * 110)

    for i, r in enumerate(results, 1):
        m = r["method"]
        c = r["cores"]
        conv = str(r["converged"])
        w_time = r["wall_time"]
        b_time = base_time.get(m, w_time)
        speedup = f"{b_time / w_time:.2f} x"
        rf_y = f"{r['rf_y_driver']:.2f}"
        rf_x = f"{r['rf_x_driver']:.2f}"
        fn = f"{r['fn_total']:.2f}"
        print(f"Case {i:<2} | {m:<12} | {c:<6} | {conv:<10} | {w_time:<10.2f} | {speedup:<10} | {rf_y:<16} | {rf_x:<16} | {fn:<14}")

    print("====================================================================================================\n")
    print(f"[+] Detailed benchmark JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
