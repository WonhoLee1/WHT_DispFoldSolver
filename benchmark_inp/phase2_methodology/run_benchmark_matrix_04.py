"""
run_benchmark_matrix_04.py
==========================
Executes a 6-case benchmark matrix for 04_elastomer_seal_compression_3d.inp:
Comparing Contact Methods (NITSCHE, DUAL_MORTAR, PENALTY) and CPU Core Scaling (1-Core vs 4-Cores).
"""

import os
import sys
import time
import json
import numpy as np

# Set environment variable before importing numba/mkl where possible
os.environ["KMP_WARNINGS"] = "0"

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.solver.step_executor import MultiStepExecutor
from dispsolver.model.interaction import NITSCHE, DUAL_MORTAR, PENALTY


def run_single_case(method_name: str, num_cores: int) -> dict:
    import numba
    numba.set_num_threads(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)

    inp_path = os.path.join(os.path.dirname(__file__), "04_elastomer_seal_compression_3d.inp")
    reader, model = _parse_source(inp_path, dim=3, strict=False)

    # Override contact method in model interactions
    for inter_name, inter in model.interactions.items():
        if hasattr(inter, "contact_enforcement"):
            inter.contact_enforcement = method_name
        if hasattr(inter, "interaction_type") and inter.interaction_type == "CONTACT_PAIR":
            inter.contact_enforcement = method_name

    for step_name, step in model.steps.items():
        for entry in step.interactions.values():
            if hasattr(entry, "entity") and hasattr(entry.entity, "contact_enforcement"):
                entry.entity.contact_enforcement = method_name

    executor = MultiStepExecutor(model, verbose=False, dim=3)
    solver, sys3d = executor.initialize_solver()

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
    u_pts = solver.u.reshape(-1, 3)
    max_abs_ux = float(max(abs(np.max(u_pts[:, 0])), abs(np.min(u_pts[:, 0]))))
    max_abs_uy = float(max(abs(np.max(u_pts[:, 1])), abs(np.min(u_pts[:, 1]))))

    # Evaluate internal forces & contact stats
    K_g, f_int = solver.assemble_system(solver.u, dt=1.0)
    
    plate_top_indices = sys3d.global_nsets.get("PLATE_1.PLATE_TOP", None)
    rf_y_plate = 0.0
    if plate_top_indices is not None:
        for g_idx in plate_top_indices:
            rf_y_plate += f_int[3 * g_idx + 1]

    seal_bot_indices = sys3d.global_nsets.get("SEAL_1.SEAL_BOTTOM", None)
    rf_y_seal = 0.0
    if seal_bot_indices is not None:
        for g_idx in seal_bot_indices:
            rf_y_seal += f_int[3 * g_idx + 1]

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
        "rf_y_plate": float(rf_y_plate),
        "rf_y_seal": float(rf_y_seal),
        "fn_total": fn_total,
        "max_pen_um": max_pen,
        "max_ux_mm": max_abs_ux,
        "max_uy_mm": max_abs_uy,
    }


def main():
    print("=" * 100)
    print(" Benchmark Matrix Execution: 04_elastomer_seal_compression_3d")
    print(" Comparing Contact Methods (NITSCHE, DUAL_MORTAR, PENALTY) x Core Counts (1-Core, 4-Cores)")
    print("=" * 100)

    methods = [NITSCHE, DUAL_MORTAR, PENALTY]
    cores_list = [1, 4]

    results = []

    for method in methods:
        for cores in cores_list:
            print(f"[*] Running Case: Method={method:12s} | Cores={cores}...", end="", flush=True)
            res = run_single_case(method, cores)
            results.append(res)
            print(f" Done! (Converged={res['converged']}, Time={res['wall_time']:.2f}s, Iters={res['total_iters']})")

    # Print Report Table
    print("\n" + "=" * 100)
    print(" BENCHMARK MATRIX SUMMARY REPORT")
    print("=" * 100)
    header = f"{'Case':<6} | {'Method':<12} | {'Cores':<6} | {'Converged':<10} | {'Time (s)':<10} | {'Speedup':<8} | {'RF_y Plate (N)':<14} | {'Fn Contact (N)':<14} | {'Bulge Ux (mm)':<13}"
    print(header)
    print("-" * 100)

    # Compute 1-core baseline times per method
    t_1core = {}
    for r in results:
        if r["cores"] == 1:
            t_1core[r["method"]] = r["wall_time"]

    for idx, r in enumerate(results, 1):
        sp = t_1core[r["method"]] / r["wall_time"]
        line = (
            f"Case {idx:<2} | {r['method']:<12} | {r['cores']:<6} | {str(r['converged']):<10} | "
            f"{r['wall_time']:<10.2f} | {sp:<8.2f}x | {r['rf_y_plate']:<14.2f} | {r['fn_total']:<14.2f} | {r['max_ux_mm']:<13.4f}"
        )
        print(line)

    print("=" * 100)

    out_file = os.path.join(os.path.dirname(__file__), "benchmark_04_matrix_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Detailed benchmark JSON saved to: {out_file}")


if __name__ == "__main__":
    main()
