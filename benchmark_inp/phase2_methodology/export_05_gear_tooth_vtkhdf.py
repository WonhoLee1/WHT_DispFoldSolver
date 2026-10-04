"""
export_05_gear_tooth_vtkhdf.py
==============================
Generates transient and snapshot VTKHDF (.vtkhdf) files for Deck #05:
2D Gear Tooth Finite Sliding & Friction Contact Benchmark.

Supports:
1. DUAL_MORTAR (Wohlmuth Biorthogonal Dual Mortar)
2. NITSCHE (Multiplier-Free Symmetric Nitsche)
3. PENALTY (Node-to-Segment Penalty with Coulomb Friction)

Outputs:
- output_05_gear_tooth_finite_sliding_2d/05_gear_tooth_dual_mortar_transient.vtkhdf
- output_05_gear_tooth_finite_sliding_2d/05_gear_tooth_nitsche_transient.vtkhdf
- output_05_gear_tooth_finite_sliding_2d/05_gear_tooth_penalty_transient.vtkhdf
- Snapshot .vtkhdf files with von Mises stress and reaction forces
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np

from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.solver.step_executor import MultiStepExecutor
from dispsolver.output.vtkhdf_writer import TransientVTKHDFWriter
from dispsolver.export.vtkhdf_exporter import export_vtkhdf


def run_and_export_vtkhdf(method_name: str = "DUAL_MORTAR", out_dir: str = "output_05_gear_tooth_finite_sliding_2d"):
    """Run Deck #05 with the specified contact method and export VTKHDF files."""
    inp_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "05_gear_tooth_finite_sliding_2d.inp"
    )
    if not os.path.exists(inp_path):
        inp_path = os.path.join(
            "benchmark_inp", "phase2_methodology", "05_gear_tooth_finite_sliding_2d.inp"
        )

    print(f"\n==================================================================================")
    print(f"  [Deck #05 VTKHDF Generator] Running method: {method_name}")
    print(f"  Input deck: {inp_path}")
    print(f"==================================================================================")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Parse .inp deck
    reader, model = _parse_source(inp_path, dim=2, strict=False)

    # 2. Configure contact formulation on ContactPair
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

    # 3. Create MultiStepExecutor
    executor = MultiStepExecutor(model, verbose=False)
    executor.initialize_solver()
    solver = executor.solver

    # Set parallel cores
    import numba
    numba.set_num_threads(4)

    # Track all increments across steps
    all_times = [0.0]
    all_displacements = [solver.u.copy()]

    # Execute Step 1: ENGAGE
    print("  [*] Executing Step 1: ENGAGE (Approaching & Initial Tooth Contact)...", flush=True)
    step1 = model.steps["ENGAGE"]
    ok1 = executor.execute_step(step1, from_parent_checkpoint=False)
    if not ok1:
        print("  [!] Step 1 failed to converge!")
        return False

    # Collect step 1 history
    t_engage = step1.time_period
    for t_inc, u_inc in zip(solver.history_t[1:], solver.history_u[1:]):
        all_times.append(t_inc)
        all_displacements.append(u_inc.copy())

    print(f"      Step 1 complete ({len(solver.history_t)-1} increments converged).")

    # Execute Step 2: DRIVE_SLIDE
    print("  [*] Executing Step 2: DRIVE_SLIDE (Finite Sliding & Friction Rotation)...", flush=True)
    step2 = model.steps["DRIVE_SLIDE"]
    ok2 = executor.execute_step(step2, from_parent_checkpoint=True)
    if not ok2:
        print("  [!] Step 2 failed to converge!")
        return False

    # Collect step 2 history (offset time by t_engage)
    for t_inc, u_inc in zip(solver.history_t[1:], solver.history_u[1:]):
        all_times.append(t_engage + t_inc)
        all_displacements.append(u_inc.copy())

    print(f"      Step 2 complete ({len(solver.history_t)-1} increments converged).")
    print(f"  [+] Total solution history: {len(all_times)} time steps.")

    # 4. Export Transient Animation VTKHDF (Kitware VTKHDF 2.2 format)
    method_slug = method_name.lower()
    transient_vtkhdf = os.path.join(out_dir, f"05_gear_tooth_{method_slug}_transient.vtkhdf")
    print(f"  [*] Writing Transient Animation VTKHDF: {transient_vtkhdf} ...", flush=True)

    writer = TransientVTKHDFWriter(filepath=transient_vtkhdf, mesh=solver.mesh)
    for step_idx, (t_val, u_vec) in enumerate(zip(all_times, all_displacements)):
        writer.add_step(time_val=float(t_val), u=u_vec)
    writer.close()
    print(f"  [OK] Successfully saved Transient VTKHDF ({os.path.getsize(transient_vtkhdf):,} bytes)!")

    # 5. Export Snapshot Final State VTKHDF (with von Mises stress & reaction forces)
    final_vtkhdf = os.path.join(out_dir, f"05_gear_tooth_{method_slug}_final.vtkhdf")
    print(f"  [*] Writing Final State VTKHDF: {final_vtkhdf} ...", flush=True)

    nodes_arr = solver.mesh.nodes_array()
    nid_to_idx = solver.mesh.node_id_to_index()
    elem_conn = np.array([[nid_to_idx[nid] for nid in elem.node_ids] for elem in solver.mesh.elements.values()], dtype=np.int64)
    stresses = getattr(solver, "elem_stresses", None)
    sdvs = getattr(solver, "elem_sdvs", None)

    export_vtkhdf(
        file_path=final_vtkhdf,
        nodes=nodes_arr,
        elements=elem_conn,
        u=solver.u,
        element_type="CPE4",
        stresses=stresses,
        sdvs=sdvs,
        time_value=all_times[-1],
    )
    print(f"  [OK] Successfully saved Final VTKHDF ({os.path.getsize(final_vtkhdf):,} bytes)!")

    return {
        "method": method_name,
        "transient_file": transient_vtkhdf,
        "final_file": final_vtkhdf,
        "n_steps": len(all_times),
        "total_time": all_times[-1],
    }


def main():
    out_dir = os.path.join(os.getcwd(), "output_05_gear_tooth_finite_sliding_2d")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Warmup pass
    print("[*] Performing JIT Warmup pass...")
    _ = run_and_export_vtkhdf("DUAL_MORTAR", out_dir=out_dir)

    # 2. Export DUAL_MORTAR (Primary high-accuracy formulation)
    res_mortar = run_and_export_vtkhdf("DUAL_MORTAR", out_dir=out_dir)

    # 3. Export NITSCHE (Ultra-fast symmetric formulation)
    res_nitsche = run_and_export_vtkhdf("NITSCHE", out_dir=out_dir)

    # 4. Export PENALTY (Baseline Node-to-Segment formulation)
    res_penalty = run_and_export_vtkhdf("PENALTY", out_dir=out_dir)

    print("\n==================================================================================")
    print("  ALL DECK #05 VTKHDF EXPORTS COMPLETED SUCCESSFULLY!")
    print("==================================================================================")
    print(f"  Output directory: {os.path.abspath(out_dir)}")
    print(f"  Files generated:")
    for res in [res_mortar, res_nitsche, res_penalty]:
        if res:
            print(f"    - {res['method']}:")
            print(f"        Transient (Animation, {res['n_steps']} steps): {res['transient_file']}")
            print(f"        Final Snapshot:                               {res['final_file']}")
    print("==================================================================================\n")


if __name__ == "__main__":
    main()
