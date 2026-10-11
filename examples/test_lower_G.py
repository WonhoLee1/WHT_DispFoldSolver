"""
test_lower_G.py
================
Tests ultra-low PSA shear modulus G (0.001x and 0.0001x) to observe
convergence towards the exact theoretical pure-slip lower bound (0.9479 N*mm).
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dispsolver.fold_model_config import make_teardrop_config
from examples.gen_ex12_inp import generate as generate_ex12_inp
from examples.ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding
from examples.verify_theoretical_mechanics import solve_teardrop_elastica, compute_theoretical_stiffness
from examples.verify_psa_shear_sensitivity import compute_pivot_moment_from_solver


def test_ultra_low_g():
    print("=" * 115)
    print(" TESTING ULTRA-LOW PSA SHEAR MODULUS G (< 0.17 kPa)")
    print("=" * 115)

    D_slip, D_mono, y_NA = compute_theoretical_stiffness()
    elastica = solve_teardrop_elastica(D=D_slip, L_half=7.5, w_half=1.4)
    M_lower_bound = elastica["M_pivot_pure_slip"]  # 0.9479 N*mm
    print(f"[THEORY] Pure Free-Slip Lower Bound (G = 0): {M_lower_bound:.4f} N*mm\n")

    scales = [0.005, 0.001, 0.0001]

    for scale_G in scales:
        g_val = 16.78 * scale_G
        print(f">>> Running scale_G = {scale_G:.4f} (G = {g_val:.4f} kPa) ...")

        cfg = make_teardrop_config()
        base_mu = 0.015614
        cfg.materials.definitions["PSA"].params["mu"] = base_mu * scale_G

        temp_inp = f"examples/temp_ultra_low_{scale_G}.inp"
        temp_res = f"examples/temp_ultra_low_{scale_G}.pkl"
        temp_png_b = f"examples/temp_ultra_low_{scale_G}_b.png"
        temp_png_a = f"examples/temp_ultra_low_{scale_G}_a.png"

        try:
            inp_text = generate_ex12_inp(cfg)
            with open(temp_inp, "w", encoding="utf-8") as f:
                f.write(inp_text)

            t0 = time.time()
            res_dict = run_abaqus_inp_folding(
                inp_path=temp_inp,
                before_png_name=temp_png_b,
                after_png_name=temp_png_a,
                result_name=temp_res,
                elem_jit="numba",
                config=cfg,
                preset="ADAPTIVE",
            )
            elapsed = time.time() - t0

            solver = res_dict["solver"]
            mesh = res_dict["result"].mesh
            nid_to_idx = mesh.node_id_to_index()
            M_pivot, Fx = compute_pivot_moment_from_solver(solver, mesh, nid_to_idx)
            ratio = M_pivot / M_lower_bound
            err_abs = M_pivot - M_lower_bound

            print(f"--> Done in {elapsed:.1f}s | G = {g_val:.4f} kPa | M_pivot = {M_pivot:.4f} N*mm | Ratio = {ratio:.3f}x | Diff = {err_abs:+.4f} N*mm\n")

        except Exception as e:
            print(f"--> Failed for scale_G = {scale_G}: {e}\n")

        finally:
            for p in [temp_inp, temp_res, temp_png_b, temp_png_a]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass


if __name__ == "__main__":
    test_ultra_low_g()
