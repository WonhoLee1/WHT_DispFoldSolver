"""
scratch/compare_90deg_conforming_refinement.py
Generates a comprehensive 1:1 comparison report and figures for:
1. FrontISTR Baseline 361 FI (100% Size: nx=30, h=0.500mm)
2. FrontISTR Conforming Baseline (100% Size: nx=30, h=0.500mm)
3. FrontISTR Conforming Refined (~55% Size: nx=55, h=0.273mm)
4. WHT Solver 2D (Opt1 Continuum Reference at 90 deg)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.run_fistr_conforming_teardrop import STACKUP_14

def main():
    conf_base_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop", "01_conforming_hybrid_14layer", "fistr_summary.json")
    fistr_fi_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_fistr", "FISTR_2D_teardrop_FI_FI", "fistr_summary.json")
    wht_2d_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_14layer_ny1_vs_ny2", "WHT_2D_ny1", "wht_summary.json")
    conf_ref_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_refined_90deg", "refined_conforming_90deg_summary.json")

    if not os.path.exists(conf_ref_path):
        print("Waiting for refined 90 deg run to complete...")
        return

    with open(conf_base_path, "r", encoding="utf-8") as f:
        conf_base = json.load(f)
    with open(fistr_fi_path, "r", encoding="utf-8") as f:
        fistr_fi = json.load(f)
    with open(wht_2d_path, "r", encoding="utf-8") as f:
        wht_2d = json.load(f)
    with open(conf_ref_path, "r", encoding="utf-8") as f:
        conf_ref = json.load(f)

    # 1. Macro Table
    print("=" * 140)
    print("14-LAYER 90° TEARDROP FOLDING (180° TOTAL BEND): MACRO METRICS COMPARISON TABLE")
    print("=" * 140)
    header = f"{'Case Name':<35} | {'Elements':<10} | {'Nodes':<10} | {'Wall Time':<12} | {'Moment |M_L|':<16} | {'Force |F_x|':<14} | {'Loop Width':<14} | {'Apex Y':<12}"
    print(header)
    print("-" * 140)

    rows = [
        ("FrontISTR Baseline (361 FI)", fistr_fi.get("num_real_elements", 1030), fistr_fi.get("num_real_nodes", 2382), f"{fistr_fi.get('wall_time_s', 0):.1f} s", f"{abs(fistr_fi.get('M_L_final_Nmm', 0)):.4f} N*mm", f"{abs(fistr_fi.get('Fx_L_final_N', 0)):.4f} N", f"{fistr_fi.get('loop_width_x_mm', 0):.4f} mm", f"{fistr_fi.get('apex_y_mm', 0):.4f} mm"),
        ("FrontISTR Conforming (100% Size)", conf_base.get("num_real_elements", 2650), conf_base.get("num_real_nodes", 7554), f"{conf_base.get('wall_time_s', 0):.1f} s", f"{abs(conf_base.get('M_L_final_Nmm', 0)):.4f} N*mm", f"{abs(conf_base.get('Fx_L_final_N', 0)):.4f} N", f"{conf_base.get('loop_width_x_mm', 0):.4f} mm", f"{conf_base.get('apex_y_mm', 0):.4f} mm"),
        ("FrontISTR Conforming (55% Refined)", conf_ref.get("num_real_elements", 4715), conf_ref.get("num_real_nodes", 13404), f"{conf_ref.get('wall_time_s', 0):.1f} s", f"{abs(conf_ref.get('bar_M_L', 0)):.4f} N*mm", f"{abs(conf_ref.get('bar_Fx_L', 0)):.4f} N", f"{conf_ref.get('loop_width_x_mm', 0):.4f} mm", f"{conf_ref.get('apex_y_mm', 0):.4f} mm"),
        ("WHT Solver 2D (Opt1 Reference)", wht_2d.get("elements", 1030), wht_2d.get("nodes", 1162), f"{wht_2d.get('wall_time_s', 0):.1f} s", f"{abs(wht_2d.get('M_L_final_Nmm', 0)):.4f} N*mm", f"{abs(wht_2d.get('Fx_L_final_N', 0)):.4f} N", f"{wht_2d.get('loop_width_x_mm', 0):.4f} mm", f"{wht_2d.get('apex_y_mm', 0):.4f} mm"),
    ]

    for r in rows:
        print(f"{r[0]:<35} | {str(r[1]):<10} | {str(r[2]):<10} | {r[3]:<12} | {r[4]:<16} | {r[5]:<14} | {r[6]:<14} | {r[7]:<12}")
    print("=" * 140)

    # 2. Layerwise Stresses Table (100% vs 55% Refined Conforming)
    base_layers = {
        "GLASS_Top": {"mises": 1272.12, "sig_tt": 2359.11, "eps": 2.36, "gam": 0.11},
        "PSA_6":     {"mises": 0.05,    "sig_tt": 4.16,    "eps": 54.72, "gam": 73.88},
        "PET_6":     {"mises": 142.47,  "sig_tt": 121.24,  "eps": 5.31, "gam": 0.93},
        "PSA_5":     {"mises": 0.06,    "sig_tt": 4.23,    "eps": 56.24, "gam": 75.61},
        "PET_5":     {"mises": 192.97,  "sig_tt": 72.97,   "eps": 6.60, "gam": 0.95},
        "PSA_4":     {"mises": 0.06,    "sig_tt": 4.29,    "eps": 57.75, "gam": 76.06},
        "PET_4":     {"mises": 242.14,  "sig_tt": 43.06,   "eps": 7.80, "gam": 0.92},
        "PSA_3":     {"mises": 0.06,    "sig_tt": 4.33,    "eps": 59.45, "gam": 74.30},
        "PET_3":     {"mises": 288.92,  "sig_tt": 15.60,   "eps": 8.72, "gam": 0.79},
        "PSA_2":     {"mises": 0.07,    "sig_tt": 4.38,    "eps": 61.10, "gam": 70.43},
        "PET_2":     {"mises": 339.26,  "sig_tt": 43.64,   "eps": 9.54, "gam": 0.56},
        "PSA_1":     {"mises": 0.07,    "sig_tt": 4.48,    "eps": 62.46, "gam": 63.93},
        "PET_1":     {"mises": 399.37,  "sig_tt": 80.06,   "eps": 10.32, "gam": 0.45},
        "PSA_0":     {"mises": 0.00,    "sig_tt": 0.00,    "eps": 0.00,  "gam": 0.00},
    }

    ref_layers = conf_ref.get("layers", {})

    print("\n" + "=" * 140)
    print("14-LAYER 90° TEARDROP FOLDING: LOCAL FRAME LAYERWISE STRESS & STRAIN COMPARISON (100% vs 55% Size)")
    print("=" * 140)
    l_header = f"{'Layer':<12} | {'Material':<8} | {'100% Conforming Mises':<22} | {'55% Refined Mises':<20} | {'Mises Delta':<14} | {'100% sig_tt':<14} | {'55% sig_tt':<14} | {'55% eps_eng':<12}"
    print(l_header)
    print("-" * 140)

    for lyr in STACKUP_14:
        lname = lyr["name"]
        mat = lyr["mat"]
        b = base_layers.get(lname, {})
        r = ref_layers.get(lname, {})
        b_m = b.get("mises", 0.0)
        r_m = r.get("max_mises", 0.0)
        diff_pct = (r_m - b_m) / b_m * 100.0 if b_m > 0 else 0.0
        b_tt = b.get("sig_tt", 0.0)
        r_tt = r.get("max_sig_tt", 0.0)
        r_eps = r.get("max_eps_eng", 0.0) * 100.0
        print(f"{lname:<12} | {mat:<8} | {b_m:18.2f} MPa | {r_m:16.2f} MPa | {diff_pct:+12.1f}% | {b_tt:10.2f} MPa | {r_tt:10.2f} MPa | {r_eps:10.2f}%")
    print("=" * 140)

    # 3. Plotting comparison figures
    out_png = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_refined_90deg", "fistr_conforming_90deg_mesh_refinement_comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Ax 1: Moment vs Theta
    ax = axes[0]
    th_fi = fistr_fi.get("history_theta_deg", [])
    m_fi = [abs(m) for m in fistr_fi.get("history_M_L", [])]
    ax.plot(th_fi, m_fi, "r--", lw=1.8, label=f"FrontISTR Baseline (361 FI) [{m_fi[-1]:.2f} N·mm]")

    th_base = conf_base.get("history_theta_deg", [])
    m_base = [abs(m) for m in conf_base.get("history_M_L", [])]
    ax.plot(th_base, m_base, color="#0284c7", lw=2.0, label=f"Conforming (100% Size) [{m_base[-1]:.2f} N·mm]")

    th_wht = wht_2d.get("history_theta", [])
    m_wht = [abs(m) for m in wht_2d.get("history_M_L", [])]
    if th_wht and m_wht:
        ax.plot(th_wht, m_wht, "g-.", lw=2.0, label=f"WHT Solver 2D (Opt1) [{m_wht[-1]:.2f} N·mm]")

    # Refined conforming
    with open(os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_refined_90deg", "fistr_summary.json"), "r", encoding="utf-8") as f:
        conf_ref_summary = json.load(f)
    th_ref = conf_ref_summary.get("history_theta_deg", [])
    m_ref = [abs(m) for m in conf_ref_summary.get("history_M_L", [])]
    ax.plot(th_ref, m_ref, color="#8b5cf6", lw=2.2, label=f"Conforming Refined (55% Size) [{m_ref[-1]:.2f} N·mm]")

    ax.set_xlabel("플레이트 접힘 회전각 θ [deg]")
    ax.set_ylabel("피벗 구동 모멘트 |M_L| [N·mm]")
    ax.set_title("90°(180° 접힘) 힌지 구동 모멘트 비교")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")

    # Ax 2: Force vs Theta
    ax = axes[1]
    fx_fi = [abs(f) for f in fistr_fi.get("history_Fx_L", [])]
    ax.plot(th_fi, fx_fi, "r--", lw=1.8, label=f"Baseline 361 FI [{fx_fi[-1]:.2f} N]")
    fx_base = [abs(f) for f in conf_base.get("history_Fx_L", [])]
    ax.plot(th_base, fx_base, color="#0284c7", lw=2.0, label=f"Conforming 100% [{fx_base[-1]:.2f} N]")
    fx_ref = [abs(f) for f in conf_ref_summary.get("history_Fx_L", [])]
    ax.plot(th_ref, fx_ref, color="#8b5cf6", lw=2.2, label=f"Conforming 55% Refined [{fx_ref[-1]:.2f} N]")
    if th_wht:
        fx_wht = [abs(f) for f in wht_2d.get("history_Fx_L", [])]
        if fx_wht:
            ax.plot(th_wht, fx_wht, "g-.", lw=2.0, label=f"WHT Solver 2D [{fx_wht[-1]:.2f} N]")

    ax.set_xlabel("플레이트 접힘 회전각 θ [deg]")
    ax.set_ylabel("수평 반력 |F_x| [N]")
    ax.set_title("90° 힌지 수평 반력 추이 비교")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")

    # Ax 3: Layerwise Mises Stresses Bar Chart
    ax = axes[2]
    stiff_layers = [lyr["name"] for lyr in STACKUP_14 if lyr["is_stiff"]]
    y_pos = np.arange(len(stiff_layers))
    m_base_stiff = [base_layers[l]["mises"] for l in stiff_layers]
    m_ref_stiff = [ref_layers.get(l, {}).get("max_mises", 0.0) for l in stiff_layers]

    width = 0.35
    ax.barh(y_pos - width/2, m_base_stiff, width, label="100% Size (nx=30)", color="#0284c7", alpha=0.85)
    ax.barh(y_pos + width/2, m_ref_stiff, width, label="55% Refined (nx=55)", color="#8b5cf6", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stiff_layers, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("정점부 Mises 응력 [MPa]")
    ax.set_title("14층 강성 필름(PET/Glass) 정점부 응력 변화")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"\n[*] Comparison plot saved to: {out_png}")

if __name__ == "__main__":
    main()
