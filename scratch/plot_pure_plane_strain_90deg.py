"""
scratch/plot_pure_plane_strain_90deg.py
Generates comparison plots between:
1. Baseline 361 FI
2. Conforming 100% (Free-Z)
3. Conforming 55% Refined (Free-Z)
4. Conforming 55% Pure Plane Strain (All u_z = 0)
5. WHT 2D (Opt1)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    conf_base_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop", "01_conforming_hybrid_14layer", "fistr_summary.json")
    fistr_fi_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_fistr", "FISTR_2D_teardrop_FI_FI", "fistr_summary.json")
    wht_2d_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_14layer_ny1_vs_ny2", "WHT_2D_ny1", "wht_summary.json")
    conf_ref_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_refined_90deg", "refined_conforming_90deg_summary.json")
    conf_pps_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_pure_plane_strain_90deg", "pure_plane_strain_conforming_90deg_summary.json")
    conf_pps_sum_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_pure_plane_strain_90deg", "fistr_summary.json")

    with open(conf_base_path, "r", encoding="utf-8") as f: conf_base = json.load(f)
    with open(fistr_fi_path, "r", encoding="utf-8") as f: fistr_fi = json.load(f)
    with open(wht_2d_path, "r", encoding="utf-8") as f: wht_2d = json.load(f)
    with open(conf_ref_path, "r", encoding="utf-8") as f: conf_ref = json.load(f)
    with open(conf_pps_path, "r", encoding="utf-8") as f: conf_pps = json.load(f)
    with open(conf_pps_sum_path, "r", encoding="utf-8") as f: conf_pps_sum = json.load(f)

    out_png = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_pure_plane_strain_90deg", "fistr_conforming_90deg_pure_plane_strain_comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Ax 1: Moment vs Theta
    ax = axes[0]
    th_fi = fistr_fi.get("history_theta_deg", [])
    m_fi = [abs(m) for m in fistr_fi.get("history_M_L", [])]
    ax.plot(th_fi, m_fi, "r--", lw=1.8, label=f"Baseline 361 FI [{m_fi[-1]:.2f} N·mm]")

    th_base = conf_base.get("history_theta_deg", [])
    m_base = [abs(m) for m in conf_base.get("history_M_L", [])]
    ax.plot(th_base, m_base, color="#0284c7", lw=2.0, label=f"Conforming 100% Free-Z [{m_base[-1]:.2f} N·mm]")

    with open(os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_refined_90deg", "fistr_summary.json"), "r", encoding="utf-8") as f:
        conf_ref_summary = json.load(f)
    th_ref = conf_ref_summary.get("history_theta_deg", [])
    m_ref = [abs(m) for m in conf_ref_summary.get("history_M_L", [])]
    ax.plot(th_ref, m_ref, color="#8b5cf6", lw=2.0, label=f"Conforming 55% Free-Z [{m_ref[-1]:.2f} N·mm]")

    th_pps = conf_pps_sum.get("history_theta_deg", [])
    m_pps = [abs(m) for m in conf_pps_sum.get("history_M_L", [])]
    ax.plot(th_pps, m_pps, color="#10b981", lw=2.4, label=f"Conforming 55% Pure-PS [{m_pps[-1]:.2f} N·mm]")

    th_wht = wht_2d.get("history_theta", [])
    m_wht = [abs(m) for m in wht_2d.get("history_M_L", [])]
    if th_wht and m_wht:
        ax.plot(th_wht, m_wht, "g-.", lw=2.0, label=f"WHT 2D (Opt1) [{m_wht[-1]:.2f} N·mm]")

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
    ax.plot(th_base, fx_base, color="#0284c7", lw=2.0, label=f"Conforming 100% Free-Z [{fx_base[-1]:.2f} N]")
    fx_ref = [abs(f) for f in conf_ref_summary.get("history_Fx_L", [])]
    ax.plot(th_ref, fx_ref, color="#8b5cf6", lw=2.0, label=f"Conforming 55% Free-Z [{fx_ref[-1]:.2f} N]")
    fx_pps = [abs(f) for f in conf_pps_sum.get("history_Fx_L", [])]
    ax.plot(th_pps, fx_pps, color="#10b981", lw=2.4, label=f"Conforming 55% Pure-PS [{fx_pps[-1]:.2f} N]")
    if th_wht:
        fx_wht = [abs(f) for f in wht_2d.get("history_Fx_L", [])]
        if fx_wht: ax.plot(th_wht, fx_wht, "g-.", lw=2.0, label=f"WHT 2D [{fx_wht[-1]:.2f} N]")

    ax.set_xlabel("플레이트 접힘 회전각 θ [deg]")
    ax.set_ylabel("수평 반력 |F_x| [N]")
    ax.set_title("90° 힌지 수평 반력 추이 비교")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")

    # Ax 3: Layerwise Mises Bar Chart
    ax = axes[2]
    layers = ["GLASS_Top", "PET_6", "PET_5", "PET_4", "PET_3", "PET_2", "PET_1"]
    y_pos = np.arange(len(layers))

    base_layers = {
        "GLASS_Top": 1272.12, "PET_6": 142.47, "PET_5": 192.97, "PET_4": 242.14, "PET_3": 288.92, "PET_2": 339.26, "PET_1": 399.37
    }
    m_base = [base_layers[l] for l in layers]
    m_ref = [conf_ref.get("layers", {}).get(l, {}).get("max_mises", 0.0) for l in layers]
    m_pps = [conf_pps.get("layers", {}).get(l, {}).get("max_mises", 0.0) for l in layers]

    w = 0.25
    ax.barh(y_pos - w, m_base, w, label="100% Free-Z (nx=30)", color="#0284c7", alpha=0.85)
    ax.barh(y_pos,     m_ref,  w, label="55% Free-Z (nx=55)",  color="#8b5cf6", alpha=0.85)
    ax.barh(y_pos + w, m_pps,  w, label="55% Pure-PS (all uz=0)", color="#10b981", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(layers, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("정점부 Mises 응력 [MPa]")
    ax.set_title("강성층(Glass/PET) 정점부 Mises 응력 비교")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[*] Pure Plane Strain comparison plot saved to: {out_png}")

    # Copy to brain
    brain_dst = os.path.join(r"C:\Users\GOODMAN\.gemini\antigravity-cli\brain\6e0be6af-fd3f-4b54-b349-ef02946cf0c2", "fistr_conforming_90deg_pure_plane_strain_comparison.png")
    import shutil
    shutil.copyfile(out_png, brain_dst)
    print(f"[*] Copied to brain dir: {brain_dst}")

if __name__ == "__main__":
    main()
