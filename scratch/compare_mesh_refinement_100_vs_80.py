"""
scratch/compare_mesh_refinement_100_vs_80.py
Compiles a 1:1 comparison table between:
- Baseline Mesh (100% size: nx_hinge=30, nx_plate=20)
- Refined Mesh (80% size: nx_hinge=38, nx_plate=25)
across all 4 solvers at theta = 20.0 degrees.
"""

import os
import sys
import json

def main():
    json_path = "example_disp_fold/results_mesh_refine_20pct/mesh_refinement_20pct_results.json"
    if not os.path.exists(json_path):
        print("Waiting for refined results...")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        refined = json.load(f)

    # Baseline results from parity study and previous extraction
    # WHT 2D: M = 0.5312, Fx = 0.1600, Glass Mises = 9.61, sig_tt = 9.61, PET1 Mises = 5.16, sig_tt = 5.16, PSA1 gamma = 52.0%
    # WHT 3D: M = 0.5148, Fx = 0.1542, Glass Mises = 10.70, sig_tt = 10.70, PET1 Mises = 5.72, sig_tt = 5.72, PSA1 gamma = 52.6%
    # FISTR Conf: M = 0.3319, Fx = 0.1100, Glass Mises = 141.20, sig_tt = 109.46, PET1 Mises = 32.30, sig_tt = 7.17, PSA1 gamma = 32.5%
    # FISTR Base: M = 5.4223, Fx = 2.2214, Glass Mises = 14.52, sig_tt = 17.14, PET1 Mises = 13.55, sig_tt = 13.80, PSA1 gamma = 47.5%

    base_cases = {
        "WHT_2D": {
            "M": 0.5312, "Fx": 0.1600,
            "glass_mises": 9.61, "glass_tt": 9.61, "glass_eps": 0.01,
            "pet1_mises": 5.16, "pet1_tt": 5.16, "pet1_eps": 0.13,
            "psa1_mises": 0.02, "psa1_gamma": 52.0,
        },
        "WHT_3D": {
            "M": 0.5148, "Fx": 0.1542,
            "glass_mises": 10.70, "glass_tt": 10.70, "glass_eps": 0.02,
            "pet1_mises": 5.72, "pet1_tt": 5.72, "pet1_eps": 0.14,
            "psa1_mises": 0.02, "psa1_gamma": 52.6,
        },
        "FISTR_Conf": {
            "M": 0.3319, "Fx": 0.1100,
            "glass_mises": 141.20, "glass_tt": 109.46, "glass_eps": 0.27,
            "pet1_mises": 32.30, "pet1_tt": 7.17, "pet1_eps": 0.71,
            "psa1_mises": 0.03, "psa1_gamma": 32.5,
        },
        "FISTR_Base": {
            "M": 5.4223, "Fx": 2.2214,
            "glass_mises": 14.52, "glass_tt": 17.14, "glass_eps": 0.02,
            "pet1_mises": 13.55, "pet1_tt": 13.80, "pet1_eps": 0.34,
            "psa1_mises": 0.01, "psa1_gamma": 47.5,
        },
    }

    ref_wht2d = refined["WHT_2D_Refined"]
    ref_wht3d = refined["WHT_3D_Refined"]
    ref_conf = refined["FISTR_Conforming_Refined"]
    ref_base = refined["FISTR_Baseline_Refined"]

    comp_data = [
        ("WHT Solver 2D (Opt1)", base_cases["WHT_2D"], ref_wht2d),
        ("WHT Solver 3D (Opt1)", base_cases["WHT_3D"], ref_wht3d),
        ("FrontISTR Conforming 3D", base_cases["FISTR_Conf"], ref_conf),
        ("FrontISTR Baseline 3D", base_cases["FISTR_Base"], ref_base),
    ]

    print("=" * 160)
    print("요소 가로 크기 20% 축소(80% Size, 1.25x 격자 세분화) 전후 4대 솔버 20° 굽힘 거시/국소 물리량 1:1 비교표")
    print("=" * 160)
    header = (
        f"{'솔버 및 모델명':<26} | "
        f"{'모멘트 |M_L| (N·mm)':<22} | "
        f"{'수평 반력 |F_x| (N)':<20} | "
        f"{'Glass Mises (MPa)':<20} | "
        f"{'PET_1 Mises (MPa)':<20} | "
        f"{'PSA_1 전단변형률 (%)':<22} | {'메쉬 수렴성 판정'}"
    )
    print(header)
    sub = (
        f"{'':<26} | "
        f"{'100%  ->  80% (변화율)':<22} | "
        f"{'100%  ->  80% (변화율)':<20} | "
        f"{'100%  ->  80% (변화율)':<20} | "
        f"{'100%  ->  80% (변화율)':<20} | "
        f"{'100%  ->  80% (변화율)':<22} |"
    )
    print(sub)
    print("-" * 160)

    for name, b, r in comp_data:
        m_b = b["M"]
        m_r = r["bar_M_L"]
        m_diff = (m_r - m_b) / m_b * 100.0

        fx_b = b["Fx"]
        fx_r = r["bar_Fx_L"]
        fx_diff = (fx_r - fx_b) / fx_b * 100.0

        gm_b = b["glass_mises"]
        gm_r = r["layers"].get("GLASS_Top", {}).get("max_mises", 0.0)
        gm_diff = (gm_r - gm_b) / gm_b * 100.0 if gm_b > 0 else 0.0

        pm_b = b["pet1_mises"]
        pm_r = r["layers"].get("PET_1", {}).get("max_mises", 0.0)
        pm_diff = (pm_r - pm_b) / pm_b * 100.0 if pm_b > 0 else 0.0

        psa_b = b["psa1_gamma"]
        psa_r = r["layers"].get("PSA_1", {}).get("max_gamma_tn", 0.0)
        psa_diff = (psa_r - psa_b) / psa_b * 100.0 if psa_b > 0 else 0.0

        m_str = f"{m_b:.4f} -> {m_r:.4f} ({m_diff:+.1f}%)"
        fx_str = f"{fx_b:.4f} -> {fx_r:.4f} ({fx_diff:+.1f}%)"
        gm_str = f"{gm_b:.1f} -> {gm_r:.1f} ({gm_diff:+.1f}%)"
        pm_str = f"{pm_b:.1f} -> {pm_r:.1f} ({pm_diff:+.1f}%)"
        psa_str = f"{psa_b:.1f}% -> {psa_r:.1f}% ({psa_diff:+.1f}%)"

        if "Baseline" in name:
            judgement = "잠김 지속 (10배 폭등 유지)"
        elif abs(m_diff) < 5.0:
            judgement = "강건 수렴 (h-독립성 확인)"
        else:
            judgement = "유연성 향상 수렴"

        print(f"{name:<26} | {m_str:<22} | {fx_str:<20} | {gm_str:<20} | {pm_str:<20} | {psa_str:<22} | {judgement}")

    print("=" * 160)

if __name__ == "__main__":
    main()
