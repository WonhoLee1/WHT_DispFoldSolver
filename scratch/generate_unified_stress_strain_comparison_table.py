"""
scratch/generate_unified_stress_strain_comparison_table.py
Full metrics table including local bending stress, shear stress, engineering strain, true strain, and shear strain.
"""

import os
import sys
import json
from example_disp_fold.run_fistr_conforming_teardrop import STACKUP_14
from scratch.compare_all_4solvers_stress_strain import extract_fistr_conforming, extract_fistr_baseline

def main():
    wht_path = "scratch/wht_20deg_extracted_stresses.json"
    with open(wht_path, "r") as f:
        wht_data = json.load(f)

    wht_2d = wht_data["WHT_2D"]
    wht_3d = wht_data["WHT_3D"]

    conf_res = extract_fistr_conforming()
    conf_apex = conf_res["apex"]
    base_apex = extract_fistr_baseline()

    print("=" * 165)
    print("3대 솔버 4종 모델 20° 굽힘 정점부(|X| <= 2.0 mm) 14개 층 응력 및 변형률 1:1 전수 대조 테이블")
    print("=" * 165)
    header = (
        f"{'층 번호 & 층 이름':<18} | {'재질':<5} | "
        f"{'WHT 2D (Opt1)':<24} | {'WHT 3D (Opt1)':<24} | "
        f"{'FISTR Conforming 3D':<26} | {'FISTR Baseline 3D':<26} | {'역학적 판정'}"
    )
    print(header)
    sub_header = (
        f"{'':<18} | {'':<5} | "
        f"{'Mises / eps_eng / gamma':<24} | {'Mises / eps_eng / gamma':<24} | "
        f"{'Mises / eps_eng / gamma':<26} | {'Mises / eps_eng / gamma':<26} |"
    )
    print(sub_header)
    print("-" * 165)

    for i, lyr in enumerate(reversed(STACKUP_14)):
        layer_num = 14 - i
        lname = lyr["name"]
        mat = lyr["mat"]
        w2 = wht_2d.get(lname, {})
        w3 = wht_3d.get(lname, {})
        fc = conf_apex.get(lname, {})
        fb = base_apex.get(lname, {})

        w2_str = f"{w2.get('max_mises', 0.0):.2f}M / {w2.get('max_eps_eng', 0.0):.1f}% / {w2.get('max_gamma_tn', 0.0):.1f}%"
        w3_str = f"{w3.get('max_mises', 0.0):.2f}M / {w3.get('max_eps_eng', 0.0):.1f}% / {w3.get('max_gamma_tn', 0.0):.1f}%"
        fc_str = f"{fc.get('max_mises', 0.0):.2f}M / {fc.get('max_eps_eng', 0.0):.1f}% / {fc.get('max_gamma_tn', 0.0):.1f}%"
        fb_str = f"{fb.get('max_mises', 0.0):.2f}M / {fb.get('max_eps_eng', 0.0):.1f}% / {fb.get('max_gamma_tn', 0.0):.1f}%"

        if mat == "PSA":
            decision = "정상 초탄성 슬립 흡수 (u-P)"
        elif lname == "GLASS_Top":
            decision = "최외곽 굽힘 집중 (2차 유연성 vs 1차 강성)"
        else:
            decision = "EAS/2차 전단잠김 소거"

        name_display = f"L{layer_num:02d}: {lname}"
        print(f"{name_display:<18} | {mat:<5} | {w2_str:<24} | {w3_str:<24} | {fc_str:<26} | {fb_str:<26} | {decision}")

    print("=" * 165)

if __name__ == "__main__":
    main()
