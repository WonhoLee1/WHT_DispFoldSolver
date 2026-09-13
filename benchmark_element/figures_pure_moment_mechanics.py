"""
figures_pure_moment_mechanics.py
================================
Generates 4-panel publication-quality figures for Pure Bending Moment Roll-Up
Mechanics: Reaction Moment, Strain Energy, Interlayer Shear Stress, and Model Comparisons.

Panels:
(a) Bending Reaction Moment vs Roll-up Angle M_root(theta) [N*mm vs deg]
(b) Elastic Strain Energy vs Roll-up Angle U_strain(theta) [mJ vs deg]
(c) PSA Interlayer Shear Stress Profile along Beam Length tau_xy(x) [MPa vs mm]
(d) Final Reaction Moment and Strain Energy Comparison across all 9 Candidates
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# Set global font size
plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 300

repo_root = Path(__file__).resolve().parent.parent
results_json = repo_root / "benchmark_element" / "results" / "benchmark_pure_moment_rollup.json"
output_png = repo_root / "dev_log" / "figures" / "benchmark_pure_moment_mechanics.png"


def plot_rollup_mechanics():
    if not results_json.exists():
        print(f"Results file {results_json} not found. Run benchmark first.")
        return

    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Analytical Reference Limits
    # Monolithic upper bound (no-slip): ~0.2317 N*mm
    # Independent slip lower bound (free slip): ~0.0108 N*mm
    M_mono = 0.2317
    M_slip = 0.0108

    # Color Palette
    colors = {
        "2D-CR   (CPE4_CR+CPE4_CR)": "#008080",  # Teal
        "2D-Opt1 (CPE4I+CPE4H)":     "#1f77b4",  # Blue
        "2D-CPE6M (CPE6M+CPE4H)":   "#2ca02c",  # Green
        "2D-Opt2 (CPE4R+CPE4H)":     "#ff7f0e",  # Orange
        "2D-Base (CPE4+CPE4)":       "#7f7f7f",  # Gray
        "3D-CR   (C3D8_CR+C3D8_CR)": "#9467bd",  # Purple
        "3D-Opt1 (C3D8I+C3D8H)":     "#d62728",  # Red
        "3D-Opt2-H (C3D8R+C3D8H)":   "#e377c2",  # Pink
        "3D-Opt2 (C3D8R+C3D8R)":     "#8c564b",  # Brown
        "3D-Base (C3D8+C3D8)":       "#bcbd22",  # Olive
    }

    # ----------------------------------------------------
    # Panel (a): Moment vs Angle M_root(theta)
    # ----------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.axhline(M_mono, color="black", linestyle="--", linewidth=1.5, label=f"이론 상한선 (일체 굽힘 {M_mono:.3f} N·mm)")
    ax_a.axhline(M_slip, color="gray", linestyle=":", linewidth=1.5, label=f"이론 하한선 (완전 미끄럼 {M_slip:.3f} N·mm)")

    # Plot 2D cases
    for label, res in data["2D"].items():
        if "history_theta" in res and "history_M_root" in res:
            th = np.array(res["history_theta"])
            M = np.abs(np.array(res["history_M_root"]))
            short_lbl = label.split()[0]
            ax_a.plot(th, M, label=f"2D {short_lbl}", color=colors.get(label, "#333333"), linewidth=1.8)

    # Plot 3D Opt1, 3D Opt2-H, and 3D CR for comparison
    plot_3d_keys = ["3D-Opt1 (C3D8I+C3D8H)", "3D-Opt2-H (C3D8R+C3D8H)", "3D-CR   (C3D8_CR+C3D8_CR)"]
    for k in plot_3d_keys:
        if k in data["3D"]:
            res = data["3D"][k]
            if "history_theta" in res and "history_M_root" in res:
                th = np.array(res["history_theta"])
                M = np.abs(np.array(res["history_M_root"]))
                short_lbl = k.split()[0]
                ax_a.plot(th, M, label=f"3D {short_lbl}", color=colors.get(k, "#d62728"), linewidth=2.0, linestyle="--")

    # Add 2D-Opt1 Tip Moment curve for 1:1 equilibrium comparison
    if "2D-Opt1 (CPE4I+CPE4H)" in data["2D"]:
        res_opt1 = data["2D"]["2D-Opt1 (CPE4I+CPE4H)"]
        if "history_theta" in res_opt1 and "history_M_tip" in res_opt1:
            th = np.array(res_opt1["history_theta"])
            M_tip = np.abs(np.array(res_opt1["history_M_tip"]))
            ax_a.plot(th, M_tip, label="2D Opt1 (구동단 |M_tip|)", color="#1f77b4", linewidth=2.0, linestyle=":")

    ax_a.set_title("(a) 회전각에 따른 고정단/구동단 모멘트 |M| vs theta", fontweight="bold")
    ax_a.set_xlabel("끝단 회전각 theta [deg]")
    ax_a.set_ylabel("반력 모멘트 |M| [N·mm]")
    ax_a.set_xlim(0, 185)
    ax_a.set_ylim(0, 2.0)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="best", framealpha=0.9, fontsize=8)

    # ----------------------------------------------------
    # Panel (b): Elastic Strain Energy vs Angle U(theta)
    # ----------------------------------------------------
    ax_b = axes[0, 1]
    for label, res in data["2D"].items():
        if "history_theta" in res and "history_U_strain" in res:
            th = np.array(res["history_theta"])
            U = np.array(res["history_U_strain"])
            short_lbl = label.split()[0]
            ax_b.plot(th, U, label=f"2D {short_lbl}", color=colors.get(label, "#333333"), linewidth=1.8)

    for k in plot_3d_keys:
        if k in data["3D"]:
            res = data["3D"][k]
            if "history_theta" in res and "history_U_strain" in res:
                th = np.array(res["history_theta"])
                U = np.array(res["history_U_strain"])
                short_lbl = k.split()[0]
                ax_b.plot(th, U, label=f"3D {short_lbl}", color=colors.get(k, "#d62728"), linewidth=2.0, linestyle="--")

    ax_b.set_title("(b) 롤업 과정 중 저장된 탄성 변형 에너지 U_strain vs theta", fontweight="bold")
    ax_b.set_xlabel("끝단 회전각 theta [deg]")
    ax_b.set_ylabel("변형 에너지 U_strain [mJ]")
    ax_b.set_xlim(0, 185)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="best", framealpha=0.9, fontsize=8)

    # ----------------------------------------------------
    # Panel (c): PSA Shear Stress Profile tau_xy(x)
    # ----------------------------------------------------
    ax_c = axes[1, 0]
    for label, res in data["2D"].items():
        if "tau_x_coords" in res and len(res["tau_x_coords"]) > 0:
            x_pts = np.array(res["tau_x_coords"])
            tau = np.abs(np.array(res["tau_xy_lower"]))
            short_lbl = label.split()[0]
            ax_c.plot(x_pts, tau, label=f"2D {short_lbl} (하부 PSA)", color=colors.get(label, "#333333"), linewidth=1.8)

    for k in ["3D-Opt1 (C3D8I+C3D8H)", "3D-Opt2-H (C3D8R+C3D8H)"]:
        if k in data["3D"]:
            res_3d = data["3D"][k]
            if "tau_x_coords" in res_3d and len(res_3d["tau_x_coords"]) > 0:
                x_pts = np.array(res_3d["tau_x_coords"])
                tau = np.abs(np.array(res_3d["tau_xy_lower"]))
                short_lbl = k.split()[0]
                ax_c.plot(x_pts, tau, label=f"3D {short_lbl} (하부 PSA)", color=colors.get(k, "#d62728"), linewidth=2.0, linestyle="--")

    ax_c.set_title("(c) 180° 완주 후 PSA 층간 전단 응력 분포 |tau_xy(x)|", fontweight="bold")
    ax_c.set_xlabel("보 길이 방향 좌표 x [mm]")
    ax_c.set_ylabel("전단 응력 |tau_xy| [MPa]")
    ax_c.set_xlim(0, 40)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="best", framealpha=0.9, fontsize=8)

    # ----------------------------------------------------
    # Panel (d): 10 Candidates Final M_root, M_tip, and Energy Comparison
    # ----------------------------------------------------
    ax_d = axes[1, 1]
    all_names = []
    m_roots = []
    m_tips = []
    u_energies = []
    bar_colors = []

    for label, res in data["2D"].items():
        short_lbl = label.split()[0]
        all_names.append(short_lbl)
        m_roots.append(abs(res.get("final_M_root", 0.0)))
        m_tips.append(abs(res.get("final_M_tip", 0.0)))
        u_energies.append(res.get("final_U_strain_mJ", 0.0))
        bar_colors.append(colors.get(label, "#1f77b4"))

    for label, res in data["3D"].items():
        short_lbl = label.split()[0]
        all_names.append(short_lbl)
        m_roots.append(abs(res.get("final_M_root", 0.0)))
        m_tips.append(abs(res.get("final_M_tip", 0.0)))
        u_energies.append(res.get("final_U_strain_mJ", 0.0))
        bar_colors.append(colors.get(label, "#d62728"))

    x_indices = np.arange(len(all_names))
    width = 0.28

    ax_d1 = ax_d
    ax_d2 = ax_d.twinx()

    b1 = ax_d1.bar(x_indices - width, m_roots, width, label="고정단 |M_root| [N·mm]", color=bar_colors, alpha=0.9, edgecolor="black")
    b2 = ax_d1.bar(x_indices, m_tips, width, label="구동단 |M_tip| [N·mm]", color=bar_colors, alpha=0.45, edgecolor="black", hatch="//")
    b3 = ax_d2.bar(x_indices + width, u_energies, width, label="변형 에너지 U_strain [mJ]", color="gold", alpha=0.65, edgecolor="black", hatch="..")

    ax_d1.axhline(M_mono, color="black", linestyle="--", linewidth=1.2, label=f"이론 상한 M ({M_mono:.3f} N·mm)")

    ax_d1.set_title("(d) 10개 후보군 고정단/구동단 모멘트 및 변형 에너지 정량 대조", fontweight="bold")
    ax_d1.set_xticks(x_indices)
    ax_d1.set_xticklabels(all_names, rotation=30, ha="right", fontsize=8)
    ax_d1.set_ylabel("반력 모멘트 |M| [N·mm]", color="blue")
    ax_d2.set_ylabel("변형 에너지 U_strain [mJ]", color="goldenrod")
    ax_d1.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Combine legends
    lines1, labels1 = ax_d1.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d1.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Pure moment mechanics figure saved to {output_png}", flush=True)


if __name__ == "__main__":
    plot_rollup_mechanics()
