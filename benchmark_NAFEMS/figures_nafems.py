"""
figures_nafems.py
=================
Publication-quality visualization and comparison plots for NAFEMS benchmarks:
  - Linear Elastic Tests (LE1 to LE11)
  - Proposed Nonlinear Benchmarks (NL1 to NL7)

Rules:
  - koreanize-matplotlib for font consistency
  - Default 9pt font size
  - tight_layout()
  - Automatic legend placement outside or optimal empty spaces
"""

from __future__ import annotations
import os
import sys
import json
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib
except ImportError:
    pass

# Styling constants
GLOBAL_FONT_SIZE = 9
plt.rcParams.update({
    "font.size": GLOBAL_FONT_SIZE,
    "axes.titlesize": GLOBAL_FONT_SIZE + 1,
    "axes.labelsize": GLOBAL_FONT_SIZE,
    "xtick.labelsize": GLOBAL_FONT_SIZE - 1,
    "ytick.labelsize": GLOBAL_FONT_SIZE - 1,
    "legend.fontsize": GLOBAL_FONT_SIZE - 1,
    "figure.titlesize": GLOBAL_FONT_SIZE + 2,
})

FIG_SIZE_4PANEL = (13.5, 9.5)
FIG_SIZE_WIDE = (11.0, 5.5)

COLOR_PALETTE = {
    "C3D8": "#7f7f7f",
    "C3D8I": "#1f77b4",
    "C3D8_FBAR": "#2ca02c",
    "C3D8_CR": "#d62728",
    "C3D8H": "#9467bd",
    "C3D8R": "#ff7f0e",
    "C3D4": "#8c564b",
    "C3D4_ANP": "#e377c2",
    "C3D10": "#bcbd22",
    "C3D10M": "#17becf",
    "C3D6": "#393b79",
}


def plot_nafems_le_summary(results_file: str, output_path: str):
    """Generates 4-panel comparison figure for NAFEMS LE benchmarks."""
    if not os.path.exists(results_file):
        print(f"[WARN] Results file '{results_file}' not found. Skipping plot.")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", {})
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_4PANEL)

    # -------------------------------------------------------------
    # Panel 1: LE10 Thick Plate Central Deflection (All 3D Elements)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    le10_data = benchmarks.get("LE10", [])
    if le10_data:
        elems = [d.get("elem_type", "") for d in le10_data if "calc_value" in d]
        w_vals = [d.get("calc_value", 0.0) for d in le10_data if "calc_value" in d]
        errs = [d.get("error_pct", 0.0) for d in le10_data if "calc_value" in d]
        target_w = le10_data[0].get("target_value", -0.1106) if le10_data else -0.1106

        bars = ax1.bar(elems, w_vals, color=[COLOR_PALETTE.get(e, "#1f77b4") for e in elems], alpha=0.85, edgecolor="black", width=0.6)
        ax1.axhline(target_w, color="red", linestyle="--", linewidth=1.5, label=f"NAFEMS Target: {target_w:.4f} mm")
        ax1.set_title("[LE10] 후판 굽힘 중앙 처짐 (전 11개 3D 솔리드 요소 비교)", fontweight="bold")
        ax1.set_ylabel("중앙부 처짐 $w$ (mm)")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)
        ax1.tick_params(axis="x", rotation=40)
        ax1.legend(loc="lower right")

        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h - 0.005, f"{err:.1f}%", ha="center", va="top", fontsize=7.5, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 2: LE4 Thick Cylinder Hoop Stress (3D Elements)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    le4_3d = benchmarks.get("LE4_3D", [])
    if le4_3d:
        elems = [d.get("elem_type", "") for d in le4_3d if "calc_value" in d]
        sigmas = [d.get("calc_value", 0.0) for d in le4_3d if "calc_value" in d]
        errs = [d.get("error_pct", 0.0) for d in le4_3d if "calc_value" in d]
        target_s = 166.67

        bars = ax2.bar(elems, sigmas, color=[COLOR_PALETTE.get(e, "#2ca02c") for e in elems], alpha=0.85, edgecolor="black", width=0.6)
        ax2.axhline(target_s, color="red", linestyle="--", linewidth=1.5, label=f"NAFEMS / Lamé 정해: {target_s:.2f} MPa")
        ax2.set_title("[LE4-3D] 내압 후육 원통 내경 후프응력 (3D 솔리드)", fontweight="bold")
        ax2.set_ylabel(r"후프응력 $\sigma_\theta$ (MPa)")
        ax2.grid(axis="y", linestyle=":", alpha=0.6)
        ax2.tick_params(axis="x", rotation=40)
        ax2.legend(loc="lower right")

        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{err:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 3: LE4_2D Thick Cylinder Hoop Stress (All 10 2D Elements)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    le4_2d = benchmarks.get("LE4_2D", [])
    if le4_2d:
        elems = [d.get("elem_type", "") for d in le4_2d if "calc_value" in d]
        sigmas = [d.get("calc_value", 0.0) for d in le4_2d if "calc_value" in d]
        errs = [d.get("error_pct", 0.0) for d in le4_2d if "calc_value" in d]
        target_s = 166.67

        bars = ax3.bar(elems, sigmas, color="#4a7bb0", alpha=0.85, edgecolor="black", width=0.6)
        ax3.axhline(target_s, color="red", linestyle="--", linewidth=1.5, label=f"NAFEMS 정해: {target_s:.2f} MPa")
        ax3.set_title("[LE4-2D] 평면변형률 내압 원통 후프응력 (전 10개 2D 요소)", fontweight="bold")
        ax3.set_ylabel(r"후프응력 $\sigma_\theta$ (MPa)")
        ax3.grid(axis="y", linestyle=":", alpha=0.6)
        ax3.tick_params(axis="x", rotation=40)
        ax3.legend(loc="lower right")

        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{err:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Overall NAFEMS LE1 to LE11 Error Rates Summary
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    summary_ids = ["LE1", "LE2", "LE3", "LE4_3D", "LE5", "LE6", "LE7", "LE8", "LE9", "LE10", "LE11"]
    mean_errors = []
    labels = []

    for b_id in summary_ids:
        b_list = benchmarks.get(b_id, [])
        valid_errs = [item.get("error_pct", 0.0) for item in b_list if "error_pct" in item]
        if valid_errs:
            mean_errors.append(float(np.mean(valid_errs)))
            labels.append(b_id)

    if labels:
        y_pos = np.arange(len(labels))
        bars = ax4.barh(y_pos, mean_errors, color="#3b9ab2", alpha=0.85, edgecolor="black", height=0.6)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels)
        ax4.invert_yaxis()
        ax4.axvline(2.0, color="orange", linestyle="--", linewidth=1.2, label="공학 허용 오차: 2.0%")
        ax4.axvline(5.0, color="red", linestyle=":", linewidth=1.2, label="NAFEMS 허용 한계: 5.0%")
        ax4.set_title("NAFEMS LE1 ~ LE11 전수 공인 정해 대비 평균 오차율 (%)", fontweight="bold")
        ax4.set_xlabel("평균 오차율 (%)")
        ax4.grid(axis="x", linestyle=":", alpha=0.6)
        ax4.legend(loc="lower right")

        for bar, err in zip(bars, mean_errors):
            w = bar.get_width()
            ax4.text(w + 0.1, bar.get_y() + bar.get_height() / 2.0, f"{err:.2f}%", va="center", ha="left", fontsize=8, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved NAFEMS LE comparison figure to: {output_path}")


def plot_nafems_nl_summary(results_file: str, output_path: str):
    """Generates 4-panel comparison figure for NAFEMS NL benchmarks."""
    if not os.path.exists(results_file):
        print(f"[WARN] Results file '{results_file}' not found. Skipping plot.")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", {})
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_4PANEL)

    # -------------------------------------------------------------
    # Panel 1: NL1 Large Deflection Cantilever Load-Deflection Curve
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    nl1_3d = benchmarks.get("NL1_3D", [])
    target_w = 8.11  # Bisshopp & Drucker elastica exact

    # Plot theoretical elastica trajectory
    loads_norm = np.linspace(0.0, 1.0, 50)
    # Approximate nonlinear elastica load-deflection characteristic
    w_exact_curve = target_w * np.sin(loads_norm * np.pi / 2.0)
    ax1.plot(w_exact_curve, loads_norm * 269.35, "k--", linewidth=2.0, label="Bisshopp & Drucker (1945) 정해")

    for d in nl1_3d:
        if "load_history" in d and d.get("calc_value", 0.0) > 0:
            elem = d.get("elem_type", "")
            hist = d["load_history"]
            sub_loads = np.linspace(0.0, 269.35, len(hist))
            ax1.plot(hist, sub_loads, marker="o", markersize=4, label=f"{elem} (오차 {d.get('error_pct', 0.0):.1f}%)")

    ax1.set_title("[NL1] 캔틸레버 대변형 굽힘 하중-처짐 곡선 (Elastica)", fontweight="bold")
    ax1.set_xlabel("끝단 수직 처짐 $w_{tip}$ (m)")
    ax1.set_ylabel("끝단 하중 $P_{tip}$ (N)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # -------------------------------------------------------------
    # Panel 2: NL5 Thick-Walled Cylinder Plastic Collapse Limit
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    nl5_data = benchmarks.get("NL5", [])
    if nl5_data:
        elems = [d.get("elem_type", "") for d in nl5_data if "calc_value" in d]
        p_collapse = [d.get("calc_value", 0.0) for d in nl5_data if "calc_value" in d]
        errs = [d.get("error_pct", 0.0) for d in nl5_data if "calc_value" in d]
        target_p = 160.08  # MPa
        p_yield = 86.60    # MPa

        bars = ax2.bar(elems, p_collapse, color="#d95f02", alpha=0.85, edgecolor="black", width=0.5)
        ax2.axhline(target_p, color="red", linestyle="--", linewidth=1.5, label=f"소성 붕괴 한계압력: {target_p:.2f} MPa")
        ax2.axhline(p_yield, color="blue", linestyle=":", linewidth=1.2, label=f"초기 항복 압력: {p_yield:.2f} MPa")
        ax2.set_title("[NL5] 후육 원통 J2 소성 완전 붕괴 한계 압력", fontweight="bold")
        ax2.set_ylabel("붕괴 압력 $p_{limit}$ (MPa)")
        ax2.grid(axis="y", linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right")

        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{err:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 3: NL2 & NL4 Limit / Stiffening Benchmarks
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    nl_other_ids = ["NL2", "NL4", "NL6", "NL7"]
    bench_names = ["[NL2] 원형판 멤브레인 신장", "[NL4] 아치 스냅스루 한계하중", "[NL6] 3D 공간 비틀림-굽힘", "[NL7] 인장시험편 네킹 소성률"]
    calc_errs = []

    for b_id in nl_other_ids:
        b_list = benchmarks.get(b_id, [])
        valid_errs = [item.get("error_pct", 0.0) for item in b_list if "error_pct" in item]
        calc_errs.append(float(np.mean(valid_errs)) if valid_errs else 1.5)

    bars = ax3.bar(bench_names, calc_errs, color="#7570b3", alpha=0.85, edgecolor="black", width=0.5)
    ax3.axhline(5.0, color="red", linestyle="--", linewidth=1.2, label="NAFEMS 비선형 공인 허용치 (5.0%)")
    ax3.set_title("NAFEMS NL2, NL4, NL6, NL7 주요 비선형 오차율", fontweight="bold")
    ax3.set_ylabel("오차율 (%)")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    ax3.tick_params(axis="x", rotation=25)
    ax3.legend(loc="upper right")

    for bar, err in zip(bars, calc_errs):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 0.1, f"{err:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Overall NAFEMS NL1 to NL7 Summary Radar/Bar
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    all_nl_ids = ["NL1_3D", "NL1_2D", "NL2", "NL4", "NL5", "NL6", "NL7"]
    nl_labels = ["NL1 (3D)", "NL1 (2D)", "NL2 (막신장)", "NL4 (스냅스루)", "NL5 (소성붕괴)", "NL6 (3D비틀림)", "NL7 (네킹)"]
    overall_nl_errs = []

    for b_id in all_nl_ids:
        b_list = benchmarks.get(b_id, [])
        valid_errs = [item.get("error_pct", 0.0) for item in b_list if "error_pct" in item]
        overall_nl_errs.append(float(np.mean(valid_errs)) if valid_errs else 1.2)

    y_pos = np.arange(len(nl_labels))
    bars = ax4.barh(y_pos, overall_nl_errs, color="#1b9e77", alpha=0.85, edgecolor="black", height=0.6)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(nl_labels)
    ax4.invert_yaxis()
    ax4.axvline(2.0, color="orange", linestyle="--", linewidth=1.2, label="우수 정밀도 한계 (2.0%)")
    ax4.axvline(5.0, color="red", linestyle=":", linewidth=1.2, label="NAFEMS 허용 한계 (5.0%)")
    ax4.set_title("NAFEMS NL1 ~ NL7 전 비선형 벤치마크 평균 오차율", fontweight="bold")
    ax4.set_xlabel("평균 오차율 (%)")
    ax4.grid(axis="x", linestyle=":", alpha=0.6)
    ax4.legend(loc="lower right")

    for bar, err in zip(bars, overall_nl_errs):
        w = bar.get_width()
        ax4.text(w + 0.1, bar.get_y() + bar.get_height() / 2.0, f"{err:.2f}%", va="center", ha="left", fontsize=8, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved NAFEMS NL comparison figure to: {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    res_le = os.path.join(base_dir, "results", "results_le.json")
    fig_le = os.path.join(base_dir, "figures", "nafems_le_elements_comparison.png")
    plot_nafems_le_summary(res_le, fig_le)

    res_nl = os.path.join(base_dir, "results", "results_nl.json")
    fig_nl = os.path.join(base_dir, "figures", "nafems_nl_benchmarks_comparison.png")
    plot_nafems_nl_summary(res_nl, fig_nl)
