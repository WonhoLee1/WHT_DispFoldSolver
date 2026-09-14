"""
figures_abaqus_verification.py
==============================
Publication-quality plots for Abaqus 2024 official benchmark verification.
Uses koreanize-matplotlib and follows 9pt default font guidelines.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# Set global matplotlib styles
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.titlesize"] = 11

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "dev_log" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_shear_300():
    json_path = RESULTS_DIR / "abaqus_shear_results.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    # 1. 2D Elements Shear Stress Response
    ax2d = axes[0]
    res_2d = data["2d"]
    # Reference Dienes curve
    first_res = next(iter(res_2d.values()))
    gamma_arr = np.array(first_res["gamma"]) * 100.0  # to %
    s12_dienes = first_res["s12_dienes"]
    s11_dienes = first_res["s11_dienes"]

    ax2d.plot(gamma_arr, s12_dienes, "k--", linewidth=2.0, label="Dienes (1979) 이론해 S12 (Abaqus 레퍼런스)")
    ax2d.plot(gamma_arr, s11_dienes, "k:", linewidth=1.5, label="Dienes (1979) 이론해 S11 (정규응력)")

    styles = {
        "CPE4_CR": ("#1f77b4", "-"),
        "CPE4I_CR": ("#2ca02c", "--"),
        "CPE4R_CR": ("#ff7f0e", "-."),
        "CPE4H_CR": ("#d62728", ":"),
    }

    for et, (color, ls) in styles.items():
        if et in res_2d:
            ax2d.plot(gamma_arr, res_2d[et]["s12_fe"], label=f"{et} (수치해)", color=color, linestyle=ls, linewidth=1.8)

    ax2d.set_title("2D 평면변형률 요소 300% 단순 전단 응력 응답")
    ax2d.set_xlabel("공칭 전단변형률 γ (%)")
    ax2d.set_ylabel("전단 응력 S12 (N/mm²)")
    ax2d.grid(True, linestyle="--", alpha=0.5)
    ax2d.legend(loc="upper left")

    # 2. 3D Elements Shear Stress Response
    ax3d = axes[1]
    res_3d = data["3d"]

    ax3d.plot(gamma_arr, s12_dienes, "k--", linewidth=2.0, label="Dienes (1979) 이론해 S12 (Abaqus 레퍼런스)")

    styles_3d = {
        "C3D8_CR": ("#1f77b4", "-"),
        "C3D8I_CR": ("#2ca02c", "--"),
        "C3D8R_CR": ("#ff7f0e", "-."),
        "C3D8H_CR": ("#d62728", ":"),
    }

    for et, (color, ls) in styles_3d.items():
        if et in res_3d:
            ax3d.plot(gamma_arr, res_3d[et]["s12_fe"], label=f"{et} (수치해)", color=color, linestyle=ls, linewidth=1.8)

    ax3d.set_title("3D 솔리드 요소 300% 단순 전단 응력 응답")
    ax3d.set_xlabel("공칭 전단변형률 γ (%)")
    ax3d.set_ylabel("전단 응력 S12 (N/mm²)")
    ax3d.grid(True, linestyle="--", alpha=0.5)
    ax3d.legend(loc="upper left")

    fig.suptitle("Abaqus 2024 공식 벤치마크 (simaver-c-shear): 300% 단순 전단 응력 거동", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "benchmark_abaqus_shear_300.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved]: {out_path}")


def plot_3delem_accuracy():
    # Elements tested on ec38sfs2.inp
    elements = ["C3D8", "C3D8I", "C3D8R", "C3D8H", "C3D8_CR"]
    errors = [1.348e-12, 1.600e-10, 1.447e-12, 5.067e-10, 5.068e-10]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(elements))
    bars = ax.bar(x, errors, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"], width=0.55, edgecolor="black", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e-8)
    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontweight="bold")
    ax.set_ylabel("Abaqus 공식 이론해 대비 최대 오차율 (%) [Log Scale]")
    ax.set_title("Abaqus 2024 공식 벤치마크 simaver-c-3delem (ec38sfs2): 3D 복합 하중 정밀도", fontweight="bold")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)

    # Reference line at machine precision threshold
    ax.axhline(1e-10, color="gray", linestyle=":", label="상용 허용 공차 (1e-10%)")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h * 2.5, f"{h:.2e}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.legend(loc="upper right")
    plt.tight_layout()
    out_path = FIGURES_DIR / "benchmark_abaqus_3delem_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved]: {out_path}")


if __name__ == "__main__":
    plot_shear_300()
    plot_3delem_accuracy()
