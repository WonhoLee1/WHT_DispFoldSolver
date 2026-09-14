"""
figures_macneal_harder.py
=========================
Publication-quality comparison plots for MacNeal-Harder (1985) benchmark results.
Complies with project rules:
  - koreanize-matplotlib
  - global font size = 9pt
  - tight_layout()
  - clean multi-panel layout
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import koreanize_matplotlib
except ImportError:
    pass

plt.rcParams["font.size"] = 9.0

DEV_LOG_FIGURES_DIR = Path("dev_log/figures")
DEV_LOG_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_macneal_harder_figures(json_path: str = "benchmark_element/results/macneal_harder_results.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tb_data = data["twisted_beam_3d"]
    c3d_data = data["cooks_membrane_3d"]
    c2d_data = data["cooks_membrane_2d"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -------------------------------------------------------------
    # Panel 1: MacNeal-Harder 3D Twisted Beam
    # -------------------------------------------------------------
    elems_3d = list(tb_data.keys())
    x = np.arange(len(elems_3d))
    width = 0.38

    ratios_oop = []
    ratios_ip = []
    for el in elems_3d:
        r_oop = tb_data[el]["out_of_plane"]["ratio"] * 100.0 if tb_data[el]["out_of_plane"]["status"] == "PASS" else 0.0
        r_ip = tb_data[el]["in_plane"]["ratio"] * 100.0 if tb_data[el]["in_plane"]["status"] == "PASS" else 0.0
        ratios_oop.append(r_oop)
        ratios_ip.append(r_ip)

    rects1 = ax1.bar(x - width / 2, ratios_oop, width, label="면외 굽힘 (Out-of-Plane $u_z$)", color="#3498db")
    rects2 = ax1.bar(x + width / 2, ratios_ip, width, label="면내 굽힘 (In-Plane $u_y$)", color="#2ecc71")

    ax1.axhline(100.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="이론 정해 (100% Target)")
    ax1.axhline(90.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax1.set_ylabel("타겟 이론해 대비 정밀도 (%)")
    ax1.set_title("[MacNeal-Harder 3D] 비틀린 보(Twisted Beam) 요소별 정밀도 대조", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(elems_3d, rotation=35, ha="right")
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper left")

    # Annotate top performers (C3D8R, C3D10M)
    for i, el in enumerate(elems_3d):
        if el in ("C3D8R", "C3D10M"):
            ax1.annotate(f"{ratios_oop[i]:.1f}%", (x[i] - width / 2, ratios_oop[i]),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, fontweight="bold")
            ax1.annotate(f"{ratios_ip[i]:.1f}%", (x[i] + width / 2, ratios_ip[i]),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 2: Cook's Membrane (2D vs 3D)
    # -------------------------------------------------------------
    elems_2d = list(c2d_data.keys())
    x2 = np.arange(len(elems_2d))
    ratios_2d = [c2d_data[el]["ratio"] * 100.0 if c2d_data[el]["status"] == "PASS" else 0.0 for el in elems_2d]

    colors_2d = ["#9b59b6" if "6" in el or "8" in el or "I" in el or "FBAR" in el else "#95a5a6" for el in elems_2d]
    bars2 = ax2.bar(x2, ratios_2d, width=0.55, color=colors_2d, edgecolor="#2c3e50")

    ax2.axhline(100.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="수렴 레퍼런스 해 (23.91 mm)")
    ax2.set_ylabel("레퍼런스 해 대비 비율 (%)")
    ax2.set_title("[Cook's Membrane 2D] 사다리꼴 전단 굽힘 메쉬 왜곡 감도", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(elems_2d, rotation=35, ha="right")
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper left")

    for i, rect in enumerate(bars2):
        h = rect.get_height()
        if h > 85.0:
            ax2.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2, h),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out_png = DEV_LOG_FIGURES_DIR / "benchmark_macneal_harder.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[+] Figure saved to {out_png}")


if __name__ == "__main__":
    generate_macneal_harder_figures()
