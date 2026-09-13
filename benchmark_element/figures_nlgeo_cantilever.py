"""
figures_nlgeo_cantilever.py
===========================
Generates publication-quality figures reproducing the 6 official figures of
Abaqus benchmark `simabmk-c-nlgeocantilever`:
  - Figure 1: Displacement plots for coarse mesh with transverse loading
  - Figure 2: Displacement plots for fine mesh with transverse loading
  - Figure 3: Displacement history of tip of coarse mesh (vs Bisshopp & Drucker 1945)
  - Figure 4: Displacement history of tip of fine mesh (vs Bisshopp & Drucker 1945)
  - Figure 5: Displacement plots of cantilever with moment loading (intermediate increments)
  - Figure 6: Displacement plots of cantilever with moment loading (final 2-turn circle loop)
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.titlesize"] = 12


def generate_nlgeo_cantilever_figures(
    results_coarse: Dict[str, Any],
    results_fine: Dict[str, Any],
    results_moment: Dict[str, Any],
    exact_elastica: Dict[str, Any],
    output_dir: str = "dev_log/figures"
) -> str:
    """Generate 6-panel comprehensive publication-quality figure matching Abaqus manual.
    
    Parameters:
        results_coarse: dict of element_name -> dict(p_history, uy_history, ux_history, profile_x, profile_y)
        results_fine: dict of element_name -> dict(p_history, uy_history, ux_history, profile_x, profile_y)
        results_moment: dict of element_name -> dict(increments: list of profiles at theta=pi, 2pi, 3pi, 4pi)
        exact_elastica: dict(L, H, EI, P, y_tip, x_tip, theta_tip, p_curve, uy_curve, ux_curve, exact_ring_R)
        output_dir: directory to save output PNG
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / "simabmk_nlgeocantilever_fig1_to_fig6.png"

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    color_map = {
        # 2D elements
        "CPE4": "#7f7f7f",
        "CPE4I": "#1f77b4",
        "CPE4R": "#aec7e8",
        "CPE4H": "#17becf",
        "CPE4_FBAR": "#9edae5",
        "CPE4_CR": "#393b79",
        "CPE3": "#d62728",
        "CPE6": "#2ca02c",
        "CPE6M": "#98df8a",
        "CPE8": "#ff7f0e",
        # 3D elements
        "C3D8": "#c7c7c7",
        "C3D8I": "#2b5c8f",
        "C3D8R": "#4b9cd3",
        "C3D8H": "#008080",
        "C3D8_FBAR": "#20b2aa",
        "C3D8_CR": "#5254a3",
        "C3D4": "#ff9896",
        "C3D4_ANP": "#e377c2",
        "C3D10": "#bcbd22",
        "C3D10M": "#8c564b",
        "C3D6": "#e7ba52",
    }

    # ----------------------------------------------------
    # Figure 1: Coarse Mesh Transverse Deformed Profiles
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_title("Figure 1. Coarse Mesh Deformed Profiles (P = 269.35 N)", fontweight="bold")
    ax1.set_xlabel("X 좌표 (m)")
    ax1.set_ylabel("Y 좌표 (m)")
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 10.0)
    ax1.set_aspect("equal", "box")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Initial undeformed beam reference
    L_ref = exact_elastica.get("L", 10.0)
    ax1.plot([0, L_ref], [0, 0], "k--", linewidth=1.2, label="Undeformed (초기형상)", alpha=0.7)

    for elem_name, res in results_coarse.items():
        if "profile_x" in res and "profile_y" in res and len(res["profile_x"]) > 0:
            c = color_map.get(elem_name, "#333333")
            lw = 2.0 if ("I" in elem_name or "8" in elem_name or "6" in elem_name or "10" in elem_name) else 1.2
            ax1.plot(res["profile_x"], res["profile_y"], label=elem_name, color=c, linewidth=lw, alpha=0.85)

    ax1.legend(loc="upper right", ncol=3, fontsize=6.5, framealpha=0.85)

    # ----------------------------------------------------
    # Figure 2: Fine Mesh Transverse Deformed Profiles
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_title("Figure 2. Fine Mesh Deformed Profiles (P = 269.35 N)", fontweight="bold")
    ax2.set_xlabel("X 좌표 (m)")
    ax2.set_ylabel("Y 좌표 (m)")
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(-0.5, 10.0)
    ax2.set_aspect("equal", "box")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax2.plot([0, L_ref], [0, 0], "k--", linewidth=1.2, label="Undeformed (초기형상)", alpha=0.7)

    for elem_name, res in results_fine.items():
        if "profile_x" in res and "profile_y" in res and len(res["profile_x"]) > 0:
            c = color_map.get(elem_name, "#333333")
            lw = 2.0 if ("I" in elem_name or "8" in elem_name or "6" in elem_name or "10" in elem_name) else 1.2
            ax2.plot(res["profile_x"], res["profile_y"], label=elem_name, color=c, linewidth=lw, alpha=0.85)

    ax2.legend(loc="upper right", ncol=3, fontsize=6.5, framealpha=0.85)

    # ----------------------------------------------------
    # Figure 3: Coarse Mesh Tip Displacement History
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_title("Figure 3. Coarse Mesh Tip Displacement History (vs Bisshopp & Drucker)", fontweight="bold")
    ax3.set_xlabel("연직 하중 분율 (P / P_total)")
    ax3.set_ylabel("끝단 연직 처짐 |uy| (m)")
    ax3.set_xlim(0.0, 1.05)
    ax3.set_ylim(-0.2, 9.0)
    ax3.grid(True, linestyle="--", alpha=0.6)

    if "p_curve" in exact_elastica and "uy_curve" in exact_elastica:
        ax3.plot(exact_elastica["p_curve"], exact_elastica["uy_curve"], "k-", linewidth=2.5, label="Bisshopp & Drucker (1945) Exact |uy|", zorder=10)

    for elem_name, res in results_coarse.items():
        if "p_history" in res and "uy_history" in res and len(res["p_history"]) > 0:
            c = color_map.get(elem_name, "#333333")
            ls = "-" if not ("CPE4" == elem_name or "C3D8" == elem_name or "CPE3" == elem_name) else "--"
            ax3.plot(res["p_history"], np.abs(res["uy_history"]), label=f"{elem_name} uy", color=c, linestyle=ls, linewidth=1.3, alpha=0.8)

    ax3.legend(loc="upper left", ncol=3, fontsize=6.5, framealpha=0.85)

    # ----------------------------------------------------
    # Figure 4: Fine Mesh Tip Displacement History
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_title("Figure 4. Fine Mesh Tip Displacement History (vs Bisshopp & Drucker)", fontweight="bold")
    ax4.set_xlabel("연직 하중 분율 (P / P_total)")
    ax4.set_ylabel("끝단 연직 처짐 |uy| (m)")
    ax4.set_xlim(0.0, 1.05)
    ax4.set_ylim(-0.2, 9.0)
    ax4.grid(True, linestyle="--", alpha=0.6)

    if "p_curve" in exact_elastica and "uy_curve" in exact_elastica:
        ax4.plot(exact_elastica["p_curve"], exact_elastica["uy_curve"], "k-", linewidth=2.5, label="Bisshopp & Drucker (1945) Exact |uy|", zorder=10)

    for elem_name, res in results_fine.items():
        if "p_history" in res and "uy_history" in res and len(res["p_history"]) > 0:
            c = color_map.get(elem_name, "#333333")
            ls = "-" if not ("CPE4" == elem_name or "C3D8" == elem_name or "CPE3" == elem_name) else "--"
            ax4.plot(res["p_history"], np.abs(res["uy_history"]), label=f"{elem_name} uy", color=c, linestyle=ls, linewidth=1.3, alpha=0.8)

    ax4.legend(loc="upper left", ncol=3, fontsize=6.5, framealpha=0.85)

    # ----------------------------------------------------
    # Figure 5: Moment Loading Intermediate Increments (theta = pi..4pi)
    # ----------------------------------------------------
    ax5 = axes[2, 0]
    ax5.set_title("Figure 5. Moment Loading Intermediate Profiles (theta = pi, 2pi, 3pi, 4pi)", fontweight="bold")
    ax5.set_xlabel("X 좌표 (m)")
    ax5.set_ylabel("Y 좌표 (m)")
    ax5.set_xlim(-1.0, 11.0)
    ax5.set_ylim(-1.0, 7.5)
    ax5.set_aspect("equal", "box")
    ax5.grid(True, linestyle="--", alpha=0.6)

    rep_keys = ["CPE4I", "CPE8", "C3D8I", "C3D8R"]
    angles = ["pi", "2pi", "3pi", "4pi"]
    styles = ["-", "--", "-.", ":"]

    for elem_name in rep_keys:
        if elem_name in results_moment and "stages" in results_moment[elem_name]:
            c = color_map.get(elem_name, "#1f77b4")
            stages = results_moment[elem_name]["stages"]
            for s_idx, ang in enumerate(angles):
                if ang in stages:
                    sx, sy = stages[ang]
                    lbl = f"{elem_name} ({ang})" if elem_name == rep_keys[0] else None
                    ax5.plot(sx, sy, linestyle=styles[s_idx % len(styles)], color=c, label=lbl, alpha=0.85, linewidth=1.5)

    ax5.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.85)

    # ----------------------------------------------------
    # Figure 6: Final 2-Turn Roll-up Circle (theta = 4pi) vs Theory
    # ----------------------------------------------------
    ax6 = axes[2, 1]
    ax6.set_title("Figure 6. Final 2-Turn Roll-up Circle (theta = 4pi, R = L / 4pi)", fontweight="bold")
    ax6.set_xlabel("X 좌표 (m)")
    ax6.set_ylabel("Y 좌표 (m)")
    ax6.set_xlim(-1.0, 11.0)
    ax6.set_ylim(-1.0, 3.5)
    ax6.set_aspect("equal", "box")
    ax6.grid(True, linestyle="--", alpha=0.6)

    R_exact = L_ref / (4.0 * np.pi)
    phi = np.linspace(0, 4.0 * np.pi, 200)
    cx_exact = R_exact * np.sin(phi)
    cy_exact = R_exact * (1.0 - np.cos(phi))
    ax6.plot(cx_exact, cy_exact, "k-", linewidth=2.5, label=f"Exact 2-Turn Circle (R={R_exact:.3f}m)", zorder=10)

    for elem_name, res in results_moment.items():
        if "final_x" in res and "final_y" in res:
            c = color_map.get(elem_name, "#333333")
            lw = 1.8 if ("I" in elem_name or "8" in elem_name or "6" in elem_name or "10" in elem_name) else 1.2
            ax6.plot(res["final_x"], res["final_y"], label=elem_name, color=c, linewidth=lw, alpha=0.85)

    ax6.legend(loc="upper right", ncol=3, fontsize=7, framealpha=0.85)

    fig.suptitle("Abaqus Official Benchmark SIMACAEBMKRefMap/simabmk-c-nlgeocantilever (2D & 3D Elements)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(str(save_file), dpi=200)
    plt.close(fig)

    print(f"[figures_nlgeo_cantilever] Successfully generated 6-panel figure: {save_file}")
    return str(save_file)
