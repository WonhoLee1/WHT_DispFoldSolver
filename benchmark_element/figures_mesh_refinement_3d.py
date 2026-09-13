"""
figures_mesh_refinement_3d.py
=============================
Publication-grade 4-panel visualizer for 3D mesh refinement and runtime scaling.
Adheres strictly to project visualization guidelines:
- koreanize-matplotlib for font consistency.
- Global 9pt font size.
- tight_layout() to prevent overlapping.
- High-contrast colors and clean legends.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.rcParams["font.size"] = 9.0
plt.rcParams["axes.labelsize"] = 9.0
plt.rcParams["xtick.labelsize"] = 8.0
plt.rcParams["ytick.labelsize"] = 8.0
plt.rcParams["legend.fontsize"] = 8.0
plt.rcParams["figure.titlesize"] = 11.0


def plot_mesh_refinement_figures(json_path: Path, output_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    th_cases = data["thickness_study"]
    w_cases = data["width_study_3d"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=300)
    fig.suptitle("3D Mesh Refinement & Runtime Scaling Study (C3D8I + C3D8H Conformal)", fontweight="bold")

    # -------------------------------------------------------------
    # Panel (a): Through-Thickness Convergence (Deflection)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_title("(a) Through-Thickness Refinement: Tip Deflection", fontweight="bold")
    nz_vals = [c["nz_per_layer"] for c in th_cases]
    defl_vals = [c["final_deflection_mm"] for c in th_cases]
    dofs_th = [c["num_dofs"] for c in th_cases]

    ax_a.plot(nz_vals, defl_vals, "o-", color="#1f77b4", lw=2.0, ms=7, label="Tip Deflection |w| [mm]")
    for x, y, dof in zip(nz_vals, defl_vals, dofs_th):
        ax_a.annotate(f"{y:.4f} mm\n({dof} DOFs)", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    ax_a.set_xlabel("Number of Elements per Layer (Thickness Direction)")
    ax_a.set_ylabel("Tip Deflection [mm]", color="#1f77b4", fontweight="bold")
    ax_a.set_xticks(nz_vals)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.set_ylim(min(defl_vals) * 0.95, max(defl_vals) * 1.05)
    ax_a.legend(loc="lower right")

    # -------------------------------------------------------------
    # Panel (b): Through-Thickness Refinement (Interlayer Shear Slip)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_title("(b) Through-Thickness Refinement: Interlayer Shear Slip", fontweight="bold")
    slip_vals = [c["shear_slip_um"] for c in th_cases]

    ax_b.plot(nz_vals, slip_vals, "s-", color="#ff7f0e", lw=2.0, ms=7, label="Shear Slip $\\Delta u$ [$\\mu$m]")
    for x, y in zip(nz_vals, slip_vals):
        ax_b.annotate(f"{y:.2f} $\\mu$m", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    ax_b.set_xlabel("Number of Elements per Layer (Thickness Direction)")
    ax_b.set_ylabel("Interlayer Shear Slip [$\\mu$m]", color="#ff7f0e", fontweight="bold")
    ax_b.set_xticks(nz_vals)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.set_ylim(min(slip_vals) * 0.95, max(slip_vals) * 1.05)
    ax_b.legend(loc="lower right")

    # -------------------------------------------------------------
    # Panel (c): True 3D Width Study (Anticlastic & Poisson Edge Effect)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_title("(c) True 3D Width Study: Poisson Anticlastic Curvature", fontweight="bold")
    ny_vals = [c["ny"] for c in w_cases]
    anticlastic_vals = [c["anticlastic_delta_uz_um"] for c in w_cases]
    w_defl_vals = [c["final_deflection_mm"] for c in w_cases]

    ax_c2 = ax_c.twinx()
    l1 = ax_c.plot(ny_vals, anticlastic_vals, "d-", color="#2ca02c", lw=2.0, ms=7, label="Anticlastic $\\Delta u_z$ [$\\mu$m]")
    l2 = ax_c2.plot(ny_vals, w_defl_vals, "^--", color="#d62728", lw=2.0, ms=7, label="3D Deflection [mm]")

    for x, y in zip(ny_vals, anticlastic_vals):
        ax_c.annotate(f"{y:.2f} $\\mu$m", (x, y), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, color="#2ca02c")

    ax_c.set_xlabel("Width Subdivisions $N_y$ (Width W = 1.0 mm)")
    ax_c.set_ylabel("Anticlastic Curvature $\\Delta u_z$ [$\\mu$m]", color="#2ca02c", fontweight="bold")
    ax_c2.set_ylabel("3D Tip Deflection [mm]", color="#d62728", fontweight="bold")
    ax_c.set_xticks(ny_vals)
    ax_c.grid(True, linestyle=":", alpha=0.6)

    lines_c = l1 + l2
    labels_c = [l.get_label() for l in lines_c]
    ax_c.legend(lines_c, labels_c, loc="center right")

    # -------------------------------------------------------------
    # Panel (d): Computational Performance Scaling (DOFs vs Time)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.set_title("(d) Computational Scaling: Pure Solve Time vs System DOFs", fontweight="bold")

    all_cases = th_cases + w_cases
    # Deduplicate by num_dofs
    seen_dofs = set()
    plot_cases = []
    for c in sorted(all_cases, key=lambda x: x["num_dofs"]):
        if c["num_dofs"] not in seen_dofs:
            seen_dofs.add(c["num_dofs"])
            plot_cases.append(c)

    dofs_all = [c["num_dofs"] for c in plot_cases]
    times_all = [c["t_wall_sec"] for c in plot_cases]
    elems_all = [c["num_elements"] for c in plot_cases]

    ax_d.plot(dofs_all, times_all, "p-", color="#9467bd", lw=2.2, ms=8, label="PARDISO + Numba DOD Time [s]")
    for x, y, el in zip(dofs_all, times_all, elems_all):
        ax_d.annotate(f"{y:.2f}s\n({el} el)", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    ax_d.set_xlabel("Total System Degrees of Freedom (DOFs)")
    ax_d.set_ylabel("Pure 15-Step Solve Time [s]", color="#9467bd", fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.set_ylim(0, max(times_all) * 1.25)
    ax_d.legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure Generated] Refinement figure saved to {output_path}")


if __name__ == "__main__":
    res_json = Path(__file__).resolve().parent / "results" / "benchmark_mesh_refinement_3d.json"
    out_fig = Path(__file__).resolve().parent.parent / "dev_log" / "figures" / "benchmark_mesh_refinement_3d.png"
    plot_mesh_refinement_figures(res_json, out_fig)
