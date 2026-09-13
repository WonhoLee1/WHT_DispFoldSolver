"""
figures_pure_moment_rollup.py
=============================
Publication-grade 4-panel visualizer for Pure Moment Roll-Up Benchmark.
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

R_THEORY = 40.0 / np.pi  # 12.732395 mm


def plot_pure_moment_rollup_figures(json_path: Path, output_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=300)
    fig.suptitle("5-Layer Composite Pure Bending Moment Roll-Up (180° Half-Circle / U-Turn)", fontweight="bold")

    Z_MID = 0.105  # mm
    s_vals = np.linspace(0, 40.0, 200)
    x_exact = R_THEORY * np.sin(s_vals / R_THEORY)
    y_exact = Z_MID + R_THEORY * (1.0 - np.cos(s_vals / R_THEORY))

    colors_2d = {
        "2D-CR   (CPE4_CR+CPE4_CR)": "#008080",
        "2D-Opt1 (CPE4I+CPE4H)": "#1f77b4",
        "2D-CPE6M (CPE6M+CPE4H)": "#9467bd",
        "2D-Opt2 (CPE4R+CPE4H)": "#2ca02c",
        "2D-Base (CPE4+CPE4)": "#d62728",
    }
    colors_3d = {
        "3D-CR   (C3D8_CR+C3D8_CR)": "#800080",
        "3D-Opt1 (C3D8I+C3D8H)": "#e377c2",
        "3D-Opt2 (C3D8R+C3D8R)": "#ff7f0e",
        "3D-Base (C3D8+C3D8)": "#8c564b",
    }

    # -------------------------------------------------------------
    # Panel (a): 2D Roll-Up Arc Profiles vs Theory
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_title("(a) 2D Elements: 180° Roll-Up Deformed Arcs", fontweight="bold")
    ax_a.plot(x_exact, y_exact, "k--", lw=2.2, label=f"Exact Theory (R={R_THEORY:.2f}mm)", zorder=5)

    for name, res in data["2D"].items():
        c = colors_2d.get(name, "blue")
        lbl = f"{name.split('(')[0].strip()} ($\\theta={res['max_theta_deg']:.0f}^\\circ$)"
        ax_a.plot(res["profile_x"], res["profile_y"], "-", color=c, lw=2.0, label=lbl)

    ax_a.set_xlabel("X Coordinate [mm]")
    ax_a.set_ylabel("Z Coordinate [mm]")
    ax_a.set_xlim(-5, 45)
    ax_a.set_ylim(-2, 28)
    ax_a.set_aspect("equal", "box")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="best", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (b): 3D Elements: 180° Roll-Up Deformed Arcs
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_title("(b) 3D Elements: 180° Roll-Up Deformed Arcs", fontweight="bold")
    ax_b.plot(x_exact, y_exact, "k--", lw=2.2, label=f"Exact Theory (R={R_THEORY:.2f}mm)", zorder=5)

    for name, res in data["3D"].items():
        c = colors_3d.get(name, "purple")
        lbl = f"{name.split('(')[0].strip()} ($\\theta={res['max_theta_deg']:.0f}^\\circ$)"
        ax_b.plot(res["profile_x"], res["profile_y"], "-", color=c, lw=2.0, label=lbl)

    ax_b.set_xlabel("X Coordinate [mm]")
    ax_b.set_ylabel("Z Coordinate [mm]")
    ax_b.set_xlim(-5, 45)
    ax_b.set_ylim(-2, 28)
    ax_b.set_aspect("equal", "box")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="best", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (c): Circularity Error & Mid-Span Shear Slip Comparison
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_title("(c) Circularity Error & Mid-Span Shear Slip (at x = 20 mm)", fontweight="bold")

    all_labels = list(data["2D"].keys()) + list(data["3D"].keys())
    short_labels = [l.split("(")[0].strip() for l in all_labels]
    circ_errors = [data["2D"][k]["circularity_rmse_pct"] for k in data["2D"]] + [data["3D"][k]["circularity_rmse_pct"] for k in data["3D"]]
    shear_slips = [data["2D"][k]["shear_slip_mid_um"] for k in data["2D"]] + [data["3D"][k]["shear_slip_mid_um"] for k in data["3D"]]

    x_idx = np.arange(len(all_labels))
    width = 0.38

    ax_c2 = ax_c.twinx()
    b1 = ax_c.bar(x_idx - width/2, circ_errors, width, label="Circularity RMSE [%]", color="#1f77b4", alpha=0.85, edgecolor="black")
    b2 = ax_c2.bar(x_idx + width/2, shear_slips, width, label="Mid-Span Shear Slip $\\Delta u$ [$\\mu$m]", color="#ff7f0e", alpha=0.85, edgecolor="black")

    ax_c.set_xticks(x_idx)
    ax_c.set_xticklabels(short_labels, rotation=35, ha="right")
    ax_c.set_ylabel("Circularity RMSE [%]", color="#1f77b4", fontweight="bold")
    ax_c2.set_ylabel("Mid-Span Shear Slip [$\\mu$m]", color="#ff7f0e", fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.6, axis="y")

    lines_c = [b1, b2]
    labels_c = [b1.get_label(), b2.get_label()]
    ax_c.legend(lines_c, labels_c, loc="best", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (d): Solve Time & Total Newton Iterations
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.set_title("(d) Computational Efficiency: Solve Time & Newton Iterations", fontweight="bold")

    times = [data["2D"][k]["t_wall_sec"] for k in data["2D"]] + [data["3D"][k]["t_wall_sec"] for k in data["3D"]]
    iters = [data["2D"][k]["total_iters"] for k in data["2D"]] + [data["3D"][k]["total_iters"] for k in data["3D"]]

    ax_d2 = ax_d.twinx()
    b3 = ax_d.bar(x_idx - width/2, times, width, label="Solve Time [s]", color="#2ca02c", alpha=0.85, edgecolor="black")
    b4 = ax_d2.bar(x_idx + width/2, iters, width, label="Total Newton Iterations", color="#9467bd", alpha=0.85, edgecolor="black")

    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels(short_labels, rotation=35, ha="right")
    ax_d.set_ylabel("Wall-clock Solve Time [s]", color="#2ca02c", fontweight="bold")
    ax_d2.set_ylabel("Total Newton Iterations", color="#9467bd", fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.6, axis="y")

    lines_d = [b3, b4]
    labels_d = [b3.get_label(), b4.get_label()]
    ax_d.legend(lines_d, labels_d, loc="best", framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure Generated] Pure moment roll-up figure saved to {output_path}")


if __name__ == "__main__":
    res_json = Path(__file__).resolve().parent / "results" / "benchmark_pure_moment_rollup.json"
    out_fig = Path(__file__).resolve().parent.parent / "dev_log" / "figures" / "benchmark_pure_moment_rollup.png"
    plot_pure_moment_rollup_figures(res_json, out_fig)
