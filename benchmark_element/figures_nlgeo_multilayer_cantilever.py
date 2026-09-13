"""
figures_nlgeo_multilayer_cantilever.py
======================================
Publication-grade 4-panel visualizer for 5-layer composite cantilever benchmark.
Adheres strictly to project visualization guidelines:
- koreanize-matplotlib for font consistency.
- Global 9pt font size.
- tight_layout() to prevent overlaps.
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


def plot_multilayer_cantilever_figures(json_path: Path, output_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=300)
    fig.suptitle("5-Layer Composite (PET-PSA-PET-PSA-PET) Geometrically Nonlinear Benchmark", fontweight="bold")

    colors_2d = {
        "2D-Opt1 (CPE4I+CPE4H Conformal)": "#1f77b4",
        "2D-Tie  (CPE4I+CPE4H SurfaceTie)": "#17becf",
        "2D-Opt2 (CPE4R+CPE4H Conformal)": "#2ca02c",
        "2D-CPE6M (CPE6M+CPE4H Conformal)": "#9467bd",
        "2D-CPE6M-Tie (CPE6M+CPE4H SurfaceTie)": "#8c564b",
        "2D-Base (CPE4+CPE4 Baseline)": "#d62728"
    }
    colors_3d = {
        "3D-Opt1 (C3D8I+C3D8H Conformal)": "#9467bd",
        "3D-Tie  (C3D8I+C3D8H SurfaceTie)": "#e377c2",
        "3D-Opt2 (C3D8R+C3D8R Conformal)": "#ff7f0e",
        "3D-Base (C3D8+C3D8 Baseline)": "#8c564b"
    }

    # -------------------------------------------------------------
    # Panel (a): 2D Deformed Profiles
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_title("(a) 2D Elements: Deformed Beam Profiles (P = 1.5 mN)", fontweight="bold")
    ax_a.plot([0, 40], [0, 0], "k--", alpha=0.4, label="Initial Flat")
    for name, res in data["2D"].items():
        c = colors_2d.get(name, "gray")
        ls = "--" if "Tie" in name else "-"
        ax_a.plot(res["profile_x"], res["profile_y"], label=name.split("(")[0].strip(), color=c, lw=2.0, linestyle=ls)
    ax_a.set_xlabel("X Coordinate [mm]")
    ax_a.set_ylabel("Deformed Elevation [mm]")
    ax_a.set_xlim(-1, 41)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="best", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (b): 3D Elements Deformed Profiles
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_title("(b) 3D Elements: Deformed Beam Profiles (P = 1.5 mN)", fontweight="bold")
    ax_b.plot([0, 40], [0, 0], "k--", alpha=0.4, label="Initial Flat")
    for name, res in data["3D"].items():
        c = colors_3d.get(name, "gray")
        ls = "--" if "Tie" in name else "-"
        ax_b.plot(res["profile_x"], res["profile_y"], label=name.split("(")[0].strip(), color=c, lw=2.0, linestyle=ls)
    ax_b.set_xlabel("X Coordinate [mm]")
    ax_b.set_ylabel("Deformed Elevation [mm]")
    ax_b.set_xlim(-1, 41)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="best", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (c): Mechanical Metrics (Deflection & Interlayer Slip)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_title("(c) Tip Deflection (|w_tip|) and Free-End Shear Slip", fontweight="bold")
    all_names = list(data["2D"].keys()) + list(data["3D"].keys())
    short_labels = [n.split("(")[0].strip() for n in all_names]
    deflections = [abs(data["2D"][k]["final_uy"]) for k in data["2D"]] + [abs(data["3D"][k]["final_uy"]) for k in data["3D"]]
    slips = [data["2D"][k]["shear_slip_um"] for k in data["2D"]] + [data["3D"][k]["shear_slip_um"] for k in data["3D"]]

    x_idx = np.arange(len(short_labels))
    width = 0.38

    ax_c2 = ax_c.twinx()
    b1 = ax_c.bar(x_idx - width/2, deflections, width, label="Tip Deflection |w| [mm]", color="#1f77b4", alpha=0.85, edgecolor="black")
    b2 = ax_c2.bar(x_idx + width/2, slips, width, label="Shear Slip $\\Delta u$ [$\\mu$m]", color="#ff7f0e", alpha=0.85, edgecolor="black")

    ax_c.set_xticks(x_idx)
    ax_c.set_xticklabels(short_labels, rotation=35, ha="right")
    ax_c.set_ylabel("Tip Deflection [mm]", color="#1f77b4", fontweight="bold")
    ax_c2.set_ylabel("Interlayer Shear Slip [$\\mu$m]", color="#ff7f0e", fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Combined legend
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (d): Performance & Computational Cost (Solve Time & DOFs)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.set_title("(d) Computational Cost: Solve Time [s] & System DOFs", fontweight="bold")
    solve_times = [data["2D"][k]["t_wall_sec"] for k in data["2D"]] + [data["3D"][k]["t_wall_sec"] for k in data["3D"]]
    dofs = [data["2D"][k]["num_dofs"] for k in data["2D"]] + [data["3D"][k]["num_dofs"] for k in data["3D"]]

    ax_d2 = ax_d.twinx()
    b3 = ax_d.bar(x_idx - width/2, solve_times, width, label="Solve Time [s]", color="#2ca02c", alpha=0.85, edgecolor="black")
    b4 = ax_d2.bar(x_idx + width/2, dofs, width, label="Total DOFs", color="#9467bd", alpha=0.85, edgecolor="black")

    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels(short_labels, rotation=35, ha="right")
    ax_d.set_ylabel("Wall-clock Solve Time [s]", color="#2ca02c", fontweight="bold")
    ax_d2.set_ylabel("Total System DOFs", color="#9467bd", fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.6, axis="y")

    lines3, labels3 = ax_d.get_legend_handles_labels()
    lines4, labels4 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines3 + lines4, labels3 + labels4, loc="best", framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure Generated] Publication figure saved to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    json_file = base_dir / "results" / "benchmark_nlgeo_multilayer_cantilever.json"
    out_png = base_dir.parent / "dev_log" / "figures" / "benchmark_nlgeo_multilayer_cantilever.png"
    if json_file.exists():
        plot_multilayer_cantilever_figures(json_file, out_png)
    else:
        print(f"[Error] Results JSON not found at {json_file}. Run benchmark first!")
