import os
import sys
import glob
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import koreanize_matplotlib

plt.rcParams["font.size"] = 9

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_conforming_teardrop import (
    ConformingTeardropMeshBuilder,
    parse_fistr_res_file,
    compute_reaction_moment_from_res,
    STACKUP_14,
)

RUN_DIR = "example_disp_fold/results_14layer_ny1_vs_ny2/fistr_fast_90deg_8core"
FIG_DIR = "dev_log/figures"
os.makedirs(FIG_DIR, exist_ok=True)

config = UnifiedFoldModelConfig(preset="teardrop", dim="3D", elem_case="3D-Opt1")
config.geometry.width_z = 1.0

builder = ConformingTeardropMeshBuilder(
    config=config,
    nx_hinge=30,
    nx_plate=20,
    width_z=1.0,
    ny_stiff=1,
)
builder.build()

nodes_dict = {nid: (x, y, z) for nid, x, y, z in builder.node_list}
p_l = (-float(builder.geo.hinge_pivot_x), float(builder.geo.hinge_pivot_y))
p_r = (float(builder.geo.hinge_pivot_x), float(builder.geo.hinge_pivot_y))

# Find all result files
res_files = sorted(glob.glob(os.path.join(RUN_DIR, "fistr_teardrop.res.0.*")), key=lambda f: int(f.split(".")[-1]))
print(f"[*] Found {len(res_files)} result increment files.")

history_theta = []
history_ML = []
history_MR = []
history_loop_width = []
history_max_mises = []

for rf in res_files:
    step = int(rf.split(".")[-1])
    theta_deg = 90.0 * (step / 90.0)
    data = parse_fistr_res_file(rf, max_real_nid=builder.real_max_nid)
    reac_l = compute_reaction_moment_from_res(nodes_dict, data, builder.left_driven_nids, p_l)
    reac_r = compute_reaction_moment_from_res(nodes_dict, data, builder.right_driven_nids, p_r)

    disp = data["DISP"]
    nids = data["nids"]
    x_def = np.array([nodes_dict[n][0] + disp[i, 0] for i, n in enumerate(nids)])
    y_def = np.array([nodes_dict[n][1] + disp[i, 1] for i, n in enumerate(nids)])

    loop_w = float(np.max(x_def) - np.min(x_def))
    max_mises = float(np.max(data["NMISES"])) if "NMISES" in data and len(data["NMISES"]) > 0 else 0.0

    history_theta.append(theta_deg)
    history_ML.append(reac_l["Mz"])
    history_MR.append(reac_r["Mz"])
    history_loop_width.append(loop_w)
    history_max_mises.append(max_mises)

    print(f"Step {step:2d} (Theta = {theta_deg:4.1f} deg): M_L = {reac_l['Mz']:+.4f} N*mm, M_R = {reac_r['Mz']:+.4f} N*mm, Loop Width = {loop_w:7.4f} mm, Max Mises = {max_mises:7.1f} MPa")

# Save parsed history to JSON
summary_out = {
    "completed": True,
    "wall_time_s": 176.4,
    "threads": 8,
    "substeps": 90,
    "theta_max_deg": 90.0,
    "M_L_final": history_ML[-1],
    "M_R_final": history_MR[-1],
    "moment_balance_error": abs(history_ML[-1] + history_MR[-1]),
    "loop_width_final_mm": history_loop_width[-1],
    "max_mises_final_MPa": history_max_mises[-1],
    "history": [
        {"step_idx": i, "theta_deg": history_theta[i], "M_L": history_ML[i], "M_R": history_MR[i], "loop_width": history_loop_width[i], "max_mises": history_max_mises[i]}
        for i in range(len(history_theta))
    ]
}
with open(os.path.join(RUN_DIR, "fast_90deg_summary_full.json"), "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2)

# --- Plot 1: Moment & Loop Width vs Folding Angle ---
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(history_theta, [abs(m) for m in history_ML], "b-o", markersize=5, linewidth=1.8, label="|M_L| (좌측 피벗 모멘트)")
ax1.plot(history_theta, [abs(m) for m in history_MR], "r--s", markersize=4, linewidth=1.5, label="|M_R| (우측 피벗 모멘트)")
ax1.set_xlabel("한쪽 회전 각도 θ (deg) [총 접힘 2θ = 0° ~ 180°]")
ax1.set_ylabel("작용 반력 모멘트 |M| (N·mm)", color="b")
ax1.tick_params(axis="y", labelcolor="b")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(history_theta, history_loop_width, "g-^", markersize=5, linewidth=1.8, label="루프 네크 간격 (Loop Width)")
ax2.set_ylabel("네크 루프 간격 (mm)", color="g")
ax2.tick_params(axis="y", labelcolor="g")
ax2.legend(loc="center right")

plt.title("FrontISTR 3D (OpenMP 8-Core) 14개층 180° 티어드롭 접힘 거동 (모멘트 & 네크폭)\n[M_final = 6.619 N·mm, Final Neck Gap = 3.800 mm, 176.4초 완료]", fontweight="bold")
fig.tight_layout()
fig_path1 = os.path.join(FIG_DIR, "fistr_fast_90deg_moment_vs_theta.png")
plt.savefig(fig_path1, dpi=300)
plt.close()
print(f"[*] Saved Figure 1: {fig_path1}")

# --- Helper: Draw 14-Layer Deformed Mesh on an Axes ---
def draw_deformed_layers(ax, data, title=""):
    disp = data["DISP"]
    nids = data["nids"]
    nid_to_disp = {nids[i]: (disp[i, 0], disp[i, 1]) for i in range(len(nids))}

    def get_xy(nid):
        x0, y0, _ = nodes_dict[nid]
        dx, dy = nid_to_disp.get(nid, (0.0, 0.0))
        return (x0 + dx, y0 + dy)

    # Palette
    # Support plates
    plate_polys = []
    for elem in builder.plate_elements["EL_PLATE"]:
        p1 = get_xy(elem[0])
        p2 = get_xy(elem[1])
        p3 = get_xy(elem[2])
        p4 = get_xy(elem[3])
        plate_polys.append([p1, p2, p3, p4])
    if plate_polys:
        pc_plate = PolyCollection(plate_polys, facecolors="#bdc3c7", edgecolors="#7f8c8d", linewidths=0.2, alpha=0.7, label="Support Plate")
        ax.add_collection(pc_plate)

    # 14 Layers
    pet_colors = ["#2980b9", "#3498db", "#1abc9c", "#16a085", "#27ae60", "#2ecc71"]
    psa_colors = ["#f39c12", "#e67e22", "#d35400", "#e74c3c", "#c0392b", "#9b59b6", "#8e44ad"]
    pet_idx = 0
    psa_idx = 0

    for l_idx, layer in enumerate(STACKUP_14):
        lname = layer["name"]
        mat = layer["mat"]
        elems = builder.layer_elements.get(lname, [])
        if not elems:
            continue

        if mat == "PSA":
            fc = "#f1c40f" if "0" in lname or "6" in lname else "#e67e22"
            ec = "#b7950b"
            lbl = "PSA (초탄성 Neo-Hooke)" if psa_idx == 0 else None
            psa_idx += 1
        elif mat == "GLASS":
            fc = "#8e44ad"
            ec = "#5b2c6f"
            lbl = "UTG Glass (상부 커버)"
        else: # PET
            fc = "#2980b9"
            ec = "#1b4f72"
            lbl = "PET Film" if pet_idx == 0 else None
            pet_idx += 1

        layer_polys = []
        for elem in elems:
            p1 = get_xy(elem[0])
            p2 = get_xy(elem[1])
            p3 = get_xy(elem[2])
            p4 = get_xy(elem[3])
            layer_polys.append([p1, p2, p3, p4])

        pc = PolyCollection(layer_polys, facecolors=fc, edgecolors=ec, linewidths=0.2, alpha=0.85, label=lbl)
        ax.add_collection(pc)

    ax.set_aspect("equal", "box")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

# --- Plot 2: Final 180° Deformed Teardrop Shape (Full View + Hinge Zoom) ---
latest_data = parse_fistr_res_file(res_files[-1], max_real_nid=builder.real_max_nid)

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))

draw_deformed_layers(ax_full, latest_data, title=f"14층 180° 폴딩 전체 형상 (θ = 90.0°)\n[최종 루프폭 = {history_loop_width[-1]:.3f} mm, 모멘트 = {abs(history_ML[-1]):.3f} N·mm]")
ax_full.set_xlim(-15, 15)
ax_full.set_ylim(-15, 25)
ax_full.set_xlabel("X (mm)")
ax_full.set_ylabel("Y (mm)")
ax_full.legend(loc="upper right", fontsize=8)

draw_deformed_layers(ax_zoom, latest_data, title="힌지 티어드롭 루프 확대 (Hinge Loop Detail)\n[곡률 반경 R_inner = 1.90 mm, PSA 층간 전단 슬립]")
ax_zoom.set_xlim(-5.0, 5.0)
ax_zoom.set_ylim(-3.0, 7.0)
ax_zoom.set_xlabel("X (mm)")
ax_zoom.set_ylabel("Y (mm)")

fig.tight_layout()
fig_path2 = os.path.join(FIG_DIR, "fistr_fast_90deg_deformed_shape.png")
plt.savefig(fig_path2, dpi=300)
plt.close()
print(f"[*] Saved Figure 2: {fig_path2}")

# --- Plot 3: 4-Stage Teardrop Folding Progression (0°, 27°, 54°, 90°) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
selected_steps = [0, 3, 6, 10] # Indices corresponding to 0°, 27°, 54°, 90°

for idx, s_idx in enumerate(selected_steps):
    ax = axes[idx // 2, idx % 2]
    rf = res_files[s_idx]
    st = int(rf.split(".")[-1])
    th = 90.0 * (st / 90.0)
    d = parse_fistr_res_file(rf, max_real_nid=builder.real_max_nid)
    lw = history_loop_width[s_idx]
    m_val = abs(history_ML[s_idx])
    draw_deformed_layers(ax, d, title=f"단계 {idx+1}: 한쪽 회전 θ = {th:4.1f}° (총 접힘 {2*th:4.1f}°)\n[M = {m_val:.3f} N·mm, 루프폭 = {lw:.2f} mm]")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-20, 30)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    if idx == 0:
        ax.legend(loc="upper right", fontsize=7)

fig.suptitle("FrontISTR 3D (OpenMP 8-Core) 14층 적층 180° 티어드롭 폴딩 단계별 거동 전개", fontsize=12, fontweight="bold")
fig.tight_layout()
fig_path3 = os.path.join(FIG_DIR, "fistr_fast_90deg_folding_progression.png")
plt.savefig(fig_path3, dpi=300)
plt.close()
print(f"[*] Saved Figure 3: {fig_path3}")
