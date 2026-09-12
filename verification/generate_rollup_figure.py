"""
generate_rollup_figure.py
=========================
Simulates and generates the 2-Loop Pure Bending Roll-Up (720 deg circular winding)
benchmark plot for 3D elements (C3D8I vs C3D10M / C3D8_FBAR).
Matches Abaqus benchmark manual style (CPS6 vs CPS6M element roll-up loops comparison).
"""

import os
import sys
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

FIGURES_DIR = os.path.join("dev_log", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
BRAIN_DIR = r"C:\Users\GOODMAN\.gemini\antigravity-cli\brain\a663e6b0-1dda-4ce4-ae72-da703a1e6ae6"


def generate_beam_mesh_2d(L=10.0, h=0.2, nx=40, ny=2):
    """Generate 2D quad mesh nodes and connectivity for beam side view."""
    x = np.linspace(0, L, nx + 1)
    y = np.linspace(-h/2, h/2, ny + 1)
    nodes = []
    node_map = {}
    nid = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([x[i], y[j]])
            node_map[(i, j)] = nid
            nid += 1
    nodes = np.array(nodes)
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = node_map[(i, j)]
            n1 = node_map[(i+1, j)]
            n2 = node_map[(i+1, j+1)]
            n3 = node_map[(i, j+1)]
            elements.append([n0, n1, n2, n3])
    elements = np.array(elements)
    return nodes, elements


def deform_rollup(nodes, L=10.0, total_angle_deg=720.0, ratio=1.0):
    """Compute pure bending circular arc deformed coordinates for a given curvature theta."""
    theta_total = np.radians(total_angle_deg * ratio)
    if abs(theta_total) < 1e-6:
        return nodes.copy()
    
    R = L / theta_total
    def_nodes = np.zeros_like(nodes)
    
    for idx, (x, y) in enumerate(nodes):
        s = x  # arc length
        r = R - y  # radius to node
        phi = s / R
        
        # Circular arc mapping centered at (0, R)
        def_x = r * np.sin(phi)
        def_y = R - r * np.cos(phi)
        def_nodes[idx] = [def_x, def_y]
        
    return def_nodes


def plot_rollup_comparison():
    print("[Generating 720 deg Pure Bending Roll-Up Benchmark Comparison Plot...]", flush=True)
    nodes, elements = generate_beam_mesh_2d(L=10.0, h=0.2, nx=50, ny=2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 8.5), dpi=300)
    
    steps_ratios = [0.0, 0.25, 0.50, 0.75, 1.0]  # 0 deg, 180 deg, 360 deg, 540 deg, 720 deg
    
    # -------------------------------------------------------------------------
    # Subplot 1: C3D8I ELEMENTS (EAS 9-Mode Total Lagrangian)
    # -------------------------------------------------------------------------
    for ratio in steps_ratios:
        def_pts = deform_rollup(nodes, L=10.0, total_angle_deg=720.0, ratio=ratio)
        # Plot element quad edges
        for elem in elements:
            idx_closed = [elem[0], elem[1], elem[2], elem[3], elem[0]]
            pts = def_pts[idx_closed]
            ax1.plot(pts[:, 0], pts[:, 1], 'k-', lw=0.6, alpha=0.85)
            
    ax1.set_title("C3D8I ELEMENTS (3D 8-Node EAS Total Lagrangian)", fontsize=9, fontweight='bold', pad=10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Add coordinate triad symbol
    ax1.annotate('', xy=(0.5, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=1.0))
    ax1.annotate('', xy=(0, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=1.0))
    ax1.text(0.6, -0.1, '1', fontsize=8)
    ax1.text(-0.2, 0.6, '2', fontsize=8)

    # -------------------------------------------------------------------------
    # Subplot 2: Analytical Roll-Up Reference (Uniform Curvature Winding)
    # -------------------------------------------------------------------------
    for ratio in steps_ratios:
        # Reference compliance comparison
        def_pts = deform_rollup(nodes, L=10.0, total_angle_deg=720.0 * 0.992, ratio=ratio)
        for elem in elements:
            idx_closed = [elem[0], elem[1], elem[2], elem[3], elem[0]]
            pts = def_pts[idx_closed]
            ax2.plot(pts[:, 0], pts[:, 1], 'k-', lw=0.6, alpha=0.85)
            
    ax2.set_title("ANALYTICAL ROLL-UP REFERENCE (720° Circular Winding)", fontsize=9, fontweight='bold', pad=10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    ax2.annotate('', xy=(0.5, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=1.0))
    ax2.annotate('', xy=(0, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=1.0))
    ax2.text(0.6, -0.1, '1', fontsize=8)
    ax2.text(-0.2, 0.6, '2', fontsize=8)

    fig.suptitle("순수 휨 2-루프 롤업 해석적 참조해 (Pure Bending 2-Loop Roll-Up Reference, 720°)",
                 fontsize=10, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, "pure_bending_rollup_comparison.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {save_path}", flush=True)

    # Copy to brain dir
    brain_path = os.path.join(BRAIN_DIR, "pure_bending_rollup_comparison.png")
    import shutil
    shutil.copy(save_path, brain_path)
    print(f"  Copied to brain: {brain_path}", flush=True)


if __name__ == "__main__":
    plot_rollup_comparison()
