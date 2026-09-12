"""Example 17: Commercial Multi-Step CAE & Branching Tree Execution Demo

Demonstrates Abaqus-like Multi-Step Modeling & Branching Tree Execution:
1. Initial Step with Predefined Fields (Initial Velocity / Base Constraints)
2. Step 1 (Folding_Phase): Primary 3D folding deformation -> Captures Checkpoint 1
3. Branch A (Step 2-1: Springback_Unloading): Branches off Step 1, deactivates drive for springback relaxation
4. Branch B (Step 2-2: Crushing_Overload): Branches off Step 1, modifies loading for transverse crushing
5. Solves both branches independently and outputs distinct result bundles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

from dispsolver.model import Model, Part, GeneralSet, DisplacementBC, ConcentratedLoad, PredefinedField
from dispsolver.solver.step_executor import MultiStepExecutor

plt.rcParams["font.size"] = 9


def build_folding_display_model() -> Model:
    """Construct a 3D Foldable Display Panel Model with Multi-Step Tree configuration."""
    model = Model(name="MultiStepFoldingDisplay", dim=3)
    
    mat = model.Material(name="PET_Cover", mat_type="ELASTIC")
    mat.elastic = (4000.0, 0.3)
    sec = model.SolidSection(name="PET_Sec", material="PET_Cover")

    part = model.Part(name="DisplayPanel3D", dim=3)

    nx, ny, nz = 16, 2, 2
    x_coords = np.linspace(-20.0, 20.0, nx + 1)
    y_coords = np.linspace(0.0, 0.4, ny + 1)
    z_coords = np.linspace(0.0, 4.0, nz + 1)

    node_grid = {}
    nid = 1
    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                part.add_node(nid, [x, y, z])
                node_grid[(i, j, k)] = nid
                nid += 1

    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn = [
                    node_grid[(i, j, k)],
                    node_grid[(i + 1, j, k)],
                    node_grid[(i + 1, j + 1, k)],
                    node_grid[(i, j + 1, k)],
                    node_grid[(i, j, k + 1)],
                    node_grid[(i + 1, j, k + 1)],
                    node_grid[(i + 1, j + 1, k + 1)],
                    node_grid[(i, j + 1, k + 1)],
                ]
                part.add_element(eid, "C3D8", conn)
                eid += 1

    # Node sets
    left_nids = [nid for nid, c in part.nodes.items() if c[0] < -19.9]
    right_nids = [nid for nid, c in part.nodes.items() if c[0] > 19.9]
    top_nids = [nid for nid, c in part.nodes.items() if c[1] > 0.39]

    part.create_element_set("ALL_ELEMS", list(part.elements.keys()))
    part.create_node_set("LEFT_END", left_nids)
    part.create_node_set("RIGHT_END", right_nids)
    part.create_node_set("TOP_SURFACE", top_nids)
    part.assign_section("ALL_ELEMS", "PET_Sec")

    inst = model.root_assembly.Instance(name="DISP_1", part=part)

    # ─── Initial Step (t=0) ───────────────────────────────────────────
    model.initial_step.add_boundary_condition(
        DisplacementBC(name="LEFT_FIX", region="DISP_1.LEFT_END", u1=0.0, u2=0.0, u3=0.0)
    )
    model.PredefinedField(
        name="INIT_VEL", field_type="VELOCITY", region="ALL", values=[0.0, 0.0, 0.0]
    )

    # ─── Step 1: Primary Folding Phase ─────────────────────────────────
    step1 = model.Step(
        name="Step-1_Folding",
        previous="Initial",
        procedure="STATIC",
        time_period=0.2,
        dt_init=0.05
    )
    # Right end pushed down and left by bending drive
    bc_fold = DisplacementBC(
        name="RIGHT_FOLD_DRIVE", region="DISP_1.RIGHT_END", u1=-2.0, u2=-3.0, u3=0.0
    )
    step1.add_boundary_condition(bc_fold)

    # ─── Branch A (Step 2-1): Springback Unloading Scenario ──────────
    step2_1 = model.create_step_branch(
        branch_name="Step-2-1_Springback",
        parent_step_name="Step-1_Folding",
        procedure="STATIC",
        time_period=0.1,
        dt_init=0.05
    )
    # Deactivate folding drive to allow elastic/plastic springback relaxation
    step2_1.deactivate_bc("RIGHT_FOLD_DRIVE")

    # ─── Branch B (Step 2-2): Crushing Overload Scenario ────────────
    step2_2 = model.create_step_branch(
        branch_name="Step-2-2_Crushing",
        parent_step_name="Step-1_Folding",
        procedure="STATIC",
        time_period=0.1,
        dt_init=0.05
    )
    # Modify folding drive to apply intense secondary crushing displacement
    step2_2.modify_bc("RIGHT_FOLD_DRIVE", u1=-5.0, u2=-8.0)

    return model


def main():
    print("======================================================================")
    print("  Ex17: Commercial Multi-Step CAE & Branching Tree Execution Demo")
    print("======================================================================")

    model = build_folding_display_model()
    print(f"\n[Model Info] '{model.name}' constructed:")
    print(f"  - Registered Steps: {list(model.steps.keys())}")
    print(f"  - Initial Predefined Fields: {list(model.initial_step.predefined_fields.keys())}")

    out_dir = os.path.join(os.path.dirname(__file__), "branch_results")
    executor = MultiStepExecutor(model=model)
    results = executor.run_step_tree(save_dir=out_dir, verbose=True)

    print("\n" + "=" * 70)
    print("  BRANCHING STEP TREE EXECUTION SUMMARY")
    print("=" * 70)
    for branch_name, res in results.items():
        max_u = np.max(np.abs(res["displacement"]))
        print(f"  * Branch '{branch_name:22s}': Parent='{res['parent_step']:16s}', t_accum={res['accumulated_time']:.2f}s, Max |u|={max_u:.4f} mm")
        print(f"    Active BCs: {res['active_bcs']}")

    # ─── Visualization Comparison ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    steps_to_plot = [
        ("Step-1_Folding", "Step 1: 1차 굽힘 (Folding)", "blue"),
        ("Step-2-1_Springback", "Branch A: 스프링백 (Springback)", "green"),
        ("Step-2-2_Crushing", "Branch B: 추가 과하중 (Crushing)", "red"),
    ]

    sys = model.build_solver_system()

    for idx, (bname, title, col) in enumerate(steps_to_plot):
        ax = axes[idx]
        if bname in results:
            r = results[bname]
            u = r["displacement"]
            coords_deformed = sys.coords + u.reshape(-1, 3)
            
            # Plot 2D side projection (X vs Y)
            ax.scatter(sys.coords[:, 0], sys.coords[:, 1], c="lightgrey", s=10, alpha=0.4, label="초기 (Initial)")
            ax.scatter(coords_deformed[:, 0], coords_deformed[:, 1], c=col, s=15, label=f"변형 ({bname})")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_aspect("equal", "datalim")

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "ex17_multi_step_branching_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[Saved Branch Comparison Plot] {plot_path}")
    print("======================================================================")


if __name__ == "__main__":
    main()
